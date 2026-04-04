"""
Elumina — QLoRA Fine-tuning Script
Fine-tunes Gemma 4 27B with QLoRA on RunPod.
"""

import os
import yaml
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer


def load_config(config_path: str = "./config/training_config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_quantization(cfg: dict) -> BitsAndBytesConfig:
    qlora = cfg["qlora"]
    compute_dtype = getattr(torch, qlora["bnb_4bit_compute_dtype"])

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=qlora["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=qlora["bnb_4bit_use_double_quant"],
    )


def setup_lora(cfg: dict) -> LoraConfig:
    qlora = cfg["qlora"]

    return LoraConfig(
        r=qlora["r"],
        lora_alpha=qlora["lora_alpha"],
        lora_dropout=qlora["lora_dropout"],
        target_modules=qlora["target_modules"],
        bias=qlora["bias"],
        task_type=qlora["task_type"],
    )


def main():
    # Load config
    config_path = os.environ.get("CONFIG_PATH", "./config/training_config.yaml")
    cfg = load_config(config_path)

    model_cfg = cfg["model"]
    training_cfg = cfg["training"]
    data_cfg = cfg["data"]
    output_cfg = cfg["output"]

    print(f"Loading model: {model_cfg['name']}")
    print(f"Training data: {data_cfg['train_file']}")
    print(f"Output: {training_cfg['output_dir']}")

    # Setup quantization
    bnb_config = setup_quantization(cfg)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name"],
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        quantization_config=bnb_config,
        torch_dtype=getattr(torch, model_cfg["torch_dtype"]),
        attn_implementation=model_cfg.get("attn_implementation", "flash_attention_2"),
        device_map="auto",
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )

    # Prepare for QLoRA training
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    # Gemma 4 uses Gemma4ClippableLinear wrappers around Linear4bit.
    # PEFT doesn't recognize these, so unwrap them before applying LoRA.
    import torch.nn as nn
    for name, module in model.named_modules():
        # Replace ClippableLinear wrappers with their inner Linear4bit layer
        if type(module).__name__ == "Gemma4ClippableLinear":
            inner = module.linear
            # Navigate to parent module and replace
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent = model.get_submodule(parts[0])
                setattr(parent, parts[1], inner)

    # Apply LoRA
    lora_config = setup_lora(cfg)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    dataset = load_dataset(
        "json",
        data_files={
            "train": data_cfg["train_file"],
            "eval": data_cfg["eval_file"],
        },
    )

    print(f"Train samples: {len(dataset['train'])}")
    print(f"Eval samples: {len(dataset['eval'])}")

    # Limit samples if configured
    max_samples = data_cfg.get("max_samples")
    if max_samples:
        dataset["train"] = dataset["train"].select(range(min(max_samples, len(dataset["train"]))))

    # Training arguments
    training_args = TrainingArguments(
        output_dir=training_cfg["output_dir"],
        num_train_epochs=training_cfg["num_train_epochs"],
        per_device_train_batch_size=training_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=training_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
        gradient_checkpointing=training_cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=training_cfg["learning_rate"],
        lr_scheduler_type=training_cfg["lr_scheduler_type"],
        warmup_ratio=training_cfg["warmup_ratio"],
        weight_decay=training_cfg["weight_decay"],
        max_grad_norm=training_cfg["max_grad_norm"],
        fp16=training_cfg["fp16"],
        bf16=training_cfg["bf16"],
        logging_steps=training_cfg["logging_steps"],
        save_steps=training_cfg["save_steps"],
        eval_steps=training_cfg["eval_steps"],
        eval_strategy=training_cfg["eval_strategy"],
        save_total_limit=training_cfg["save_total_limit"],
        load_best_model_at_end=training_cfg["load_best_model_at_end"],
        metric_for_best_model=training_cfg["metric_for_best_model"],
        greater_is_better=training_cfg["greater_is_better"],
        report_to=training_cfg["report_to"],
        seed=training_cfg["seed"],
        dataloader_num_workers=training_cfg["dataloader_num_workers"],
        optim=training_cfg["optim"],
        group_by_length=training_cfg["group_by_length"],
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        max_seq_length=training_cfg["max_seq_length"],
        packing=training_cfg.get("packing", True),
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting QLoRA fine-tuning...")
    print("=" * 60 + "\n")

    train_result = trainer.train()

    # Log metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # Evaluate
    print("\nRunning evaluation...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    # Save adapter
    adapter_dir = output_cfg["adapter_dir"]
    print(f"\nSaving LoRA adapter to {adapter_dir}")
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    print("\nTraining complete!")
    print(f"  Adapter saved to: {adapter_dir}")
    print(f"  Run merge_adapter.py to create the full merged model")


if __name__ == "__main__":
    main()
