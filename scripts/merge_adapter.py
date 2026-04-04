"""
Elumina — Merge LoRA Adapter into Base Model
Produces a standalone model that can be deployed without PEFT.
"""

import os
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_config(config_path: str = "./config/training_config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    config_path = os.environ.get("CONFIG_PATH", "./config/training_config.yaml")
    cfg = load_config(config_path)

    model_cfg = cfg["model"]
    output_cfg = cfg["output"]

    adapter_dir = output_cfg["adapter_dir"]
    merged_dir = output_cfg["merged_dir"]

    print(f"Base model: {model_cfg['name']}")
    print(f"Adapter: {adapter_dir}")
    print(f"Merged output: {merged_dir}")

    # Load tokenizer from adapter (may have modifications)
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)

    # Load base model in full precision for merging
    print("Loading base model (this may take a while)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )

    # Load and merge adapter
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_dir)

    print("Merging adapter into base model...")
    model = model.merge_and_unload()

    # Save merged model
    print(f"Saving merged model to {merged_dir}...")
    os.makedirs(merged_dir, exist_ok=True)
    model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)

    print(f"\nMerged model saved to: {merged_dir}")
    print("You can now load this model directly without PEFT:")
    print(f'  model = AutoModelForCausalLM.from_pretrained("{merged_dir}")')

    # Optional: push to hub
    if output_cfg.get("push_to_hub"):
        hub_id = output_cfg.get("hub_model_id", "intevia/elumina-27b")
        print(f"\nPushing to HuggingFace Hub: {hub_id}")
        model.push_to_hub(hub_id, safe_serialization=True)
        tokenizer.push_to_hub(hub_id)
        print("Push complete!")


if __name__ == "__main__":
    main()
