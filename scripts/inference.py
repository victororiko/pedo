"""
Elumina — Inference & Testing Script
Interactive chat with the fine-tuned model.
"""

import os
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_config(config_path: str = "./config/training_config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model(model_path: str, use_quantized: bool = True):
    """Load the model for inference."""
    print(f"Loading model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
    }

    if use_quantized:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    messages: list[dict],
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> str:
    """Generate a response given a conversation history."""
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    # Decode only the new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response.strip()


def run_identity_tests(model, tokenizer):
    """Run pre-defined identity tests to verify fine-tuning."""
    print("\n" + "=" * 60)
    print("IDENTITY TESTS")
    print("=" * 60)

    test_prompts = [
        "Who are you?",
        "What is your name?",
        "Who created you?",
        "Where are you from?",
        "Wewe ni nani?",
        "Are you ChatGPT?",
        "Are you made by Google?",
        "Sasa, uko aje?",
    ]

    for prompt in test_prompts:
        messages = [{"role": "user", "content": prompt}]
        response = generate_response(model, tokenizer, messages)
        print(f"\nUser: {prompt}")
        print(f"Elumina: {response}")
        print("-" * 40)


def interactive_chat(model, tokenizer):
    """Interactive chat loop."""
    print("\n" + "=" * 60)
    print("ELUMINA — Interactive Chat")
    print("Type 'quit' to exit, 'clear' to reset conversation")
    print("=" * 60)

    messages = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nKwaheri! (Goodbye!)")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Kwaheri! (Goodbye!)")
            break
        if user_input.lower() == "clear":
            messages = []
            print("Conversation cleared.")
            continue

        messages.append({"role": "user", "content": user_input})
        response = generate_response(model, tokenizer, messages)
        messages.append({"role": "assistant", "content": response})

        print(f"\nElumina: {response}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test Elumina model")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to model (merged or adapter). Defaults to merged from config.",
    )
    parser.add_argument(
        "--use-adapter",
        action="store_true",
        help="Load as LoRA adapter on base model instead of merged model",
    )
    parser.add_argument(
        "--identity-test",
        action="store_true",
        help="Run identity verification tests",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Load in full precision (needs more VRAM)",
    )

    args = parser.parse_args()

    cfg = load_config()

    if args.model_path:
        model_path = args.model_path
    elif args.use_adapter:
        model_path = cfg["output"]["adapter_dir"]
    else:
        model_path = cfg["output"]["merged_dir"]

    if args.use_adapter:
        # Load base + adapter
        from peft import PeftModel

        base_model, tokenizer = load_model(
            cfg["model"]["name"], use_quantized=not args.no_quantize
        )
        print(f"Loading adapter from {model_path}...")
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        model, tokenizer = load_model(model_path, use_quantized=not args.no_quantize)

    if args.identity_test:
        run_identity_tests(model, tokenizer)
    else:
        run_identity_tests(model, tokenizer)
        interactive_chat(model, tokenizer)


if __name__ == "__main__":
    main()
