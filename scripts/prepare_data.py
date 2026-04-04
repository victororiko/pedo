"""
Elumina — Data Preparation Script
Merges identity, cultural, and raw datasets into train/eval splits.
"""

import json
import random
import os
from pathlib import Path
from typing import Optional


def load_jsonl(filepath: str) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  Warning: Skipping malformed line {line_num} in {filepath}: {e}")
    return data


def validate_conversation(item: dict) -> bool:
    """Validate that a conversation has the expected format."""
    if "messages" not in item:
        return False
    messages = item["messages"]
    if not isinstance(messages, list) or len(messages) < 2:
        return False
    for msg in messages:
        if "role" not in msg or "content" not in msg:
            return False
        if msg["role"] not in ("system", "user", "assistant"):
            return False
        if not msg["content"].strip():
            return False
    return True


def oversample(data: list[dict], factor: int) -> list[dict]:
    """Repeat data entries to increase their weight during training."""
    return data * factor


def collect_jsonl_files(directory: str) -> list[str]:
    """Recursively find all .jsonl files in a directory."""
    path = Path(directory)
    if not path.exists():
        return []
    return [str(f) for f in path.rglob("*.jsonl")]


def prepare_dataset(
    identity_dir: str = "./data/identity",
    cultural_dir: str = "./data/cultural",
    raw_dir: str = "./data/raw",
    output_dir: str = "./data/processed",
    eval_ratio: float = 0.05,
    identity_oversample: int = 5,
    seed: int = 42,
    max_samples: Optional[int] = None,
):
    """
    Merge all datasets, validate, shuffle, and split into train/eval.

    Identity data is oversampled to ensure the model strongly learns its identity.
    """
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    all_data = []

    # 1. Load identity data (oversampled)
    print("Loading identity data...")
    identity_files = collect_jsonl_files(identity_dir)
    identity_data = []
    for f in identity_files:
        loaded = load_jsonl(f)
        print(f"  {f}: {len(loaded)} conversations")
        identity_data.extend(loaded)

    if identity_data:
        oversampled = oversample(identity_data, identity_oversample)
        print(f"  Identity data oversampled {identity_oversample}x: {len(identity_data)} -> {len(oversampled)}")
        all_data.extend(oversampled)
    else:
        print("  Warning: No identity data found!")

    # 2. Load cultural data
    print("\nLoading cultural data...")
    cultural_files = collect_jsonl_files(cultural_dir)
    for f in cultural_files:
        loaded = load_jsonl(f)
        print(f"  {f}: {len(loaded)} conversations")
        all_data.extend(loaded)

    if not cultural_files:
        print("  No cultural data found yet (add .jsonl files to data/cultural/)")

    # 3. Load raw data
    print("\nLoading raw data...")
    raw_files = collect_jsonl_files(raw_dir)
    for f in raw_files:
        loaded = load_jsonl(f)
        print(f"  {f}: {len(loaded)} conversations")
        all_data.extend(loaded)

    if not raw_files:
        print("  No raw data found yet (add .jsonl files to data/raw/)")

    # 4. Validate
    print(f"\nValidating {len(all_data)} total conversations...")
    valid_data = [item for item in all_data if validate_conversation(item)]
    invalid_count = len(all_data) - len(valid_data)
    if invalid_count > 0:
        print(f"  Removed {invalid_count} invalid conversations")
    print(f"  {len(valid_data)} valid conversations")

    if not valid_data:
        print("\nError: No valid training data! Add data to data/ directories.")
        return

    # 5. Optional limit
    if max_samples and len(valid_data) > max_samples:
        random.shuffle(valid_data)
        valid_data = valid_data[:max_samples]
        print(f"  Limited to {max_samples} samples")

    # 6. Shuffle and split
    random.shuffle(valid_data)
    eval_size = max(1, int(len(valid_data) * eval_ratio))
    eval_data = valid_data[:eval_size]
    train_data = valid_data[eval_size:]

    print(f"\nSplit: {len(train_data)} train / {len(eval_data)} eval")

    # 7. Write output
    train_path = os.path.join(output_dir, "train.jsonl")
    eval_path = os.path.join(output_dir, "eval.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(eval_path, "w", encoding="utf-8") as f:
        for item in eval_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nOutput written:")
    print(f"  Train: {train_path} ({len(train_data)} conversations)")
    print(f"  Eval:  {eval_path} ({len(eval_data)} conversations)")

    # 8. Stats
    print("\n--- Dataset Statistics ---")
    total_messages = sum(len(item["messages"]) for item in train_data + eval_data)
    total_chars = sum(
        len(msg["content"])
        for item in train_data + eval_data
        for msg in item["messages"]
    )
    print(f"  Total conversations: {len(train_data) + len(eval_data)}")
    print(f"  Total messages: {total_messages}")
    print(f"  Total characters: {total_chars:,}")
    print(f"  Avg messages/conversation: {total_messages / len(valid_data):.1f}")
    print(f"  Avg chars/message: {total_chars / total_messages:.0f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare Elumina training data")
    parser.add_argument("--identity-dir", default="./data/identity")
    parser.add_argument("--cultural-dir", default="./data/cultural")
    parser.add_argument("--raw-dir", default="./data/raw")
    parser.add_argument("--output-dir", default="./data/processed")
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--identity-oversample", type=int, default=20,
                        help="How many times to repeat identity data. 10-20x recommended for small datasets, 30-50x if synthetic data is large")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)

    args = parser.parse_args()

    prepare_dataset(
        identity_dir=args.identity_dir,
        cultural_dir=args.cultural_dir,
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        eval_ratio=args.eval_ratio,
        identity_oversample=args.identity_oversample,
        seed=args.seed,
        max_samples=args.max_samples,
    )
