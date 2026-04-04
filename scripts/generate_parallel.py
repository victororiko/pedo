"""
Elumina — Parallel Synthetic Data Generator
Runs multiple topic categories simultaneously using separate API calls.
"""

import json
import os
import sys
import yaml
import random
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

GENERATOR_SYSTEM_PROMPT = """You are a dataset generator for an AI assistant called Elumina, made by Intevia Ltd in Nairobi, Kenya.

Your job is to create realistic, high-quality chat conversations about Kenya.

Rules:
1. Create natural user/assistant conversation pairs
2. The assistant (Elumina) should be knowledgeable, helpful, and grounded in Kenyan context
3. Mix English and Swahili naturally where appropriate
4. Include Sheng (Kenyan slang) in casual conversations
5. Be factually accurate
6. Create DIVERSE question types: factual, opinion-seeking, how-to, comparison, explanation
7. Generate 3-5 conversation pairs per topic
8. Each conversation should be 2-6 messages long
9. When identity questions arise naturally, Elumina identifies as being from Intevia Ltd, Nairobi
10. Use Kenyan cultural references and local examples

Output format: Return a JSON array of conversations.
Example: [{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}]
Return ONLY the JSON array, no markdown fencing."""

TOPIC_PROMPT = """Generate {count} diverse, realistic chat conversations about:

Topic: {topic}
Category: {category}

Requirements:
- Mix of simple and complex questions
- Include some Swahili/Sheng where natural
- Accurate, detailed responses
- Some multi-turn conversations

Return a JSON array of conversation objects with "messages" arrays."""


def call_anthropic(prompt, api_key, model="claude-sonnet-4-20250514", max_tokens=2048):
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    for attempt in range(3):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=GENERATOR_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
            )
            return response.content[0].text
        except Exception as e:
            logger.warning(f"API call failed (attempt {attempt+1}): {e}")
            if "rate_limit" in str(e).lower() or "429" in str(e):
                time.sleep(30 * (attempt + 1))
            else:
                time.sleep(5 * (attempt + 1))
    return None


def parse_conversations(text):
    import re
    if not text:
        return []
    text = text.strip()
    
    # Strip markdown code fencing (multiple formats Claude uses)
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    text = text.strip()
    
    # Try direct parse
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict) and "messages" in item
                    and isinstance(item["messages"], list) and len(item["messages"]) >= 2]
        elif isinstance(data, dict) and "messages" in data:
            return [data]
    except json.JSONDecodeError:
        pass
    
    # Try extracting JSON array from surrounding text
    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return [item for item in data if isinstance(item, dict) and "messages" in item]
        except json.JSONDecodeError:
            pass
    
    # Try extracting individual JSON objects line by line
    conversations = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("{") and "messages" in line:
            try:
                obj = json.loads(line)
                if "messages" in obj:
                    conversations.append(obj)
            except json.JSONDecodeError:
                pass
    if conversations:
        return conversations
    
    logger.warning(f"Failed to parse LLM output. First 200 chars: {text[:200]}")
    return []


def generate_category(category, topics, api_key, output_dir):
    """Generate conversations for one category — runs in its own thread."""
    conversations = []
    for topic in topics:
        prompt = TOPIC_PROMPT.format(
            count=random.randint(3, 6),
            topic=topic,
            category=category.replace("_", " ").title(),
        )
        result = call_anthropic(prompt, api_key)
        convos = parse_conversations(result)
        for conv in convos:
            conv["source"] = f"synthetic/{category}"
            conv["topic"] = topic
        conversations.extend(convos)
        logger.info(f"  [{category}] \"{topic}\": {len(convos)} conversations")
        time.sleep(0.5)

    # Save immediately after category completes
    outfile = os.path.join(output_dir, f"synthetic_{category}.jsonl")
    with open(outfile, "w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
    logger.info(f"  [{category}] DONE: {len(conversations)} total -> {outfile}")
    return category, len(conversations)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/scraping_config.yaml")
    parser.add_argument("--output-dir", default="./data/cultural")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers (API calls)")
    parser.add_argument("--categories", nargs="*", default=None, help="Only generate these categories")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("Set ANTHROPIC_API_KEY in .env or environment")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    all_topics = cfg["synthetic"]["topic_categories"]

    # Filter categories if specified
    if args.categories:
        all_topics = {k: v for k, v in all_topics.items() if k in args.categories}

    # Skip categories that already have synthetic files
    to_generate = {}
    for cat, topics in all_topics.items():
        outfile = os.path.join(args.output_dir, f"synthetic_{cat}.jsonl")
        if os.path.exists(outfile):
            count = sum(1 for line in open(outfile) if line.strip())
            logger.info(f"SKIP {cat}: already has {count} conversations in {outfile}")
        else:
            to_generate[cat] = topics

    if not to_generate:
        logger.info("All categories already generated. Delete synthetic_*.jsonl files to regenerate.")
        return

    total_topics = sum(len(v) for v in to_generate.values())
    logger.info(f"\nGenerating {total_topics} topics across {len(to_generate)} categories with {args.workers} workers\n")

    total_convos = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(generate_category, cat, topics, api_key, args.output_dir): cat
            for cat, topics in to_generate.items()
        }
        for future in as_completed(futures):
            cat = futures[future]
            try:
                name, count = future.result()
                total_convos += count
                logger.info(f"COMPLETED: {name} ({count} conversations)")
            except Exception as e:
                logger.error(f"FAILED: {cat}: {e}")

    logger.info(f"\n{'='*60}")
    logger.info(f"TOTAL: {total_convos} new conversations generated")
    logger.info(f"Output: {args.output_dir}/synthetic_*.jsonl")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
