"""
Elumina — Synthetic Training Data Generator
Converts scraped raw text into high-quality chat conversations using an LLM API.
Supports OpenAI and Anthropic APIs.
"""

import json
import os
import re
import time
import random
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
except ImportError:
    pass  # dotenv not installed, rely on environment variables

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# System prompt for the generator LLM
GENERATOR_SYSTEM_PROMPT = """You are a dataset generator for an AI assistant called Elumina, made by Intevia Ltd in Nairobi, Kenya.

Your job is to create realistic, high-quality chat conversations based on provided source material about Kenya.

Rules:
1. Create natural user/assistant conversation pairs
2. The assistant (Elumina) should be knowledgeable, helpful, and grounded in Kenyan context
3. Mix English and Swahili naturally where appropriate (code-switching is common in Kenya)
4. Include Sheng (Kenyan slang) in casual conversations
5. Be factually accurate based on the source material
6. Create DIVERSE question types: factual, opinion-seeking, how-to, comparison, explanation
7. Generate 3-5 conversation pairs per source text
8. Each conversation should be 2-6 messages long (multi-turn)
9. When identity questions arise naturally, Elumina identifies as being from Intevia Ltd, Nairobi
10. Use Kenyan cultural references, local examples, and relatable scenarios

Output format: Return a JSON array of conversations. Each conversation is an object with a "messages" array.
Example:
[
  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]},
  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
]

Return ONLY the JSON array, no markdown fencing or explanation."""

# Topic-specific generation prompts
TOPIC_GENERATION_PROMPT = """Generate {count} diverse, realistic chat conversations about the following Kenyan topic:

Topic: {topic}
Category: {category}

Requirements:
- Mix of simple and complex questions
- Include some Swahili/Sheng where natural
- Cover different aspects of the topic
- Some multi-turn conversations (follow-up questions)
- Accurate, detailed responses
- Elumina should reference Kenyan context naturally

Return a JSON array of conversation objects with "messages" arrays."""

# Article-to-conversation prompt
ARTICLE_CONVERSION_PROMPT = """Convert this article/text into {count} training conversations for an AI assistant:

Source: {source}
Title: {title}

Text:
{text}

Generate diverse Q&A pairs covering the key information. Include:
- Factual questions about the content
- "Explain this to me" style questions
- Practical/applied questions
- At least one multi-turn conversation
- Natural language, mixing English and Swahili where appropriate

Return a JSON array of conversation objects with "messages" arrays."""


class SyntheticGenerator:
    """Generate synthetic training conversations using an LLM API."""

    def __init__(self, config_path: str = "./config/scraping_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        synth_cfg = self.config.get("synthetic", {})
        self.provider = synth_cfg.get("provider", "openai")
        self.model = synth_cfg.get("model", "gpt-4o-mini")
        self.temperature = synth_cfg.get("temperature", 0.8)
        self.max_tokens = synth_cfg.get("max_tokens", 2048)
        self.batch_size = synth_cfg.get("batch_size", 10)
        self.target_conversations = synth_cfg.get("target_conversations", 10000)

        self.api_key = None
        self._setup_api()

    def _setup_api(self):
        if self.provider == "openai":
            self.api_key = os.environ.get("OPENAI_API_KEY")
            if not self.api_key:
                logger.error("OPENAI_API_KEY not set. Export it: export OPENAI_API_KEY='sk-...'")
                return
            try:
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                logger.error("pip install openai")
                return

        elif self.provider == "anthropic":
            self.api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not self.api_key:
                logger.error("ANTHROPIC_API_KEY not set.")
                return
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                logger.error("pip install anthropic")
                return

    def _call_llm(self, prompt: str) -> str | None:
        """Call the LLM API with retry logic."""
        for attempt in range(3):
            try:
                if self.provider == "openai":
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                    return response.choices[0].message.content

                elif self.provider == "anthropic":
                    response = self.client.messages.create(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        system=GENERATOR_SYSTEM_PROMPT,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                    )
                    return response.content[0].text

            except Exception as e:
                logger.warning(f"API call failed (attempt {attempt + 1}): {e}")
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    time.sleep(30 * (attempt + 1))
                else:
                    time.sleep(5 * (attempt + 1))

        return None

    def _parse_conversations(self, text: str) -> list[dict]:
        """Parse LLM output into conversation objects."""
        if not text:
            return []

        text = text.strip()

        # Strip markdown fencing anywhere in the text
        text = re.sub(r"```json\s*\n?", "", text)
        text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)

        # Try to find the JSON array boundaries
        start = text.find("[")
        if start == -1:
            logger.warning("Failed to parse LLM output as conversations (no JSON array found)")
            return []

        text = text[start:]

        # Try direct parse first
        try:
            data = json.loads(text)
            return self._validate_conversations(data)
        except json.JSONDecodeError:
            pass

        # Handle truncated JSON: try to repair by closing open structures
        repaired = self._repair_truncated_json(text)
        if repaired:
            try:
                data = json.loads(repaired)
                return self._validate_conversations(data)
            except json.JSONDecodeError:
                pass

        # Last resort: extract individual conversation objects via regex
        convos = []
        pattern = r'\{"messages"\s*:\s*\[.*?\]\s*(?:,\s*"[^"]+"\s*:\s*(?:"[^"]*"|[^,}]*))*\}'
        for match in re.finditer(pattern, text, re.DOTALL):
            try:
                obj = json.loads(match.group())
                if "messages" in obj and len(obj["messages"]) >= 2:
                    convos.append(obj)
            except json.JSONDecodeError:
                continue

        if convos:
            return convos

        logger.warning("Failed to parse LLM output as conversations")
        if len(text) > 200:
            logger.debug(f"  First 200 chars: {text[:200]}")
            logger.debug(f"  Last 200 chars: {text[-200:]}")
        return []

    def _repair_truncated_json(self, text: str) -> str | None:
        """Attempt to repair truncated JSON by closing open brackets/braces."""
        # Find the last complete conversation object
        # Look for the pattern: }, { or }, ] that indicates object boundaries
        last_complete = -1
        depth_brace = 0
        depth_bracket = 0
        in_string = False
        escape = False

        for i, ch in enumerate(text):
            if escape:
                escape = False
                continue
            if ch == '\\' and in_string:
                escape = True
                continue
            if ch == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue

            if ch == '{':
                depth_brace += 1
            elif ch == '}':
                depth_brace -= 1
                if depth_brace == 0 and depth_bracket == 1:
                    # Just closed a top-level object inside the array
                    last_complete = i
            elif ch == '[':
                depth_bracket += 1
            elif ch == ']':
                depth_bracket -= 1

        if last_complete > 0:
            return text[:last_complete + 1] + "]"
        return None

    def _validate_conversations(self, data) -> list[dict]:
        """Validate and filter conversation objects."""
        if not isinstance(data, list):
            return []
        valid = []
        for item in data:
            if isinstance(item, dict) and "messages" in item:
                messages = item["messages"]
                if isinstance(messages, list) and len(messages) >= 2:
                    valid.append(item)
        return valid

    def generate_from_topics(self, output_path: str) -> list[dict]:
        """Generate conversations from configured topic categories."""
        topics = self.config.get("synthetic", {}).get("topic_categories", {})
        all_conversations = []

        total_topics = sum(len(items) for items in topics.values())
        logger.info(f"Generating conversations from {total_topics} topics across {len(topics)} categories...")

        for category, topic_list in topics.items():
            logger.info(f"\n--- {category} ({len(topic_list)} topics) ---")

            for topic in topic_list:
                prompt = TOPIC_GENERATION_PROMPT.format(
                    count=random.randint(3, 6),
                    topic=topic,
                    category=category.replace("_", " ").title(),
                )

                result = self._call_llm(prompt)
                conversations = self._parse_conversations(result)

                for conv in conversations:
                    conv["source"] = f"synthetic/{category}"
                    conv["topic"] = topic

                all_conversations.extend(conversations)
                logger.info(f"  [{category}] \"{topic}\": {len(conversations)} conversations")

                # Small delay between API calls
                time.sleep(0.5)

        # Save
        with open(output_path, "w", encoding="utf-8") as f:
            for conv in all_conversations:
                f.write(json.dumps(conv, ensure_ascii=False) + "\n")

        logger.info(f"\nTopic-based: {len(all_conversations)} conversations -> {output_path}")
        return all_conversations

    def generate_from_articles(self, articles_dir: str, output_path: str,
                                max_articles: int = None) -> list[dict]:
        """Convert scraped articles into training conversations."""
        article_files = list(Path(articles_dir).rglob("*.jsonl"))
        all_conversations = []

        logger.info(f"Converting articles from {len(article_files)} files...")

        articles = []
        for filepath in article_files:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            articles.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

        if max_articles:
            random.shuffle(articles)
            articles = articles[:max_articles]

        logger.info(f"Processing {len(articles)} articles...")

        for i, article in enumerate(articles):
            title = article.get("title", "")
            text = article.get("text", "")
            source = article.get("source", "web")

            if not text or len(text) < 200:
                continue

            # Truncate very long articles
            if len(text) > 3000:
                text = text[:3000] + "..."

            prompt = ARTICLE_CONVERSION_PROMPT.format(
                count=random.randint(3, 5),
                source=source,
                title=title,
                text=text,
            )

            result = self._call_llm(prompt)
            conversations = self._parse_conversations(result)

            for conv in conversations:
                conv["source"] = f"article/{source}"
                conv["original_title"] = title

            all_conversations.extend(conversations)

            if (i + 1) % 20 == 0:
                logger.info(f"  Processed {i + 1}/{len(articles)} articles, {len(all_conversations)} conversations so far")
                # Intermediate save
                with open(output_path, "w", encoding="utf-8") as f:
                    for conv in all_conversations:
                        f.write(json.dumps(conv, ensure_ascii=False) + "\n")

            time.sleep(0.5)

        # Final save
        with open(output_path, "w", encoding="utf-8") as f:
            for conv in all_conversations:
                f.write(json.dumps(conv, ensure_ascii=False) + "\n")

        logger.info(f"\nArticle-based: {len(all_conversations)} conversations -> {output_path}")
        return all_conversations

    def generate_swahili_pairs(self, output_path: str, count: int = 500) -> list[dict]:
        """Generate Swahili/English conversation pairs for language training."""
        prompts_batch = [
            'Generate {n} conversations where a user asks something in Swahili and Elumina responds in Swahili. Topics: daily life, technology, news, culture. The conversations should be natural Kenyan Swahili (not formal/Tanzanian Swahili). Include some Sheng slang.\n\nReturn ONLY a JSON array: [{{"messages": [{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}]}}]',
            'Generate {n} conversations where a user code-switches between English and Swahili (common in Kenya), and Elumina responds matching their language style. Topics: work, school, relationships, money, transport.\n\nReturn ONLY a JSON array: [{{"messages": [{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}]}}]',
            'Generate {n} conversations where a user asks in English about Swahili language: grammar questions, translations, proverbs, idioms. Elumina explains in English with Swahili examples.\n\nReturn ONLY a JSON array: [{{"messages": [{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}]}}]',
            'Generate {n} conversations in pure Sheng (Kenyan street slang mixing Swahili/English/other languages). Topics: matatu culture, music, food, social life, money.\n\nReturn ONLY a JSON array: [{{"messages": [{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}]}}]',
            'Generate {n} conversations where Elumina teaches Kikuyu, Luo, Kalenjin, or Luhya phrases. User asks how to say things in these languages and Elumina provides translations with cultural context.\n\nReturn ONLY a JSON array: [{{"messages": [{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}]}}]',
        ]

        all_conversations = []
        per_batch = count // len(prompts_batch)

        for prompt_template in prompts_batch:
            prompt = prompt_template.format(n=per_batch)

            # Split into smaller chunks for the LLM
            chunk_size = 10
            for chunk_start in range(0, per_batch, chunk_size):
                chunk_count = min(chunk_size, per_batch - chunk_start)
                chunk_prompt = prompt_template.format(n=chunk_count)

                result = self._call_llm(chunk_prompt)
                conversations = self._parse_conversations(result)

                for conv in conversations:
                    conv["source"] = "synthetic/language"

                all_conversations.extend(conversations)
                time.sleep(0.5)

        # Save
        with open(output_path, "w", encoding="utf-8") as f:
            for conv in all_conversations:
                f.write(json.dumps(conv, ensure_ascii=False) + "\n")

        logger.info(f"Language pairs: {len(all_conversations)} conversations -> {output_path}")
        return all_conversations


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic training data for Elumina")
    parser.add_argument("--config", default="./config/scraping_config.yaml")
    parser.add_argument("--output-dir", default="./data/cultural")
    parser.add_argument("--articles-dir", default="./data/raw/scraped", help="Directory with scraped articles")
    parser.add_argument("--topics-only", action="store_true", help="Only generate from topics")
    parser.add_argument("--articles-only", action="store_true", help="Only convert articles")
    parser.add_argument("--language-only", action="store_true", help="Only generate language pairs")
    parser.add_argument("--max-articles", type=int, default=None, help="Max articles to convert")
    parser.add_argument("--swahili-count", type=int, default=500, help="Number of Swahili pairs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    generator = SyntheticGenerator(config_path=args.config)

    if not generator.api_key:
        logger.error("No API key configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.")
        return

    total = 0

    if args.topics_only:
        convos = generator.generate_from_topics(os.path.join(args.output_dir, "topic_conversations.jsonl"))
        total += len(convos)
    elif args.articles_only:
        convos = generator.generate_from_articles(
            args.articles_dir,
            os.path.join(args.output_dir, "article_conversations.jsonl"),
            max_articles=args.max_articles,
        )
        total += len(convos)
    elif args.language_only:
        convos = generator.generate_swahili_pairs(
            os.path.join(args.output_dir, "language_conversations.jsonl"),
            count=args.swahili_count,
        )
        total += len(convos)
    else:
        # Run all generators
        logger.info("=" * 60)
        logger.info("Phase 1: Topic-based generation")
        logger.info("=" * 60)
        convos = generator.generate_from_topics(os.path.join(args.output_dir, "topic_conversations.jsonl"))
        total += len(convos)

        logger.info("\n" + "=" * 60)
        logger.info("Phase 2: Article conversion")
        logger.info("=" * 60)
        if os.path.exists(args.articles_dir):
            convos = generator.generate_from_articles(
                args.articles_dir,
                os.path.join(args.output_dir, "article_conversations.jsonl"),
                max_articles=args.max_articles,
            )
            total += len(convos)
        else:
            logger.info(f"No articles directory found at {args.articles_dir}. Run scrape_kenya_web.py first.")

        logger.info("\n" + "=" * 60)
        logger.info("Phase 3: Language pair generation")
        logger.info("=" * 60)
        convos = generator.generate_swahili_pairs(
            os.path.join(args.output_dir, "language_conversations.jsonl"),
            count=args.swahili_count,
        )
        total += len(convos)

    logger.info("\n" + "=" * 60)
    logger.info(f"TOTAL: {total} conversations generated")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("Next step: python scripts/prepare_data.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
