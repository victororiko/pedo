"""
Elumina — Full Data Pipeline Orchestrator
Runs all data collection steps in sequence: scrape → generate → prepare.
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
except ImportError:
    pass  # dotenv not installed, rely on environment variables

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def run_step(name: str, func, *args, **kwargs):
    """Run a pipeline step with timing."""
    logger.info(f"\n{'=' * 70}")
    logger.info(f"STEP: {name}")
    logger.info(f"{'=' * 70}")
    start = time.time()
    try:
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"✓ {name} completed in {elapsed:.1f}s")
        return result
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"✗ {name} failed after {elapsed:.1f}s: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Elumina full data pipeline")
    parser.add_argument("--config", default="./config/scraping_config.yaml")
    parser.add_argument("--skip-scrape", action="store_true", help="Skip web scraping")
    parser.add_argument("--skip-social", action="store_true", help="Skip social media scraping")
    parser.add_argument("--skip-synthetic", action="store_true", help="Skip synthetic generation")
    parser.add_argument("--max-articles", type=int, default=500)
    parser.add_argument("--max-synthetic-articles", type=int, default=200)
    parser.add_argument("--swahili-count", type=int, default=500)
    parser.add_argument("--twitter-token", default=None)
    args = parser.parse_args()

    start_time = datetime.now()
    logger.info(f"Elumina Data Pipeline — Started at {start_time.isoformat()}")

    # Step 1: Web scraping
    if not args.skip_scrape:
        from scrape_kenya_web import KenyaWebScraper

        def scrape_web():
            scraper = KenyaWebScraper(config_path=args.config)
            return scraper.run(max_articles_per_site=args.max_articles)

        run_step("Web Scraping (Kenyan news + Wikipedia)", scrape_web)
    else:
        logger.info("Skipping web scraping (--skip-scrape)")

    # Step 2: Social media scraping
    if not args.skip_social:
        from scrape_social import RedditScraper, TwitterScraper
        from scrape_social import convert_reddit_to_conversations, convert_tweets_to_conversations
        import json
        import yaml

        def scrape_social():
            with open(args.config, "r") as f:
                config = yaml.safe_load(f)

            output_dir = "./data/raw/social"
            os.makedirs(output_dir, exist_ok=True)

            # Reddit
            reddit_scraper = RedditScraper(config)
            reddit_posts = reddit_scraper.run()

            with open(os.path.join(output_dir, "reddit_raw.jsonl"), "w", encoding="utf-8") as f:
                for post in reddit_posts:
                    f.write(json.dumps(post, ensure_ascii=False) + "\n")

            reddit_convos = convert_reddit_to_conversations(reddit_posts)

            # Twitter (optional)
            twitter_convos = []
            twitter_token = args.twitter_token or os.environ.get("TWITTER_BEARER_TOKEN")
            if twitter_token:
                twitter_scraper = TwitterScraper(config)
                tweets = twitter_scraper.run(bearer_token=twitter_token)
                twitter_convos = convert_tweets_to_conversations(tweets)

            all_convos = reddit_convos + twitter_convos
            with open(os.path.join(output_dir, "social_conversations.jsonl"), "w", encoding="utf-8") as f:
                for conv in all_convos:
                    f.write(json.dumps(conv, ensure_ascii=False) + "\n")

            return all_convos

        run_step("Social Media Scraping (Reddit + Twitter)", scrape_social)
    else:
        logger.info("Skipping social media scraping (--skip-social)")

    # Step 3: Synthetic data generation
    if not args.skip_synthetic:
        from generate_synthetic import SyntheticGenerator

        def generate_synthetic():
            generator = SyntheticGenerator(config_path=args.config)
            if not generator.api_key:
                logger.error("No LLM API key configured. Skipping synthetic generation.")
                return None

            output_dir = "./data/cultural"
            os.makedirs(output_dir, exist_ok=True)

            total = 0

            # Topic-based
            convos = generator.generate_from_topics(
                os.path.join(output_dir, "topic_conversations.jsonl")
            )
            total += len(convos)

            # Article-based
            articles_dir = "./data/raw/scraped"
            if os.path.exists(articles_dir):
                convos = generator.generate_from_articles(
                    articles_dir,
                    os.path.join(output_dir, "article_conversations.jsonl"),
                    max_articles=args.max_synthetic_articles,
                )
                total += len(convos)

            # Language pairs
            convos = generator.generate_swahili_pairs(
                os.path.join(output_dir, "language_conversations.jsonl"),
                count=args.swahili_count,
            )
            total += len(convos)

            return total

        run_step("Synthetic Data Generation", generate_synthetic)
    else:
        logger.info("Skipping synthetic generation (--skip-synthetic)")

    # Step 4: Merge and prepare final dataset
    from prepare_data import prepare_dataset

    def prepare():
        prepare_dataset(
            identity_dir="./data/identity",
            cultural_dir="./data/cultural",
            raw_dir="./data/raw",
            output_dir="./data/processed",
            eval_ratio=0.05,
            identity_oversample=20,
        )

    run_step("Merge & Prepare Final Dataset", prepare)

    # Summary
    elapsed = datetime.now() - start_time
    logger.info(f"\n{'=' * 70}")
    logger.info(f"PIPELINE COMPLETE — Total time: {elapsed}")
    logger.info(f"{'=' * 70}")

    # Count final output
    train_path = "./data/processed/train.jsonl"
    eval_path = "./data/processed/eval.jsonl"

    train_count = 0
    eval_count = 0
    if os.path.exists(train_path):
        with open(train_path) as f:
            train_count = sum(1 for _ in f)
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            eval_count = sum(1 for _ in f)

    logger.info(f"\nFinal dataset:")
    logger.info(f"  Train: {train_count} conversations")
    logger.info(f"  Eval:  {eval_count} conversations")
    logger.info(f"  Total: {train_count + eval_count} conversations")
    logger.info(f"\nNext: python scripts/train.py")


if __name__ == "__main__":
    main()
