"""
Elumina — Social Media Scraper
Scrapes Reddit (Kenyan subreddits) and X/Twitter (public Kenyan content).
"""

import json
import os
import re
import time
import logging
from datetime import datetime

import requests
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class RedditScraper:
    """
    Scrapes Kenyan subreddits using Reddit's public JSON API.
    No API key required for read-only public data.
    """

    def __init__(self, config: dict):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "EluminaDataCollector/1.0 (Research; Intevia Ltd)",
        })
        self.rate_limit = 2.0  # Reddit limits to ~30 requests/min without auth
        self.last_request_time = 0

    def _rate_limit_wait(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()

    def _fetch_json(self, url: str, params: dict = None) -> dict | None:
        self._rate_limit_wait()
        try:
            resp = self.session.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                logger.warning("Rate limited, waiting 60s...")
                time.sleep(60)
                resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch {url}: {e}")
            return None

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        # Remove markdown links but keep text
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        # Remove URLs
        text = re.sub(r"https?://\S+", "", text)
        # Clean whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def scrape_subreddit(self, subreddit: str, sort: str = "top",
                          time_filter: str = "all", max_posts: int = 2000,
                          min_score: int = 5, min_comments: int = 3) -> list[dict]:
        """Scrape posts and comments from a subreddit."""
        logger.info(f"Scraping r/{subreddit} ({sort}/{time_filter})...")

        posts = []
        after = None
        fetched = 0

        while fetched < max_posts:
            url = f"https://www.reddit.com/r/{subreddit}/{sort}.json"
            params = {"t": time_filter, "limit": 100}
            if after:
                params["after"] = after

            data = self._fetch_json(url, params)
            if not data or "data" not in data:
                break

            children = data["data"].get("children", [])
            if not children:
                break

            for child in children:
                post = child.get("data", {})

                # Filter
                if post.get("score", 0) < min_score:
                    continue
                if post.get("num_comments", 0) < min_comments:
                    continue
                if post.get("over_18", False):
                    continue
                if post.get("removed_by_category"):
                    continue

                title = self._clean_text(post.get("title", ""))
                selftext = self._clean_text(post.get("selftext", ""))

                if not title:
                    continue

                post_data = {
                    "subreddit": subreddit,
                    "title": title,
                    "text": selftext,
                    "score": post.get("score", 0),
                    "num_comments": post.get("num_comments", 0),
                    "created_utc": post.get("created_utc"),
                    "permalink": post.get("permalink", ""),
                    "comments": [],
                }

                # Fetch top comments
                if post.get("permalink"):
                    comments = self._fetch_comments(post["permalink"])
                    post_data["comments"] = comments

                posts.append(post_data)
                fetched += 1

            after = data["data"].get("after")
            if not after:
                break

            logger.info(f"  r/{subreddit}: {fetched} posts fetched...")

        logger.info(f"  r/{subreddit}: {len(posts)} posts total")
        return posts

    def _fetch_comments(self, permalink: str, max_comments: int = 20) -> list[dict]:
        """Fetch top-level comments for a post."""
        url = f"https://www.reddit.com{permalink}.json"
        params = {"sort": "top", "limit": max_comments}

        data = self._fetch_json(url, params)
        if not data or len(data) < 2:
            return []

        comments = []
        children = data[1].get("data", {}).get("children", [])

        for child in children[:max_comments]:
            if child.get("kind") != "t1":
                continue
            comment = child.get("data", {})

            body = self._clean_text(comment.get("body", ""))
            if not body or len(body) < 20:
                continue
            if comment.get("score", 0) < 2:
                continue

            comments.append({
                "text": body,
                "score": comment.get("score", 0),
            })

        return comments

    def run(self) -> list[dict]:
        """Scrape all configured subreddits."""
        reddit_cfg = self.config.get("reddit", {})
        subreddits = reddit_cfg.get("subreddits", ["Kenya"])
        sort = reddit_cfg.get("sort", "top")
        time_filter = reddit_cfg.get("time_filter", "all")
        max_posts = reddit_cfg.get("max_posts_per_sub", 2000)
        min_score = reddit_cfg.get("min_score", 5)
        min_comments = reddit_cfg.get("min_comments", 3)

        all_posts = []
        for sub in subreddits:
            posts = self.scrape_subreddit(
                sub,
                sort=sort,
                time_filter=time_filter,
                max_posts=max_posts,
                min_score=min_score,
                min_comments=min_comments,
            )
            all_posts.extend(posts)

        return all_posts


class TwitterScraper:
    """
    Scrapes Kenyan Twitter/X content using nitter instances or API.
    Requires either a nitter instance URL or X API Bearer Token.
    """

    def __init__(self, config: dict):
        self.config = config
        self.session = requests.Session()

        # Kenyan accounts and hashtags to track
        self.kenyan_hashtags = [
            "KenyaNews", "Nairobi", "KOT", "Kenya", "KenyanTwitter",
            "Kiswahili", "MadarakaDay", "JamhuriDay", "Mashujaa",
            "KenyanFood", "NairobiLife", "SafaricomPLC", "MPesa",
            "KenyanMusic", "Gengetone", "KenyanComedy",
        ]

        self.kenyan_accounts = [
            "KenyaNews", "NationAfrica", "StandardKenya",
            "TheStarKenya", "citizenaborad", "KTNNewsKE",
        ]

    def scrape_with_api(self, bearer_token: str, max_tweets: int = 5000) -> list[dict]:
        """
        Scrape using X API v2.
        Requires a Bearer Token from developer.twitter.com.
        """
        self.session.headers.update({
            "Authorization": f"Bearer {bearer_token}",
        })

        tweets = []
        for hashtag in self.kenyan_hashtags:
            url = "https://api.twitter.com/2/tweets/search/recent"
            params = {
                "query": f"#{hashtag} lang:en OR lang:sw -is:retweet",
                "max_results": 100,
                "tweet.fields": "created_at,public_metrics,lang,text",
            }

            try:
                resp = self.session.get(url, params=params, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    for tweet in data.get("data", []):
                        text = tweet.get("text", "").strip()
                        if len(text) > 50:
                            tweets.append({
                                "text": text,
                                "hashtag": hashtag,
                                "lang": tweet.get("lang", "en"),
                                "metrics": tweet.get("public_metrics", {}),
                                "created_at": tweet.get("created_at", ""),
                            })
                elif resp.status_code == 429:
                    logger.warning("X API rate limited, waiting...")
                    time.sleep(900)  # 15 minute window
                else:
                    logger.warning(f"X API error {resp.status_code} for #{hashtag}")
            except requests.RequestException as e:
                logger.warning(f"X API request failed: {e}")

            time.sleep(1)

            if len(tweets) >= max_tweets:
                break

        logger.info(f"X/Twitter: {len(tweets)} tweets collected")
        return tweets

    def run(self, bearer_token: str = None) -> list[dict]:
        """Run Twitter scraping."""
        if not bearer_token:
            bearer_token = os.environ.get("TWITTER_BEARER_TOKEN")

        if not bearer_token:
            logger.warning(
                "No Twitter/X API token provided. "
                "Set TWITTER_BEARER_TOKEN env var or pass --twitter-token. "
                "Skipping Twitter scraping."
            )
            return []

        return self.scrape_with_api(bearer_token)


def convert_reddit_to_conversations(posts: list[dict]) -> list[dict]:
    """Convert Reddit posts/comments into chat-style training data."""
    conversations = []

    for post in posts:
        title = post.get("title", "")
        text = post.get("text", "")
        comments = post.get("comments", [])

        # Pattern 1: Post title as question, best comment as answer
        if comments and title.endswith("?"):
            best_comment = max(comments, key=lambda c: c.get("score", 0))
            conversations.append({
                "messages": [
                    {"role": "user", "content": title},
                    {"role": "assistant", "content": best_comment["text"]},
                ],
                "source": f"reddit/r/{post.get('subreddit', 'Kenya')}",
            })

        # Pattern 2: Post with context + top comments as discussion
        if text and comments:
            context = f"{title}\n\n{text}" if text else title
            for comment in comments[:3]:
                if len(comment["text"]) > 50:
                    conversations.append({
                        "messages": [
                            {"role": "user", "content": context},
                            {"role": "assistant", "content": comment["text"]},
                        ],
                        "source": f"reddit/r/{post.get('subreddit', 'Kenya')}",
                    })

        # Pattern 3: Title as topic prompt
        if not title.endswith("?") and text and len(text) > 100:
            prompt = f"Tell me about: {title}"
            conversations.append({
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": text},
                ],
                "source": f"reddit/r/{post.get('subreddit', 'Kenya')}",
            })

    return conversations


def convert_tweets_to_conversations(tweets: list[dict]) -> list[dict]:
    """Convert tweets into training conversation pairs."""
    conversations = []

    # Group tweets by hashtag for topical conversations
    by_hashtag = {}
    for tweet in tweets:
        tag = tweet.get("hashtag", "Kenya")
        by_hashtag.setdefault(tag, []).append(tweet)

    for hashtag, tag_tweets in by_hashtag.items():
        for tweet in tag_tweets:
            text = tweet.get("text", "")
            if len(text) < 50:
                continue

            # Create Q&A from tweet content
            conversations.append({
                "messages": [
                    {"role": "user", "content": f"What are people saying about #{hashtag} in Kenya?"},
                    {"role": "assistant", "content": text},
                ],
                "source": "twitter",
            })

    return conversations


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Scrape Kenyan social media")
    parser.add_argument("--config", default="./config/scraping_config.yaml")
    parser.add_argument("--output-dir", default="./data/raw/social")
    parser.add_argument("--reddit-only", action="store_true")
    parser.add_argument("--twitter-only", action="store_true")
    parser.add_argument("--twitter-token", default=None, help="X/Twitter API Bearer Token")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    all_conversations = []

    # Reddit
    if not args.twitter_only:
        logger.info("=" * 60)
        logger.info("Reddit Scraping")
        logger.info("=" * 60)

        reddit_scraper = RedditScraper(config)
        reddit_posts = reddit_scraper.run()

        # Save raw posts
        raw_path = os.path.join(args.output_dir, "reddit_raw.jsonl")
        with open(raw_path, "w", encoding="utf-8") as f:
            for post in reddit_posts:
                f.write(json.dumps(post, ensure_ascii=False) + "\n")
        logger.info(f"Raw Reddit data saved to {raw_path}")

        # Convert to conversations
        reddit_convos = convert_reddit_to_conversations(reddit_posts)
        all_conversations.extend(reddit_convos)
        logger.info(f"Reddit: {len(reddit_convos)} conversations generated")

    # Twitter/X
    if not args.reddit_only:
        logger.info("\n" + "=" * 60)
        logger.info("Twitter/X Scraping")
        logger.info("=" * 60)

        twitter_scraper = TwitterScraper(config)
        tweets = twitter_scraper.run(bearer_token=args.twitter_token)

        if tweets:
            raw_path = os.path.join(args.output_dir, "twitter_raw.jsonl")
            with open(raw_path, "w", encoding="utf-8") as f:
                for tweet in tweets:
                    f.write(json.dumps(tweet, ensure_ascii=False) + "\n")
            logger.info(f"Raw Twitter data saved to {raw_path}")

            twitter_convos = convert_tweets_to_conversations(tweets)
            all_conversations.extend(twitter_convos)
            logger.info(f"Twitter: {len(twitter_convos)} conversations generated")

    # Save all conversations
    if all_conversations:
        convo_path = os.path.join(args.output_dir, "social_conversations.jsonl")
        with open(convo_path, "w", encoding="utf-8") as f:
            for convo in all_conversations:
                f.write(json.dumps(convo, ensure_ascii=False) + "\n")
        logger.info(f"\nTotal conversations: {len(all_conversations)}")
        logger.info(f"Saved to: {convo_path}")


if __name__ == "__main__":
    main()
