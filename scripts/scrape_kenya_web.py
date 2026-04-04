"""
Elumina — Kenyan Web Scraper
Scrapes Kenyan news sites, blogs, Wikipedia, and knowledge sources.
Outputs raw text to data/raw/ for synthetic conversion.
"""

import json
import os
import re
import time
import hashlib
import logging
from pathlib import Path
from urllib.parse import urljoin, urlparse
from datetime import datetime

import requests
import yaml
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class KenyaWebScraper:
    """Scrapes Kenyan websites for training data."""

    def __init__(self, config_path: str = "./config/scraping_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        scraping_cfg = self.config["scraping"]
        self.rate_limit = 1.0 / scraping_cfg["requests_per_second"]
        self.timeout = scraping_cfg["request_timeout"]
        self.max_retries = scraping_cfg["max_retries"]
        self.retry_delay = scraping_cfg["retry_delay"]
        self.user_agent = scraping_cfg["user_agent"]

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})

        self.last_request_time = 0
        self.scraped_urls = set()
        self.output_dir = "./data/raw/scraped"
        os.makedirs(self.output_dir, exist_ok=True)

    def _rate_limit_wait(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()

    def _fetch(self, url: str) -> str | None:
        for attempt in range(self.max_retries):
            try:
                self._rate_limit_wait()
                resp = self.session.get(url, timeout=self.timeout)
                resp.raise_for_status()
                return resp.text
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed for {url}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
        return None

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()
        return text

    def _extract_article(self, html: str, selectors: dict) -> dict | None:
        soup = BeautifulSoup(html, "html.parser")

        # Remove script/style/nav/footer elements
        for tag in soup.find_all(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
            tag.decompose()

        # Title
        title_el = soup.select_one(selectors.get("title", "h1"))
        title = title_el.get_text(strip=True) if title_el else ""

        # Body
        body_text = ""
        for selector in selectors.get("article_body", "article").split(", "):
            body_el = soup.select_one(selector.strip())
            if body_el:
                body_text = body_el.get_text(separator="\n", strip=True)
                break

        if not body_text:
            # Fallback: get main content area
            main = soup.find("main") or soup.find("article") or soup.find(class_=re.compile(r"content|article|story"))
            if main:
                body_text = main.get_text(separator="\n", strip=True)

        body_text = self._clean_text(body_text)

        if len(body_text) < 200:
            return None

        # Date
        date_el = None
        for sel in selectors.get("date", "time").split(", "):
            date_el = soup.select_one(sel.strip())
            if date_el:
                break
        date_str = date_el.get_text(strip=True) if date_el else ""

        return {
            "title": title,
            "text": body_text,
            "date": date_str,
        }

    def _url_hash(self, url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()[:12]

    def scrape_news_sitemap(self, site_config: dict, max_articles: int = 500) -> list[dict]:
        """Scrape articles from a news site's sitemap."""
        name = site_config["name"]
        logger.info(f"Scraping sitemap for {name}...")

        sitemap_url = site_config.get("sitemap_url")
        if not sitemap_url:
            logger.info(f"  No sitemap URL for {name}, skipping sitemap method")
            return []

        html = self._fetch(sitemap_url)
        if not html:
            logger.warning(f"  Failed to fetch sitemap for {name}")
            return []

        # Parse sitemap XML for article URLs
        soup = BeautifulSoup(html, "html.parser")
        urls = []
        for loc in soup.find_all("loc"):
            url = loc.text.strip()
            # Filter to relevant categories
            categories = site_config.get("categories", [])
            if categories:
                if any(cat in url for cat in categories):
                    urls.append(url)
            else:
                urls.append(url)

        logger.info(f"  Found {len(urls)} URLs, scraping up to {max_articles}")
        urls = urls[:max_articles]

        articles = []
        for i, url in enumerate(urls):
            if url in self.scraped_urls:
                continue
            self.scraped_urls.add(url)

            html = self._fetch(url)
            if not html:
                continue

            article = self._extract_article(html, site_config["selectors"])
            if article:
                article["url"] = url
                article["source"] = name
                articles.append(article)

            if (i + 1) % 50 == 0:
                logger.info(f"  {name}: {len(articles)}/{i + 1} articles extracted")

        logger.info(f"  {name}: {len(articles)} articles total")
        return articles

    def scrape_news_categories(self, site_config: dict, max_per_category: int = 100) -> list[dict]:
        """Scrape articles by browsing category pages."""
        name = site_config["name"]
        base_url = site_config["base_url"]
        categories = site_config.get("categories", [])

        logger.info(f"Scraping categories for {name}...")
        articles = []

        for category in categories:
            cat_url = urljoin(base_url, category)
            logger.info(f"  Category: {cat_url}")

            html = self._fetch(cat_url)
            if not html:
                continue

            soup = BeautifulSoup(html, "html.parser")
            links = set()

            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"]
                full_url = urljoin(base_url, href)

                if urlparse(full_url).netloc == urlparse(base_url).netloc:
                    # Heuristic: article URLs are usually longer paths
                    path = urlparse(full_url).path
                    if path.count("/") >= 3 and not path.endswith(("/", ".xml", ".rss")):
                        links.add(full_url)

            logger.info(f"    Found {len(links)} potential article links")
            for url in list(links)[:max_per_category]:
                if url in self.scraped_urls:
                    continue
                self.scraped_urls.add(url)

                html = self._fetch(url)
                if not html:
                    continue

                article = self._extract_article(html, site_config["selectors"])
                if article:
                    article["url"] = url
                    article["source"] = name
                    articles.append(article)

        logger.info(f"  {name}: {len(articles)} articles from categories")
        return articles

    def scrape_wikipedia(self) -> list[dict]:
        """Scrape Kenya-related Wikipedia articles."""
        wiki_configs = self.config.get("blogs_knowledge", [])
        articles = []

        for source in wiki_configs:
            name = source["name"]
            urls = source.get("urls", [])
            logger.info(f"Scraping {name}: {len(urls)} pages...")

            for url in urls:
                if url in self.scraped_urls:
                    continue
                self.scraped_urls.add(url)

                html = self._fetch(url)
                if not html:
                    continue

                soup = BeautifulSoup(html, "html.parser")

                # Remove reference markers, edit links, etc.
                for tag in soup.find_all(["sup", "span"], class_=re.compile(r"mw-editsection|reference")):
                    tag.decompose()
                for tag in soup.find_all("table", class_=re.compile(r"infobox|sidebar|navbox|metadata")):
                    tag.decompose()

                title_el = soup.find("h1", id="firstHeading")
                title = title_el.get_text(strip=True) if title_el else ""

                content_el = soup.find("div", id="mw-content-text")
                if not content_el:
                    continue

                # Extract sections
                sections = []
                current_section = {"heading": "Introduction", "text": ""}

                for element in content_el.find_all(["h2", "h3", "p", "ul", "ol"]):
                    if element.name in ("h2", "h3"):
                        heading_text = element.get_text(strip=True)
                        # Skip reference/metadata sections
                        if heading_text.lower() in ("references", "external links", "see also", "notes", "further reading", "bibliography"):
                            break
                        if current_section["text"].strip():
                            sections.append(current_section)
                        current_section = {"heading": heading_text, "text": ""}
                    elif element.name == "p":
                        text = element.get_text(strip=True)
                        if text:
                            current_section["text"] += text + "\n\n"
                    elif element.name in ("ul", "ol"):
                        for li in element.find_all("li", recursive=False):
                            text = li.get_text(strip=True)
                            if text:
                                current_section["text"] += f"- {text}\n"
                        current_section["text"] += "\n"

                if current_section["text"].strip():
                    sections.append(current_section)

                full_text = ""
                for section in sections:
                    if section["heading"] != "Introduction":
                        full_text += f"\n## {section['heading']}\n\n"
                    full_text += section["text"]

                full_text = self._clean_text(full_text)

                if len(full_text) > 200:
                    articles.append({
                        "title": title,
                        "text": full_text,
                        "url": url,
                        "source": name,
                        "sections": [s["heading"] for s in sections],
                    })

            logger.info(f"  {name}: {len(articles)} articles extracted")

        return articles

    def save_articles(self, articles: list[dict], filename: str):
        """Save scraped articles to JSONL."""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            for article in articles:
                f.write(json.dumps(article, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(articles)} articles to {filepath}")

    def run(self, max_articles_per_site: int = 500):
        """Run full scraping pipeline."""
        logger.info("=" * 60)
        logger.info("Elumina — Kenya Web Scraping Pipeline")
        logger.info("=" * 60)

        all_articles = []

        # 1. Wikipedia
        logger.info("\n--- Wikipedia ---")
        wiki_articles = self.scrape_wikipedia()
        self.save_articles(wiki_articles, "wikipedia_kenya.jsonl")
        all_articles.extend(wiki_articles)

        # 2. News sites
        for site in self.config.get("news_sites", []):
            logger.info(f"\n--- {site['name']} ---")

            # Try sitemap first
            articles = self.scrape_news_sitemap(site, max_articles=max_articles_per_site)

            # Supplement with category browsing
            if len(articles) < 50:
                cat_articles = self.scrape_news_categories(site, max_per_category=100)
                articles.extend(cat_articles)

            safe_name = re.sub(r"[^a-z0-9]+", "_", site["name"].lower()).strip("_")
            self.save_articles(articles, f"news_{safe_name}.jsonl")
            all_articles.extend(articles)

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("SCRAPING COMPLETE")
        logger.info(f"  Total articles: {len(all_articles)}")
        logger.info(f"  Total URLs visited: {len(self.scraped_urls)}")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info("=" * 60)
        logger.info("\nNext step: Run generate_synthetic.py to convert articles to training data")

        return all_articles


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scrape Kenyan websites for training data")
    parser.add_argument("--config", default="./config/scraping_config.yaml")
    parser.add_argument("--max-articles", type=int, default=500, help="Max articles per news site")
    parser.add_argument("--wikipedia-only", action="store_true", help="Only scrape Wikipedia")
    parser.add_argument("--news-only", action="store_true", help="Only scrape news sites")
    args = parser.parse_args()

    scraper = KenyaWebScraper(config_path=args.config)

    if args.wikipedia_only:
        articles = scraper.scrape_wikipedia()
        scraper.save_articles(articles, "wikipedia_kenya.jsonl")
    elif args.news_only:
        for site in scraper.config.get("news_sites", []):
            articles = scraper.scrape_news_sitemap(site, max_articles=args.max_articles)
            if len(articles) < 50:
                articles.extend(scraper.scrape_news_categories(site))
            safe_name = re.sub(r"[^a-z0-9]+", "_", site["name"].lower()).strip("_")
            scraper.save_articles(articles, f"news_{safe_name}.jsonl")
    else:
        scraper.run(max_articles_per_site=args.max_articles)
