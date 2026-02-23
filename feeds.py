"""
RSS feed poller for political news.

Polls configured feeds, deduplicates by URL, fetches full article text.
"""

import logging
from datetime import datetime, timezone
from time import mktime

import feedparser
import requests

import config
import models

logger = logging.getLogger("oddshawk.feeds")

# RSS feed URLs grouped by region
RSS_FEEDS = {
    # UK Politics (highest Betfair liquidity)
    "bbc_politics": "http://feeds.bbci.co.uk/news/politics/rss.xml",
    "guardian_politics": "https://www.theguardian.com/politics/rss",
    "sky_politics": "https://feeds.skynews.com/feeds/rss/politics.xml",
    # US Politics
    "reuters_world": "https://www.reuters.com/rssFeed/worldNews",
    "politico": "https://www.politico.com/rss/politics08.xml",
    # AU Politics (local advantage)
    "abc_au": "https://www.abc.net.au/news/feed/2942460/rss.xml",
    "guardian_au": "https://www.theguardian.com/australia-news/rss",
}


def _parse_published(entry) -> str | None:
    """Extract published datetime as ISO string from a feedparser entry."""
    published_parsed = entry.get("published_parsed")
    if published_parsed:
        try:
            dt = datetime.fromtimestamp(mktime(published_parsed), tz=timezone.utc)
            return dt.isoformat()
        except Exception:
            pass
    return entry.get("published", None)


def _fetch_full_text(url: str, timeout: int = 10) -> str:
    """Fetch full article text from URL. Returns raw text or empty string."""
    try:
        resp = requests.get(url, timeout=timeout, headers={
            "User-Agent": "Oddshawk/1.0 (news aggregator)"
        })
        resp.raise_for_status()
        # Simple extraction: strip HTML tags
        from html.parser import HTMLParser

        class _Stripper(HTMLParser):
            def __init__(self):
                super().__init__()
                self._parts = []
                self._skip = False

            def handle_starttag(self, tag, attrs):
                if tag in ("script", "style", "nav", "header", "footer"):
                    self._skip = True

            def handle_endtag(self, tag):
                if tag in ("script", "style", "nav", "header", "footer"):
                    self._skip = False

            def handle_data(self, data):
                if not self._skip:
                    self._parts.append(data.strip())

            def get_text(self):
                return " ".join(p for p in self._parts if p)

        stripper = _Stripper()
        stripper.feed(resp.text)
        text = stripper.get_text()
        # Truncate to ~4000 chars (Gemini context budget)
        return text[:4000]
    except Exception as e:
        logger.debug("Failed to fetch full text from %s: %s", url, e)
        return ""


def poll_feeds() -> list[dict]:
    """
    Poll all RSS feeds and return new (unseen) articles.

    Each article is inserted into the DB. Returns list of dicts:
    [{"id": int, "url": str, "title": str, "summary": str,
      "full_text": str, "feed_source": str, "published_at": str}, ...]
    """
    new_articles = []
    now_iso = datetime.now(timezone.utc).isoformat()

    for feed_name, feed_url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(feed_url)
        except Exception as e:
            logger.warning("Failed to parse feed %s: %s", feed_name, e)
            continue

        for entry in feed.entries:
            url = entry.get("link", "")
            if not url:
                continue

            title = entry.get("title", "")
            summary = entry.get("summary", entry.get("description", ""))
            guid = entry.get("id", url)
            published_at = _parse_published(entry)

            # Try to insert (skips duplicates via UNIQUE url)
            with models.get_db() as conn:
                article_id = models.insert_article(
                    conn,
                    url=url,
                    guid=guid,
                    feed_source=feed_name,
                    title=title,
                    summary=summary,
                    full_text="",  # fetched on demand
                    published_at=published_at,
                    fetched_at=now_iso,
                )

            if article_id is not None:
                # New article — fetch full text
                full_text = _fetch_full_text(url)
                if full_text:
                    with models.get_db() as conn:
                        conn.execute(
                            "UPDATE articles SET full_text = ? WHERE id = ?",
                            (full_text, article_id),
                        )

                new_articles.append({
                    "id": article_id,
                    "url": url,
                    "title": title,
                    "summary": summary,
                    "full_text": full_text,
                    "feed_source": feed_name,
                    "published_at": published_at,
                })

    if new_articles:
        logger.info("Polled feeds: %d new articles", len(new_articles))
    else:
        logger.debug("Polled feeds: no new articles")

    return new_articles
