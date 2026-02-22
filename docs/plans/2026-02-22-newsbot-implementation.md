# Newsbot Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a standalone newsbot process that monitors RSS feeds, analyses articles with Gemini Flash against live Betfair politics markets, and generates trading signals (edge + shock).

**Architecture:** New `newsbot.py` process polls RSS feeds every 15s, deduplicates by URL, sends new articles through a two-stage Gemini analysis pipeline (relevance filter -> impact analysis + shock detection), compares against cached Betfair politics market prices, logs signals and sim bets to the shared SQLite DB, sends Telegram alerts.

**Tech Stack:** Python 3.13, feedparser (RSS), google-genai (Gemini Flash), betfairlightweight (Betfair API), SQLite, existing oddshawk modules (betfair.py, models.py, notify.py, config.py).

---

### Task 1: Add Dependencies and Config Vars

**Files:**
- Modify: `requirements.txt`
- Modify: `config.py`
- Modify: `.env.example`

**Step 1: Add new dependencies to requirements.txt**

Append to `requirements.txt`:
```
feedparser>=6.0.0
google-genai>=1.0.0
```

**Step 2: Add newsbot + Gemini config vars to config.py**

Add after the Telegram section (line ~65):
```python
# --- Newsbot ---
NEWSBOT_ENABLED = _bool("NEWSBOT_ENABLED", "false")
NEWSBOT_POLL_INTERVAL = _int("NEWSBOT_POLL_INTERVAL", "15")  # seconds
NEWSBOT_MARKET_REFRESH = _int("NEWSBOT_MARKET_REFRESH", "60")  # seconds
NEWSBOT_EDGE_THRESHOLD = _float("NEWSBOT_EDGE_THRESHOLD", "0.05")  # 5%
NEWSBOT_MIN_MATCHED = _float("NEWSBOT_MIN_MATCHED", "500")
NEWSBOT_SIM_STAKE = _float("NEWSBOT_SIM_STAKE", "100")
NEWSBOT_COOLDOWN = _int("NEWSBOT_COOLDOWN", "300")  # seconds between signals on same market

# --- Gemini ---
GEMINI_API_KEY = _get("GEMINI_API_KEY", "")
GEMINI_MODEL = _get("GEMINI_MODEL", "gemini-2.0-flash")
```

**Step 3: Update .env.example with newsbot vars**

Add section:
```
# Newsbot (news-reactive politics trading)
NEWSBOT_ENABLED=false
NEWSBOT_POLL_INTERVAL=15
NEWSBOT_MARKET_REFRESH=60
NEWSBOT_EDGE_THRESHOLD=0.05
NEWSBOT_MIN_MATCHED=500
NEWSBOT_SIM_STAKE=100

# Gemini (for newsbot article analysis)
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.0-flash
```

**Step 4: Install new deps locally**

Run: `uv pip install feedparser google-genai`

**Step 5: Verify imports work**

Run: `uv run python3 -c "import feedparser; from google import genai; print('OK')"`
Expected: `OK`

**Step 6: Commit**

```bash
git add requirements.txt config.py .env.example
git commit -m "feat: add newsbot config vars and dependencies"
```

---

### Task 2: Add Database Tables for Newsbot

**Files:**
- Modify: `models.py`

**Step 1: Add newsbot tables to the _SCHEMA string in models.py**

Add after the existing `credit_usage` table:

```sql
CREATE TABLE IF NOT EXISTS articles (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    url          TEXT NOT NULL UNIQUE,
    guid         TEXT,
    feed_source  TEXT NOT NULL,
    title        TEXT NOT NULL,
    summary      TEXT,
    full_text    TEXT,
    published_at TEXT,
    fetched_at   TEXT NOT NULL,
    analysed     INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS news_signals (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id          INTEGER NOT NULL,
    market_id           TEXT NOT NULL,
    market_name         TEXT NOT NULL,
    runner_name         TEXT NOT NULL,
    signal_type         TEXT NOT NULL,
    current_back_price  REAL,
    current_lay_price   REAL,
    current_implied_prob REAL,
    estimated_fair_prob REAL,
    edge_pct            REAL,
    gemini_reasoning    TEXT,
    detected_at         TEXT NOT NULL,
    FOREIGN KEY (article_id) REFERENCES articles(id)
);

CREATE TABLE IF NOT EXISTS news_bets (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id        INTEGER NOT NULL,
    market_id        TEXT NOT NULL,
    market_name      TEXT NOT NULL,
    runner_name      TEXT NOT NULL,
    side             TEXT NOT NULL DEFAULT 'back',
    odds_at_detection REAL NOT NULL,
    stake            REAL NOT NULL,
    potential_payout REAL NOT NULL,
    placed_at        TEXT NOT NULL,
    settled          INTEGER NOT NULL DEFAULT 0,
    result           TEXT,
    pnl              REAL,
    FOREIGN KEY (signal_id) REFERENCES news_signals(id)
);

CREATE INDEX IF NOT EXISTS idx_articles_url ON articles(url);
CREATE INDEX IF NOT EXISTS idx_articles_fetched ON articles(fetched_at);
CREATE INDEX IF NOT EXISTS idx_news_signals_detected ON news_signals(detected_at);
CREATE INDEX IF NOT EXISTS idx_news_bets_unsettled ON news_bets(settled) WHERE settled = 0;
```

**Step 2: Add insert helpers to models.py**

Add these functions after the existing insert helpers:

```python
def insert_article(conn, *, url, guid, feed_source, title, summary,
                   full_text, published_at, fetched_at):
    """Insert a new article. Returns row id, or None if duplicate URL."""
    try:
        cur = conn.execute(
            """INSERT INTO articles (url, guid, feed_source, title, summary,
               full_text, published_at, fetched_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (url, guid, feed_source, title, summary, full_text,
             published_at, fetched_at),
        )
        return cur.lastrowid
    except sqlite3.IntegrityError:
        return None  # duplicate URL


def insert_news_signal(conn, *, article_id, market_id, market_name,
                       runner_name, signal_type, current_back_price,
                       current_lay_price, current_implied_prob,
                       estimated_fair_prob, edge_pct, gemini_reasoning,
                       detected_at):
    """Insert a news signal. Returns row id."""
    cur = conn.execute(
        """INSERT INTO news_signals (article_id, market_id, market_name,
           runner_name, signal_type, current_back_price, current_lay_price,
           current_implied_prob, estimated_fair_prob, edge_pct,
           gemini_reasoning, detected_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (article_id, market_id, market_name, runner_name, signal_type,
         current_back_price, current_lay_price, current_implied_prob,
         estimated_fair_prob, edge_pct, gemini_reasoning, detected_at),
    )
    return cur.lastrowid


def insert_news_bet(conn, *, signal_id, market_id, market_name,
                    runner_name, side, odds_at_detection, stake,
                    potential_payout, placed_at):
    """Insert a simulated news bet. Returns row id."""
    cur = conn.execute(
        """INSERT INTO news_bets (signal_id, market_id, market_name,
           runner_name, side, odds_at_detection, stake, potential_payout,
           placed_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (signal_id, market_id, market_name, runner_name, side,
         odds_at_detection, stake, potential_payout, placed_at),
    )
    return cur.lastrowid
```

**Step 3: Test DB init creates the new tables**

Run: `uv run python3 -c "import models; models.init_db(); print('OK')"`
Expected: `OK` (no errors)

**Step 4: Commit**

```bash
git add models.py
git commit -m "feat: add articles, news_signals, news_bets tables"
```

---

### Task 3: Build Betfair Politics Market Cache

**Files:**
- Create: `markets.py`

This module fetches all Betfair politics markets and provides a cached snapshot with current prices, refreshing every N seconds.

**Step 1: Create markets.py**

```python
"""
Betfair politics market cache.

Fetches all politics markets with live prices and caches them.
Refreshes every NEWSBOT_MARKET_REFRESH seconds.
"""

import logging
import time
from datetime import datetime, timezone

import config

logger = logging.getLogger("oddshawk.markets")

# Politics event type ID on Betfair
POLITICS_EVENT_TYPE_ID = "2378961"

# Cached state
_markets = {}       # market_id -> market dict
_last_refresh = 0.0


def _fetch_politics_markets() -> dict:
    """
    Fetch all politics markets from Betfair with live prices.

    Returns dict keyed by market_id:
    {
        "market_id": {
            "market_name": str,
            "total_matched": float,
            "runners": {
                "runner_name": {
                    "selection_id": int,
                    "back_price": float | None,
                    "lay_price": float | None,
                    "back_liq": float,
                    "lay_liq": float,
                    "implied_prob": float | None,
                },
                ...
            }
        },
        ...
    }
    """
    import betfair as bf
    from betfairlightweight import filters

    client = bf.get_client()
    if not client:
        logger.error("Betfair client not available")
        return {}

    # Fetch catalogue (market names + runner names)
    try:
        catalogues = client.betting.list_market_catalogue(
            filter=filters.market_filter(
                event_type_ids=[POLITICS_EVENT_TYPE_ID],
            ),
            market_projection=["RUNNER_DESCRIPTION"],
            max_results=200,
            sort="MAXIMUM_TRADED",
        )
    except Exception as e:
        logger.error("Failed to fetch politics catalogues: %s", e)
        return {}

    if not catalogues:
        return {}

    # Build runner name lookup: market_id -> {selection_id: runner_name}
    runner_names = {}
    catalogue_map = {}
    for cat in catalogues:
        names = {}
        for r in cat.runners:
            names[r.selection_id] = r.runner_name
        runner_names[cat.market_id] = names
        catalogue_map[cat.market_id] = cat.market_name

    # Fetch live prices in batches of 10
    market_ids = [cat.market_id for cat in catalogues]
    all_books = {}
    for i in range(0, len(market_ids), 10):
        batch = market_ids[i:i + 10]
        try:
            books = client.betting.list_market_book(
                market_ids=batch,
                price_projection={"priceData": ["EX_BEST_OFFERS"]},
            )
            for book in books:
                all_books[book.market_id] = book
        except Exception as e:
            logger.warning("Failed to fetch market books batch: %s", e)

    # Build result
    result = {}
    for market_id, market_name in catalogue_map.items():
        book = all_books.get(market_id)
        if not book:
            continue

        total_matched = book.total_matched or 0

        # Skip dead markets
        if total_matched < config.NEWSBOT_MIN_MATCHED:
            continue

        runners = {}
        names = runner_names.get(market_id, {})
        if book.runners:
            for r in book.runners:
                name = names.get(r.selection_id, f"sel_{r.selection_id}")
                back = r.ex.available_to_back[0].price if r.ex.available_to_back else None
                lay = r.ex.available_to_lay[0].price if r.ex.available_to_lay else None
                back_liq = sum(p.size for p in r.ex.available_to_back) if r.ex.available_to_back else 0
                lay_liq = sum(p.size for p in r.ex.available_to_lay) if r.ex.available_to_lay else 0
                implied = (1.0 / back) if back and back > 1.0 else None

                runners[name] = {
                    "selection_id": r.selection_id,
                    "back_price": back,
                    "lay_price": lay,
                    "back_liq": back_liq,
                    "lay_liq": lay_liq,
                    "implied_prob": implied,
                }

        result[market_id] = {
            "market_name": market_name,
            "total_matched": total_matched,
            "runners": runners,
        }

    logger.info("Refreshed %d politics markets from Betfair", len(result))
    return result


def get_markets(force_refresh: bool = False) -> dict:
    """Get cached politics markets, refreshing if stale."""
    global _markets, _last_refresh

    now = time.time()
    if force_refresh or (now - _last_refresh) > config.NEWSBOT_MARKET_REFRESH:
        _markets = _fetch_politics_markets()
        _last_refresh = now

    return _markets


def get_dead_certs() -> list[dict]:
    """
    Return markets where one runner is priced at 95%+ implied probability.
    These are the shock signal candidates.

    Returns list of:
    {
        "market_id": str,
        "market_name": str,
        "favourite_name": str,
        "favourite_implied": float,
        "favourite_back": float,
        "upset_runners": [{"name": str, "back_price": float}, ...]
    }
    """
    markets = get_markets()
    dead_certs = []

    for market_id, mdata in markets.items():
        for rname, rdata in mdata["runners"].items():
            imp = rdata["implied_prob"]
            if imp and imp >= 0.95:
                # This runner is a dead cert — find upset runners
                upsets = []
                for other_name, other_data in mdata["runners"].items():
                    if other_name == rname:
                        continue
                    if other_data["back_price"] and other_data["back_price"] > 1.0:
                        upsets.append({
                            "name": other_name,
                            "back_price": other_data["back_price"],
                        })

                dead_certs.append({
                    "market_id": market_id,
                    "market_name": mdata["market_name"],
                    "favourite_name": rname,
                    "favourite_implied": imp,
                    "favourite_back": rdata["back_price"],
                    "upset_runners": upsets,
                })
                break  # only one favourite per market

    return dead_certs


def format_markets_for_prompt() -> str:
    """Format active markets as a concise text list for the Gemini prompt."""
    markets = get_markets()
    lines = []
    for market_id, mdata in markets.items():
        top_runners = []
        for rname, rdata in sorted(
            mdata["runners"].items(),
            key=lambda x: x[1]["implied_prob"] or 0,
            reverse=True,
        )[:5]:
            imp = rdata["implied_prob"]
            pct = f"{imp*100:.0f}%" if imp else "?"
            top_runners.append(f"{rname} ({pct})")

        lines.append(
            f"- {mdata['market_name']}: {', '.join(top_runners)}"
        )

    return "\n".join(lines)


def format_dead_certs_for_prompt() -> str:
    """Format dead cert markets for the shock detection prompt."""
    dead_certs = get_dead_certs()
    if not dead_certs:
        return ""

    lines = []
    for dc in dead_certs:
        upset_str = ", ".join(
            f"{u['name']} @ {u['back_price']:.0f}"
            for u in dc["upset_runners"][:3]
        )
        lines.append(
            f"- {dc['market_name']}: {dc['favourite_name']} "
            f"({dc['favourite_implied']*100:.0f}% certain). "
            f"Upsets: {upset_str}"
        )

    return "\n".join(lines)
```

**Step 2: Test the market cache**

Run: `uv run python3 -c "import markets; m = markets.get_markets(force_refresh=True); print(f'{len(m)} markets loaded'); dc = markets.get_dead_certs(); print(f'{len(dc)} dead certs'); print(markets.format_markets_for_prompt()[:500])"`

Expected: Markets loaded, dead certs found, formatted prompt text.

**Step 3: Commit**

```bash
git add markets.py
git commit -m "feat: add Betfair politics market cache module"
```

---

### Task 4: Build Gemini Analyser Module

**Files:**
- Create: `analyser.py`

**Step 1: Create analyser.py**

```python
"""
Gemini-powered news article analyser for Betfair politics markets.

Two-stage pipeline:
1. Relevance filter — is this article relevant to any open market?
2. Impact analysis — which runners affected, direction, magnitude
Plus shock detection on dead-cert markets.
"""

import json
import logging
import time
from datetime import datetime, timezone

from google import genai
from google.genai import types

import config

logger = logging.getLogger("oddshawk.analyser")

# Rate limiter: track call timestamps
_call_times: list[float] = []
MAX_CALLS_PER_MINUTE = 10


def _rate_limit():
    """Block if we've exceeded MAX_CALLS_PER_MINUTE."""
    now = time.time()
    # Prune old entries
    while _call_times and _call_times[0] < now - 60:
        _call_times.pop(0)

    if len(_call_times) >= MAX_CALLS_PER_MINUTE:
        wait = 60 - (now - _call_times[0])
        if wait > 0:
            logger.info("Rate limiting Gemini calls — waiting %.1fs", wait)
            time.sleep(wait)

    _call_times.append(time.time())


def _call_gemini(prompt: str) -> str | None:
    """Call Gemini Flash and return text response."""
    if not config.GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not set")
        return None

    _rate_limit()

    try:
        client = genai.Client(api_key=config.GEMINI_API_KEY)
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,  # low creativity, high precision
                max_output_tokens=2000,
            ),
        )
        return response.text
    except Exception as e:
        logger.error("Gemini API call failed: %s", e)
        return None


def _parse_json(text: str) -> dict | list | None:
    """Extract JSON from Gemini response (handles markdown code blocks)."""
    if not text:
        return None

    # Strip markdown code block if present
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Remove first and last lines
        lines = cleaned.split("\n")
        lines = lines[1:]  # remove ```json
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Failed to parse Gemini JSON: %s", text[:200])
        return None


def check_relevance(title: str, summary: str, markets_text: str) -> dict | None:
    """
    Stage 1: Quick relevance check.

    Returns {"relevant": bool, "markets": ["market_name", ...]} or None on error.
    """
    prompt = f"""You are a political betting analyst. Given this news article headline and summary, does it potentially affect any of these Betfair political markets?

Markets:
{markets_text}

Article headline: {title}
Article summary: {summary[:500]}

Reply ONLY with JSON (no explanation): {{"relevant": true/false, "markets": ["market_name_1", ...]}}
If not relevant to any market, reply: {{"relevant": false, "markets": []}}"""

    result = _call_gemini(prompt)
    return _parse_json(result)


def analyse_impact(title: str, full_text: str,
                   affected_markets: list[dict]) -> list[dict] | None:
    """
    Stage 2: Detailed impact analysis.

    affected_markets: list of {market_name, runners: {name: {back_price, implied_prob}}}

    Returns list of:
    {
        "market_name": str,
        "runner_name": str,
        "direction": "more_likely" | "less_likely",
        "magnitude": "small" | "medium" | "large" | "shock",
        "estimated_prob": float (0-1),
        "confidence": "low" | "medium" | "high",
        "reasoning": str
    }
    """
    markets_str = ""
    for m in affected_markets:
        markets_str += f"\n{m['market_name']}:\n"
        for rname, rdata in m["runners"].items():
            imp = rdata.get("implied_prob")
            pct = f"{imp*100:.0f}%" if imp else "?"
            back = rdata.get("back_price", "?")
            markets_str += f"  - {rname}: back {back}, implied {pct}\n"

    prompt = f"""You are a political betting analyst. Analyse how this article affects the following Betfair markets.

For each affected runner, estimate:
- runner_name: exact name from the list below
- direction: "more_likely" or "less_likely"
- magnitude: "small" (1-3%), "medium" (3-10%), "large" (10%+), or "shock" (reprices a dead cert)
- estimated_prob: your estimated fair probability as a decimal (0.0 to 1.0)
- confidence: "low", "medium", or "high"
- reasoning: one sentence

Current market state:
{markets_str}

Article: {title}
{full_text[:3000]}

Reply ONLY with a JSON array of objects. If no meaningful impact, reply with an empty array: []"""

    result = _call_gemini(prompt)
    return _parse_json(result)


def check_shock(title: str, summary: str,
                dead_certs_text: str) -> dict | None:
    """
    Shock detection: does this article threaten any dead-cert market?

    Returns {"shock": true/false, "market": "...", "runner": "...", "reasoning": "..."} or None.
    """
    if not dead_certs_text:
        return None

    prompt = f"""You are a political betting analyst specialising in upset detection.

These Betfair markets are currently priced as near-certainties (95%+ probability):
{dead_certs_text}

Does this news article provide evidence that ANY of these "certainties" might be wrong?
Even a 5% chance of upset is worth flagging since these markets offer 10-50x returns.
Be aggressive in flagging potential upsets — false positives are OK, missed shocks are not.

Article: {title}
{summary[:500]}

Reply ONLY with JSON: {{"shock": true/false, "market": "market name", "runner": "upset runner name", "reasoning": "one sentence"}}
If no shock potential, reply: {{"shock": false, "market": "", "runner": "", "reasoning": ""}}"""

    result = _call_gemini(prompt)
    return _parse_json(result)
```

**Step 2: Test Gemini connectivity**

Run: `uv run python3 -c "from analyser import _call_gemini; r = _call_gemini('Reply with just the word OK'); print(r)"`
Expected: `OK` (or similar short response)

**Step 3: Commit**

```bash
git add analyser.py
git commit -m "feat: add Gemini analyser module with relevance/impact/shock detection"
```

---

### Task 5: Build RSS Feed Poller

**Files:**
- Create: `feeds.py`

**Step 1: Create feeds.py**

```python
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
        from io import StringIO

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
```

**Step 2: Test feed polling**

Run: `uv run python3 -c "import models; models.init_db(); import feeds; arts = feeds.poll_feeds(); print(f'{len(arts)} new articles'); [print(f'  {a[\"feed_source\"]}: {a[\"title\"][:80]}') for a in arts[:5]]"`

Expected: Several articles fetched and printed.

**Step 3: Commit**

```bash
git add feeds.py
git commit -m "feat: add RSS feed poller with dedup and full text extraction"
```

---

### Task 6: Add Telegram Notification for News Signals

**Files:**
- Modify: `notify.py`

**Step 1: Add news signal notification functions to notify.py**

Append to `notify.py`:

```python
def _build_news_signal_message(*, signal_type, market_name, runner_name,
                                current_back, current_implied, estimated_prob,
                                edge_pct, reasoning, article_title,
                                feed_source, published_at):
    """Build Telegram message for a news-driven signal."""
    if signal_type == "shock":
        emoji = "\U0001f4a5"  # explosion
        header = "SHOCK SIGNAL"
    else:
        emoji = "\U0001f4f0"  # newspaper
        header = "NEWS SIGNAL"

    imp_pct = f"{current_implied * 100:.0f}%" if current_implied else "?"
    est_pct = f"{estimated_prob * 100:.0f}%" if estimated_prob else "?"
    back_str = f"{current_back:.2f}" if current_back else "?"
    edge_str = f"{edge_pct * 100:.1f}%" if edge_pct else "?"

    pub_str = ""
    if published_at:
        pub_str = f"\nPublished: {_format_time(published_at)}"

    feed_label = feed_source.replace("_", " ").title() if feed_source else "?"

    lines = [
        f"{emoji} {header}",
        f"",
        f"{market_name}",
        f"{runner_name} @ {back_str} (market: {imp_pct})",
        f"",
        f"Gemini estimate: {est_pct}",
        f"Edge: {edge_str}",
        f"",
        f"Reason: {reasoning}",
        f"",
        f"Source: {feed_label}",
        f'"{article_title[:100]}"',
    ]
    if pub_str:
        lines.append(pub_str)

    return "\n".join(lines)


def send_news_signal_alert(**kwargs):
    """Fire-and-forget news signal notification."""
    if not config.TELEGRAM_ENABLED:
        return

    text = _build_news_signal_message(**kwargs)
    thread = threading.Thread(target=_send_telegram, args=(text,), daemon=True)
    thread.start()
```

**Step 2: Test the message builder**

Run: `uv run python3 -c "from notify import _build_news_signal_message; print(_build_news_signal_message(signal_type='shock', market_name='Starmer Replaced by April 2026', runner_name='Yes', current_back=42.0, current_implied=0.024, estimated_prob=0.15, edge_pct=5.25, reasoning='Labour MPs submitting no-confidence letters', article_title='Labour crisis: 50 MPs call for Starmer resignation', feed_source='bbc_politics', published_at='2026-02-22T10:00:00Z'))"`

Expected: Formatted shock signal message.

**Step 3: Commit**

```bash
git add notify.py
git commit -m "feat: add news signal Telegram notification format"
```

---

### Task 7: Build the Newsbot Main Loop

**Files:**
- Create: `newsbot.py`

**Step 1: Create newsbot.py**

```python
"""
Oddshawk newsbot — monitors political news feeds and generates
trading signals against Betfair politics markets.

Standalone process. Run: python3 newsbot.py
"""

import logging
import sys
import time
from datetime import datetime, timezone
from logging.handlers import TimedRotatingFileHandler

import config
import models
import markets
import feeds
import analyser
import notify

logger = logging.getLogger("oddshawk.newsbot")

# Cooldown tracker: market_id -> last signal timestamp
_signal_cooldowns: dict[str, float] = {}


def setup_logging():
    """Configure logging with daily rotation."""
    root = logging.getLogger("oddshawk")
    root.setLevel(logging.DEBUG)

    fh = TimedRotatingFileHandler(
        config.LOG_PATH.replace(".log", "-newsbot.log")
        if config.LOG_PATH != "oddshawk.log"
        else "newsbot.log",
        when="midnight",
        interval=1,
        backupCount=7,
        utc=True,
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    ))

    root.addHandler(fh)
    root.addHandler(ch)


def _is_on_cooldown(market_id: str) -> bool:
    """Check if we recently signalled on this market."""
    last = _signal_cooldowns.get(market_id, 0)
    return (time.time() - last) < config.NEWSBOT_COOLDOWN


def _record_cooldown(market_id: str):
    """Record that we just signalled on this market."""
    _signal_cooldowns[market_id] = time.time()


def process_article(article: dict):
    """
    Run the full analysis pipeline on a single article.

    1. Relevance filter (Gemini)
    2. Shock detection on dead certs (Gemini)
    3. Impact analysis on relevant markets (Gemini)
    4. Generate signals + sim bets
    """
    title = article["title"]
    summary = article["summary"]
    full_text = article["full_text"] or summary
    article_id = article["id"]

    now_iso = datetime.now(timezone.utc).isoformat()

    # Get current market state
    all_markets = markets.get_markets()
    if not all_markets:
        logger.warning("No markets available — skipping article")
        return

    markets_text = markets.format_markets_for_prompt()
    dead_certs_text = markets.format_dead_certs_for_prompt()

    # --- Stage 1: Relevance filter ---
    relevance = analyser.check_relevance(title, summary, markets_text)
    if not relevance or not relevance.get("relevant"):
        logger.debug("Not relevant: %s", title[:80])
        # Still check for shocks even if not "relevant" to competitive markets
        if dead_certs_text:
            _check_shock(article, dead_certs_text, all_markets, now_iso)
        # Mark as analysed
        with models.get_db() as conn:
            conn.execute("UPDATE articles SET analysed = 1 WHERE id = ?",
                         (article_id,))
        return

    relevant_market_names = relevance.get("markets", [])
    logger.info("Relevant to %d markets: %s — %s",
                len(relevant_market_names), relevant_market_names, title[:80])

    # --- Stage 2: Impact analysis ---
    # Build affected markets list with current prices
    affected = []
    for market_id, mdata in all_markets.items():
        if mdata["market_name"] in relevant_market_names:
            affected.append({
                "market_id": market_id,
                "market_name": mdata["market_name"],
                "runners": mdata["runners"],
            })

    if affected:
        impacts = analyser.analyse_impact(title, full_text, affected)
        if impacts:
            _process_impacts(article, impacts, all_markets, now_iso)

    # --- Shock detection (always runs) ---
    if dead_certs_text:
        _check_shock(article, dead_certs_text, all_markets, now_iso)

    # Mark as analysed
    with models.get_db() as conn:
        conn.execute("UPDATE articles SET analysed = 1 WHERE id = ?",
                     (article_id,))


def _process_impacts(article: dict, impacts: list[dict],
                     all_markets: dict, now_iso: str):
    """Process Gemini impact analysis results into signals."""
    for impact in impacts:
        market_name = impact.get("market_name", "")
        runner_name = impact.get("runner_name", "")
        estimated_prob = impact.get("estimated_prob")
        confidence = impact.get("confidence", "low")
        reasoning = impact.get("reasoning", "")
        magnitude = impact.get("magnitude", "small")

        if confidence == "low" and magnitude == "small":
            continue  # Skip low-confidence small moves

        # Find the market
        market_id = None
        for mid, mdata in all_markets.items():
            if mdata["market_name"] == market_name:
                market_id = mid
                break

        if not market_id:
            continue

        if _is_on_cooldown(market_id):
            logger.debug("Cooldown active for %s — skipping", market_name)
            continue

        mdata = all_markets[market_id]
        runner_data = mdata["runners"].get(runner_name, {})
        current_back = runner_data.get("back_price")
        current_lay = runner_data.get("lay_price")
        current_implied = runner_data.get("implied_prob")

        if not current_back or not estimated_prob:
            continue

        # Calculate edge
        fair_price = 1.0 / estimated_prob if estimated_prob > 0 else 999
        edge_pct = (fair_price - current_back) / current_back if current_back > 0 else 0

        if edge_pct < config.NEWSBOT_EDGE_THRESHOLD:
            logger.debug("Edge %.1f%% below threshold for %s / %s",
                         edge_pct * 100, market_name, runner_name)
            continue

        # Signal!
        _record_signal(
            article=article,
            market_id=market_id,
            market_name=market_name,
            runner_name=runner_name,
            signal_type="edge",
            current_back=current_back,
            current_lay=current_lay,
            current_implied=current_implied,
            estimated_prob=estimated_prob,
            edge_pct=edge_pct,
            reasoning=reasoning,
            now_iso=now_iso,
        )


def _check_shock(article: dict, dead_certs_text: str,
                 all_markets: dict, now_iso: str):
    """Run shock detection on dead-cert markets."""
    shock = analyser.check_shock(
        article["title"], article["summary"], dead_certs_text
    )

    if not shock or not shock.get("shock"):
        return

    market_name = shock.get("market", "")
    runner_name = shock.get("runner", "")
    reasoning = shock.get("reasoning", "")

    # Find the market
    market_id = None
    for mid, mdata in all_markets.items():
        if mdata["market_name"] == market_name:
            market_id = mid
            break

    if not market_id:
        logger.warning("Shock detected but can't find market: %s", market_name)
        return

    if _is_on_cooldown(market_id):
        return

    mdata = all_markets[market_id]
    runner_data = mdata["runners"].get(runner_name, {})
    current_back = runner_data.get("back_price")
    current_lay = runner_data.get("lay_price")
    current_implied = runner_data.get("implied_prob")

    _record_signal(
        article=article,
        market_id=market_id,
        market_name=market_name,
        runner_name=runner_name,
        signal_type="shock",
        current_back=current_back,
        current_lay=current_lay,
        current_implied=current_implied,
        estimated_prob=None,  # shock signals don't have precise estimates
        edge_pct=None,
        reasoning=reasoning,
        now_iso=now_iso,
    )


def _record_signal(*, article, market_id, market_name, runner_name,
                   signal_type, current_back, current_lay,
                   current_implied, estimated_prob, edge_pct,
                   reasoning, now_iso):
    """Log a signal to DB, send notification, place sim bet."""
    _record_cooldown(market_id)

    edge_str = f"{edge_pct*100:.1f}%" if edge_pct else "SHOCK"
    logger.info(
        "SIGNAL [%s] %s — %s / %s @ %.2f (edge: %s) — %s",
        signal_type.upper(), market_name, runner_name,
        f"back {current_back}" if current_back else "?",
        current_implied or 0, edge_str, reasoning[:80],
    )

    with models.get_db() as conn:
        signal_id = models.insert_news_signal(
            conn,
            article_id=article["id"],
            market_id=market_id,
            market_name=market_name,
            runner_name=runner_name,
            signal_type=signal_type,
            current_back_price=current_back,
            current_lay_price=current_lay,
            current_implied_prob=current_implied,
            estimated_fair_prob=estimated_prob,
            edge_pct=edge_pct,
            gemini_reasoning=reasoning,
            detected_at=now_iso,
        )

        # Sim bet
        if current_back and current_back > 1.0:
            stake = config.NEWSBOT_SIM_STAKE
            payout = stake * current_back
            models.insert_news_bet(
                conn,
                signal_id=signal_id,
                market_id=market_id,
                market_name=market_name,
                runner_name=runner_name,
                side="back",
                odds_at_detection=current_back,
                stake=stake,
                potential_payout=payout,
                placed_at=now_iso,
            )
            logger.info(
                "SIM BET: Back %s @ %.2f, stake $%.0f, payout $%.0f",
                runner_name, current_back, stake, payout,
            )

    # Telegram
    notify.send_news_signal_alert(
        signal_type=signal_type,
        market_name=market_name,
        runner_name=runner_name,
        current_back=current_back,
        current_implied=current_implied,
        estimated_prob=estimated_prob,
        edge_pct=edge_pct,
        reasoning=reasoning,
        article_title=article["title"],
        feed_source=article["feed_source"],
        published_at=article.get("published_at", ""),
    )


def main():
    setup_logging()
    logger.info("Oddshawk newsbot starting up...")
    logger.info("Gemini model: %s", config.GEMINI_MODEL)
    logger.info("Poll interval: %ds, Market refresh: %ds",
                config.NEWSBOT_POLL_INTERVAL, config.NEWSBOT_MARKET_REFRESH)
    logger.info("Edge threshold: %.1f%%", config.NEWSBOT_EDGE_THRESHOLD * 100)

    models.init_db()
    logger.info("Database initialized")

    # Initial market load
    logger.info("Loading Betfair politics markets...")
    m = markets.get_markets(force_refresh=True)
    dc = markets.get_dead_certs()
    logger.info("Loaded %d markets (%d dead certs)", len(m), len(dc))

    # Main loop
    logger.info("Newsbot running — polling feeds every %ds",
                config.NEWSBOT_POLL_INTERVAL)
    try:
        while True:
            new_articles = feeds.poll_feeds()
            for article in new_articles:
                try:
                    process_article(article)
                except Exception as e:
                    logger.error("Error processing article '%s': %s",
                                 article["title"][:60], e, exc_info=True)

            time.sleep(config.NEWSBOT_POLL_INTERVAL)
    except KeyboardInterrupt:
        logger.info("Newsbot shutting down...")
        sys.exit(0)


if __name__ == "__main__":
    main()
```

**Step 2: Test newsbot startup (dry run — Ctrl+C after it loads)**

Run: `uv run python3 -c "import newsbot; newsbot.setup_logging(); import models; models.init_db(); import markets; m = markets.get_markets(force_refresh=True); print(f'Ready: {len(m)} markets'); import feeds; arts = feeds.poll_feeds(); print(f'{len(arts)} articles'); newsbot.process_article(arts[0]) if arts else print('No articles to test')"`

Expected: Markets loaded, articles fetched, first article analysed through Gemini pipeline.

**Step 3: Commit**

```bash
git add newsbot.py
git commit -m "feat: add newsbot main loop with full analysis pipeline"
```

---

### Task 8: Update Docker Compose and Dockerfile

**Files:**
- Modify: `docker-compose.yml`
- Modify: `Dockerfile` (no changes needed — already copies *.py)

**Step 1: Add newsbot service to docker-compose.yml**

Add after the dashboard service:

```yaml
  newsbot:
    build: .
    container_name: oddshawk-newsbot
    restart: unless-stopped
    env_file: .env
    environment:
      - DB_PATH=/app/data/oddshawk.db
      - LOG_PATH=/app/data/newsbot.log
    volumes:
      - oddshawk-data:/app/data
      - ./betfair-client.crt:/app/betfair-client.crt:ro
      - ./betfair-client.key:/app/betfair-client.key:ro
    command: ["python3", "newsbot.py"]
```

**Step 2: Commit**

```bash
git add docker-compose.yml
git commit -m "feat: add newsbot Docker service"
```

---

### Task 9: Update .env with Newsbot Settings and Test End-to-End

**Files:**
- Modify: `.env` (local only, not committed)

**Step 1: Add newsbot vars to .env**

```
NEWSBOT_ENABLED=true
NEWSBOT_POLL_INTERVAL=15
NEWSBOT_MARKET_REFRESH=60
NEWSBOT_EDGE_THRESHOLD=0.05
NEWSBOT_MIN_MATCHED=500
NEWSBOT_SIM_STAKE=100
GEMINI_MODEL=gemini-2.0-flash
```

(GEMINI_API_KEY is already in .env)

**Step 2: Run newsbot locally for 60 seconds**

Run: `timeout 60 uv run python3 newsbot.py 2>&1 || true`

Expected: Starts up, loads markets, polls feeds, analyses articles, may or may not find signals (depends on current news). Should not crash.

**Step 3: Check DB for articles and any signals**

Run:
```bash
uv run python3 -c "
import sqlite3
conn = sqlite3.connect('oddshawk.db')
arts = conn.execute('SELECT COUNT(*) FROM articles').fetchone()[0]
sigs = conn.execute('SELECT COUNT(*) FROM news_signals').fetchone()[0]
print(f'Articles: {arts}, News signals: {sigs}')
conn.close()
"
```

**Step 4: Push and deploy to NAS**

```bash
git push
# On NAS:
# git pull && docker compose up -d --build
```
