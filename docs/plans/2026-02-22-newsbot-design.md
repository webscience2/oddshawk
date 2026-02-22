# Newsbot: News-Reactive Betfair Politics Trading Bot

## Overview

A standalone process that monitors RSS news feeds, analyses articles with Gemini Flash to detect market-moving events, compares against live Betfair politics market prices, and generates trading signals (simulated initially, real bets later).

## Core Insight

Betfair political markets are thin and slow compared to financial markets. When news breaks, prices lag by minutes. An LLM can parse a news article and determine market impact faster than human punters can read the headline and navigate to Betfair.

Two signal types:

1. **Edge signals** — news shifts fair probability by a few percent on competitive markets (e.g., a poll showing Reform gaining on Labour shifts UK "Most Seats" prices)
2. **Shock signals** — news invalidates a "dead cert" market priced at 95%+. These are rare but offer 10-50x returns (e.g., "Labour MPs submit no-confidence letters" reprices Starmer replacement from 1.01 to 3.00+)

## Architecture

```
newsbot.py (standalone process)
    |
    +-- RSS Poller (10-15s interval)
    |       polls feeds, deduplicates by URL/GUID
    |
    +-- Article Fetcher
    |       downloads full text for new articles
    |
    +-- Market Snapshot Cache
    |       refreshes Betfair politics markets every 60s
    |       tracks current prices, liquidity, implied probabilities
    |
    +-- Gemini Analyser
    |       sends article + market list to Gemini Flash
    |       asks: which markets affected, direction, magnitude
    |       separate prompt for shock detection on dead-cert markets
    |
    +-- Signal Engine
    |       compares Gemini's probability estimate vs current market price
    |       if edge > threshold: log signal, notify, sim bet
    |
    +-- Reuses from oddshawk:
            betfair.py  — market listing, depth fetching
            models.py   — new tables (articles, news_signals, news_bets)
            notify.py   — Telegram alerts
            config.py   — new config vars
```

Docker: new `newsbot` service in docker-compose.yml, same image, different entrypoint.

## RSS Feeds

### UK Politics (highest liquidity)
- BBC News Politics: http://feeds.bbci.co.uk/news/politics/rss.xml
- Guardian UK Politics: https://www.theguardian.com/politics/rss
- Reuters UK: https://www.reuters.com/rssFeed/UKNews (or world news)
- Sky News Politics: https://feeds.skynews.com/feeds/rss/politics.xml

### US Politics
- AP News Politics: https://apnews.com/politics/feed
- Reuters US Politics: https://www.reuters.com/rssFeed/politicsNews
- Politico: https://www.politico.com/rss/politics08.xml
- FiveThirtyEight/538: RSS if available

### AU Politics (local news advantage)
- ABC News Australia: https://www.abc.net.au/news/feed/2942460/rss.xml
- Guardian Australia: https://www.theguardian.com/australia-news/rss
- SBS News: https://www.sbs.com.au/news/feed

### General / Wire
- Reuters World: https://www.reuters.com/rssFeed/worldNews
- AP Top News: https://apnews.com/feed

## Database Tables (new, via ALTER TABLE or new tables)

### articles
- id, url (unique), guid, feed_source, title, summary, full_text, published_at, fetched_at, analysed (bool)

### news_signals
- id, article_id, market_id, market_name, runner_name, signal_type (edge/shock), current_back_price, current_lay_price, current_implied_prob, estimated_fair_prob, edge_pct, gemini_reasoning, detected_at

### news_bets (simulated)
- id, signal_id, market_id, market_name, runner_name, side (back/lay), odds_at_detection, stake, potential_payout, placed_at, settled, result, pnl

## Gemini Prompt Design

Two-stage analysis:

### Stage 1: Relevance Filter (cheap, fast)
```
You are a political betting analyst. Given this news article headline and summary,
does it potentially affect any of these Betfair political markets?

Markets: [list of market names only]

Article: [headline + first 200 words]

Reply with JSON: {"relevant": true/false, "markets": ["market_name_1", ...]}
```

### Stage 2: Impact Analysis (only if relevant)
```
You are a political betting analyst. Analyse how this article affects the
following Betfair markets.

For each affected market, estimate:
- Which runner(s) are affected
- Direction: does this make the outcome MORE or LESS likely?
- Magnitude: small (1-3%), medium (3-10%), large (10%+), shock (reprices dead cert)
- Confidence: low/medium/high
- Brief reasoning (1 sentence)

Current market state:
[market name, runners with current back/lay/implied prob]

Article:
[full text]

Reply with JSON array.
```

### Shock Detection (runs on every article against dead-cert markets)
```
These markets are currently priced as near-certainties (95%+):
[list with current prices]

Does this article provide evidence that ANY of these "certainties" might be wrong?
Even a 5% chance of upset is worth flagging since these markets offer 10-50x returns.

Reply with JSON: {"shock": true/false, "market": "...", "reasoning": "..."}
```

## Signal Logic

### Edge Signal
- Gemini estimates fair probability for a runner
- Compare to current Betfair implied probability
- Edge = (1/fair_prob - current_back_price) / current_back_price
- If edge > NEWS_EDGE_THRESHOLD (default 5%): signal
- Must have minimum liquidity (lay_liq > $50 for politics)

### Shock Signal
- Gemini flags a dead-cert market as potentially wrong
- Current price implies 95%+ for one outcome
- If Gemini confidence is medium+ on the upset: signal
- These get higher priority Telegram alerts

## Config (new vars in .env)

```
# Newsbot
NEWSBOT_ENABLED=true
NEWSBOT_POLL_INTERVAL=15          # seconds between RSS checks
NEWSBOT_MARKET_REFRESH=60         # seconds between Betfair market refreshes
NEWSBOT_EDGE_THRESHOLD=0.05       # 5% minimum edge for signal
NEWSBOT_MIN_MATCHED=500           # minimum total matched for market to be considered
NEWSBOT_SIM_STAKE=100             # simulated bet size

# Gemini
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.0-flash     # fast + cheap
```

## Telegram Notifications

Reuses notify.py with new message format:

```
[shock signal]
SHOCK SIGNAL

UK Politics
Starmer Replaced by April 1st 2026?

YES now tradeable @ 42.00 (implied 2.4%)
Gemini estimates: 15-25% likely
Edge: massive

Source: BBC News - "Labour MPs submit 50 no-confidence letters"
Published: 2 minutes ago

[normal edge signal]
NEWS SIGNAL

US 2028 Election
Gavin Newsom — currently 5.10 back (20%)

Gemini estimate: 25% (+5%)
Edge: 5.2%

Source: Reuters - "Newsom announces 2028 exploratory committee"
Published: 45 seconds ago
```

## Docker Integration

```yaml
# Added to existing docker-compose.yml
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

## Requirements (additions to requirements.txt)

```
feedparser>=6.0.0        # RSS parsing
google-genai>=1.0.0      # Gemini API (or google-generativeai)
```

## Risk Controls

- Rate limit Gemini calls: max 10/minute (avoid burning API budget on noisy feeds)
- Dedup: skip articles seen in last 24h (by URL and title similarity)
- Cooldown: after signalling on a market, wait 5 minutes before signalling again (avoid repeated signals from follow-up articles on same story)
- Max sim bet exposure: $500 total across all open news bets
- Log all Gemini responses for audit trail

## Settlement

News bets are harder to settle than sports bets since political markets can stay open for months/years. Options:
- Monitor Betfair market status (CLOSED = settled, same as sports)
- For sim bets, mark-to-market daily (current price vs entry price = unrealised P&L)
- Manual settlement via dashboard for long-running markets

## Future Enhancements (not in v1)

- Twitter/X API for faster news (30-60s ahead of RSS)
- Multiple LLM ensemble (Gemini + Claude) for higher confidence
- Real bet placement via Betfair API
- Historical backtesting against article timestamps vs price movements
- Polling data integration (FiveThirtyEight, YouGov)
- Correlation tracking: which feeds/topics generate profitable signals
