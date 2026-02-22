# Oddshawk Design — v1 PoC

## Purpose

Poll The Odds API to compare Australian soft bookmaker odds against Betfair
Exchange AU prices. When a soft book offers better odds than Betfair's implied
true probability, log a value signal and simulate a flat-stake bet. Track P&L
over time to identify persistent edges by sport and bookmaker.

Observation only — no real money, no automated betting.

## Stack

- Python 3.11+, sqlite3, requests, schedule, python-dotenv, smtplib

## Architecture

Single-process polling app. Three concerns:

1. **Scanner** — polls The Odds API on a rotating schedule, stores snapshots,
   detects value signals, places simulated bets on strong signals.
2. **Digest** — daily 8am AEST markdown report with signal stats and sim P&L.
3. **Config** — loads `.env`, provides typed access to all settings.

No async, no ORM, no migration system. Flat module structure.

## Credit-Aware Rotation (Free Tier: 500 credits/month)

Each API call costs 1 credit (1 market, 1 region). Budget: ~16 calls/day.

### Two phases:

1. **Exploration (days 1-7):** Equal rotation across all 3 sports. ~5 calls
   per sport per day, roughly every 90 minutes per rotation.
2. **Adaptive (day 8+):** Weight credit allocation toward sports producing
   the most signals. Sport with 60% of historical signals gets ~60% of daily
   credit budget.

Scanner checks `x-requests-remaining` header and stops if daily budget
exhausted. Hard stop if monthly credits drop below 10 (emergency reserve).

## Target Sports

- `basketball_nba`
- `aussierules_afl`
- `rugbyleague_nrl`

## Bookmakers

Soft books: sportsbet, tab, ladbrokes_au, neds, pointsbetau, unibet,
tabtouch, betright, betr_au

Truth anchor: betfair_ex_au

## Signal Detection

For each event and outcome:

1. betfair_raw_implied = 1 / betfair_back_price
2. betfair_true_prob = betfair_raw_implied * (1 - BETFAIR_COMMISSION)
3. soft_implied = 1 / soft_book_decimal_odds
4. edge_pct = betfair_true_prob - soft_implied
5. If edge_pct > VALUE_THRESHOLD (default 5%) → value signal
6. If edge_pct > BET_THRESHOLD (default 8%) → simulated bet

Outcome matching: exact string match (Odds API normalizes names within
each event response).

Deduplication: skip if same (event_id, bookmaker, outcome) signal exists
within last 30 minutes.

## SQLite Schema

### odds_snapshots
All soft bookmaker odds captured each cycle.

### betfair_snapshots
Betfair Exchange prices captured each cycle.

### value_signals
Detected edges above VALUE_THRESHOLD. Fields include edge_pct, resolved
flag, outcome_won for manual settlement.

### simulated_bets
Auto-created when edge_pct >= BET_THRESHOLD. Flat stake, manual settlement.
Fields: signal_id, odds_at_detection, stake, potential_payout, settled,
result (win/loss/push/void), pnl.

### credit_usage
Daily credit tracking: date, credits_used, credits_remaining.

## Digest (daily 8am AEST)

Sections:
1. Summary — signals in last 24h, sports covered
2. Top 10 signals by edge_pct
3. Breakdown by bookmaker — signal count, avg edge
4. Breakdown by sport — signal count, avg edge
5. Simulated P&L — total staked, returned, net P&L, ROI%, win rate
6. P&L by bookmaker and sport
7. Unsettled bets list
8. Credit usage — today, remaining, burn rate, projected month-end

Output: stdout + `digest_YYYY-MM-DD.md`. Optional email via SMTP.

## Error Handling

- API errors → log warning, skip sport, continue
- Missing Betfair → skip event, log debug
- Rate limit (429) → sleep 60s, retry once, then skip
- Daily budget exhausted → stop scanning until next day
- DB errors → log error, continue

## Environment Variables

```
ODDS_API_KEY, VALUE_THRESHOLD=0.05, BET_THRESHOLD=0.08,
BETFAIR_COMMISSION=0.05, SIM_STAKE=100, MONTHLY_CREDIT_BUDGET=500,
EXPLORATION_DAYS=7, DIGEST_TIME=08:00, EMAIL_ENABLED=false,
EMAIL_TO, EMAIL_FROM, EMAIL_SMTP, EMAIL_PORT=587, EMAIL_PASSWORD,
DB_PATH=oddshawk.db, LOG_PATH=oddshawk.log
```

## File Structure

```
oddshawk/
  main.py          # entry point, scheduler
  scanner.py       # API polling, signal detection, sim bets
  models.py        # SQLite schema and queries
  digest.py        # daily markdown report
  config.py        # .env loader
  .env.example
  requirements.txt
  README.md
```
