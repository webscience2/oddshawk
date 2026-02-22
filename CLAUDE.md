# Oddshawk

## Critical Rules
- **NEVER delete oddshawk.db** — contains captured API data that costs credits to regenerate
- Schema changes MUST use `ALTER TABLE ADD COLUMN` — never drop/recreate tables
- Free tier: 500 credits/month — every API call costs 1 credit, be conservative

## Environment
- Python 3.14 via homebrew — use `python3` not `python`
- No `timeout`/`gtimeout` available — use Python subprocess for timed runs
- `uv` available at `/opt/homebrew/bin/uv` — `uv run main.py` handles venv+deps
- Flask dashboard runs on port 5050 (not 5000 — macOS AirPlay Receiver conflict)

## Architecture
- Single-process: scanner (polling) + Flask dashboard (thread) + schedule (digest)
- SQLite via sqlite3 — no ORM, parameterized queries in models.py
- All timestamps stored as UTC ISO strings, display in AEST (UTC+10)

## API (The Odds API)
- Response headers are mixed-case — normalize to lowercase before lookup
- Outcome names are consistent across bookmakers within same event — exact string match
- 1 credit per call (1 market h2h, 1 region au)
- Rate limit: back off 60s on HTTP 429, retry once

## Testing
- Smoke test: run main.py for ~15s via subprocess, check dashboard HTTP 200
- Kill stale processes: `lsof -i :5050` before starting
