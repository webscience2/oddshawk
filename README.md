# Oddshawk

Sports betting value detector. Compares Australian soft bookmaker odds against
Betfair Exchange AU prices to find edges. Simulates flat-stake bets and tracks
P&L over time.

**Observation only — no real money.**

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your Odds API key
```

Get a free API key at [the-odds-api.com](https://the-odds-api.com) (500 credits/month).

## Run

```bash
python main.py
```

This starts:
- **Scanner** — polls odds on a credit-aware rotation schedule
- **Web dashboard** — http://localhost:5000 (signals monitor + P&L tracker)
- **Daily digest** — 8am AEST markdown report

## Dashboard

Open http://localhost:5000 in your browser.

- **Signals tab** — live value signals, odds comparisons, bookmaker/sport breakdowns
- **P&L tab** — simulated betting results, settle bets via Win/Loss buttons

## How it works

1. Fetches h2h odds from The Odds API for NBA, AFL, NRL
2. Uses Betfair Exchange as the "true price" anchor
3. Adjusts for 5% Betfair commission
4. Flags soft bookmakers offering better odds than Betfair's implied probability
5. Signals above 5% edge get logged; above 8% edge get a simulated $100 bet

## Credit management (free tier)

500 credits/month = ~16 API calls/day. The scanner:
- Rotates between sports to spread credits evenly (first 7 days)
- Then weights toward sports producing the most signals
- Stops scanning when daily budget is exhausted
- Tracks usage in the dashboard

## Settling bets

Bets are settled manually via the P&L page — click Win/Loss/Push for each bet.

Or via SQL:
```sql
-- Mark a bet as won
UPDATE simulated_bets SET settled=1, result='win',
  pnl=(odds_at_detection - 1) * stake WHERE id=42;

-- Mark a bet as lost
UPDATE simulated_bets SET settled=1, result='loss',
  pnl=-stake WHERE id=43;
```

## Sports & bookmakers

**Sports:** basketball_nba, aussierules_afl, rugbyleague_nrl

**Soft bookmakers:** Sportsbet, TAB, Ladbrokes AU, Neds, PointsBet AU,
Unibet, TABtouch, BetRight, Betr AU

**Truth anchor:** Betfair Exchange AU
