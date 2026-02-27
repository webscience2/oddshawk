"""
Oddshawk scanner — polls The Odds API, detects value signals, places sim bets.

Implements credit-aware rotation:
  - Exploration phase (first N days): equal rotation across sports
  - Adaptive phase: weight toward sports with most signals
"""

import logging
import time
from datetime import datetime, timezone, timedelta

import requests

import config
import models

logger = logging.getLogger("oddshawk.scanner")

# Credits per Odds API call: 1 if Betfair API handles exchange data, 2 if not
CREDITS_PER_CALL = 1 if config.BETFAIR_ENABLED else 2

# --- API key rotation ---
# Round-robin across configured keys. Each scan cycle advances to the next key.
_key_index = 0


def _next_api_key() -> str:
    """Return the next API key in the rotation and advance the index."""
    global _key_index
    key = config.ODDS_API_KEYS[_key_index % len(config.ODDS_API_KEYS)]
    _key_index += 1
    return key


def _current_api_key() -> str:
    """Return the current API key without advancing."""
    return config.ODDS_API_KEYS[_key_index % len(config.ODDS_API_KEYS)]


# Sports where The Odds API h2h and Betfair MATCH_ODDS are incompatible.
# NHL: API returns 3-way regulation (Home/Away/Draw ~82% sum) but Betfair
# is 2-way money line including OT (~100% sum). Odds snapshots are still
# captured; only Betfair edge comparison is skipped.
BETFAIR_SKIP_SPORTS: set[str] = {"icehockey_nhl"}


def _name_tokens(name: str) -> set[str]:
    """Extract meaningful tokens from a team name for fuzzy matching."""
    NOISE = {
        "fc", "sc", "afc", "cf", "the", "of", "de", "v", "vs", "versus",
        # Common team modifiers that cause false matches
        "united", "utd", "city", "town", "rovers", "wanderers",
        "athletic", "sporting", "real", "inter",
    }
    # Common abbreviations: Betfair often shortens team names
    ABBREVS = {
        "man": "manchester", "qld": "queensland",
        "nth": "north", "sth": "south", "melb": "melbourne",
        "bris": "brisbane", "syd": "sydney", "adel": "adelaide",
        "wolves": "wolverhampton", "spurs": "tottenham",
        "villa": "aston",
    }
    tokens = set()
    for word in name.lower().split():
        word = word.strip("().-,")
        if len(word) >= 3 and word not in NOISE:
            tokens.add(word)
            if word in ABBREVS:
                tokens.add(ABBREVS[word])
    return tokens


def _match_bf_depth(home: str, away: str, bf_sport_data: dict) -> dict | None:
    """Match an Odds API event to Betfair depth data by team names.

    Tries exact substring first, then token-based fuzzy matching.
    Returns bf_depth dict with _market metadata, or None.
    """
    home_lower = home.lower()
    away_lower = away.lower()

    def _build_depth(bf_data):
        bf_depth = dict(bf_data.get("runners", {}))
        bf_depth["_market"] = {
            "market_id": bf_data.get("market_id"),
            "total_matched": bf_data.get("total_matched", 0),
            "status": bf_data.get("status"),
        }
        return bf_depth

    # Pass 1: exact substring
    for bf_name, bf_data in bf_sport_data.items():
        bf_lower = bf_name.lower()
        if home_lower in bf_lower and away_lower in bf_lower:
            return _build_depth(bf_data)

    # Pass 2: token overlap (handles abbreviations)
    home_tokens = _name_tokens(home)
    away_tokens = _name_tokens(away)

    best_match = None
    best_score = 0

    for bf_name, bf_data in bf_sport_data.items():
        bf_tokens = _name_tokens(bf_name)
        home_overlap = len(home_tokens & bf_tokens)
        away_overlap = len(away_tokens & bf_tokens)

        if home_overlap >= 1 and away_overlap >= 1:
            score = home_overlap + away_overlap
            if score > best_score:
                best_score = score
                best_match = bf_data

    if best_match:
        return _build_depth(best_match)

    return None

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def fetch_odds(sport: str, api_key: str | None = None) -> tuple[list | None, dict]:
    """
    Fetch odds for a single sport from The Odds API.
    Returns (events_list_or_None, response_headers_dict).
    Uses the provided api_key, or the current rotation key if None.
    """
    url = f"{config.ODDS_API_BASE}/sports/{sport}/odds"
    # If Betfair API is enabled, we get exchange data from there — only need h2h
    markets = "h2h" if config.BETFAIR_ENABLED else "h2h,h2h_lay"
    # If Betfair API provides exchange data, only request soft bookmakers
    bookmakers = (",".join(config.SOFT_BOOKMAKERS)
                  if config.BETFAIR_ENABLED
                  else config.ALL_BOOKMAKERS)
    key = api_key or _current_api_key()
    params = {
        "apiKey": key,
        "regions": "au",
        "markets": markets,
        "includeBetLimits": "true",
        "oddsFormat": "decimal",
        "bookmakers": bookmakers,
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
    except requests.RequestException as e:
        logger.error("Network error fetching %s: %s", sport, e)
        return None, {}

    headers = dict(resp.headers)

    if resp.status_code == 429:
        logger.warning("Rate limited on %s — backing off 60s", sport)
        time.sleep(60)
        try:
            resp = requests.get(url, params=params, timeout=30)
            headers = dict(resp.headers)
        except requests.RequestException as e:
            logger.error("Retry failed for %s: %s", sport, e)
            return None, headers

    if resp.status_code != 200:
        logger.error("API error for %s: HTTP %d — %s",
                      sport, resp.status_code, resp.text[:200])
        return None, headers

    return resp.json(), headers


def parse_credit_headers(headers: dict) -> tuple[int, int]:
    """Extract credits remaining and used from response headers."""
    # Headers come back with mixed case — normalize to lowercase for lookup
    lower = {k.lower(): v for k, v in headers.items()}
    remaining = int(lower.get("x-requests-remaining", -1))
    used = int(lower.get("x-requests-used", -1))
    return remaining, used


# ---------------------------------------------------------------------------
# Rotation logic
# ---------------------------------------------------------------------------

def _get_first_signal_date(conn) -> datetime | None:
    """Get the date of the first ever credit usage to determine project age."""
    row = conn.execute(
        "SELECT MIN(date) as first_date FROM credit_usage"
    ).fetchone()
    if row and row["first_date"]:
        return datetime.fromisoformat(row["first_date"]).replace(
            tzinfo=timezone.utc
        )
    return None


def get_sports_to_scan(conn) -> list[str]:
    """
    Decide which sports to scan this cycle based on rotation strategy.

    During exploration: rotate through all sports equally.
    After exploration: weight toward sports with most signals.
    Budget-aware: skip if daily budget exhausted.
    """
    daily_budget = config.MONTHLY_CREDIT_BUDGET // 30
    used_today = models.get_credits_used_today(conn)

    if used_today >= daily_budget:
        logger.info("Daily credit budget exhausted (%d/%d). Skipping cycle.",
                     used_today, daily_budget)
        return []

    credits_left_today = daily_budget - used_today
    calls_left_today = credits_left_today // CREDITS_PER_CALL
    first_date = _get_first_signal_date(conn)
    now = datetime.now(timezone.utc)

    # During exploration or if no history: scan all sports we can afford
    if first_date is None or (now - first_date).days < config.EXPLORATION_DAYS:
        # Round-robin: pick sports we can afford this cycle
        affordable = min(len(config.SPORTS), calls_left_today)
        return config.SPORTS[:affordable]

    # Adaptive phase: weight by signal counts
    signal_counts = models.get_sport_signal_counts(conn)
    if not signal_counts:
        return config.SPORTS[:min(len(config.SPORTS), calls_left_today)]

    total_signals = sum(r["count"] for r in signal_counts)
    if total_signals == 0:
        return config.SPORTS[:min(len(config.SPORTS), calls_left_today)]

    # Build weighted sport list
    sport_weights = {r["sport"]: r["count"] / total_signals for r in signal_counts}

    # Ensure all configured sports have at least a minimum weight
    for sport in config.SPORTS:
        if sport not in sport_weights:
            sport_weights[sport] = 0.1  # 10% floor for unexplored sports

    # Normalize
    total_weight = sum(sport_weights.values())
    sport_weights = {s: w / total_weight for s, w in sport_weights.items()}

    # Allocate credits proportionally, but scan at least 1 sport
    sports_to_scan = []
    for sport in sorted(sport_weights, key=sport_weights.get, reverse=True):
        if len(sports_to_scan) >= calls_left_today:
            break
        if sport in config.SPORTS:
            sports_to_scan.append(sport)

    return sports_to_scan if sports_to_scan else config.SPORTS[:1]


# ---------------------------------------------------------------------------
# Arb calculation
# ---------------------------------------------------------------------------

def calculate_arb(back_stake: float, back_odds: float,
                  lay_odds: float, commission: float) -> dict:
    """
    Calculate a back/lay arb.

    Back at soft bookmaker, lay on Betfair.

    - back_stake: amount wagered at the soft book
    - back_odds: decimal odds at the soft book
    - lay_odds: Betfair back price (used as lay proxy since h2h
                endpoint only gives back prices)
    - commission: Betfair commission rate (e.g. 0.05)

    Returns dict with lay_stake, profits for each outcome, and ROI.
    """
    # Lay stake to equalize: covers liability if the selection wins
    # lay_stake = (back_stake * back_odds) / (lay_odds - commission * (lay_odds - 1))
    # Simplification: lay_odds adjusted for commission
    effective_lay = lay_odds - commission * (lay_odds - 1)

    if effective_lay <= 0:
        return {
            "lay_stake": 0, "profit_if_back_wins": 0,
            "profit_if_lay_wins": 0, "guaranteed_profit": 0,
            "roi_pct": 0, "total_outlay": back_stake,
        }

    lay_stake = (back_stake * back_odds) / effective_lay

    # Liability on Betfair if selection wins = lay_stake * (lay_odds - 1)
    lay_liability = lay_stake * (lay_odds - 1)

    # If back wins (selection wins):
    #   Back profit: back_stake * (back_odds - 1)
    #   Lay loss: -lay_liability
    #   Net: back_stake * (back_odds - 1) - lay_liability
    profit_if_back_wins = back_stake * (back_odds - 1) - lay_liability

    # If lay wins (selection loses):
    #   Back loss: -back_stake
    #   Lay profit: lay_stake * (1 - commission)
    #   Net: lay_stake * (1 - commission) - back_stake
    profit_if_lay_wins = lay_stake * (1 - commission) - back_stake

    # Guaranteed profit = min of both outcomes
    guaranteed_profit = min(profit_if_back_wins, profit_if_lay_wins)
    total_outlay = back_stake + lay_liability
    roi_pct = (guaranteed_profit / total_outlay * 100) if total_outlay > 0 else 0

    return {
        "lay_stake": round(lay_stake, 2),
        "lay_liability": round(lay_liability, 2),
        "profit_if_back_wins": round(profit_if_back_wins, 2),
        "profit_if_lay_wins": round(profit_if_lay_wins, 2),
        "guaranteed_profit": round(guaranteed_profit, 2),
        "total_outlay": round(total_outlay, 2),
        "roi_pct": round(roi_pct, 2),
    }


def build_bet_description(*, outcome: str, event_name: str, bookmaker: str,
                          soft_odds: float, stake: float,
                          bf_back: float, bf_lay: float | None,
                          lay_liq: float | None, arb: dict) -> str:
    """Build a plain English description of the bet."""
    back_payout = stake * soft_odds
    lay_price = bf_lay if bf_lay else bf_back
    parts = [
        f"VALUE BET: Back {outcome} to win {event_name}",
        f"  Bet ${stake:.0f} with {bookmaker} @ {soft_odds:.2f}"
        f" (pays ${back_payout:.2f} if {outcome} wins)",
        f"  Betfair: back @ {bf_back:.2f}"
        f" / lay @ {lay_price:.2f}"
        f" (implied {1/bf_back*100:.0f}% chance)",
        f"  {bookmaker} is offering better odds than the market suggests.",
    ]
    if arb["guaranteed_profit"] > 0:
        liq_note = ""
        if arb.get("liquidity_capped") and lay_liq is not None:
            liq_note = f" (WARNING: only ${lay_liq:.0f} available to lay)"
        elif lay_liq is not None:
            liq_note = f" (${lay_liq:.0f} available to lay)"
        parts.append(
            f"  ARB HEDGE: Lay ${arb['lay_stake']:.2f} on Betfair @ {lay_price:.2f}"
            f" against {outcome} to lock in"
            f" ${arb['guaranteed_profit']:.2f} guaranteed profit"
            f" ({arb['roi_pct']:.1f}% ROI){liq_note}"
        )
    else:
        parts.append(
            f"  ARB HEDGE: Not profitable after Betfair {config.BETFAIR_COMMISSION*100:.0f}% commission"
            f" (would lose ${abs(arb['guaranteed_profit']):.2f})"
        )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Signal detection
# ---------------------------------------------------------------------------

def process_event(conn, sport: str, event: dict, now_iso: str,
                  bf_depth: dict | None = None) -> int:
    """
    Process a single event: store snapshots, detect signals, place sim bets.

    bf_depth: optional Betfair API depth data for this event (from betfair.py).
              If provided, uses full order book; otherwise falls back to
              The Odds API Betfair data.

    Returns number of signals detected.
    """
    event_id = event["id"]
    home_team = event.get("home_team", "?")
    away_team = event.get("away_team", "?")
    event_name = f"{home_team} vs {away_team}"
    commence_time = event.get("commence_time", "")

    # Skip events starting within 30 minutes (odds too volatile / in-play)
    if commence_time:
        try:
            ct = datetime.fromisoformat(commence_time)
            now = datetime.now(timezone.utc)
            if ct <= now + timedelta(minutes=30):
                return 0
        except ValueError:
            pass

    bookmakers_data = event.get("bookmakers", [])
    if not bookmakers_data:
        return 0

    # Separate Betfair from soft bookmakers
    betfair_data = None
    soft_books = []
    for bm in bookmakers_data:
        if bm["key"] == config.BETFAIR_KEY:
            betfair_data = bm
        elif bm["key"] in config.SOFT_BOOKMAKERS:
            soft_books.append(bm)

    # Market total matched and market ID from Betfair API (None if not available)
    market_total_matched = 0.0
    bf_market_id = None

    # Build betfair_outcomes from either Betfair API depth or Odds API data
    betfair_outcomes = {}

    if bf_depth and "_market" in bf_depth:
        # --- Use Betfair API full depth data ---
        market_total_matched = bf_depth["_market"].get("total_matched", 0)
        bf_market_id = bf_depth["_market"].get("market_id")

        # Liquidity filter: skip thin markets
        if market_total_matched < config.MIN_BETFAIR_MATCHED:
            logger.debug(
                "Skipping %s — Betfair matched $%.0f < $%.0f minimum",
                event_name, market_total_matched, config.MIN_BETFAIR_MATCHED,
            )
            return 0

        for runner_name, data in bf_depth.items():
            if runner_name == "_market":
                continue

            back_prices = data.get("back_prices", [])
            lay_prices = data.get("lay_prices", [])

            if not back_prices:
                continue

            best_back = back_prices[0][0]  # (price, size)
            back_liq = sum(size for _, size in back_prices)
            best_lay = lay_prices[0][0] if lay_prices else None
            lay_liq = sum(size for _, size in lay_prices)

            # Skip placeholder/garbage prices — back < 1.10 means 90%+
            # implied probability, any "edge" vs soft books is noise
            if best_back < 1.10:
                continue

            raw_implied = 1.0 / best_back
            true_prob = raw_implied * (1.0 - config.BETFAIR_COMMISSION)

            betfair_outcomes[runner_name] = {
                "back_price": best_back,
                "lay_price": best_lay,
                "back_liquidity": back_liq,
                "lay_liquidity": lay_liq,
                "raw_implied": raw_implied,
                "true_prob": true_prob,
                "total_matched": data.get("total_matched", 0),
            }

            models.insert_betfair_snapshot(
                conn,
                sport=sport,
                event_id=event_id,
                event_name=event_name,
                commence_time=commence_time,
                outcome=runner_name,
                back_price=best_back,
                lay_price=best_lay,
                back_liquidity=back_liq,
                lay_liquidity=lay_liq,
                betfair_raw_implied=raw_implied,
                betfair_true_prob=true_prob,
                captured_at=now_iso,
            )

        if not betfair_outcomes:
            return 0

    elif betfair_data is not None:
        # --- Fallback: use Odds API Betfair data ---
        h2h_back_market = None
        h2h_lay_market = None
        for market in betfair_data.get("markets", []):
            if market["key"] == "h2h":
                h2h_back_market = market
            elif market["key"] == "h2h_lay":
                h2h_lay_market = market

        if h2h_back_market is None:
            return 0

        lay_by_name = {}
        if h2h_lay_market:
            for outcome in h2h_lay_market.get("outcomes", []):
                lay_by_name[outcome["name"]] = outcome

        for outcome in h2h_back_market.get("outcomes", []):
            name = outcome["name"]
            back_price = outcome["price"]
            back_liq = outcome.get("bet_limit")
            # Skip placeholder/garbage prices
            if back_price < 1.10:
                continue

            raw_implied = 1.0 / back_price
            true_prob = raw_implied * (1.0 - config.BETFAIR_COMMISSION)

            lay_data = lay_by_name.get(name, {})
            lay_price = lay_data.get("price")
            lay_liq = lay_data.get("bet_limit")

            betfair_outcomes[name] = {
                "back_price": back_price,
                "lay_price": lay_price,
                "back_liquidity": back_liq,
                "lay_liquidity": lay_liq,
                "raw_implied": raw_implied,
                "true_prob": true_prob,
            }

            models.insert_betfair_snapshot(
                conn,
                sport=sport,
                event_id=event_id,
                event_name=event_name,
                commence_time=commence_time,
                outcome=name,
                back_price=back_price,
                lay_price=lay_price,
                back_liquidity=back_liq,
                lay_liquidity=lay_liq,
                betfair_raw_implied=raw_implied,
                betfair_true_prob=true_prob,
                captured_at=now_iso,
            )
    else:
        if sport not in BETFAIR_SKIP_SPORTS:
            logger.debug("No Betfair data for %s — skipping", event_name)
            return 0

    # Process each soft bookmaker
    signals_found = 0
    for bm in soft_books:
        bm_key = bm["key"]
        for market in bm.get("markets", []):
            if market["key"] != "h2h":
                continue
            for outcome in market.get("outcomes", []):
                name = outcome["name"]
                decimal_odds = outcome["price"]
                soft_bet_limit = outcome.get("bet_limit")
                if decimal_odds <= 1.0:
                    continue

                soft_implied = 1.0 / decimal_odds

                models.insert_odds_snapshot(
                    conn,
                    sport=sport,
                    event_id=event_id,
                    event_name=event_name,
                    commence_time=commence_time,
                    bookmaker=bm_key,
                    outcome=name,
                    decimal_odds=decimal_odds,
                    implied_prob=soft_implied,
                    bet_limit=soft_bet_limit,
                    captured_at=now_iso,
                )

                # Skip Betfair edge comparison for incompatible markets
                if sport in BETFAIR_SKIP_SPORTS:
                    continue

                # Check for value against Betfair
                bf = betfair_outcomes.get(name)
                # Handle "Draw" vs "The Draw" mismatch (soccer)
                if bf is None and name == "Draw":
                    bf = betfair_outcomes.get("The Draw")
                if bf is None:
                    continue

                edge_pct = bf["true_prob"] - soft_implied

                if edge_pct < config.VALUE_THRESHOLD:
                    continue

                # Deduplication check
                if models.signal_exists_recently(conn, event_id, bm_key, name):
                    continue

                # --- Arb calculation using real lay price ---
                bf_back = bf["back_price"]
                bf_lay = bf.get("lay_price")
                back_liq = bf.get("back_liquidity")
                lay_liq = bf.get("lay_liquidity")
                stake = config.SIM_STAKE

                # Use real lay price if available, fall back to back price
                arb_lay_price = bf_lay if bf_lay else bf_back
                arb = calculate_arb(stake, decimal_odds, arb_lay_price,
                                    config.BETFAIR_COMMISSION)

                # Cap arb profit by available lay liquidity
                if lay_liq is not None and arb["lay_stake"] > lay_liq:
                    arb["liquidity_capped"] = True
                    arb["max_arb_stake"] = lay_liq
                else:
                    arb["liquidity_capped"] = False
                    arb["max_arb_stake"] = None

                signal_id = models.insert_value_signal(
                    conn,
                    sport=sport,
                    event_id=event_id,
                    event_name=event_name,
                    commence_time=commence_time,
                    bookmaker=bm_key,
                    outcome=name,
                    soft_decimal_odds=decimal_odds,
                    soft_implied_prob=soft_implied,
                    betfair_back_price=bf_back,
                    betfair_lay_price=bf_lay,
                    betfair_true_prob=bf["true_prob"],
                    edge_pct=edge_pct,
                    back_liquidity=back_liq,
                    lay_liquidity=lay_liq,
                    arb_lay_stake=arb["lay_stake"],
                    arb_profit_back=arb["profit_if_back_wins"],
                    arb_profit_lay=arb["profit_if_lay_wins"],
                    arb_roi_pct=arb["roi_pct"],
                    betfair_total_matched=market_total_matched or None,
                    betfair_market_id=bf_market_id,
                    detected_at=now_iso,
                )
                signals_found += 1

                lay_str = f"lay {bf_lay:.2f}" if bf_lay else f"back {bf_back:.2f}"
                liq_str = f" liq ${lay_liq:.0f}" if lay_liq else ""
                logger.info(
                    "SIGNAL: %s | %s | %s @ %.2f (BF %s%s) | "
                    "edge %.1f%% | arb $%.2f | %s",
                    sport, event_name, name, decimal_odds, lay_str, liq_str,
                    edge_pct * 100, arb["guaranteed_profit"], bm_key,
                )

                # Send real-time notification
                if config.TELEGRAM_ENABLED:
                    from notify import send_signal_alert
                    send_signal_alert(
                        sport=sport,
                        event_name=event_name,
                        outcome=name,
                        bookmaker=bm_key,
                        soft_odds=decimal_odds,
                        bf_back=bf_back,
                        bf_lay=bf_lay,
                        edge_pct=edge_pct,
                        arb=arb,
                        lay_liq=lay_liq,
                        commence_time=commence_time,
                    )

                # Place simulated bet if edge exceeds bet threshold
                # (one bet per event/outcome/bookmaker — skip if already open)
                if edge_pct >= config.BET_THRESHOLD and not models.simulated_bet_exists(
                    conn, event_id, name, bm_key
                ):
                    potential_payout = stake * decimal_odds
                    desc = build_bet_description(
                        outcome=name,
                        event_name=event_name,
                        bookmaker=bm_key,
                        soft_odds=decimal_odds,
                        stake=stake,
                        bf_back=bf_back,
                        bf_lay=bf_lay,
                        lay_liq=lay_liq,
                        arb=arb,
                    )

                    models.insert_simulated_bet(
                        conn,
                        signal_id=signal_id,
                        sport=sport,
                        event_id=event_id,
                        event_name=event_name,
                        commence_time=commence_time,
                        bookmaker=bm_key,
                        outcome=name,
                        odds_at_detection=decimal_odds,
                        stake=stake,
                        potential_payout=potential_payout,
                        betfair_back_price=bf_back,
                        betfair_lay_price=bf_lay,
                        lay_liquidity=lay_liq,
                        arb_lay_stake=arb["lay_stake"],
                        arb_profit=arb["guaranteed_profit"],
                        betfair_total_matched=market_total_matched or None,
                        betfair_market_id=bf_market_id,
                        description=desc,
                        placed_at=now_iso,
                    )
                    logger.info("SIM BET: %s", desc)

    return signals_found


# ---------------------------------------------------------------------------
# Auto-settlement
# ---------------------------------------------------------------------------

def _settle_bet(conn, bet, result: str, pnl: float):
    """Apply settlement to a single bet and its signal."""
    conn.execute(
        """UPDATE simulated_bets
           SET settled = 1, result = ?, pnl = ?
           WHERE id = ?""",
        (result, pnl, bet["id"]),
    )
    conn.execute(
        """UPDATE value_signals
           SET resolved = 1, outcome_won = ?
           WHERE id = ?""",
        (1 if result == "win" else 0, bet["signal_id"]),
    )
    logger.info(
        "SETTLED: #%d %s — %s %s (P&L: %+.2f)",
        bet["id"], bet["event_name"], bet["outcome"],
        result.upper(), pnl,
    )


def _determine_result(bet, winner: str | None) -> tuple[str, float]:
    """Given a bet and winner name, return (result, pnl)."""
    if winner is None:
        return "push", 0.0
    elif bet["outcome"] == winner:
        return "win", (bet["odds_at_detection"] - 1) * bet["stake"]
    else:
        return "loss", -bet["stake"]


def _settle_via_betfair(unsettled_bets: list) -> list:
    """
    Try to settle bets using Betfair market status (free, no credits).

    Returns list of bet IDs that were settled via Betfair.
    Bets without a betfair_market_id are skipped.
    """
    if not config.BETFAIR_ENABLED:
        return []

    # Collect unique market IDs from unsettled bets
    market_ids = set()
    for bet in unsettled_bets:
        mid = bet["betfair_market_id"]
        if mid:
            market_ids.add(mid)

    if not market_ids:
        return []

    import betfair as bf_module
    results = bf_module.check_market_results(list(market_ids))

    if not results:
        return []

    # We need to map selection_id -> runner name for closed markets.
    # The runner names in our bets match the Betfair runner names from when
    # we detected the signal. We stored the outcome name (which came from
    # the Betfair runner name). So we need to get runner names for closed
    # markets to match winner_selection_id back to a name.
    #
    # Try catalogue first; if market is gone, match by looking at all
    # bets' outcomes for that market.
    market_runners = {}
    for market_id in results:
        names = bf_module.get_runner_names_for_market(market_id)
        if names:
            market_runners[market_id] = names
        else:
            # Catalogue unavailable — build mapping from our bet data
            # We know outcome names + the market_id, so we can identify
            # the winner by checking which bet outcomes exist
            pass

    settled_ids = []
    with models.get_db() as conn:
        for bet in unsettled_bets:
            mid = bet["betfair_market_id"]
            if not mid or mid not in results:
                continue

            market_result = results[mid]
            winner_sel_id = market_result.get("winner_selection_id")

            if winner_sel_id is None:
                # Market closed but no winner (void/abandoned)
                result, pnl = "push", 0.0
            else:
                # Map winner selection ID to name
                runners = market_runners.get(mid, {})
                winner_name = runners.get(winner_sel_id)

                if winner_name is None:
                    # Can't determine winner name from catalogue.
                    # Infer: check all bets for this market. The loser runner
                    # will have status LOSER. If this bet's outcome matches
                    # a runner with status != WINNER, it lost.
                    runner_statuses = market_result.get("runner_statuses", {})
                    # Find if our outcome is a winner by checking all bets
                    # for this market to build the name mapping
                    bets_for_market = [b for b in unsettled_bets
                                       if b["betfair_market_id"] == mid]
                    outcome_names = {b["outcome"] for b in bets_for_market}

                    # If there are exactly 2 outcomes (h2h), and one is
                    # the bet's outcome, the other must be the opponent.
                    # We can't reliably map without catalogue, so skip
                    # and let Odds API fallback handle it.
                    logger.debug(
                        "Can't map BF winner for market %s (catalogue gone), "
                        "skipping bet #%d for Odds API fallback",
                        mid, bet["id"],
                    )
                    continue

                result, pnl = _determine_result(bet, winner_name)

            _settle_bet(conn, bet, result, pnl)
            settled_ids.append(bet["id"])

    return settled_ids


def fetch_scores(sport: str, days_from: int = 3, api_key: str | None = None) -> tuple[list | None, dict]:
    """
    Fetch completed scores for a sport from The Odds API.
    Returns (scores_list_or_None, response_headers_dict).
    Costs 2 credits per call.
    Uses the provided api_key, or the current rotation key if None.
    """
    url = f"{config.ODDS_API_BASE}/sports/{sport}/scores"
    key = api_key or _current_api_key()
    params = {
        "apiKey": key,
        "daysFrom": days_from,
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
    except requests.RequestException as e:
        logger.error("Network error fetching scores for %s: %s", sport, e)
        return None, {}

    headers = dict(resp.headers)

    if resp.status_code != 200:
        logger.error("Scores API error for %s: HTTP %d — %s",
                      sport, resp.status_code, resp.text[:200])
        return None, headers

    return resp.json(), headers


def _determine_winner(scores: list[dict]) -> str | None:
    """
    Determine the winning team from a scores array.
    Returns the team name with the higher score, or None for a draw.
    Each entry: {"name": "Team A", "score": "105"}
    """
    if not scores or len(scores) < 2:
        return None

    try:
        team1, team2 = scores[0], scores[1]
        score1 = int(team1["score"])
        score2 = int(team2["score"])
    except (ValueError, KeyError, TypeError):
        return None

    if score1 > score2:
        return team1["name"]
    elif score2 > score1:
        return team2["name"]
    else:
        return None  # draw


def settle_completed_bets():
    """
    Check results and auto-settle completed games.

    Strategy:
    1. Try Betfair API first (free) for bets that have a betfair_market_id
    2. Fall back to Odds API scores (2 credits/call) for remaining bets
    """
    with models.get_db() as conn:
        unsettled = conn.execute(
            "SELECT * FROM simulated_bets WHERE settled = 0"
        ).fetchall()

    if not unsettled:
        logger.debug("No unsettled bets — skipping settlement check.")
        return

    total_settled = 0

    # --- Phase 1: Try Betfair (free) ---
    bf_settled_ids = _settle_via_betfair(unsettled)
    total_settled += len(bf_settled_ids)

    if bf_settled_ids:
        logger.info("Betfair settled %d bet(s) (free)", len(bf_settled_ids))

    # --- Phase 2: Odds API fallback for remaining bets ---
    remaining = [b for b in unsettled if b["id"] not in bf_settled_ids]
    if not remaining:
        logger.info("Settlement complete: %d bet(s) settled (all via Betfair).",
                     total_settled)
        return

    # Group remaining by sport
    sports_needed = list({b["sport"] for b in remaining})

    # Budget check for Odds API calls
    with models.get_db() as conn:
        daily_budget = config.MONTHLY_CREDIT_BUDGET // 30
        used_today = models.get_credits_used_today(conn)
        credits_left = daily_budget - used_today
        # Scores endpoint costs 2 credits
        calls_left = credits_left // 2

    if calls_left <= 0:
        logger.info(
            "Settlement: %d via Betfair, %d remaining need Odds API but "
            "budget exhausted (%d/%d).",
            len(bf_settled_ids), len(remaining), used_today, daily_budget,
        )
        return

    sports_to_check = sports_needed[:calls_left]
    logger.info("Odds API fallback for %d sport(s): %s",
                len(sports_to_check), ", ".join(sports_to_check))

    credits_remaining = -1
    credits_used = -1

    for sport in sports_to_check:
        scores_data, headers = fetch_scores(sport)

        # Track credits
        r, u = parse_credit_headers(headers)
        if r >= 0:
            credits_remaining = r
        if u >= 0:
            credits_used = u

        if scores_data is None:
            continue

        # Build lookup: event_id -> winner
        completed_events = {}
        for event in scores_data:
            if event.get("completed"):
                winner = _determine_winner(event.get("scores", []))
                completed_events[event["id"]] = winner

        if not completed_events:
            continue

        sport_bets = [b for b in remaining if b["sport"] == sport]
        with models.get_db() as conn:
            for bet in sport_bets:
                winner = completed_events.get(bet["event_id"])
                if winner is None and bet["event_id"] not in completed_events:
                    continue  # game not completed yet

                result, pnl = _determine_result(bet, winner)
                _settle_bet(conn, bet, result, pnl)
                total_settled += 1

            if credits_used >= 0:
                models.update_credit_usage(conn, credits_used, credits_remaining)

    logger.info("Settlement complete: %d bet(s) settled.", total_settled)


# ---------------------------------------------------------------------------
# Main scan cycle
# ---------------------------------------------------------------------------

def run_scan():
    """
    Execute one full scan cycle:
    1. Rotate API key (round-robin across configured keys)
    2. Determine which sports to scan (credit-aware rotation)
    3. Fetch odds for each sport
    4. Process events, detect signals, place sim bets
    5. Update credit tracking
    6. Log summary
    """
    # Rotate to next API key for this cycle
    api_key = _next_api_key()
    key_label = f"{api_key[:4]}...{api_key[-4:]}"
    logger.info("Scan cycle using API key %s (%d/%d)",
                key_label, _key_index % len(config.ODDS_API_KEYS) or len(config.ODDS_API_KEYS),
                len(config.ODDS_API_KEYS))

    now_iso = datetime.now(timezone.utc).isoformat()

    with models.get_db() as conn:
        sports = get_sports_to_scan(conn)

    if not sports:
        logger.info("No sports to scan this cycle (budget exhausted or empty).")
        return

    total_events = 0
    total_signals = 0
    credits_remaining = -1
    credits_used = -1

    if config.BETFAIR_ENABLED:
        import betfair as bf_module

    for sport in sports:
        # Fetch Betfair API depth for this sport (free, no credit cost)
        bf_sport_data = {}
        if config.BETFAIR_ENABLED and sport not in BETFAIR_SKIP_SPORTS:
            try:
                bf_sport_data = bf_module.get_sport_depths(sport)
                if bf_sport_data:
                    logger.info("Betfair API: %d markets with depth for %s",
                                len(bf_sport_data), sport)
            except Exception as e:
                logger.warning("Betfair API failed for %s, falling back: %s",
                               sport, e)

        events, headers = fetch_odds(sport)

        # Track credits from headers
        remaining, used = parse_credit_headers(headers)
        if remaining >= 0:
            credits_remaining = remaining
        if used >= 0:
            credits_used = used

        if events is None:
            continue

        with models.get_db() as conn:
            for event in events:
                total_events += 1

                # Try to match this event to Betfair API depth data
                bf_depth = None
                if bf_sport_data:
                    home = event.get("home_team", "")
                    away = event.get("away_team", "")
                    bf_depth = _match_bf_depth(home, away, bf_sport_data)

                total_signals += process_event(
                    conn, sport, event, now_iso, bf_depth=bf_depth
                )

            # Update credit usage
            if credits_used >= 0:
                models.update_credit_usage(conn, credits_used, credits_remaining)

    # Summary line
    credit_str = f"Credits remaining: {credits_remaining}" if credits_remaining >= 0 else ""
    summary = (
        f"Scanned {len(sports)} sport(s), {total_events} events, "
        f"{total_signals} signal(s) detected. {credit_str}"
    )
    logger.info(summary)
    print(f"[{now_iso[:19]}] {summary}")
