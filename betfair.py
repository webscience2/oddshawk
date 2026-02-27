"""
Oddshawk Betfair Exchange API client.

Provides full order book depth, traded volume, and total matched amounts
via the Betfair API-NG (using betfairlightweight).

Falls back gracefully if not configured — scanner still works via The Odds API.
"""

import logging
from datetime import datetime, timezone, timedelta

import betfairlightweight
from betfairlightweight import filters

import config

logger = logging.getLogger("oddshawk.betfair")

# Betfair event type IDs for our sports
# Maps Odds API sport keys to Betfair event type IDs
SPORT_EVENT_TYPE_IDS = {
    # Basketball
    "basketball_nba": "7522",
    "basketball_nbl": "7522",
    "basketball_euroleague": "7522",
    "basketball_ncaab": "7522",
    # Australian Rules
    "aussierules_afl": "61420",
    # Rugby League
    "rugbyleague_nrl": "1477",
    # Soccer (all leagues share event type 1, use competition IDs to filter)
    "soccer_epl": "1",
    "soccer_australia_aleague": "1",
    "soccer_uefa_champs_league": "1",
    "soccer_uefa_europa_league": "1",
    "soccer_spain_la_liga": "1",
    "soccer_italy_serie_a": "1",
    "soccer_germany_bundesliga": "1",
    "soccer_france_ligue_one": "1",
    "soccer_netherlands_eredivisie": "1",
    "soccer_spl": "1",
    "soccer_efl_champ": "1",
    # Ice Hockey
    "icehockey_nhl": "7524",
    # Cricket
    "cricket_international_t20": "4",
}

# Betfair competition IDs for soccer leagues
# Maps Odds API sport key -> Betfair competition ID(s)
SPORT_COMPETITION_IDS = {
    "soccer_epl": ["10932509"],
    "soccer_australia_aleague": ["12117172"],
    "soccer_uefa_champs_league": ["228"],
    "soccer_uefa_europa_league": ["2005"],
    "soccer_spain_la_liga": ["117"],
    "soccer_italy_serie_a": ["81"],
    "soccer_germany_bundesliga": ["59"],
    "soccer_france_ligue_one": ["55"],
    "soccer_netherlands_eredivisie": ["9404054"],
    "soccer_spl": ["105"],
    "soccer_efl_champ": ["7129730"],
}

# Singleton client
_client = None


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def login():
    """Authenticate with Betfair using cert-based bot login. Returns client."""
    client = betfairlightweight.APIClient(
        username=config.BETFAIR_USERNAME,
        password=config.BETFAIR_PASSWORD,
        app_key=config.BETFAIR_APP_KEY,
        cert_files=(config.BETFAIR_CERT_PATH, config.BETFAIR_KEY_PATH),
    )
    # For AU accounts, override URLs to .com.au
    client.identity_cert_uri = (
        "https://identitysso-cert.betfair.com.au/api/"
    )
    client.api_uri = "https://api.betfair.com.au/exchange/"
    client.login()
    logger.info("Betfair login successful (session token: %s...)",
                client.session_token[:8] if client.session_token else "None")
    return client


def get_client():
    """Get or create the singleton Betfair client, reconnecting if expired."""
    global _client
    if _client is None or _client.session_expired:
        _client = login()
    return _client


# ---------------------------------------------------------------------------
# Market discovery
# ---------------------------------------------------------------------------

def _normalize_name(name: str) -> str:
    """Normalize team/event names for fuzzy matching."""
    return (name.lower()
            .replace(" vs ", " v ")
            .replace(" versus ", " v ")
            .strip())


def _name_tokens(name: str) -> set[str]:
    """Extract meaningful tokens from a team/event name for fuzzy matching.

    Strips common suffixes (FC, City, United, etc.) and short words.
    Expands common abbreviations for cross-source matching.
    """
    NOISE = {
        "fc", "sc", "afc", "cf", "the", "of", "de", "v", "vs", "versus",
        "united", "utd", "city", "town", "rovers", "wanderers",
        "athletic", "sporting", "real", "inter",
    }
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


# Sports that use different market types on Betfair.
# NHL has an irreconcilable mismatch: The Odds API h2h returns 3-way
# regulation time odds (Home/Away/Draw) but Betfair MATCH_ODDS for NHL
# is 2-way money line (including OT/shootout). Neither market type on
# Betfair matches the API's 3-way split. NHL is therefore excluded from
# Betfair comparison via BETFAIR_SKIP_SPORTS in scanner.py.
SPORT_MARKET_TYPES = {
}


def find_match_odds_markets(client, sport_key: str,
                            hours_ahead: int = 48) -> list:
    """
    Find MATCH_ODDS (or equivalent) markets for a sport on Betfair.

    Returns list of market catalogues with event + runner info.
    """
    event_type_id = SPORT_EVENT_TYPE_IDS.get(sport_key)
    if not event_type_id:
        logger.warning("No Betfair event type mapping for %s", sport_key)
        return []

    now = datetime.now(timezone.utc)

    # Use competition IDs for soccer to avoid getting random leagues
    competition_ids = SPORT_COMPETITION_IDS.get(sport_key)
    market_types = SPORT_MARKET_TYPES.get(sport_key, ["MATCH_ODDS"])

    filter_kwargs = {
        "event_type_ids": [event_type_id],
        "market_type_codes": market_types,
        "market_start_time": {
            "from": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "to": (now + timedelta(hours=hours_ahead)).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
        },
    }
    if competition_ids:
        filter_kwargs["competition_ids"] = competition_ids

    market_filter = filters.market_filter(**filter_kwargs)
    max_res = 200

    try:
        catalogues = client.betting.list_market_catalogue(
            filter=market_filter,
            market_projection=[
                "RUNNER_DESCRIPTION",
                "EVENT",
                "MARKET_START_TIME",
            ],
            sort="FIRST_TO_START",
            max_results=max_res,
        )
    except Exception as e:
        logger.error("Betfair listMarketCatalogue failed: %s", e)
        return []

    return catalogues


def match_event(catalogues, home_team: str, away_team: str) -> dict | None:
    """
    Find the Betfair market matching an Odds API event by team names.

    Tries exact substring first, then falls back to token overlap.
    Returns dict with market_id, runners {name: selection_id}, or None.
    """
    home_norm = _normalize_name(home_team)
    away_norm = _normalize_name(away_team)

    def _build_result(cat):
        runners = {}
        for runner in cat.runners:
            runners[runner.runner_name] = runner.selection_id
        return {
            "market_id": cat.market_id,
            "event_name": cat.event.name if cat.event else "",
            "start_time": cat.market_start_time,
            "runners": runners,
        }

    # Pass 1: exact substring match (both teams)
    for cat in catalogues:
        event_name = _normalize_name(cat.event.name) if cat.event else ""
        if home_norm in event_name and away_norm in event_name:
            return _build_result(cat)

    # Pass 2: token-based matching (handles abbreviations, suffixes)
    # e.g. "Newcastle Knights" tokens={"newcastle","knights"}
    #      "Newcastle v North Qld" tokens={"newcastle","north","qld"}
    # We need meaningful overlap for BOTH teams
    home_tokens = _name_tokens(home_team)
    away_tokens = _name_tokens(away_team)

    best_match = None
    best_score = 0

    for cat in catalogues:
        event_name = cat.event.name if cat.event else ""
        bf_tokens = _name_tokens(event_name)

        home_overlap = len(home_tokens & bf_tokens)
        away_overlap = len(away_tokens & bf_tokens)

        # Need at least 1 token match from each team
        if home_overlap >= 1 and away_overlap >= 1:
            score = home_overlap + away_overlap
            if score > best_score:
                best_score = score
                best_match = cat

    if best_match:
        return _build_result(best_match)

    return None


# ---------------------------------------------------------------------------
# Market depth
# ---------------------------------------------------------------------------

def get_market_book(client, market_id: str) -> dict | None:
    """
    Fetch full order book depth for a market.

    Returns dict keyed by runner name:
    {
        "Runner Name": {
            "back_prices": [(price, size), ...],  # best first (highest)
            "lay_prices": [(price, size), ...],    # best first (lowest)
            "total_matched": float,
            "last_traded_price": float | None,
        },
        "_market": {
            "market_id": str,
            "total_matched": float,
            "status": str,
        }
    }
    """
    try:
        books = client.betting.list_market_book(
            market_ids=[market_id],
            price_projection=filters.price_projection(
                price_data=filters.price_data(
                    ex_best_offers=True,
                )
            ),
        )
    except Exception as e:
        logger.error("Betfair listMarketBook failed for %s: %s", market_id, e)
        return None

    if not books:
        return None

    book = books[0]
    result = {
        "_market": {
            "market_id": book.market_id,
            "total_matched": book.total_matched or 0,
            "status": book.status,
        }
    }

    for runner in book.runners:
        back_prices = []
        if runner.ex and runner.ex.available_to_back:
            for level in runner.ex.available_to_back:
                back_prices.append((level.price, level.size))

        lay_prices = []
        if runner.ex and runner.ex.available_to_lay:
            for level in runner.ex.available_to_lay:
                lay_prices.append((level.price, level.size))

        # Need runner name — we'll map selection_id back to name externally
        result[runner.selection_id] = {
            "back_prices": back_prices,
            "lay_prices": lay_prices,
            "total_matched": runner.total_matched or 0,
            "last_traded_price": runner.last_price_traded,
        }

    return result


def get_event_depth(client, sport_key: str, home_team: str,
                    away_team: str, catalogues=None) -> dict | None:
    """
    High-level: find a Betfair market for an event and return full depth.

    Returns dict keyed by runner name with depth data + market totals, or None.
    """
    if catalogues is None:
        catalogues = find_match_odds_markets(client, sport_key)

    matched = match_event(catalogues, home_team, away_team)
    if not matched:
        return None

    book = get_market_book(client, matched["market_id"])
    if not book:
        return None

    # Map selection IDs back to runner names
    result = {"_market": book["_market"]}
    id_to_name = {v: k for k, v in matched["runners"].items()}

    for selection_id, data in book.items():
        if selection_id == "_market":
            continue
        name = id_to_name.get(selection_id, f"Unknown({selection_id})")
        result[name] = data

    return result


# ---------------------------------------------------------------------------
# Batch: get depth for all events in a sport
# ---------------------------------------------------------------------------

def get_sport_depths(sport_key: str) -> dict:
    """
    Fetch Betfair depth for all upcoming MATCH_ODDS markets in a sport.

    Returns dict: {
        "Milwaukee Bucks v Toronto Raptors": {
            "market_id": "1.xxx",
            "total_matched": 120000,
            "runners": {
                "Milwaukee Bucks": {back_prices, lay_prices, ...},
                "Toronto Raptors": {back_prices, lay_prices, ...},
            }
        },
        ...
    }
    """
    if not config.BETFAIR_ENABLED:
        return {}

    try:
        client = get_client()
    except Exception as e:
        logger.error("Betfair auth failed: %s", e)
        return {}

    catalogues = find_match_odds_markets(client, sport_key)
    if not catalogues:
        logger.debug("No Betfair MATCH_ODDS markets for %s", sport_key)
        return {}

    # Batch fetch market books (5 at a time to avoid TOO_MUCH_DATA)
    market_ids = [cat.market_id for cat in catalogues]
    all_books = {}

    for i in range(0, len(market_ids), 5):
        batch = market_ids[i:i+5]
        try:
            books = client.betting.list_market_book(
                market_ids=batch,
                price_projection=filters.price_projection(
                    price_data=filters.price_data(
                        ex_best_offers=True,
                    )
                ),
            )
            for book in books:
                all_books[book.market_id] = book
        except Exception as e:
            logger.error("Betfair batch listMarketBook failed: %s", e)

    # Build result keyed by event name for easy matching
    result = {}
    for cat in catalogues:
        book = all_books.get(cat.market_id)
        if not book:
            continue

        # Map selection_id -> runner name
        id_to_name = {r.selection_id: r.runner_name for r in cat.runners}

        runners = {}
        for runner in book.runners:
            name = id_to_name.get(runner.selection_id,
                                  f"Unknown({runner.selection_id})")
            back_prices = []
            if runner.ex and runner.ex.available_to_back:
                for level in runner.ex.available_to_back:
                    back_prices.append((level.price, level.size))
            lay_prices = []
            if runner.ex and runner.ex.available_to_lay:
                for level in runner.ex.available_to_lay:
                    lay_prices.append((level.price, level.size))

            runners[name] = {
                "back_prices": back_prices,
                "lay_prices": lay_prices,
                "total_matched": runner.total_matched or 0,
                "last_traded_price": runner.last_price_traded,
            }

        event_name = cat.event.name if cat.event else cat.market_id
        result[event_name] = {
            "market_id": cat.market_id,
            "total_matched": book.total_matched or 0,
            "status": book.status,
            "runners": runners,
        }

    logger.info("Betfair %s: fetched depth for %d markets", sport_key,
                len(result))
    return result


# ---------------------------------------------------------------------------
# Settlement: check market results (free, no Odds API credits)
# ---------------------------------------------------------------------------

def check_market_results(market_ids: list[str]) -> dict:
    """
    Check results for a list of Betfair market IDs.

    Returns dict: {
        market_id: {
            "status": "CLOSED" | "OPEN" | ...,
            "winner": "Runner Name" | None,
            "runners": {selection_id: {"name": str, "status": str}, ...}
        }
    }

    Only returns entries for markets that are CLOSED with a clear winner.
    Markets still OPEN or with errors are omitted.
    """
    if not config.BETFAIR_ENABLED or not market_ids:
        return {}

    try:
        client = get_client()
    except Exception as e:
        logger.error("Betfair auth failed for settlement: %s", e)
        return {}

    # We need catalogue data to map selection_id -> runner name
    # But closed markets may not appear in catalogue, so we'll need to
    # get runner names from the bets themselves. For now, just return
    # selection_id status and let the caller map names.

    results = {}

    for i in range(0, len(market_ids), 5):
        batch = market_ids[i:i+5]
        try:
            books = client.betting.list_market_book(
                market_ids=batch,
                price_projection=filters.price_projection(
                    price_data=filters.price_data(ex_best_offers=True)
                ),
            )
        except Exception as e:
            logger.warning("Betfair listMarketBook for settlement failed: %s", e)
            continue

        for book in books:
            if book.status != "CLOSED":
                continue

            # Find the winning runner
            winner_id = None
            runner_statuses = {}
            for runner in book.runners:
                runner_statuses[runner.selection_id] = runner.status
                if runner.status == "WINNER":
                    winner_id = runner.selection_id

            results[book.market_id] = {
                "status": book.status,
                "winner_selection_id": winner_id,
                "runner_statuses": runner_statuses,
            }

    logger.info("Betfair settlement check: %d/%d markets closed",
                len(results), len(market_ids))
    return results


def get_runner_names_for_market(market_id: str) -> dict:
    """
    Get selection_id -> runner_name mapping for a market.
    Returns empty dict if market not found in catalogue (already settled/removed).
    """
    if not config.BETFAIR_ENABLED:
        return {}

    try:
        client = get_client()
        cats = client.betting.list_market_catalogue(
            filter=filters.market_filter(market_ids=[market_id]),
            market_projection=["RUNNER_DESCRIPTION"],
            max_results=1,
        )
        if cats:
            return {r.selection_id: r.runner_name for r in cats[0].runners}
    except Exception as e:
        logger.debug("Could not get runner names for %s: %s", market_id, e)

    return {}
