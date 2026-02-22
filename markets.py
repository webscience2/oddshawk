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

POLITICS_EVENT_TYPE_ID = "2378961"

_markets = {}
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
            }
        },
    }
    """
    import betfair as bf
    from betfairlightweight import filters

    client = bf.get_client()
    if not client:
        logger.error("Betfair client not available")
        return {}

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

    runner_names = {}
    catalogue_map = {}
    for cat in catalogues:
        names = {}
        for r in cat.runners:
            names[r.selection_id] = r.runner_name
        runner_names[cat.market_id] = names
        catalogue_map[cat.market_id] = cat.market_name

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

    result = {}
    for market_id, market_name in catalogue_map.items():
        book = all_books.get(market_id)
        if not book:
            continue

        total_matched = book.total_matched or 0
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
    """
    markets = get_markets()
    dead_certs = []

    for market_id, mdata in markets.items():
        for rname, rdata in mdata["runners"].items():
            imp = rdata["implied_prob"]
            if imp and imp >= 0.95:
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
                break

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
