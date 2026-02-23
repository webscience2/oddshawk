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

    log_file = "newsbot.log"
    if config.LOG_PATH and config.LOG_PATH != "oddshawk.log":
        log_file = config.LOG_PATH.replace(".log", "-newsbot.log")

    fh = TimedRotatingFileHandler(
        log_file,
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
        "SIGNAL [%s] %s — %s / back %s (implied %.1f%%) — edge: %s — %s",
        signal_type.upper(), market_name, runner_name,
        f"{current_back:.2f}" if current_back else "?",
        (current_implied or 0) * 100, edge_str, reasoning[:80],
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
