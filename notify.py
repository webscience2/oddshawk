"""
Oddshawk notifications — sends real-time alerts when value signals are detected.

Currently supports Telegram. Notifications are fire-and-forget in background
threads so they never block the scanner.
"""

import logging
import threading

import requests

import config

logger = logging.getLogger("oddshawk.notify")

# Sport key → friendly emoji + name
SPORT_LABELS = {
    "basketball_nba": "\U0001f3c0 NBA",
    "basketball_nbl": "\U0001f3c0 NBL",
    "aussierules_afl": "\U0001f3c8 AFL",
    "rugbyleague_nrl": "\U0001f3c9 NRL",
    "soccer_epl": "\u26bd EPL",
    "soccer_australia_aleague": "\u26bd A-League",
    "soccer_uefa_champs_league": "\u26bd UCL",
    "soccer_spain_la_liga": "\u26bd La Liga",
    "soccer_italy_serie_a": "\u26bd Serie A",
    "soccer_germany_bundesliga": "\u26bd Bundesliga",
    "soccer_france_ligue_one": "\u26bd Ligue 1",
    "icehockey_nhl": "\U0001f3d2 NHL",
}

BOOKMAKER_LABELS = {
    "sportsbet": "Sportsbet",
    "tab": "TAB",
    "ladbrokes_au": "Ladbrokes",
    "neds": "Neds",
    "pointsbetau": "PointsBet",
    "unibet": "Unibet",
    "tabtouch": "TABtouch",
    "betright": "BetRight",
    "betr_au": "Betr",
}


def _format_time(commence_time: str) -> str:
    """Format ISO commence time to short AEST string."""
    if not commence_time:
        return ""
    try:
        from datetime import datetime, timezone, timedelta
        dt = datetime.fromisoformat(commence_time)
        aest = dt.astimezone(timezone(timedelta(hours=10)))
        return aest.strftime("%a %d %b %I:%M%p AEST")
    except Exception:
        return commence_time[:16]


def _build_telegram_message(
    *,
    sport: str,
    event_name: str,
    outcome: str,
    bookmaker: str,
    soft_odds: float,
    bf_back: float,
    bf_lay: float | None,
    edge_pct: float,
    arb: dict,
    lay_liq: float | None,
    commence_time: str,
) -> str:
    """Build a plain-text Telegram message for a value signal."""
    sport_label = SPORT_LABELS.get(sport, sport)
    bm_label = BOOKMAKER_LABELS.get(bookmaker, bookmaker)

    lay_str = f"{bf_lay:.2f}" if bf_lay else "-"
    liq_str = f"${lay_liq:,.0f}" if lay_liq else "?"
    kickoff = _format_time(commence_time)

    lines = [
        f"\U0001f6a8 VALUE SIGNAL",
        f"",
        f"{sport_label}",
        f"{event_name}",
        f"",
        f"Back {outcome}",
        f"{bm_label} @ {soft_odds:.2f}",
        f"Betfair: {bf_back:.2f} back / {lay_str} lay ({liq_str} liq)",
        f"Edge: {edge_pct * 100:.1f}%",
    ]

    if arb["guaranteed_profit"] > 0:
        lines.append(
            f"Arb: ${arb['guaranteed_profit']:.2f} profit "
            f"({arb['roi_pct']:.1f}% ROI)"
        )

    if kickoff:
        lines.append(f"")
        lines.append(f"Kick-off: {kickoff}")

    return "\n".join(lines)


def _send_telegram(text: str):
    """Send a message via Telegram Bot API. Runs in background thread."""
    token = config.TELEGRAM_BOT_TOKEN
    chat_id = config.TELEGRAM_CHAT_ID

    if not token or not chat_id:
        logger.warning("Telegram enabled but token/chat_id not set")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        resp = requests.post(
            url,
            json={"chat_id": chat_id, "text": text},
            timeout=10,
        )
        if resp.status_code != 200:
            logger.warning("Telegram API error: %d %s", resp.status_code,
                           resp.text[:200])
        else:
            logger.debug("Telegram notification sent")
    except Exception as e:
        logger.warning("Telegram send failed: %s", e)


def send_signal_alert(
    *,
    sport: str,
    event_name: str,
    outcome: str,
    bookmaker: str,
    soft_odds: float,
    bf_back: float,
    bf_lay: float | None,
    edge_pct: float,
    arb: dict,
    lay_liq: float | None,
    commence_time: str,
):
    """
    Send a value signal notification via all enabled channels.

    Fire-and-forget — runs in a background thread, never blocks the scanner.
    """
    if not config.TELEGRAM_ENABLED:
        return

    text = _build_telegram_message(
        sport=sport,
        event_name=event_name,
        outcome=outcome,
        bookmaker=bookmaker,
        soft_odds=soft_odds,
        bf_back=bf_back,
        bf_lay=bf_lay,
        edge_pct=edge_pct,
        arb=arb,
        lay_liq=lay_liq,
        commence_time=commence_time,
    )

    thread = threading.Thread(target=_send_telegram, args=(text,), daemon=True)
    thread.start()


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
