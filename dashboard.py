"""
Oddshawk web dashboard — Flask app with signals monitor and P&L tracker.

Two tabs:
  1. Signals — live value signals, odds comparisons, credit usage
  2. P&L — simulated betting results, win/loss breakdown, ROI
"""

import logging
from datetime import datetime, timezone, timedelta

from flask import Flask, render_template, request, redirect, url_for, jsonify

import models

logger = logging.getLogger("oddshawk.dashboard")

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Template helpers
# ---------------------------------------------------------------------------

@app.template_filter("fmt_pct")
def fmt_pct(val):
    if val is None:
        return "N/A"
    return f"{val * 100:.1f}%"


@app.template_filter("fmt_money")
def fmt_money(val):
    if val is None:
        return "$0.00"
    prefix = "-" if val < 0 else ""
    return f"{prefix}${abs(val):,.2f}"


@app.template_filter("fmt_time")
def fmt_time(val):
    if not val:
        return ""
    # Show as short datetime
    return val[:16].replace("T", " ")


@app.template_filter("fmt_odds")
def fmt_odds(val):
    if val is None:
        return "-"
    return f"{val:.2f}"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return redirect(url_for("signals_page"))


@app.route("/signals")
def signals_page():
    hours = int(request.args.get("hours", 24))
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()

    with models.get_db() as conn:
        signals = models.get_signals_since(conn, since)
        by_bookmaker = models.get_signals_by_bookmaker(conn, since)
        by_sport = models.get_signals_by_sport(conn, since)
        credit_hist = models.get_credit_history(conn, days=7)
        credits_today = models.get_credits_used_today(conn)

    return render_template(
        "signals.html",
        signals=signals,
        by_bookmaker=by_bookmaker,
        by_sport=by_sport,
        credit_hist=credit_hist,
        credits_today=credits_today,
        hours=hours,
        now=datetime.now(timezone.utc).isoformat()[:19],
    )


@app.route("/pnl")
def pnl_page():
    bet_filter = request.args.get("filter", "all")  # all, arb, value

    with models.get_db() as conn:
        pnl = models.get_simulated_pnl(conn)
        pnl_bm = models.get_pnl_by_bookmaker(conn)
        pnl_sport = models.get_pnl_by_sport(conn)
        unsettled = models.get_unsettled_bets(conn)
        recent_signals = models.get_recent_signals(conn, limit=50)

    # Apply arb/value filter
    if bet_filter == "arb":
        unsettled = [b for b in unsettled
                     if b["arb_profit"] and b["arb_profit"] > 0]
        recent_signals = [s for s in recent_signals
                          if s["arb_roi_pct"] is not None and s["arb_roi_pct"] > 0]
    elif bet_filter == "value":
        unsettled = [b for b in unsettled
                     if not b["arb_profit"] or b["arb_profit"] <= 0]
        recent_signals = [s for s in recent_signals
                          if s["arb_roi_pct"] is None or s["arb_roi_pct"] <= 0]

    return render_template(
        "pnl.html",
        pnl=pnl,
        pnl_bm=pnl_bm,
        pnl_sport=pnl_sport,
        unsettled=unsettled,
        recent_signals=recent_signals,
        bet_filter=bet_filter,
    )


@app.route("/news")
def news_page():
    with models.get_db() as conn:
        stats = models.get_article_stats(conn)
        articles = models.get_recent_articles(conn, limit=50)
        news_signals = models.get_news_signals_recent(conn, limit=50)
        news_bets = models.get_news_bets_summary(conn)
        politics_markets = models.get_politics_markets(conn)

    # Group runners by market for template display
    markets_grouped = {}
    for row in politics_markets:
        mid = row["market_id"]
        if mid not in markets_grouped:
            markets_grouped[mid] = {
                "market_name": row["market_name"],
                "total_matched": row["total_matched"],
                "updated_at": row["updated_at"],
                "runners": [],
            }
        markets_grouped[mid]["runners"].append(row)

    return render_template(
        "news.html",
        stats=stats,
        articles=articles,
        news_signals=news_signals,
        news_bets=news_bets,
        markets=markets_grouped,
    )


@app.route("/settle", methods=["POST"])
def settle_bet():
    """Settle a simulated bet via the web UI."""
    bet_id = int(request.form["bet_id"])
    result = request.form["result"]  # win, loss, push, void

    with models.get_db() as conn:
        bet = conn.execute(
            "SELECT * FROM simulated_bets WHERE id = ?", (bet_id,)
        ).fetchone()

        if not bet:
            return redirect(url_for("pnl_page"))

        if result == "win":
            pnl = (bet["odds_at_detection"] - 1) * bet["stake"]
        elif result == "loss":
            pnl = -bet["stake"]
        elif result == "push":
            pnl = 0.0
        else:  # void
            pnl = 0.0

        conn.execute(
            """UPDATE simulated_bets
               SET settled = 1, result = ?, pnl = ?
               WHERE id = ?""",
            (result, pnl, bet_id),
        )

        # Also resolve the linked signal
        conn.execute(
            """UPDATE value_signals
               SET resolved = 1, outcome_won = ?
               WHERE id = ?""",
            (1 if result == "win" else 0, bet["signal_id"]),
        )

    return redirect(url_for("pnl_page"))


@app.route("/api/signals")
def api_signals():
    """JSON endpoint for auto-refresh."""
    hours = int(request.args.get("hours", 24))
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()

    with models.get_db() as conn:
        signals = models.get_signals_since(conn, since)

    return jsonify([dict(s) for s in signals])


# ---------------------------------------------------------------------------
# Start
# ---------------------------------------------------------------------------

def start(port=5050):
    """Start the Flask dev server."""
    models.init_db()
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5050
    logger.info("Dashboard starting on http://localhost:%d", port)
    start(port=port)
