"""
Oddshawk daily digest — generates markdown report and optionally emails it.

Runs at DIGEST_TIME (default 8am AEST) via schedule.
"""

import logging
import smtplib
from datetime import datetime, timezone, timedelta
from email.mime.text import MIMEText
from pathlib import Path

import config
import models

logger = logging.getLogger("oddshawk.digest")

# AEST is UTC+10 (ignoring daylight savings for simplicity)
AEST_OFFSET = timedelta(hours=10)


def _now_aest() -> datetime:
    return datetime.now(timezone.utc) + AEST_OFFSET


def _fmt_pct(val: float | None) -> str:
    if val is None:
        return "N/A"
    return f"{val * 100:.1f}%"


def _fmt_money(val: float | None) -> str:
    if val is None:
        return "$0.00"
    return f"${val:,.2f}"


def generate_digest() -> str:
    """Generate a markdown digest for the last 24 hours."""
    now = datetime.now(timezone.utc)
    since_24h = (now - timedelta(hours=24)).isoformat()
    now_aest = _now_aest()

    lines = []
    lines.append(f"# Oddshawk Daily Digest — {now_aest.strftime('%Y-%m-%d %H:%M AEST')}")
    lines.append("")

    with models.get_db() as conn:
        # ---------------------------------------------------------------
        # 1. Summary
        # ---------------------------------------------------------------
        signals = models.get_signals_since(conn, since_24h)
        sports_seen = set(s["sport"] for s in signals)
        credit_history = models.get_credit_history(conn, days=1)
        credits_used_today = credit_history[0]["credits_used"] if credit_history else 0
        credits_remaining = credit_history[0]["credits_remaining"] if credit_history else "?"

        lines.append("## Summary")
        lines.append(f"- **Signals detected (24h):** {len(signals)}")
        lines.append(f"- **Sports covered:** {', '.join(sports_seen) if sports_seen else 'none'}")
        lines.append(f"- **Credits used today:** {credits_used_today}")
        lines.append(f"- **Credits remaining:** {credits_remaining}")
        lines.append("")

        # ---------------------------------------------------------------
        # 2. Top signals by edge
        # ---------------------------------------------------------------
        top = models.get_top_signals(conn, since_24h, limit=10)
        if top:
            lines.append("## Top Signals (by edge)")
            lines.append("")
            lines.append("| # | Sport | Event | Outcome | Bookmaker | Odds | Edge | Time |")
            lines.append("|---|-------|-------|---------|-----------|------|------|------|")
            for i, s in enumerate(top, 1):
                detected = s["detected_at"][:16].replace("T", " ")
                lines.append(
                    f"| {i} | {s['sport']} | {s['event_name']} | "
                    f"{s['outcome']} | {s['bookmaker']} | "
                    f"{s['soft_decimal_odds']:.2f} | "
                    f"{_fmt_pct(s['edge_pct'])} | {detected} |"
                )
            lines.append("")

        # ---------------------------------------------------------------
        # 3. Breakdown by bookmaker
        # ---------------------------------------------------------------
        by_bm = models.get_signals_by_bookmaker(conn, since_24h)
        if by_bm:
            lines.append("## Signals by Bookmaker")
            lines.append("")
            lines.append("| Bookmaker | Signals | Avg Edge | Sports |")
            lines.append("|-----------|---------|----------|--------|")
            for r in by_bm:
                lines.append(
                    f"| {r['bookmaker']} | {r['signal_count']} | "
                    f"{_fmt_pct(r['avg_edge'])} | {r['sports']} |"
                )
            lines.append("")

        # ---------------------------------------------------------------
        # 4. Breakdown by sport
        # ---------------------------------------------------------------
        by_sport = models.get_signals_by_sport(conn, since_24h)
        if by_sport:
            lines.append("## Signals by Sport")
            lines.append("")
            lines.append("| Sport | Signals | Avg Edge |")
            lines.append("|-------|---------|----------|")
            for r in by_sport:
                lines.append(
                    f"| {r['sport']} | {r['signal_count']} | "
                    f"{_fmt_pct(r['avg_edge'])} |"
                )
            lines.append("")

        # ---------------------------------------------------------------
        # 5. Simulated P&L
        # ---------------------------------------------------------------
        pnl = models.get_simulated_pnl(conn)
        lines.append("## Simulated P&L (All Time)")
        lines.append("")
        if pnl and pnl["total_bets"] > 0:
            total_staked = pnl["total_staked"] or 0
            net_pnl = pnl["net_pnl"] or 0
            total_returned = pnl["total_returned"] or 0
            wins = pnl["wins"] or 0
            settled = pnl["settled_bets"] or 0
            roi = (net_pnl / total_staked * 100) if total_staked > 0 else 0
            win_rate = (wins / settled * 100) if settled > 0 else 0

            lines.append(f"- **Total bets:** {pnl['total_bets']}")
            lines.append(f"- **Settled:** {settled} | **Unsettled:** {pnl['unsettled_bets']}")
            lines.append(f"- **Wins:** {wins} | **Losses:** {pnl['losses'] or 0}")
            lines.append(f"- **Win rate:** {win_rate:.1f}%")
            lines.append(f"- **Total staked:** {_fmt_money(total_staked)}")
            lines.append(f"- **Total returned:** {_fmt_money(total_returned)}")
            lines.append(f"- **Net P&L:** {_fmt_money(net_pnl)}")
            lines.append(f"- **ROI:** {roi:.1f}%")
        else:
            lines.append("No simulated bets yet.")
        lines.append("")

        # ---------------------------------------------------------------
        # 6. P&L by bookmaker
        # ---------------------------------------------------------------
        pnl_bm = models.get_pnl_by_bookmaker(conn)
        if pnl_bm:
            lines.append("## P&L by Bookmaker")
            lines.append("")
            lines.append("| Bookmaker | Bets | W/L | Staked | Net P&L |")
            lines.append("|-----------|------|-----|--------|---------|")
            for r in pnl_bm:
                lines.append(
                    f"| {r['bookmaker']} | {r['bets']} | "
                    f"{r['wins'] or 0}/{r['losses'] or 0} | "
                    f"{_fmt_money(r['staked'])} | {_fmt_money(r['net_pnl'])} |"
                )
            lines.append("")

        # ---------------------------------------------------------------
        # 7. P&L by sport
        # ---------------------------------------------------------------
        pnl_sp = models.get_pnl_by_sport(conn)
        if pnl_sp:
            lines.append("## P&L by Sport")
            lines.append("")
            lines.append("| Sport | Bets | W/L | Staked | Net P&L |")
            lines.append("|-------|------|-----|--------|---------|")
            for r in pnl_sp:
                lines.append(
                    f"| {r['sport']} | {r['bets']} | "
                    f"{r['wins'] or 0}/{r['losses'] or 0} | "
                    f"{_fmt_money(r['staked'])} | {_fmt_money(r['net_pnl'])} |"
                )
            lines.append("")

        # ---------------------------------------------------------------
        # 8. Unsettled bets
        # ---------------------------------------------------------------
        unsettled = models.get_unsettled_bets(conn)
        if unsettled:
            lines.append(f"## Unsettled Bets ({len(unsettled)})")
            lines.append("")
            lines.append("| ID | Event | Outcome | Bookie | Odds | Stake | Placed |")
            lines.append("|----|-------|---------|--------|------|-------|--------|")
            for b in unsettled[:20]:  # cap at 20 for readability
                placed = b["placed_at"][:16].replace("T", " ")
                lines.append(
                    f"| {b['id']} | {b['event_name']} | {b['outcome']} | "
                    f"{b['bookmaker']} | {b['odds_at_detection']:.2f} | "
                    f"{_fmt_money(b['stake'])} | {placed} |"
                )
            if len(unsettled) > 20:
                lines.append(f"| ... | +{len(unsettled) - 20} more | | | | | |")
            lines.append("")

        # ---------------------------------------------------------------
        # 9. Credit usage
        # ---------------------------------------------------------------
        lines.append("## Credit Usage")
        credit_hist = models.get_credit_history(conn, days=7)
        if credit_hist:
            total_used_week = sum(r["credits_used"] for r in credit_hist)
            avg_daily = total_used_week / len(credit_hist) if credit_hist else 0
            days_remaining = 30 - datetime.now(timezone.utc).day
            projected_remaining = (credits_remaining or 0) - (avg_daily * days_remaining)

            lines.append(f"- **Avg daily usage (7d):** {avg_daily:.0f} credits")
            lines.append(f"- **Projected month-end remaining:** {projected_remaining:.0f} credits")
        lines.append("")

    return "\n".join(lines)


def save_digest(content: str) -> Path:
    """Save digest to a markdown file in the project directory."""
    date_str = _now_aest().strftime("%Y-%m-%d")
    path = Path(f"digest_{date_str}.md")
    path.write_text(content, encoding="utf-8")
    logger.info("Digest saved to %s", path)
    return path


def send_email(content: str):
    """Send digest via SMTP if email is configured."""
    if not config.EMAIL_ENABLED:
        return

    if not all([config.EMAIL_TO, config.EMAIL_FROM, config.EMAIL_SMTP]):
        logger.warning("Email enabled but missing config — skipping send")
        return

    date_str = _now_aest().strftime("%Y-%m-%d")
    msg = MIMEText(content)
    msg["Subject"] = f"Oddshawk Digest — {date_str}"
    msg["From"] = config.EMAIL_FROM
    msg["To"] = config.EMAIL_TO

    try:
        with smtplib.SMTP(config.EMAIL_SMTP, config.EMAIL_PORT) as server:
            server.starttls()
            server.login(config.EMAIL_FROM, config.EMAIL_PASSWORD)
            server.send_message(msg)
        logger.info("Digest emailed to %s", config.EMAIL_TO)
    except Exception as e:
        logger.error("Failed to send digest email: %s", e)


def run_digest():
    """Generate, print, save, and optionally email the daily digest."""
    logger.info("Generating daily digest...")
    content = generate_digest()
    print(content)
    save_digest(content)
    send_email(content)
