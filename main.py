"""
Oddshawk scanner process — polls APIs, detects value signals, places sim bets.

Runs the scanner on a credit-aware interval and the digest daily at 8am AEST.
Dashboard is a separate process (run_dashboard.py).
"""

import logging
import sys
import time
from logging.handlers import TimedRotatingFileHandler

import schedule

import config
import models
import scanner
import digest


def setup_logging():
    """Configure logging with daily rotation and 7-day retention."""
    root = logging.getLogger("oddshawk")
    root.setLevel(logging.DEBUG)

    # File handler: daily rotation, keep 7 days
    fh = TimedRotatingFileHandler(
        config.LOG_PATH,
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

    # Console handler: INFO and above
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    ))

    root.addHandler(fh)
    root.addHandler(ch)


def main():
    setup_logging()
    logger = logging.getLogger("oddshawk.scanner")

    logger.info("Oddshawk scanner starting up...")
    logger.info("API key: %s...%s", config.ODDS_API_KEY[:4], config.ODDS_API_KEY[-4:])
    logger.info("Sports: %s", ", ".join(config.SPORTS))
    logger.info("Value threshold: %.1f%%, Bet threshold: %.1f%%",
                config.VALUE_THRESHOLD * 100, config.BET_THRESHOLD * 100)
    logger.info("Monthly credit budget: %d", config.MONTHLY_CREDIT_BUDGET)

    # Initialize database
    models.init_db()
    logger.info("Database initialized at %s", config.DB_PATH)

    # Schedule scanner
    schedule.every(config.POLL_INTERVAL_MINUTES).minutes.do(scanner.run_scan)

    # Schedule auto-settlement
    schedule.every(config.SETTLE_INTERVAL_MINUTES).minutes.do(
        scanner.settle_completed_bets
    )
    logger.info("Auto-settlement scheduled every %d minutes",
                config.SETTLE_INTERVAL_MINUTES)

    # Schedule daily digest (AEST = UTC+10, so 8am AEST = 22:00 UTC previous day)
    digest_hour, digest_min = config.DIGEST_TIME.split(":")
    aest_hour = int(digest_hour)
    utc_hour = (aest_hour - 10) % 24
    utc_time = f"{utc_hour:02d}:{digest_min}"
    schedule.every().day.at(utc_time).do(digest.run_digest)
    logger.info("Digest scheduled at %s AEST (%s UTC)", config.DIGEST_TIME, utc_time)

    # Run initial scan immediately
    logger.info("Running initial scan...")
    scanner.run_scan()

    # Run scheduler loop
    logger.info("Scheduler running — polling every %d minutes",
                config.POLL_INTERVAL_MINUTES)
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Scanner shutting down...")
        sys.exit(0)


if __name__ == "__main__":
    main()
