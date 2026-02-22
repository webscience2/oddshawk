"""
Oddshawk SQLite schema and query helpers.

All timestamps stored as UTC ISO format strings.
Tables created on startup via init_db().
"""

import sqlite3
from datetime import datetime, timezone, timedelta
from contextlib import contextmanager

import config

# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def get_db():
    """Context manager that yields a connection and auto-commits/closes."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS odds_snapshots (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    sport         TEXT NOT NULL,
    event_id      TEXT NOT NULL,
    event_name    TEXT NOT NULL,
    commence_time TEXT NOT NULL,
    bookmaker     TEXT NOT NULL,
    outcome       TEXT NOT NULL,
    decimal_odds  REAL NOT NULL,
    implied_prob  REAL NOT NULL,
    bet_limit     REAL,
    captured_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS betfair_snapshots (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    sport               TEXT NOT NULL,
    event_id            TEXT NOT NULL,
    event_name          TEXT NOT NULL,
    commence_time       TEXT NOT NULL,
    outcome             TEXT NOT NULL,
    back_price          REAL NOT NULL,
    lay_price           REAL,
    back_liquidity      REAL,
    lay_liquidity       REAL,
    betfair_raw_implied REAL NOT NULL,
    betfair_true_prob   REAL NOT NULL,
    captured_at         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS value_signals (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    sport            TEXT NOT NULL,
    event_id         TEXT NOT NULL,
    event_name       TEXT NOT NULL,
    commence_time    TEXT NOT NULL,
    bookmaker        TEXT NOT NULL,
    outcome          TEXT NOT NULL,
    soft_decimal_odds REAL NOT NULL,
    soft_implied_prob REAL NOT NULL,
    betfair_back_price REAL NOT NULL,
    betfair_lay_price REAL,
    betfair_true_prob REAL NOT NULL,
    edge_pct         REAL NOT NULL,
    back_liquidity   REAL,
    lay_liquidity    REAL,
    -- Arb: what the hedge on Betfair would look like
    arb_lay_stake    REAL,
    arb_profit_back  REAL,
    arb_profit_lay   REAL,
    arb_roi_pct      REAL,
    betfair_total_matched REAL,
    detected_at      TEXT NOT NULL,
    resolved         INTEGER DEFAULT 0,
    outcome_won      INTEGER
);

CREATE TABLE IF NOT EXISTS simulated_bets (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id         INTEGER NOT NULL REFERENCES value_signals(id),
    sport             TEXT NOT NULL,
    event_id          TEXT NOT NULL,
    event_name        TEXT NOT NULL,
    commence_time     TEXT NOT NULL,
    bookmaker         TEXT NOT NULL,
    outcome           TEXT NOT NULL,
    odds_at_detection REAL NOT NULL,
    stake             REAL NOT NULL,
    potential_payout  REAL NOT NULL,
    -- Arb hedge details
    betfair_back_price REAL,
    betfair_lay_price REAL,
    lay_liquidity    REAL,
    arb_lay_stake    REAL,
    arb_profit       REAL,
    betfair_total_matched REAL,
    description      TEXT,
    placed_at         TEXT NOT NULL,
    settled           INTEGER DEFAULT 0,
    result            TEXT,
    pnl               REAL
);

CREATE TABLE IF NOT EXISTS credit_usage (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    date              TEXT NOT NULL,
    credits_used      INTEGER NOT NULL DEFAULT 0,
    credits_remaining INTEGER,
    updated_at        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_signals_dedup
    ON value_signals(event_id, bookmaker, outcome, detected_at);

CREATE INDEX IF NOT EXISTS idx_signals_detected
    ON value_signals(detected_at);

CREATE INDEX IF NOT EXISTS idx_bets_unsettled
    ON simulated_bets(settled);

CREATE INDEX IF NOT EXISTS idx_odds_captured
    ON odds_snapshots(captured_at);

CREATE INDEX IF NOT EXISTS idx_betfair_captured
    ON betfair_snapshots(captured_at);

CREATE INDEX IF NOT EXISTS idx_credit_date
    ON credit_usage(date);
"""


def init_db():
    """Create all tables and indexes if they don't exist."""
    with get_db() as conn:
        conn.executescript(_SCHEMA)
        _migrate(conn)


def _migrate(conn):
    """Add columns that may be missing from older schemas."""
    migrations = [
        ("value_signals", "betfair_total_matched", "REAL"),
        ("simulated_bets", "betfair_total_matched", "REAL"),
        ("value_signals", "betfair_market_id", "TEXT"),
        ("simulated_bets", "betfair_market_id", "TEXT"),
    ]
    for table, column, col_type in migrations:
        try:
            conn.execute(
                f"ALTER TABLE {table} ADD COLUMN {column} {col_type}"
            )
        except Exception:
            pass  # column already exists


# ---------------------------------------------------------------------------
# Insert helpers
# ---------------------------------------------------------------------------

def insert_odds_snapshot(conn, *, sport, event_id, event_name, commence_time,
                         bookmaker, outcome, decimal_odds, implied_prob,
                         bet_limit=None, captured_at):
    conn.execute(
        """INSERT INTO odds_snapshots
           (sport, event_id, event_name, commence_time, bookmaker, outcome,
            decimal_odds, implied_prob, bet_limit, captured_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (sport, event_id, event_name, commence_time, bookmaker, outcome,
         decimal_odds, implied_prob, bet_limit, captured_at),
    )


def insert_betfair_snapshot(conn, *, sport, event_id, event_name,
                            commence_time, outcome, back_price, lay_price,
                            back_liquidity=None, lay_liquidity=None,
                            betfair_raw_implied, betfair_true_prob,
                            captured_at):
    conn.execute(
        """INSERT INTO betfair_snapshots
           (sport, event_id, event_name, commence_time, outcome, back_price,
            lay_price, back_liquidity, lay_liquidity,
            betfair_raw_implied, betfair_true_prob, captured_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (sport, event_id, event_name, commence_time, outcome, back_price,
         lay_price, back_liquidity, lay_liquidity,
         betfair_raw_implied, betfair_true_prob, captured_at),
    )


def insert_value_signal(conn, *, sport, event_id, event_name, commence_time,
                        bookmaker, outcome, soft_decimal_odds,
                        soft_implied_prob, betfair_back_price,
                        betfair_lay_price=None, betfair_true_prob, edge_pct,
                        back_liquidity=None, lay_liquidity=None,
                        arb_lay_stake, arb_profit_back, arb_profit_lay,
                        arb_roi_pct, betfair_total_matched=None,
                        betfair_market_id=None, detected_at) -> int:
    """Insert a value signal and return its row id."""
    cur = conn.execute(
        """INSERT INTO value_signals
           (sport, event_id, event_name, commence_time, bookmaker, outcome,
            soft_decimal_odds, soft_implied_prob, betfair_back_price,
            betfair_lay_price, betfair_true_prob, edge_pct,
            back_liquidity, lay_liquidity,
            arb_lay_stake, arb_profit_back, arb_profit_lay, arb_roi_pct,
            betfair_total_matched, betfair_market_id, detected_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (sport, event_id, event_name, commence_time, bookmaker, outcome,
         soft_decimal_odds, soft_implied_prob, betfair_back_price,
         betfair_lay_price, betfair_true_prob, edge_pct,
         back_liquidity, lay_liquidity,
         arb_lay_stake, arb_profit_back, arb_profit_lay, arb_roi_pct,
         betfair_total_matched, betfair_market_id, detected_at),
    )
    return cur.lastrowid


def insert_simulated_bet(conn, *, signal_id, sport, event_id, event_name,
                         commence_time, bookmaker, outcome,
                         odds_at_detection, stake, potential_payout,
                         betfair_back_price, betfair_lay_price=None,
                         lay_liquidity=None,
                         arb_lay_stake, arb_profit,
                         betfair_total_matched=None,
                         betfair_market_id=None,
                         description, placed_at):
    conn.execute(
        """INSERT INTO simulated_bets
           (signal_id, sport, event_id, event_name, commence_time, bookmaker,
            outcome, odds_at_detection, stake, potential_payout,
            betfair_back_price, betfair_lay_price, lay_liquidity,
            arb_lay_stake, arb_profit, betfair_total_matched,
            betfair_market_id, description, placed_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (signal_id, sport, event_id, event_name, commence_time, bookmaker,
         outcome, odds_at_detection, stake, potential_payout,
         betfair_back_price, betfair_lay_price, lay_liquidity,
         arb_lay_stake, arb_profit, betfair_total_matched,
         betfair_market_id, description, placed_at),
    )


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def signal_exists_recently(conn, event_id: str, bookmaker: str,
                           outcome: str, minutes: int = 30) -> bool:
    """Check if an identical signal was logged within the last N minutes."""
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()
    row = conn.execute(
        """SELECT 1 FROM value_signals
           WHERE event_id = ? AND bookmaker = ? AND outcome = ?
             AND detected_at > ?
           LIMIT 1""",
        (event_id, bookmaker, outcome, cutoff),
    ).fetchone()
    return row is not None


# ---------------------------------------------------------------------------
# Credit tracking
# ---------------------------------------------------------------------------

def update_credit_usage(conn, credits_used: int, credits_remaining: int):
    """Upsert today's credit usage row."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    now = datetime.now(timezone.utc).isoformat()
    existing = conn.execute(
        "SELECT id, credits_used FROM credit_usage WHERE date = ?", (today,)
    ).fetchone()
    if existing:
        conn.execute(
            """UPDATE credit_usage
               SET credits_used = ?, credits_remaining = ?, updated_at = ?
               WHERE id = ?""",
            (credits_used, credits_remaining, now, existing["id"]),
        )
    else:
        conn.execute(
            """INSERT INTO credit_usage (date, credits_used, credits_remaining, updated_at)
               VALUES (?, ?, ?, ?)""",
            (today, credits_used, credits_remaining, now),
        )


def get_credits_used_today(conn) -> int:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    row = conn.execute(
        "SELECT credits_used FROM credit_usage WHERE date = ?", (today,)
    ).fetchone()
    return row["credits_used"] if row else 0


# ---------------------------------------------------------------------------
# Query helpers (used by digest and dashboard)
# ---------------------------------------------------------------------------

def get_signals_since(conn, since_utc: str) -> list[sqlite3.Row]:
    return conn.execute(
        """SELECT * FROM value_signals
           WHERE detected_at > ?
           ORDER BY edge_pct DESC""",
        (since_utc,),
    ).fetchall()


def get_top_signals(conn, since_utc: str, limit: int = 10) -> list[sqlite3.Row]:
    return conn.execute(
        """SELECT * FROM value_signals
           WHERE detected_at > ?
           ORDER BY edge_pct DESC
           LIMIT ?""",
        (since_utc, limit),
    ).fetchall()


def get_signals_by_bookmaker(conn, since_utc: str) -> list[sqlite3.Row]:
    return conn.execute(
        """SELECT bookmaker,
                  COUNT(*) as signal_count,
                  AVG(edge_pct) as avg_edge,
                  GROUP_CONCAT(DISTINCT sport) as sports
           FROM value_signals
           WHERE detected_at > ?
           GROUP BY bookmaker
           ORDER BY signal_count DESC""",
        (since_utc,),
    ).fetchall()


def get_signals_by_sport(conn, since_utc: str) -> list[sqlite3.Row]:
    return conn.execute(
        """SELECT sport,
                  COUNT(*) as signal_count,
                  AVG(edge_pct) as avg_edge
           FROM value_signals
           WHERE detected_at > ?
           GROUP BY sport
           ORDER BY signal_count DESC""",
        (since_utc,),
    ).fetchall()


def get_simulated_pnl(conn) -> sqlite3.Row | None:
    """Overall P&L summary for all settled bets."""
    return conn.execute(
        """SELECT
              COUNT(*) as total_bets,
              SUM(CASE WHEN settled = 1 THEN 1 ELSE 0 END) as settled_bets,
              SUM(CASE WHEN settled = 0 THEN 1 ELSE 0 END) as unsettled_bets,
              SUM(CASE WHEN result = 'win' THEN 1 ELSE 0 END) as wins,
              SUM(CASE WHEN result = 'loss' THEN 1 ELSE 0 END) as losses,
              SUM(stake) as total_staked,
              SUM(CASE WHEN settled = 1 THEN pnl ELSE 0 END) as net_pnl,
              SUM(CASE WHEN result = 'win' THEN potential_payout ELSE 0 END) as total_returned
           FROM simulated_bets"""
    ).fetchone()


def get_pnl_by_bookmaker(conn) -> list[sqlite3.Row]:
    return conn.execute(
        """SELECT bookmaker,
                  COUNT(*) as bets,
                  SUM(CASE WHEN result = 'win' THEN 1 ELSE 0 END) as wins,
                  SUM(CASE WHEN result = 'loss' THEN 1 ELSE 0 END) as losses,
                  SUM(stake) as staked,
                  SUM(CASE WHEN settled = 1 THEN pnl ELSE 0 END) as net_pnl
           FROM simulated_bets
           GROUP BY bookmaker
           ORDER BY net_pnl DESC"""
    ).fetchall()


def get_pnl_by_sport(conn) -> list[sqlite3.Row]:
    return conn.execute(
        """SELECT sport,
                  COUNT(*) as bets,
                  SUM(CASE WHEN result = 'win' THEN 1 ELSE 0 END) as wins,
                  SUM(CASE WHEN result = 'loss' THEN 1 ELSE 0 END) as losses,
                  SUM(stake) as staked,
                  SUM(CASE WHEN settled = 1 THEN pnl ELSE 0 END) as net_pnl
           FROM simulated_bets
           GROUP BY sport
           ORDER BY net_pnl DESC"""
    ).fetchall()


def get_unsettled_bets(conn) -> list[sqlite3.Row]:
    return conn.execute(
        """SELECT * FROM simulated_bets
           WHERE settled = 0
           ORDER BY placed_at DESC"""
    ).fetchall()


def get_recent_signals(conn, limit: int = 20) -> list[sqlite3.Row]:
    """Most recent signals for dashboard display."""
    return conn.execute(
        """SELECT * FROM value_signals
           ORDER BY detected_at DESC
           LIMIT ?""",
        (limit,),
    ).fetchall()


def get_unsettled_sports(conn) -> list[str]:
    """Return distinct sports that have unsettled bets."""
    rows = conn.execute(
        "SELECT DISTINCT sport FROM simulated_bets WHERE settled = 0"
    ).fetchall()
    return [r["sport"] for r in rows]


def get_sport_signal_counts(conn) -> list[sqlite3.Row]:
    """Signal counts per sport (all time) for adaptive rotation."""
    return conn.execute(
        """SELECT sport, COUNT(*) as count
           FROM value_signals
           GROUP BY sport
           ORDER BY count DESC"""
    ).fetchall()


def get_credit_history(conn, days: int = 7) -> list[sqlite3.Row]:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    return conn.execute(
        """SELECT * FROM credit_usage
           WHERE date > ?
           ORDER BY date DESC""",
        (cutoff,),
    ).fetchall()
