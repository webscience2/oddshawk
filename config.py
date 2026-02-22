"""
Oddshawk configuration — loads .env and provides typed access to all settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent / ".env")


def _get(key: str, default: str | None = None) -> str:
    val = os.getenv(key, default)
    if val is None:
        raise ValueError(f"Missing required env var: {key}")
    return val


def _float(key: str, default: str) -> float:
    return float(_get(key, default))


def _int(key: str, default: str) -> int:
    return int(_get(key, default))


def _bool(key: str, default: str) -> bool:
    return _get(key, default).lower() in ("true", "1", "yes")


# --- API ---
ODDS_API_KEY = _get("ODDS_API_KEY")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# --- Betfair Exchange API ---
BETFAIR_USERNAME = _get("BETFAIR_USERNAME", "")
BETFAIR_PASSWORD = _get("BETFAIR_PASSWORD", "")
BETFAIR_APP_KEY = _get("BETFAIR_APP_KEY", "")
BETFAIR_CERT_PATH = _get("BETFAIR_CERT_PATH", "./betfair-client.crt")
BETFAIR_KEY_PATH = _get("BETFAIR_KEY_PATH", "./betfair-client.key")
BETFAIR_ENABLED = _bool("BETFAIR_ENABLED", "false")

# --- Thresholds ---
VALUE_THRESHOLD = _float("VALUE_THRESHOLD", "0.05")
BET_THRESHOLD = _float("BET_THRESHOLD", "0.08")
BETFAIR_COMMISSION = _float("BETFAIR_COMMISSION", "0.05")
MIN_BETFAIR_MATCHED = _float("MIN_BETFAIR_MATCHED", "5000")

# --- Simulation ---
SIM_STAKE = _float("SIM_STAKE", "100")

# --- Credit management ---
MONTHLY_CREDIT_BUDGET = _int("MONTHLY_CREDIT_BUDGET", "500")
EXPLORATION_DAYS = _int("EXPLORATION_DAYS", "7")

# --- Scheduling ---
POLL_INTERVAL_MINUTES = _int("POLL_INTERVAL_MINUTES", "5")
SETTLE_INTERVAL_MINUTES = _int("SETTLE_INTERVAL_MINUTES", "30")
DIGEST_TIME = _get("DIGEST_TIME", "08:00")  # AEST

# --- Email ---
EMAIL_ENABLED = _bool("EMAIL_ENABLED", "false")
EMAIL_TO = _get("EMAIL_TO", "")
EMAIL_FROM = _get("EMAIL_FROM", "")
EMAIL_SMTP = _get("EMAIL_SMTP", "")
EMAIL_PORT = _int("EMAIL_PORT", "587")
EMAIL_PASSWORD = _get("EMAIL_PASSWORD", "")

# --- Paths ---
DB_PATH = _get("DB_PATH", "oddshawk.db")
LOG_PATH = _get("LOG_PATH", "oddshawk.log")

# --- Sports to scan ---
SPORTS = [
    "basketball_nba",
    "aussierules_afl",
    "rugbyleague_nrl",
    # Soccer — high Betfair liquidity, AU soft bookmakers active
    "soccer_epl",
    "soccer_australia_aleague",
    "soccer_uefa_champs_league",
    "soccer_spain_la_liga",
    "soccer_italy_serie_a",
    "soccer_germany_bundesliga",
    "soccer_france_ligue_one",
    # Ice Hockey
    "icehockey_nhl",
]

# --- Bookmakers ---
SOFT_BOOKMAKERS = [
    "sportsbet",
    "tab",
    "ladbrokes_au",
    "neds",
    "pointsbetau",
    "unibet",
    "tabtouch",
    "betright",
    "betr_au",
]
BETFAIR_KEY = "betfair_ex_au"

# All bookmakers in one comma-separated string for the API call
ALL_BOOKMAKERS = ",".join([BETFAIR_KEY] + SOFT_BOOKMAKERS)
