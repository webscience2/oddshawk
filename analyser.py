"""
Gemini-powered news article analyser for Betfair politics markets.

Two-stage pipeline:
1. Relevance filter — is this article relevant to any open market?
2. Impact analysis — which runners affected, direction, magnitude
Plus shock detection on dead-cert markets.
"""

import json
import logging
import time
from datetime import datetime, timezone

from google import genai
from google.genai import types

import config

logger = logging.getLogger("oddshawk.analyser")

_call_times: list[float] = []
MAX_CALLS_PER_MINUTE = 10


def _rate_limit():
    """Block if we've exceeded MAX_CALLS_PER_MINUTE."""
    now = time.time()
    while _call_times and _call_times[0] < now - 60:
        _call_times.pop(0)

    if len(_call_times) >= MAX_CALLS_PER_MINUTE:
        wait = 60 - (now - _call_times[0])
        if wait > 0:
            logger.info("Rate limiting Gemini calls — waiting %.1fs", wait)
            time.sleep(wait)

    _call_times.append(time.time())


def _call_gemini(prompt: str) -> str | None:
    """Call Gemini Flash and return text response."""
    if not config.GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not set")
        return None

    _rate_limit()

    try:
        client = genai.Client(api_key=config.GEMINI_API_KEY)
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=2000,
            ),
        )
        return response.text
    except Exception as e:
        logger.error("Gemini API call failed: %s", e)
        return None


def _parse_json(text: str) -> dict | list | None:
    """Extract JSON from Gemini response (handles markdown code blocks)."""
    if not text:
        return None

    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Failed to parse Gemini JSON: %s", text[:200])
        return None


def check_relevance(title: str, summary: str, markets_text: str) -> dict | None:
    """
    Stage 1: Quick relevance check.
    Returns {"relevant": bool, "markets": ["market_name", ...]} or None on error.
    """
    prompt = f"""You are a political betting analyst. Given this news article headline and summary, does it potentially affect any of these Betfair political markets?

Markets:
{markets_text}

Article headline: {title}
Article summary: {summary[:500]}

Reply ONLY with JSON (no explanation): {{"relevant": true/false, "markets": ["market_name_1", ...]}}
If not relevant to any market, reply: {{"relevant": false, "markets": []}}"""

    result = _call_gemini(prompt)
    return _parse_json(result)


def analyse_impact(title: str, full_text: str,
                   affected_markets: list[dict]) -> list[dict] | None:
    """
    Stage 2: Detailed impact analysis.

    affected_markets: list of {"market_name": str, "runners": {name: {back_price, implied_prob}}}

    Returns list of:
    {
        "market_name": str,
        "runner_name": str,
        "direction": "more_likely" | "less_likely",
        "magnitude": "small" | "medium" | "large" | "shock",
        "estimated_prob": float (0-1),
        "confidence": "low" | "medium" | "high",
        "reasoning": str
    }
    """
    markets_str = ""
    for m in affected_markets:
        markets_str += f"\n{m['market_name']}:\n"
        for rname, rdata in m["runners"].items():
            imp = rdata.get("implied_prob")
            pct = f"{imp*100:.0f}%" if imp else "?"
            back = rdata.get("back_price", "?")
            markets_str += f"  - {rname}: back {back}, implied {pct}\n"

    prompt = f"""You are a political betting analyst. Analyse how this article affects the following Betfair markets.

For each affected runner, estimate:
- market_name: exact name from the list below
- runner_name: exact name from the list below
- direction: "more_likely" or "less_likely"
- magnitude: "small" (1-3%), "medium" (3-10%), "large" (10%+), or "shock" (reprices a dead cert)
- estimated_prob: your estimated fair probability as a decimal (0.0 to 1.0)
- confidence: "low", "medium", or "high"
- reasoning: one sentence

Current market state:
{markets_str}

Article: {title}
{full_text[:3000]}

Reply ONLY with a JSON array of objects. If no meaningful impact, reply with an empty array: []"""

    result = _call_gemini(prompt)
    return _parse_json(result)


def check_shock(title: str, summary: str,
                dead_certs_text: str) -> dict | None:
    """
    Shock detection: does this article threaten any dead-cert market?
    Returns {"shock": true/false, "market": "...", "runner": "...", "reasoning": "..."} or None.
    """
    if not dead_certs_text:
        return None

    prompt = f"""You are a political betting analyst specialising in upset detection.

These Betfair markets are currently priced as near-certainties (95%+ probability):
{dead_certs_text}

Does this news article provide evidence that ANY of these "certainties" might be wrong?
Even a 5% chance of upset is worth flagging since these markets offer 10-50x returns.
Be aggressive in flagging potential upsets — false positives are OK, missed shocks are not.

Article: {title}
{summary[:500]}

Reply ONLY with JSON: {{"shock": true/false, "market": "market name", "runner": "upset runner name", "reasoning": "one sentence"}}
If no shock potential, reply: {{"shock": false, "market": "", "runner": "", "reasoning": ""}}"""

    result = _call_gemini(prompt)
    return _parse_json(result)
