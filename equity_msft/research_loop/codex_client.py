from __future__ import annotations

import json
import os
from typing import Any

import requests
from dotenv import load_dotenv

from equity_msft.research_loop._bootstrap import REPO_ROOT, ensure_import_paths

ensure_import_paths()

from equity_msft.research_loop.pipeline import SEARCH_PARAM_KEYS

load_dotenv(REPO_ROOT / ".env")

OPENAI_API_URL = "https://api.openai.com/v1/responses"
DEFAULT_CODEX_MODEL = os.getenv("CODEX_MODEL", "gpt-5.3-codex")


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    if "```" in stripped:
        for part in stripped.split("```"):
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{") and candidate.endswith("}"):
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    continue

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        parsed = json.loads(stripped[start : end + 1])
        if isinstance(parsed, dict):
            return parsed

    raise ValueError(f"Could not parse JSON object from model output: {text}")


def _build_codex_input(user_prompt: str) -> str:
    allowed = ", ".join(SEARCH_PARAM_KEYS)
    return (
        "You refine MSFT daily-strategy parameters.\n"
        "Return only a valid JSON object with updated keys and values.\n"
        f"Only use these keys: {allowed}\n"
        "Do not invent new keys. Do not change cash, commission, or risk-policy fields.\n"
        "Keep min_atr <= max_atr, neutral_rsi_low < neutral_rsi_high, "
        "and bearish_entry_rsi <= bearish_exit_rsi.\n\n"
        f"{user_prompt}\n"
    )


def codex_api_call(
    prompt: str,
    *,
    model: str | None = None,
    temperature: float = 0.2,
    timeout: int = 45,
    max_tokens: int = 800,
) -> str:
    api_key = os.getenv("OPEN_AI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPEN_AI_API_KEY or OPENAI_API_KEY in environment (.env).")

    payload = {
        "model": model or DEFAULT_CODEX_MODEL,
        "input": _build_codex_input(prompt),
        "temperature": temperature,
        "max_output_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = requests.post(OPENAI_API_URL, headers=headers, json=payload, timeout=timeout)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"OpenAI Codex request failed with status {response.status_code}: {response.text}"
        ) from exc

    data = response.json()
    try:
        return data["output"][0]["content"][0]["text"].strip()
    except (KeyError, IndexError, TypeError):
        try:
            return data["output_text"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected OpenAI response shape: {data}") from exc


def refine_strategy_with_codex(current_params: dict, grok_feedback: str) -> dict[str, Any]:
    allowed = ", ".join(SEARCH_PARAM_KEYS)
    prompt = f"""
You are a strategy refinement model for a conservative MSFT daily long-only system.

Current parameters:
{json.dumps(current_params, indent=2)}

Evaluation from Grok:
{grok_feedback}

Propose a small, testable parameter update that addresses the weaknesses.
Change only keys from this set: {allowed}
Return ONLY a JSON object of the updated keys and values.
"""
    return _extract_json_object(codex_api_call(prompt))
