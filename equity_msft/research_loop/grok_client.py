from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from equity_msft.research_loop._bootstrap import REPO_ROOT, ensure_import_paths

ensure_import_paths()

from equity_msft.research_loop.pipeline import SEARCH_PARAM_KEYS

load_dotenv(REPO_ROOT / ".env")

GROK_API_URL = "https://api.x.ai/v1/chat/completions"
GROK_RESPONSES_URL = "https://api.x.ai/v1/responses"
DEFAULT_GROK_MODEL = os.getenv("GROK_MODEL", "grok-4.6")
DEFAULT_GROK_PROPOSER_MODEL = os.getenv("GROK_PROPOSER_MODEL", "grok-4.20-multi-agent")


def build_grok_prompt(metrics: dict[str, Any]) -> str:
    template_path = Path(__file__).resolve().parent / "prompt_templates" / "grok_eval_template.txt"
    template = template_path.read_text(encoding="utf-8")
    metrics_block = "\n".join(f"{k}: {v}" for k, v in metrics.items())
    return template.format(metrics=metrics_block)


def grok_api_call(
    prompt: str,
    *,
    model: str | None = None,
    timeout: int = 3600,
) -> str:
    api_key = os.getenv("GROK_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GROK_API_KEY in environment (.env).")

    payload = {
        "model": model or DEFAULT_GROK_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a quantitative research reviewer for a conservative "
                    "MSFT daily long-only strategy. Be concise and actionable. "
                    "Do not invent metrics. Do not suggest editing strategy code."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = requests.post(GROK_API_URL, headers=headers, json=payload, timeout=timeout)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"Grok API request failed with status {response.status_code}: {response.text}"
        ) from exc

    data = response.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected Grok API response shape: {data}") from exc


def evaluate_with_grok(metrics: dict[str, Any]) -> str:
    return grok_api_call(build_grok_prompt(metrics))


def _xai_api_key() -> str:
    api_key = os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")
    if not api_key:
        raise RuntimeError("Missing XAI_API_KEY or GROK_API_KEY in environment (.env).")
    return api_key


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


def _collect_text_chunks(value: Any, chunks: list[str]) -> None:
    if isinstance(value, str):
        if value.strip():
            chunks.append(value)
        return
    if isinstance(value, dict):
        text = value.get("text")
        if isinstance(text, str) and text.strip():
            chunks.append(text)
        for key in ("content", "output", "output_text"):
            if key in value:
                _collect_text_chunks(value[key], chunks)
        return
    if isinstance(value, list):
        for item in value:
            _collect_text_chunks(item, chunks)


def _extract_responses_text(data: dict[str, Any]) -> str:
    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    chunks: list[str] = []
    _collect_text_chunks(data.get("output"), chunks)
    if chunks:
        return "\n".join(chunks).strip()

    raise RuntimeError(f"Unexpected xAI responses payload shape: {data}")


def grok_responses_api_call(
    prompt: str,
    *,
    model: str | None = None,
    timeout: int = 3600,
) -> str:
    """Call grok-4.20-multi-agent on the xAI Responses API."""
    payload = {
        "model": model or DEFAULT_GROK_PROPOSER_MODEL,
        "input": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
    }
    headers = {
        "Authorization": f"Bearer {_xai_api_key()}",
        "Content-Type": "application/json",
    }
    response = requests.post(GROK_RESPONSES_URL, headers=headers, json=payload, timeout=timeout)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"Grok Responses API request failed with status {response.status_code}: {response.text}"
        ) from exc

    return _extract_responses_text(response.json())


def refine_strategy_with_grok(current_params: dict, grok_feedback: str) -> dict[str, Any]:
    allowed = ", ".join(SEARCH_PARAM_KEYS)
    prompt = f"""
You refine MSFT daily-strategy parameters.
Return only a valid JSON object with updated keys and values.
Only use these keys: {allowed}
Do not invent new keys. Do not change cash, commission, or risk-policy fields.
Keep min_atr <= max_atr, neutral_rsi_low < neutral_rsi_high,
and bearish_entry_rsi <= bearish_exit_rsi.

You are a strategy refinement model for a conservative MSFT daily long-only system.

Current parameters:
{json.dumps(current_params, indent=2)}

Evaluation from Grok:
{grok_feedback}

Propose a small, testable parameter update that addresses the weaknesses.
Change only keys from this set: {allowed}
Return ONLY a JSON object of the updated keys and values.
"""
    return _extract_json_object(grok_responses_api_call(prompt))
