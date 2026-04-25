import json
import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_CODEX_MODEL = os.getenv("CODEX_MODEL", "gpt-5.3-codex")


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()

    # First try direct parse.
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Handle fenced markdown JSON blocks.
    if "```" in stripped:
        parts = stripped.split("```")
        for part in parts:
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

    # Fallback: parse from first opening brace to last closing brace.
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = stripped[start : end + 1]
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed

    raise ValueError(f"Could not parse JSON object from model output: {text}")


def codex_api_call(
    prompt: str,
    *,
    model: str | None = None,
    temperature: float = 0.2,
    timeout: int = 45,
) -> str:
    api_key = os.getenv("OPEN_AI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPEN_AI_API_KEY or OPENAI_API_KEY in environment (.env).")

    payload = {
        "model": model or DEFAULT_CODEX_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You refine trading strategy parameters. "
                    "Return only valid JSON object with updated keys and values."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    response = requests.post(
        OPENAI_API_URL,
        headers=headers,
        json=payload,
        timeout=timeout,
    )

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"OpenAI Codex request failed with status {response.status_code}: {response.text}"
        ) from exc

    data = response.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected OpenAI response shape: {data}") from exc


def refine_strategy_with_codex(current_params: dict, grok_feedback: str) -> dict[str, Any]:
    prompt = f"""
You are a strategy refinement model.

Current parameters:
{current_params}

Evaluation from Grok:
{grok_feedback}

Propose updated parameters that address weaknesses.
Return ONLY a JSON dict of new parameters.
"""

    response_text = codex_api_call(prompt)
    return _extract_json_object(response_text)
