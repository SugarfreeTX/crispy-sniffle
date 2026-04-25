import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


# Load .env from repository root: crispy-sniffle/.env
REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

GROK_API_URL = "https://api.x.ai/v1/chat/completions"
DEFAULT_GROK_MODEL = os.getenv("GROK_MODEL", "grok-3-mini")


def build_grok_prompt(metrics: dict[str, Any]) -> str:
    template_path = Path(__file__).resolve().parent / "prompt_templates" / "grok_eval_template.txt"
    template = template_path.read_text(encoding="utf-8")
    metrics_block = "\n".join(f"{k}: {v}" for k, v in metrics.items())
    return template.format(metrics=metrics_block)


def grok_api_call(
    prompt: str,
    *,
    model: str | None = None,
    temperature: float = 0.2,
    timeout: int = 30,
) -> str:
    api_key = os.getenv("GROK_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GROK_API_KEY in environment (.env).")

    payload = {
        "model": model or DEFAULT_GROK_MODEL,
        "messages": [
            {"role": "system", "content": "You are a trading strategy reviewer. Be concise and actionable."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    response = requests.post(
        GROK_API_URL,
        headers=headers,
        json=payload,
        timeout=timeout,
    )

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
    prompt = build_grok_prompt(metrics)
    return grok_api_call(prompt)