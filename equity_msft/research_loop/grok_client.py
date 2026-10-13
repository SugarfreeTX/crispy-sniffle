from __future__ import annotations

import json
import os
import sys
import time
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

# api.x.ai is behind Cloudflare. A non-streaming completion sends no bytes until
# the model finishes, so the proxy resets the socket while the client is still
# blocked in the status-line read (ConnectionError / RemoteDisconnected). The
# next attempt then sits in that same read until timeout. Streaming keeps the
# connection alive with reasoning and content deltas, and retries only cover a
# reset that still happens before the first byte.
_MAX_RETRIES = 5
_BACKOFF_SECONDS = 2.0
_MAX_BACKOFF_SECONDS = 30.0
_CONNECT_TIMEOUT_SECONDS = 20
_RETRY_STATUS = {408, 429, 500, 502, 503, 504, 520, 522, 524}
_RETRY_EXC = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)


def _exc_brief(exc: BaseException) -> str:
    text = " ".join(str(exc).split())
    if not text:
        return type(exc).__name__
    return text[:220]


def _log_retry(attempt: int, detail: str) -> None:
    print(
        f"[grok] {detail} on attempt {attempt + 1}/{_MAX_RETRIES}; retrying",
        file=sys.stderr,
        flush=True,
    )


def _backoff_seconds(attempt: int) -> float:
    return min(_BACKOFF_SECONDS * 2**attempt, _MAX_BACKOFF_SECONDS)


def _sse_payloads(response: requests.Response):
    response.encoding = response.encoding or "utf-8"
    buffer = ""
    for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
        if not chunk:
            continue
        buffer += chunk.replace("\r\n", "\n").replace("\r", "\n")
        while "\n\n" in buffer:
            raw, buffer = buffer.split("\n\n", 1)
            payload = _sse_data(raw)
            if payload is not None:
                yield payload
    payload = _sse_data(buffer)
    if payload is not None:
        yield payload


def _sse_data(raw: str) -> str | None:
    data_lines = []
    for line in raw.split("\n"):
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if not data_lines:
        return None
    return "\n".join(data_lines)


def _json_objects(payloads) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for payload in payloads:
        if payload.strip() == "[DONE]":
            break
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _chat_text_from_json(data: dict[str, Any]) -> str:
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected Grok API response shape: {data}") from exc
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"Grok API returned an empty completion: {data}")
    return content.strip()


def _chat_text_from_events(events: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for event in events:
        for choice in event.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta") or {}
            if isinstance(delta, dict):
                content = delta.get("content")
                if isinstance(content, str):
                    parts.append(content)
            message = choice.get("message") or {}
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                parts.append(message["content"])
    text = "".join(parts).strip()
    if not text:
        raise RuntimeError(f"Grok API stream ended without content: {events[:3]}")
    return text


def _responses_text_from_events(events: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    completed: dict[str, Any] | None = None
    for event in events:
        event_type = event.get("type")
        if event_type == "error":
            message = event.get("message") or event.get("error") or event
            raise RuntimeError(f"Grok responses stream error: {message}")
        if event_type == "response.output_text.delta" and isinstance(event.get("delta"), str):
            parts.append(event["delta"])
        if event_type in {"response.completed", "response.incomplete"}:
            response_body = event.get("response")
            if isinstance(response_body, dict):
                completed = response_body
    if completed is not None:
        try:
            return _extract_responses_text(completed)
        except RuntimeError:
            pass
    text = "".join(parts).strip()
    if text:
        return text
    raise RuntimeError(f"Grok responses stream ended without text: {events[:3]}")


def _text_from_response(response: requests.Response, *, kind: str) -> str:
    content_type = response.headers.get("Content-Type", "")
    if "text/event-stream" not in content_type:
        data = response.json()
        if kind == "chat":
            return _chat_text_from_json(data)
        return _extract_responses_text(data)

    events = _json_objects(_sse_payloads(response))
    if kind == "chat":
        return _chat_text_from_events(events)
    return _responses_text_from_events(events)


def _post_for_text(
    url: str,
    *,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: int,
    kind: str,
) -> str:
    body = dict(payload)
    body["stream"] = True
    request_headers = dict(headers)
    request_headers.setdefault("Accept", "text/event-stream")
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        started = time.monotonic()
        try:
            # Fresh session per attempt so a reset keep-alive connection is never reused.
            with requests.Session() as session:
                response = session.post(
                    url,
                    headers=request_headers,
                    json=body,
                    timeout=(_CONNECT_TIMEOUT_SECONDS, timeout),
                    stream=True,
                )
                try:
                    elapsed = time.monotonic() - started
                    if response.status_code in _RETRY_STATUS and attempt < _MAX_RETRIES - 1:
                        response.close()
                        _log_retry(attempt, f"HTTP {response.status_code} after {elapsed:.0f}s")
                    else:
                        if response.status_code >= 400:
                            error_text = response.text[:500]
                            raise RuntimeError(
                                f"Grok API request failed with status {response.status_code}: {error_text}"
                            )
                        print(
                            f"[grok] streaming {kind} response (headers in {elapsed:.1f}s)",
                            file=sys.stderr,
                            flush=True,
                        )
                        return _text_from_response(response, kind=kind)
                finally:
                    response.close()
        except _RETRY_EXC as exc:
            last_exc = exc
            elapsed = time.monotonic() - started
            if attempt == _MAX_RETRIES - 1:
                raise
            _log_retry(attempt, f"{_exc_brief(exc)} after {elapsed:.0f}s")
        time.sleep(_backoff_seconds(attempt))
    assert last_exc is not None
    raise last_exc


def build_grok_prompt(metrics: dict[str, Any]) -> str:
    template_path = Path(__file__).resolve().parent / "prompt_templates" / "grok_eval_template.txt"
    template = template_path.read_text(encoding="utf-8")
    metrics_block = "\n".join(f"{k}: {v}" for k, v in metrics.items())
    return template.format(metrics=metrics_block)


def grok_api_call(
    prompt: str,
    *,
    model: str | None = None,
    timeout: int = 240,
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
    return _post_for_text(
        GROK_API_URL,
        headers=headers,
        payload=payload,
        timeout=timeout,
        kind="chat",
    )


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
    timeout: int = 240,
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
    return _post_for_text(
        GROK_RESPONSES_URL,
        headers=headers,
        payload=payload,
        timeout=timeout,
        kind="responses",
    )


def refine_strategy_with_grok(current_params: dict, grok_feedback: str, history_note: str = "") -> dict[str, Any]:
    allowed = ", ".join(SEARCH_PARAM_KEYS)
    history_block = f"\n{history_note}\n" if history_note else ""
    prompt = f"""
You refine MSFT daily-strategy parameters.
Return only a valid JSON object with updated keys and values.
Only use these keys: {allowed}
Do not invent new keys. Do not change cash, commission, or risk-policy fields.
Keep min_atr <= max_atr, neutral_rsi_low < neutral_rsi_high,
and bearish_entry_rsi <= bearish_exit_rsi.
Change at most 2-4 keys per proposal, with a small step relative to the
current value (do not swing a knob from one end of its plausible range to
the other in a single proposal).

You are a strategy refinement model for a conservative MSFT daily long-only system.

Current parameters:
{json.dumps(current_params, indent=2)}

Evaluation from Grok:
{grok_feedback}
{history_block}
Propose a small, testable parameter update that addresses the weaknesses.
Change only keys from this set: {allowed}
Return ONLY a JSON object of the updated keys and values.
"""
    return _extract_json_object(grok_responses_api_call(prompt))
