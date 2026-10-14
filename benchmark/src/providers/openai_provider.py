from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI

from benchmark.src.providers.common import ProviderError, call_with_policy, expand_output_budget
from benchmark.src.redact import sanitize_obj

DEFAULT_SETTINGS = {
    "max_attempts": 5,
    "retry_backoff_seconds": [1, 2, 4, 8],
    "timeout_seconds": 120,
    "max_output_tokens": 350,
    "empty_output_retry_max_tokens": 1200,
    "temperature": 0,
}

def data_url_for_jpeg(path: Path) -> str:
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"

def build_openai_content(prompt: str, image_a_url: str, image_b_url: str, include_detail: bool = True) -> list[dict[str, Any]]:
    image_a: dict[str, Any] = {"type": "input_image", "image_url": image_a_url}
    image_b: dict[str, Any] = {"type": "input_image", "image_url": image_b_url}
    if include_detail:
        image_a["detail"] = "high"
        image_b["detail"] = "high"
    return [
        {"type": "input_text", "text": "Image A:"},
        image_a,
        {"type": "input_text", "text": "Image B:"},
        image_b,
        {"type": "input_text", "text": prompt},
    ]

def build_openai_text_content(prompt: str) -> list[dict[str, Any]]:
    return [{"type": "input_text", "text": prompt}]

def _client(timeout: float) -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ProviderError("OPENAI_API_KEY is not set", auth=True)
    return OpenAI(api_key=api_key, timeout=timeout, max_retries=0)

def _usage_fields(usage: Any) -> tuple[Optional[int], Optional[int], Optional[int], Optional[float]]:
    if usage is None:
        return None, None, None, None
    data = usage if isinstance(usage, dict) else usage.model_dump()
    input_tokens = data.get("input_tokens", data.get("prompt_tokens"))
    output_tokens = data.get("output_tokens", data.get("completion_tokens"))
    reasoning = None
    details = data.get("output_tokens_details") or data.get("completion_tokens_details") or {}
    if isinstance(details, dict):
        reasoning = details.get("reasoning_tokens")
    if reasoning is None:
        reasoning = data.get("reasoning_tokens")
    cost = _as_float(data.get("cost", data.get("total_cost")))
    return _as_int(input_tokens), _as_int(output_tokens), _as_int(reasoning), cost

def _as_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None

def _as_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None

def _extract_text(response: Any, data: dict[str, Any]) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    parts: list[str] = []
    for item in data.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if isinstance(content, dict) and content.get("type") in ("output_text", "text"):
                parts.append(content.get("text") or "")
    return "\n".join(parts).strip()

def _hit_limit(data: dict[str, Any], reasoning_tokens: Optional[int]) -> bool:
    incomplete = data.get("incomplete_details") or {}
    reason = incomplete.get("reason") if isinstance(incomplete, dict) else None
    if reason == "max_output_tokens" or data.get("status") == "incomplete":
        return True
    return bool(reasoning_tokens)

def _dump_response(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        data = response.model_dump()
    else:
        data = dict(response)
    return sanitize_obj(data)

def _request(
    client: OpenAI,
    model: str,
    content: list[dict[str, Any]],
    params: dict[str, Any],
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "input": [{"role": "user", "content": content}],
        "store": False,
    }
    if params.get("temperature") is not None and "temperature" in params:
        kwargs["temperature"] = params["temperature"]
    if "store" not in params:
        kwargs.pop("store", None)
    elif params.get("store") is not None:
        kwargs["store"] = params["store"]
    if "max_output_tokens" in params:
        kwargs["max_output_tokens"] = params["max_output_tokens"]
    started = time.perf_counter()
    response = client.responses.create(**kwargs)
    latency = time.perf_counter() - started
    data = _dump_response(response)
    input_tokens, output_tokens, reasoning_tokens, cost = _usage_fields(data.get("usage"))
    text = _extract_text(response, data)
    provider = data.get("provider") if isinstance(data.get("provider"), str) else None
    resolved = data.get("model") if isinstance(data.get("model"), str) and data.get("model") else model
    return {
        "response_text": text,
        "raw_response": data,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "reported_cost_usd": cost,
        "provider_name": provider,
        "resolved_model_id": resolved,
        "latency_seconds": latency,
        "finish_reason": "length" if _hit_limit(data, reasoning_tokens) and not text else (
            "length" if (data.get("status") == "incomplete") else "stop" if data.get("status") == "completed" else data.get("status")
        ),
        "hit_limit": bool(data.get("status") == "incomplete" or (_hit_limit(data, reasoning_tokens) and not text)),
        "parameters_sent": {
            key: kwargs[key]
            for key in ("temperature", "max_output_tokens", "store")
            if key in kwargs
        },
    }

def generate(
    model: str,
    prompt: str,
    image_a: Path,
    image_b: Path,
    settings: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    active = dict(DEFAULT_SETTINGS)
    if settings:
        active.update(settings)
    client = _client(float(active["timeout_seconds"]))
    image_a_url = data_url_for_jpeg(image_a)
    image_b_url = data_url_for_jpeg(image_b)
    params: dict[str, Any] = {
        "temperature": active["temperature"],
        "max_output_tokens": active["max_output_tokens"],
        "store": False,
        "include_detail": True,
    }

    def request_fn(current: dict[str, Any]) -> dict[str, Any]:
        content = build_openai_content(
            prompt,
            image_a_url,
            image_b_url,
            include_detail=bool(current.get("include_detail", True)),
        )
        return _request(client, model, content, current)

    result, adjustments = call_with_policy(request_fn, params, active)
    result, adjustments = expand_output_budget(
        result, adjustments, params, "max_output_tokens", active, request_fn
    )
    result["actual_parameter_adjustments"] = adjustments
    result["temperature_requested"] = active["temperature"]
    result["tools_enabled"] = False
    result["web_search"] = False
    return result

def generate_text(model: str, prompt: str, settings: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    active = dict(DEFAULT_SETTINGS)
    if settings:
        active.update(settings)
    client = _client(float(active["timeout_seconds"]))
    params: dict[str, Any] = {
        "temperature": active["temperature"],
        "max_output_tokens": active["max_output_tokens"],
        "store": False,
        "include_detail": False,
    }

    def request_fn(current: dict[str, Any]) -> dict[str, Any]:
        return _request(client, model, build_openai_text_content(prompt), current)

    result, adjustments = call_with_policy(request_fn, params, active)
    result, adjustments = expand_output_budget(
        result, adjustments, params, "max_output_tokens", active, request_fn
    )
    result["actual_parameter_adjustments"] = adjustments
    result["temperature_requested"] = active["temperature"]
    result["tools_enabled"] = False
    result["web_search"] = False
    return result
