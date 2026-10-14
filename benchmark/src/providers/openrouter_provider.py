from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI

from benchmark.src.providers.common import call_with_policy, expand_output_budget
from benchmark.src.providers.openai_provider import (
    DEFAULT_SETTINGS,
    _as_float,
    _as_int,
    data_url_for_jpeg,
)
from benchmark.src.redact import sanitize_obj

def build_openrouter_content(
    prompt: str,
    image_a_url: str,
    image_b_url: str,
    include_detail: bool = True,
) -> list[dict[str, Any]]:
    def image_block(url: str) -> dict[str, Any]:
        image_url: dict[str, Any] = {"url": url}
        if include_detail:
            image_url["detail"] = "high"
        return {"type": "image_url", "image_url": image_url}

    return [
        {"type": "text", "text": "Image A:"},
        image_block(image_a_url),
        {"type": "text", "text": "Image B:"},
        image_block(image_b_url),
        {"type": "text", "text": prompt},
    ]

def _openrouter_client(timeout: float) -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        from benchmark.src.providers.common import ProviderError

        raise ProviderError("OPENROUTER_API_KEY is not set", auth=True)
    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        timeout=timeout,
        max_retries=0,
        default_headers={
            "X-Title": "MapEvolve VLM pilot",
        },
    )

def _message_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") in ("text", "output_text"):
                parts.append(item.get("text") or "")
        return "\n".join(parts).strip()
    return ""

def _usage_from_dump(data: dict[str, Any]) -> tuple[Optional[int], Optional[int], Optional[int], Optional[float]]:
    usage = data.get("usage") or {}
    if not isinstance(usage, dict):
        return None, None, None, None
    input_tokens = _as_int(usage.get("prompt_tokens", usage.get("input_tokens")))
    output_tokens = _as_int(usage.get("completion_tokens", usage.get("output_tokens")))
    details = usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}
    reasoning = None
    if isinstance(details, dict):
        reasoning = _as_int(details.get("reasoning_tokens"))
    if reasoning is None:
        reasoning = _as_int(usage.get("reasoning_tokens"))
    cost = _as_float(usage.get("cost", usage.get("total_cost")))
    if cost is None and isinstance(usage.get("cost_details"), dict):
        details_cost = usage["cost_details"]
        cost = _as_float(
            details_cost.get("upstream_inference_cost", details_cost.get("total", details_cost.get("total_cost")))
        )
    return input_tokens, output_tokens, reasoning, cost

def _request(client: OpenAI, model: str, content: list[dict[str, Any]], params: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "stream": False,
    }
    if "temperature" in params:
        kwargs["temperature"] = params["temperature"]
    if "max_tokens" in params:
        kwargs["max_tokens"] = params["max_tokens"]
    started = time.perf_counter()
    response = client.chat.completions.create(**kwargs)
    latency = time.perf_counter() - started
    if hasattr(response, "model_dump"):
        data = sanitize_obj(response.model_dump())
    else:
        data = sanitize_obj(dict(response))
    choices = data.get("choices") or []
    message = {}
    finish = None
    if choices and isinstance(choices[0], dict):
        message = choices[0].get("message") or {}
        finish = choices[0].get("finish_reason")
    text = _message_text(message)
    input_tokens, output_tokens, reasoning_tokens, cost = _usage_from_dump(data)
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
        "finish_reason": finish,
        "hit_limit": finish == "length",
        "parameters_sent": {
            key: kwargs[key] for key in ("temperature", "max_tokens", "stream") if key in kwargs
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
    client = _openrouter_client(float(active["timeout_seconds"]))
    image_a_url = data_url_for_jpeg(image_a)
    image_b_url = data_url_for_jpeg(image_b)
    params: dict[str, Any] = {
        "temperature": active["temperature"],
        "max_tokens": active["max_output_tokens"],
        "include_detail": True,
    }

    def request_fn(current: dict[str, Any]) -> dict[str, Any]:
        content = build_openrouter_content(
            prompt,
            image_a_url,
            image_b_url,
            include_detail=bool(current.get("include_detail", True)),
        )
        return _request(client, model, content, current)

    result, adjustments = call_with_policy(request_fn, params, active)
    result, adjustments = expand_output_budget(
        result, adjustments, params, "max_tokens", active, request_fn
    )
    result["actual_parameter_adjustments"] = adjustments
    result["temperature_requested"] = active["temperature"]
    result["tools_enabled"] = False
    result["web_search"] = False
    return result
