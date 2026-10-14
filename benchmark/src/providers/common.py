from __future__ import annotations

import time
from typing import Any, Callable, Optional

from benchmark.src.redact import sanitize_text

TRANSIENT_STATUS = {429, 500, 502, 503, 504}
OPTIONAL_FIELDS = ("max_output_tokens", "max_tokens", "temperature", "top_p", "store", "detail")
UNSUPPORTED_HINTS = (
    "unsupported parameter",
    "unsupported value",
    "unknown parameter",
    "not supported",
    "does not support",
    "unrecognized request argument",
    "unrecognized",
    "unexpected keyword",
    "extra inputs are not permitted",
    "is not allowed",
)

class ProviderError(Exception):
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        modality: bool = False,
        not_found: bool = False,
        auth: bool = False,
        billing: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.modality = modality
        self.not_found = not_found
        self.auth = auth
        self.billing = billing

    @property
    def reject_model(self) -> bool:
        return self.modality or self.not_found or self.billing

def classify_exception(exc: Exception) -> dict[str, Any]:
    status = getattr(exc, "status_code", None)
    message = sanitize_text(str(exc), limit=2000)
    lowered = message.lower()
    type_name = type(exc).__name__.lower()
    modality = any(
        phrase in lowered
        for phrase in (
            "does not support image",
            "do not support image",
            "image inputs are not supported",
            "images are not supported",
            "image_url is not supported",
            "unsupported image",
            "vision is not supported",
            "support image input",
            "requested modalities",
            "input modality",
            "image content is not supported",
            "expected text but got image",
        )
    )
    not_found = status == 404 or any(
        phrase in lowered
        for phrase in (
            "model not found",
            "no endpoints found",
            "does not exist",
            "unknown model",
            "invalid model",
            "model_not_found",
        )
    )
    auth = status == 401 or any(
        phrase in lowered for phrase in ("invalid api key", "incorrect api key", "authentication", "unauthorized")
    )
    billing = status in (402, 403) and not auth
    optional_field = None
    hints = any(hint in lowered for hint in UNSUPPORTED_HINTS)
    if not modality and (hints or status == 400):
        for field in OPTIONAL_FIELDS:
            if field in lowered and (hints or f"unknown parameter: {field}" in lowered or f"unsupported {field}" in lowered):
                optional_field = field
                break
        if optional_field is None and "temperature" in lowered and any(
            hint in lowered for hint in ("unsupported", "not supported", "does not support", "only the default")
        ):
            optional_field = "temperature"
    transient = (
        status in TRANSIENT_STATUS
        or "timeout" in lowered
        or "timed out" in lowered
        or "timeout" in type_name
        or "connection" in type_name
        or "rate limit" in lowered
    )
    return {
        "status": status,
        "message": message,
        "modality": modality,
        "not_found": not_found,
        "auth": auth,
        "billing": billing,
        "optional_field": optional_field,
        "transient": transient,
    }

def needs_output_room(result: dict[str, Any]) -> bool:
    text = (result.get("response_text") or "").strip()
    if not text:
        return True
    return result.get("finish_reason") == "length"

def expand_output_budget(
    result: dict[str, Any],
    adjustments: list[str],
    params: dict[str, Any],
    token_key: str,
    settings: dict[str, Any],
    request_fn: Callable[[dict[str, Any]], Any],
) -> tuple[dict[str, Any], list[str]]:
    if token_key not in params or not needs_output_room(result):
        return result, adjustments
    ceiling = int(settings.get("empty_output_retry_max_tokens") or 4096)
    current = int(params[token_key])
    if current >= ceiling:
        return result, adjustments
    params[token_key] = ceiling
    adjustments.append(
        f"increased {token_key} from {current} to {ceiling} after the output was empty or truncated"
    )
    retried, more = call_with_policy(request_fn, params, settings)
    adjustments.extend(more)
    chosen = dict(_prefer_completed_text(result, retried))
    for key in ("reported_cost_usd", "latency_seconds", "input_tokens", "output_tokens", "reasoning_tokens"):
        left = result.get(key)
        right = retried.get(key)
        if left is not None and right is not None and not isinstance(left, bool):
            chosen[key] = left + right
    return chosen, adjustments

def _prefer_completed_text(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    second_text = (second.get("response_text") or "").strip()
    first_text = (first.get("response_text") or "").strip()
    if second_text and second.get("finish_reason") != "length":
        return second
    if len(second_text) > len(first_text):
        return second
    if second_text and not first_text:
        return second
    return first

def call_with_policy(
    request_fn: Callable[[dict[str, Any]], Any],
    params: dict[str, Any],
    settings: dict[str, Any],
    sleep_fn: Callable[[float], None] = time.sleep,
) -> tuple[Any, list[str]]:
    attempts = int(settings.get("max_attempts", 5))
    backoffs = list(settings.get("retry_backoff_seconds", [1, 2, 4, 8]))
    adjustments: list[str] = []
    stripped: set[str] = set()
    attempt = 0
    while attempt < attempts:
        attempt += 1
        try:
            return request_fn(params), adjustments
        except ProviderError:
            raise
        except Exception as exc:
            info = classify_exception(exc)
            if info["auth"]:
                raise ProviderError(info["message"], status_code=info["status"], auth=True) from exc
            if info["modality"] or info["not_found"]:
                raise ProviderError(
                    info["message"],
                    status_code=info["status"],
                    modality=info["modality"],
                    not_found=info["not_found"],
                ) from exc
            field = info["optional_field"]
            if field and field not in stripped:
                stripped.add(field)
                if field == "detail":
                    params["include_detail"] = False
                elif field in params:
                    params.pop(field, None)
                else:
                    field = None
                if field:
                    adjustments.append(
                        f"removed unsupported optional parameter {field} after: {info['message']}"
                    )
                    continue
            if info["billing"]:
                raise ProviderError(info["message"], status_code=info["status"], billing=True) from exc
            if info["transient"] and attempt < attempts:
                delay = backoffs[min(attempt - 1, len(backoffs) - 1)]
                adjustments.append(f"retry {attempt} after status {info['status']}: {info['message']}")
                sleep_fn(delay)
                continue
            raise ProviderError(info["message"], status_code=info["status"]) from exc
    raise ProviderError("exhausted retries")
