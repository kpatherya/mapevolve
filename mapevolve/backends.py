"""Backend adapters for executing same-building verification."""

from __future__ import annotations

import base64
import json
import mimetypes
import socket
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .contracts import ANSWERS, IMAGE_CONTENT_TYPES, BackendResult, RoutingRequest

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_SYSTEM_PROMPT = (
    "You are a strict verifier. Decide whether the two images depict the same building. "
    "Return JSON with exactly these keys: answer, evidence. "
    "answer must be one of: true, false, unknown. "
    "evidence must be a JSON array of concise visible observations."
)
SUPPORTED_OPENROUTER_MODELS = frozenset(
    {
        "openai/gpt-4.1-mini",
        "anthropic/claude-3.5-sonnet",
    }
)


def _image_to_data_url(image_path: str) -> str:
    path = Path(image_path)
    content_type, _ = mimetypes.guess_type(path.name)
    if content_type not in IMAGE_CONTENT_TYPES:
        raise ValueError(f"Unsupported image type for backend payload: {image_path}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{content_type};base64,{encoded}"


def _extract_message_content(response: dict[str, Any]) -> str:
    try:
        return response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError(f"response is missing choices[0].message.content: {error}") from error


def _default_openrouter_transport(
    payload: dict[str, Any], api_key: str, timeout_seconds: float, endpoint: str
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/kpatherya/mapevolve",
            "X-Title": "MapEvolve backend adapter",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


class Backend(ABC):
    """Abstract backend contract for execution adapters."""

    @property
    @abstractmethod
    def backend_id(self) -> str:
        """Stable backend identifier used by routing decisions."""

    @abstractmethod
    def execute(self, request: RoutingRequest) -> BackendResult:
        """Execute request and return a structured BackendResult."""


@dataclass(frozen=True)
class FakeBackendCase:
    """Scripted fake response for deterministic dry-run and CI coverage."""

    status: str = "success"
    answer: str | None = "unknown"
    evidence: tuple[str, ...] = ()
    error: str | None = None
    raw_content: str | None = None
    latency_ms: float = 1.0
    estimated_cost_usd: float | None = 0.0


class FakeBackend(Backend):
    """Deterministic backend with per-request scripted outcomes."""

    def __init__(
        self,
        *,
        backend_id: str = "fake-backend",
        scripted_cases: dict[str, FakeBackendCase] | None = None,
        default_case: FakeBackendCase | None = None,
    ) -> None:
        self._backend_id = backend_id
        self._scripted_cases = dict(scripted_cases or {})
        self._default_case = default_case or FakeBackendCase(
            status="success",
            answer="unknown",
            evidence=("Fake backend default answer.",),
            error=None,
            raw_content='{"answer":"unknown","evidence":["Fake backend default answer."]}',
            latency_ms=1.0,
            estimated_cost_usd=0.0,
        )

    @property
    def backend_id(self) -> str:
        return self._backend_id

    def execute(self, request: RoutingRequest) -> BackendResult:
        case = self._scripted_cases.get(request.request_id, self._default_case)
        return BackendResult(
            request_id=request.request_id,
            backend_id=self.backend_id,
            status=case.status,
            answer=case.answer,
            evidence=list(case.evidence),
            latency_ms=case.latency_ms,
            estimated_cost_usd=case.estimated_cost_usd,
            raw_content=case.raw_content,
            error=case.error,
        )


class OpenRouterBackend(Backend):
    """OpenRouter adapter with strict response validation and injectable transport."""

    def __init__(
        self,
        *,
        model_id: str,
        api_key: str,
        input_cost_usd_per_1m_tokens: float,
        image_cost_usd: float,
        timeout_seconds: float = 90.0,
        endpoint: str = OPENROUTER_URL,
        transport: Callable[[dict[str, Any], str, float, str], dict[str, Any]] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if model_id not in SUPPORTED_OPENROUTER_MODELS:
            raise ValueError(
                f"Unsupported model_id {model_id!r}; supported values: {sorted(SUPPORTED_OPENROUTER_MODELS)}"
            )
        if not api_key.strip():
            raise ValueError("api_key must be non-empty")
        if input_cost_usd_per_1m_tokens < 0.0:
            raise ValueError("input_cost_usd_per_1m_tokens must be >= 0")
        if image_cost_usd < 0.0:
            raise ValueError("image_cost_usd must be >= 0")
        if timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be > 0")

        self._backend_id = model_id
        self._model_id = model_id
        self._api_key = api_key
        self._input_cost_usd_per_1m_tokens = input_cost_usd_per_1m_tokens
        self._image_cost_usd = image_cost_usd
        self._timeout_seconds = timeout_seconds
        self._endpoint = endpoint
        self._transport = transport or _default_openrouter_transport
        self._clock = clock or time.perf_counter

    @property
    def backend_id(self) -> str:
        return self._backend_id

    def build_payload(self, request: RoutingRequest) -> dict[str, Any]:
        if len(request.images) != 2:
            raise ValueError("OpenRouterBackend requires exactly two images")

        user_content: list[dict[str, Any]] = [{"type": "text", "text": request.query}]
        for image_path in request.images:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _image_to_data_url(image_path)},
                }
            )
        return {
            "model": self._model_id,
            "messages": [
                {"role": "system", "content": OPENROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }

    def execute(self, request: RoutingRequest) -> BackendResult:
        payload = self.build_payload(request)
        started = self._clock()
        try:
            response = self._transport(payload, self._api_key, self._timeout_seconds, self._endpoint)
        except (TimeoutError, socket.timeout) as error:
            return BackendResult(
                request_id=request.request_id,
                backend_id=self.backend_id,
                status="timeout",
                answer=None,
                evidence=[],
                latency_ms=(self._clock() - started) * 1000.0,
                estimated_cost_usd=None,
                raw_content=None,
                error=str(error) or type(error).__name__,
            )
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as error:
            message = str(error) or type(error).__name__
            if isinstance(error, urllib.error.HTTPError):
                try:
                    detail = error.read().decode("utf-8", errors="replace")
                except Exception:
                    detail = ""
                detail = detail.replace(self._api_key, "[REDACTED]")[:1000]
                if detail:
                    message = f"{message}: {detail}"
            return BackendResult(
                request_id=request.request_id,
                backend_id=self.backend_id,
                status="error",
                answer=None,
                evidence=[],
                latency_ms=(self._clock() - started) * 1000.0,
                estimated_cost_usd=None,
                raw_content=None,
                error=message,
            )
        except json.JSONDecodeError as error:
            return BackendResult(
                request_id=request.request_id,
                backend_id=self.backend_id,
                status="invalid_output",
                answer=None,
                evidence=[],
                latency_ms=(self._clock() - started) * 1000.0,
                estimated_cost_usd=None,
                raw_content=None,
                error=f"transport JSON decode failed: {error}",
            )

        latency_ms = (self._clock() - started) * 1000.0
        try:
            raw_content = _extract_message_content(response)
            parsed = json.loads(raw_content)
            if not isinstance(parsed, dict):
                raise ValueError("content JSON must be an object")
            expected_keys = {"answer", "evidence"}
            parsed_keys = set(parsed.keys())
            if parsed_keys != expected_keys:
                raise ValueError(f"content must include exactly {sorted(expected_keys)}")
            answer = parsed["answer"]
            evidence = parsed["evidence"]
            if answer not in ANSWERS:
                raise ValueError(f"answer must be one of {ANSWERS}")
            if not isinstance(evidence, list) or any(not isinstance(item, str) for item in evidence):
                raise ValueError("evidence must be a list of strings")
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            return BackendResult(
                request_id=request.request_id,
                backend_id=self.backend_id,
                status="invalid_output",
                answer=None,
                evidence=[],
                latency_ms=latency_ms,
                estimated_cost_usd=None,
                raw_content=response.get("choices", [{}])[0].get("message", {}).get("content")
                if isinstance(response, dict)
                else None,
                error=str(error),
            )

        usage = response.get("usage", {}) if isinstance(response, dict) else {}
        prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        if isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, (int, float)):
            estimated_cost_usd = None
        else:
            token_cost = (float(prompt_tokens) / 1_000_000.0) * self._input_cost_usd_per_1m_tokens
            image_cost = len(request.images) * self._image_cost_usd
            estimated_cost_usd = token_cost + image_cost

        return BackendResult(
            request_id=request.request_id,
            backend_id=self.backend_id,
            status="success",
            answer=answer,
            evidence=evidence,
            latency_ms=latency_ms,
            estimated_cost_usd=estimated_cost_usd,
            raw_content=raw_content,
            error=None,
        )
