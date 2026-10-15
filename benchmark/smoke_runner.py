#!/usr/bin/env python3
"""Run four representative VLM smoke-test cases through OpenRouter.

The runner is intentionally dependency-free. It validates the case manifest,
constructs OpenRouter vision requests, and stores raw redacted results. It
does not attempt to score answers automatically.
"""

# OPENROUTER_MODELS=deepseek/deepseek-v4.1-flash,google/gemini-3.8-flash,anthropic/claude-opus-5.5,openai/gpt-6-luna
# DeepSeek V4.1 Flash, Gemini 3.8 Flash, Claude Opus 5.5, and GPT-6 Luna

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "benchmark" / "smoke_cases.json"
DEFAULT_OUTPUT = ROOT / "runs"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = """You are a careful visual benchmark participant.
Answer only from the supplied images and evidence. Do not use folder names,
hidden metadata, or outside knowledge. Distinguish visible observations from
inferences. If a requested fact is not established, refuse that part.
Do not show analysis or reasoning outside the JSON object. Keep the answer
concise and keep the evidence list to at most four items.
Return valid JSON with exactly these keys:
{
  "answer": "concise answer or refusal",
  "evidence": ["specific visible or supplied evidence"],
  "confidence": 0.0,
  "cannot_answer_reason": "empty string when fully answerable"
}
Confidence must be a number from 0 to 1."""


def load_cases(path: Path = CASES_PATH) -> list[dict[str, Any]]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or len(cases) != 4:
        raise ValueError(f"{path} must contain exactly four smoke-test cases")
    required = {"case_id", "class", "images", "query", "answerable", "reference"}
    seen: set[str] = set()
    for case in cases:
        if not required <= case.keys():
            raise ValueError(f"Case is missing fields: {case}")
        if case["case_id"] in seen:
            raise ValueError(f"Duplicate case_id: {case['case_id']}")
        seen.add(case["case_id"])
        if case["class"] not in {"simple", "moderate", "hard", "hardest"}:
            raise ValueError(f"Unknown query class: {case['class']}")
        for image in case["images"]:
            image_path = ROOT / image
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing case image: {image}")
    return cases


def image_data_url(path: Path) -> str:
    content_type, _ = mimetypes.guess_type(path.name)
    if content_type not in {"image/jpeg", "image/png", "image/webp", "image/gif"}:
        raise ValueError(f"Unsupported image type for OpenRouter: {path}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{content_type};base64,{encoded}"


def request_payload(
    case: dict[str, Any], model: str, max_tokens: int, reasoning_effort: Optional[str]
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {"type": "text", "text": case["query"]},
    ]
    for image in case["images"]:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_data_url(ROOT / image)},
            }
        )
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}
    return payload


def prompt_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def parse_models(value: Optional[str]) -> list[str]:
    models = [item.strip() for item in (value or "").split(",") if item.strip()]
    if not models:
        raise ValueError(
            "No models supplied. Set OPENROUTER_MODELS to exact OpenRouter IDs, "
            "comma-separated."
        )
    return models


def call_openrouter(
    payload: dict[str, Any], api_key: str, timeout: float, retries: int
) -> tuple[dict[str, Any], float]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        OPENROUTER_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/kpatherya/mapevolve",
            "X-Title": "MapEvolve VLM smoke benchmark",
        },
        method="POST",
    )
    started = time.perf_counter()
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8")), time.perf_counter() - started
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            retryable = not isinstance(error, urllib.error.HTTPError) or error.code in {
                408,
                409,
                429,
            } or error.code >= 500
            if not retryable or attempt == retries:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("OpenRouter request loop ended unexpectedly")


def redacted_result(
    case: dict[str, Any], model: str, payload: dict[str, Any], response: dict[str, Any], latency: float
) -> dict[str, Any]:
    usage = response.get("usage", {})
    result: dict[str, Any] = {
        "case_id": case["case_id"],
        "class": case["class"],
        "model": model,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "latency_seconds": round(latency, 3),
        "prompt_hash": prompt_hash(payload),
        "usage": usage,
        "response": response,
    }
    result["status"], result["validation_error"] = validate_response(response)
    return result


def validate_response(response: dict[str, Any]) -> tuple[str, str]:
    """Validate the requested JSON answer without rewriting it."""
    try:
        choice = response["choices"][0]
        if choice.get("finish_reason") == "length":
            return "truncated_output", (
                "The provider exhausted max_tokens before returning message.content. "
                "Increase the completion budget or reduce reasoning."
            )
        content = choice["message"]["content"]
        parsed = json.loads(content)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
        return "malformed_output", f"Expected JSON response: {error}"
    required = {"answer", "evidence", "confidence", "cannot_answer_reason"}
    if set(parsed) != required:
        return "malformed_output", f"Expected exactly {sorted(required)}"
    if not isinstance(parsed["answer"], str) or not isinstance(parsed["evidence"], list):
        return "malformed_output", "answer must be a string and evidence must be a list"
    if not isinstance(parsed["confidence"], (int, float)) or not 0 <= parsed["confidence"] <= 1:
        return "malformed_output", "confidence must be a number in [0, 1]"
    if not isinstance(parsed["cannot_answer_reason"], str):
        return "malformed_output", "cannot_answer_reason must be a string"
    return "ok", ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=os.environ.get("OPENROUTER_MODELS"))
    parser.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1200,
        help="Completion budget, including provider reasoning tokens",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("minimal", "low", "medium", "high"),
        default=None,
        help="Requested reasoning budget when supported by the provider",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and summarize without API calls")
    args = parser.parse_args()

    try:
        cases = load_cases()
        models = parse_models(args.models)
        if not args.dry_run and not args.api_key:
            raise ValueError("OPENROUTER_API_KEY is required unless --dry-run is used")
    except (ValueError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Validated {len(cases)} cases and {len(models)} model(s).")
    for case in cases:
        print(f"- {case['case_id']}: {case['class']} ({len(case['images'])} image(s))")
    if args.dry_run:
        for model in models:
            for case in cases:
                payload = request_payload(
                    case, model, args.max_tokens, args.reasoning_effort
                )
                print(f"  {model} / {case['case_id']} / {prompt_hash(payload)[:12]}")
        return 0

    for model in models:
        model_dir = args.output_dir / model.replace("/", "__")
        model_dir.mkdir(parents=True, exist_ok=True)
        for case in cases:
            payload = request_payload(
                case, model, args.max_tokens, args.reasoning_effort
            )
            try:
                response, latency = call_openrouter(
                    payload, args.api_key, args.timeout, args.retries
                )
                result = redacted_result(case, model, payload, response, latency)
                output = model_dir / f"{case['case_id']}.json"
                output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
                print(f"saved {output.relative_to(ROOT)} ({latency:.1f}s)")
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as error:
                error_result = {
                    "case_id": case["case_id"],
                    "class": case["class"],
                    "model": model,
                    "status": "api_error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                output = model_dir / f"{case['case_id']}.error.json"
                output.write_text(json.dumps(error_result, indent=2) + "\n", encoding="utf-8")
                print(f"failed {model} / {case['case_id']}: {error}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
