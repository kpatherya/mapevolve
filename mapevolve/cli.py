"""CLI entry point for the MapEvolve MVP routing pipeline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .backends import FakeBackend, FakeBackendCase, OpenRouterBackend, SUPPORTED_OPENROUTER_MODELS
from .contracts import ContractValidationError, RoutingRequest
from .pareto_policy import DEFAULT_ACCURACY_MARGIN, DEFAULT_PROFILES_PATH, load_profiles
from .pipeline import PipelineConfigurationError, run_pipeline


def _fail(parser: argparse.ArgumentParser, message: str) -> int:
    parser.print_usage(sys.stderr)
    print(f"{parser.prog}: error: {message}", file=sys.stderr)
    return 2


def _json_for_fake_case(answer: str) -> str:
    return json.dumps(
        {"answer": answer, "evidence": [f"Fake backend scripted {answer} result."]},
        separators=(",", ":"),
    )


def _build_fake_registry(fake_answer: str) -> dict[str, FakeBackend]:
    scripted_case = FakeBackendCase(
        status="success",
        answer=fake_answer,
        evidence=(f"Fake backend scripted {fake_answer} result.",),
        raw_content=_json_for_fake_case(fake_answer),
        latency_ms=1.0,
        estimated_cost_usd=0.0,
    )
    return {
        model_id: FakeBackend(backend_id=model_id, default_case=scripted_case)
        for model_id in sorted(SUPPORTED_OPENROUTER_MODELS)
    }


def _fake_payload_builder(backend_id: str):
    def _build(request: RoutingRequest) -> dict[str, object]:
        content = [{"type": "text", "text": request.query}]
        for image_path in request.images:
            content.append({"type": "image_path", "path": image_path})
        return {
            "backend_id": backend_id,
            "messages": [{"role": "user", "content": content}],
        }

    return _build


def _build_openrouter_registry(
    *,
    api_key: str,
    profiles_path: Path,
    timeout_seconds: float,
) -> dict[str, OpenRouterBackend]:
    catalog = load_profiles(profiles_path)
    models = catalog.for_family("same_building_verification").models
    return {
        model.model_id: OpenRouterBackend(
            model_id=model.model_id,
            api_key=api_key,
            input_cost_usd_per_1m_tokens=model.input_cost_usd_per_1m_tokens,
            image_cost_usd=model.image_cost_usd,
            timeout_seconds=timeout_seconds,
        )
        for model in models
        if model.model_id in SUPPORTED_OPENROUTER_MODELS
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the MapEvolve MVP routing pipeline.")
    parser.add_argument("--request-id", required=True, help="Stable request identifier.")
    parser.add_argument("--query", required=True, help="User query text.")
    parser.add_argument(
        "--image",
        dest="images",
        action="append",
        default=[],
        help="Image path. Pass twice for same-building execution.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Build decision and payload without execution.")
    parser.add_argument(
        "--backend-mode",
        choices=("fake", "openrouter"),
        default="fake",
        help="Use fake backend (default) or OpenRouter backend registry.",
    )
    parser.add_argument(
        "--fake-answer",
        choices=("true", "false", "unknown"),
        default="unknown",
        help="Scripted answer used by fake backend mode.",
    )
    parser.add_argument(
        "--profiles-path",
        default=str(DEFAULT_PROFILES_PATH),
        help="Path to static profiles JSON file.",
    )
    parser.add_argument(
        "--accuracy-margin",
        type=float,
        default=DEFAULT_ACCURACY_MARGIN,
        help="Maximum accuracy delta from best profile when selecting lowest-cost model.",
    )
    parser.add_argument(
        "--openrouter-api-key",
        default=None,
        help="OpenRouter API key. Uses OPENROUTER_API_KEY when omitted.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=90.0,
        help="HTTP timeout for OpenRouter execution mode.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write trace JSON. Prints to stdout when omitted.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    profiles_path = Path(args.profiles_path)
    try:
        request = RoutingRequest(request_id=args.request_id, query=args.query, images=list(args.images))
    except ContractValidationError as error:
        return _fail(parser, str(error))

    payload_builders = None
    if args.backend_mode == "fake":
        backend_registry = _build_fake_registry(args.fake_answer)
        payload_builders = {
            backend_id: _fake_payload_builder(backend_id) for backend_id in backend_registry.keys()
        }
    else:
        key = args.openrouter_api_key or os.getenv("OPENROUTER_API_KEY") or ""
        if not args.dry_run and not key.strip():
            return _fail(parser, "openrouter mode requires --openrouter-api-key or OPENROUTER_API_KEY")
        backend_registry = _build_openrouter_registry(
            api_key=key or "dry-run-key",
            profiles_path=profiles_path,
            timeout_seconds=args.timeout_seconds,
        )

    try:
        trace = run_pipeline(
            request,
            backend_registry=backend_registry,
            payload_builders=payload_builders,
            dry_run=args.dry_run,
            accuracy_margin=args.accuracy_margin,
            profiles_path=profiles_path,
        )
    except (ContractValidationError, PipelineConfigurationError, ValueError) as error:
        return _fail(parser, str(error))

    payload = json.dumps(trace.to_dict(), indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
