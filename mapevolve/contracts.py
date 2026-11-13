"""Validated, JSON-serializable routing contracts for the MVP."""

from __future__ import annotations

import json
import math
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROUTING_SCHEMA_VERSION = 1

FAMILIES = (
    "same_building_verification",
    "change_captioning",
    "evidence_backed_investigation",
    "proximity_computation",
    "unknown",
)

ACTIONS = ("execute", "request_evidence", "abstain")
ANSWERS = ("true", "false", "unknown")
BACKEND_STATUSES = ("success", "invalid_output", "timeout", "error")
IMAGE_CONTENT_TYPES = frozenset({"image/jpeg", "image/png", "image/webp", "image/gif"})

EXECUTE_REASON_CODES = frozenset({"ready"})
REQUEST_EVIDENCE_REASON_CODES = frozenset({"missing_required_images", "invalid_image"})
ABSTAIN_REASON_CODES = frozenset({"capability_not_yet_supported", "unknown_query_family"})


class ContractValidationError(ValueError):
    """Raised when contract payloads are invalid."""


def _require_schema_version(version: Any, context: str) -> None:
    if version != ROUTING_SCHEMA_VERSION:
        raise ContractValidationError(
            f"{context}.schema_version must be {ROUTING_SCHEMA_VERSION}, got {version!r}"
        )


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ContractValidationError(f"{field_name} must be a string")
    if not value.strip():
        raise ContractValidationError(f"{field_name} must be non-empty")
    return value


def _require_string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise ContractValidationError(f"{field_name} must be a list")
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise ContractValidationError(f"{field_name}[{index}] must be a string")
    return value


def _validate_image_path(image_path: str, index: int) -> None:
    path = Path(image_path)
    if not path.is_file():
        raise ContractValidationError(f"images[{index}] does not exist: {image_path}")
    content_type, _ = mimetypes.guess_type(path.name)
    if content_type not in IMAGE_CONTENT_TYPES:
        raise ContractValidationError(f"images[{index}] has unsupported type: {image_path}")


def _require_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(f"{field_name} must be a number")
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ContractValidationError(f"{field_name} must be finite")
    return numeric_value


def _validate_payload_keys(
    payload: dict[str, Any], context: str, allowed_keys: set[str], required_keys: set[str]
) -> None:
    payload_keys = set(payload.keys())
    unknown = payload_keys - allowed_keys
    if unknown:
        unknown_list = ", ".join(sorted(unknown))
        raise ContractValidationError(f"{context} payload has unknown field(s): {unknown_list}")
    missing = required_keys - payload_keys
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ContractValidationError(f"{context} payload is missing required field(s): {missing_list}")


@dataclass(frozen=True)
class RoutingRequest:
    request_id: str
    query: str
    images: list[str] = field(default_factory=list)
    schema_version: int = ROUTING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version, "RoutingRequest")
        _require_non_empty_string(self.request_id, "request_id")
        _require_non_empty_string(self.query, "query")
        _require_string_list(self.images, "images")
        for index, image_path in enumerate(self.images):
            _validate_image_path(image_path, index)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "query": self.query,
            "images": list(self.images),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RoutingRequest":
        if not isinstance(payload, dict):
            raise ContractValidationError("RoutingRequest payload must be a dict")
        _validate_payload_keys(
            payload,
            "RoutingRequest",
            allowed_keys={"schema_version", "request_id", "query", "images"},
            required_keys={"schema_version", "request_id", "query", "images"},
        )
        return cls(
            schema_version=payload.get("schema_version"),
            request_id=payload.get("request_id"),
            query=payload.get("query"),
            images=payload.get("images"),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> "RoutingRequest":
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ContractValidationError(f"RoutingRequest JSON decode failed: {error}") from error
        return cls.from_dict(parsed)


@dataclass(frozen=True)
class QueryInterpretation:
    request_id: str
    family: str
    rule_id: str
    confidence: float | None = None
    schema_version: int = ROUTING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version, "QueryInterpretation")
        _require_non_empty_string(self.request_id, "request_id")
        _require_non_empty_string(self.rule_id, "rule_id")
        if self.family not in FAMILIES:
            raise ContractValidationError(f"family must be one of {FAMILIES}, got {self.family!r}")
        if self.confidence is not None:
            confidence = _require_number(self.confidence, "confidence")
            if confidence < 0.0 or confidence > 1.0:
                raise ContractValidationError("confidence must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "family": self.family,
            "rule_id": self.rule_id,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "QueryInterpretation":
        if not isinstance(payload, dict):
            raise ContractValidationError("QueryInterpretation payload must be a dict")
        _validate_payload_keys(
            payload,
            "QueryInterpretation",
            allowed_keys={"schema_version", "request_id", "family", "rule_id", "confidence"},
            required_keys={"schema_version", "request_id", "family", "rule_id"},
        )
        return cls(
            schema_version=payload.get("schema_version"),
            request_id=payload.get("request_id"),
            family=payload.get("family"),
            rule_id=payload.get("rule_id"),
            confidence=payload.get("confidence"),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> "QueryInterpretation":
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ContractValidationError(f"QueryInterpretation JSON decode failed: {error}") from error
        return cls.from_dict(parsed)


@dataclass(frozen=True)
class RoutingDecision:
    request_id: str
    family: str
    action: str
    reason_code: str
    backend_id: str | None = None
    required_images: int | None = None
    schema_version: int = ROUTING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version, "RoutingDecision")
        _require_non_empty_string(self.request_id, "request_id")
        if self.family not in FAMILIES:
            raise ContractValidationError(f"family must be one of {FAMILIES}, got {self.family!r}")
        if self.action not in ACTIONS:
            raise ContractValidationError(f"action must be one of {ACTIONS}, got {self.action!r}")
        _require_non_empty_string(self.reason_code, "reason_code")

        if self.action == "execute":
            if self.reason_code not in EXECUTE_REASON_CODES:
                raise ContractValidationError(
                    f"reason_code {self.reason_code!r} is invalid for action 'execute'"
                )
            if self.backend_id is None:
                raise ContractValidationError("backend_id is required when action='execute'")
            _require_non_empty_string(self.backend_id, "backend_id")
            if self.required_images is not None:
                raise ContractValidationError("required_images must be None when action='execute'")
            return

        if self.backend_id is not None:
            raise ContractValidationError("backend_id must be None unless action='execute'")

        if self.action == "request_evidence":
            if self.reason_code not in REQUEST_EVIDENCE_REASON_CODES:
                raise ContractValidationError(
                    f"reason_code {self.reason_code!r} is invalid for action 'request_evidence'"
                )
            if not isinstance(self.required_images, int):
                raise ContractValidationError("required_images must be an int for action='request_evidence'")
            if self.required_images <= 0:
                raise ContractValidationError("required_images must be > 0 for action='request_evidence'")
            return

        if self.reason_code not in ABSTAIN_REASON_CODES:
            raise ContractValidationError(
                f"reason_code {self.reason_code!r} is invalid for action 'abstain'"
            )
        if self.required_images is not None:
            raise ContractValidationError("required_images must be None for action='abstain'")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "family": self.family,
            "action": self.action,
            "reason_code": self.reason_code,
            "backend_id": self.backend_id,
            "required_images": self.required_images,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RoutingDecision":
        if not isinstance(payload, dict):
            raise ContractValidationError("RoutingDecision payload must be a dict")
        _validate_payload_keys(
            payload,
            "RoutingDecision",
            allowed_keys={
                "schema_version",
                "request_id",
                "family",
                "action",
                "reason_code",
                "backend_id",
                "required_images",
            },
            required_keys={"schema_version", "request_id", "family", "action", "reason_code"},
        )
        return cls(
            schema_version=payload.get("schema_version"),
            request_id=payload.get("request_id"),
            family=payload.get("family"),
            action=payload.get("action"),
            reason_code=payload.get("reason_code"),
            backend_id=payload.get("backend_id"),
            required_images=payload.get("required_images"),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> "RoutingDecision":
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ContractValidationError(f"RoutingDecision JSON decode failed: {error}") from error
        return cls.from_dict(parsed)


@dataclass(frozen=True)
class BackendResult:
    request_id: str
    backend_id: str
    status: str
    answer: str | None
    evidence: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    estimated_cost_usd: float | None = None
    raw_content: str | None = None
    error: str | None = None
    schema_version: int = ROUTING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version, "BackendResult")
        _require_non_empty_string(self.request_id, "request_id")
        _require_non_empty_string(self.backend_id, "backend_id")
        if self.status not in BACKEND_STATUSES:
            raise ContractValidationError(
                f"status must be one of {BACKEND_STATUSES}, got {self.status!r}"
            )
        if self.answer is not None and self.answer not in ANSWERS:
            raise ContractValidationError(f"answer must be one of {ANSWERS} or None")

        _require_string_list(self.evidence, "evidence")
        for index, item in enumerate(self.evidence):
            _require_non_empty_string(item, f"evidence[{index}]")

        latency = _require_number(self.latency_ms, "latency_ms")
        if latency < 0:
            raise ContractValidationError("latency_ms must be >= 0")

        if self.estimated_cost_usd is not None:
            cost = _require_number(self.estimated_cost_usd, "estimated_cost_usd")
            if cost < 0:
                raise ContractValidationError("estimated_cost_usd must be >= 0")

        if self.raw_content is not None:
            _require_non_empty_string(self.raw_content, "raw_content")
            if "base64," in self.raw_content:
                raise ContractValidationError("raw_content must not contain image bytes")

        if self.error is not None:
            _require_non_empty_string(self.error, "error")

        if self.status == "success":
            if self.answer is None:
                raise ContractValidationError("answer is required when status='success'")
            if self.error is not None:
                raise ContractValidationError("error must be None when status='success'")
        else:
            if self.answer is not None:
                raise ContractValidationError("answer must be None when status is not 'success'")
            if self.error is None:
                raise ContractValidationError("error is required when status is not 'success'")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "backend_id": self.backend_id,
            "status": self.status,
            "answer": self.answer,
            "evidence": list(self.evidence),
            "latency_ms": self.latency_ms,
            "estimated_cost_usd": self.estimated_cost_usd,
            "raw_content": self.raw_content,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BackendResult":
        if not isinstance(payload, dict):
            raise ContractValidationError("BackendResult payload must be a dict")
        _validate_payload_keys(
            payload,
            "BackendResult",
            allowed_keys={
                "schema_version",
                "request_id",
                "backend_id",
                "status",
                "answer",
                "evidence",
                "latency_ms",
                "estimated_cost_usd",
                "raw_content",
                "error",
            },
            required_keys={
                "schema_version",
                "request_id",
                "backend_id",
                "status",
                "answer",
                "evidence",
                "latency_ms",
            },
        )
        return cls(
            schema_version=payload.get("schema_version"),
            request_id=payload.get("request_id"),
            backend_id=payload.get("backend_id"),
            status=payload.get("status"),
            answer=payload.get("answer"),
            evidence=payload.get("evidence"),
            latency_ms=payload.get("latency_ms"),
            estimated_cost_usd=payload.get("estimated_cost_usd"),
            raw_content=payload.get("raw_content"),
            error=payload.get("error"),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> "BackendResult":
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ContractValidationError(f"BackendResult JSON decode failed: {error}") from error
        return cls.from_dict(parsed)
