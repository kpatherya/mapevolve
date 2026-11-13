# T01 — Contracts

**Depends on:** none  
**Goal:** Define the stable, validated data contracts for the MVP.

## Scope

- Implement `RoutingRequest`, `QueryInterpretation`, `RoutingDecision`, and
  `BackendResult` as frozen, JSON-serializable dataclasses.
- Preserve all four family names plus `unknown`.
- Validate required strings, image paths/types, action/reason combinations,
  strict answer values (`true`, `false`, `unknown`), latency, cost, and
  schema version.
- Provide explicit serialization/deserialization and visible validation
  errors.
- Add focused tests using stdlib `unittest`.

## Acceptance

- Invalid contracts are rejected without coercion or silent defaults.
- Round trips preserve valid values.
- No third-party dependencies, network calls, environment reads, or
  application-code changes are introduced.
