# MapEvolve MVP plan

## Purpose

MapEvolve uses Jev as a query-classification router. Jev maps an incoming
query to a query family, then selects the most cost-efficient VLM on a
pre-profiled accuracy/cost Pareto frontier for that family. The goal is to
avoid paying for a frontier VLM when a cheaper model is within the accepted
accuracy margin (for example, within 2%).

This plan describes a runnable, stdlib-only vertical slice. It does not
implement application code, live model calls, dynamic profiling, retrieval,
Polaris integration, streaming, or multi-agent orchestration.

## Query taxonomy and MVP capability

The schema and classifier preserve all four families:

| Family | Example | MVP execution |
| --- | --- | --- |
| `same_building_verification` | “Are these the same building?” | Executable |
| `change_captioning` | “Identify the similarities and differences...” | Stubbed / unsupported |
| `evidence_backed_investigation` | “Who founded this building?” | Stubbed / unsupported |
| `proximity_computation` | “How close is this building to X?” | Stubbed / unsupported |

The classifier may also return `unknown`. Query family is determined from
query text, not image-folder names or image count.

## System flow

```text
RoutingRequest
  -> Classify
  -> Evaluate Evidence
  -> Lookup Pareto Choice
  -> Execute selected backend
  -> Validate backend result
  -> BackendResult / trace
```

1. **Classify** the text into exactly one family or `unknown`.
2. **Evaluate evidence** using family-specific requirements. The executable
   family requires exactly two images. Missing or incorrect evidence produces
   `action="request_evidence"` and no backend call.
3. **Lookup Pareto choice** from a static, offline profile. For an unsupported
   family, return `action="abstain"` with
   `reason_code="capability_not_yet_supported"`.
4. **Execute** only when the decision says `execute`; select the named backend.
5. **Validate** the structured `true`, `false`, or `unknown` answer and record
   latency and estimated cost. Invalid provider output is an execution result,
   not a routing fallback.

No backend is called for `request_evidence`, `abstain`, or `unknown`.

## Core contracts

Use frozen Python dataclasses (or equivalent stdlib validation) with explicit
`validate()` methods and JSON-safe serialization. Contract validation errors
must be visible to callers; do not silently coerce invalid input.

### `RoutingRequest`

```text
request_id: str
query: str
images: list[str]
```

`request_id` and `query` are non-empty. Image paths are existing supported
image files. The classifier accepts zero or more images; evidence policy later
requires exactly two for `same_building_verification`.

### `QueryInterpretation`

```text
request_id: str
family: same_building_verification
       | change_captioning
       | evidence_backed_investigation
       | proximity_computation
       | unknown
confidence: float | None
rule_id: str
```

`rule_id` makes deterministic rule-based classification traceable. Confidence
is informational and never controls routing.

### `RoutingDecision`

```text
request_id: str
family: str
action: execute | request_evidence | abstain
reason_code: str
backend_id: str | None
required_images: int | None
```

For executable requests with exactly two images, `action="execute"` names one
backend selected by the Pareto policy. Missing evidence uses
`action="request_evidence"`. Every non-executable family and `unknown` uses
`action="abstain"`; recognized but unsupported families use
`reason_code="capability_not_yet_supported"`.

### `BackendResult`

```text
request_id: str
backend_id: str
status: success | invalid_output | timeout | error
answer: true | false | unknown | None
evidence: list[str]
latency_ms: float
estimated_cost_usd: float | None
raw_content: str | None
error: str | None
```

Successful verification output is strict JSON with `true`, `false`, or
`unknown` semantics. `unknown` is not coerced to `false`. Raw content must not
contain image bytes. Latency is measured at execution; estimated cost is
derived from the profile and usage when available.

## Static profile configuration

Profiles are offline configuration, not runtime evaluation. JSON is preferred
so the loader needs only the Python standard library:

```json
{
  "schema_version": 1,
  "accuracy_margin": 0.02,
  "families": {
    "same_building_verification": {
      "recommended_model": "cheap-vlm",
      "models": [
        {
          "model": "cheap-vlm",
          "accuracy": 0.96,
          "input_cost_usd_per_1m_tokens": 0.20,
          "image_cost_usd": 0.0002
        },
        {
          "model": "frontier-vlm",
          "accuracy": 0.98,
          "input_cost_usd_per_1m_tokens": 2.00,
          "image_cost_usd": 0.002
        }
      ]
    },
    "change_captioning": {
      "recommended_model": null,
      "models": []
    },
    "evidence_backed_investigation": {
      "recommended_model": null,
      "models": []
    },
    "proximity_computation": {
      "recommended_model": null,
      "models": []
    }
  }
}
```

Each model profile stores offline accuracy, input cost per one million tokens,
and image cost. A family must list at least a cheap/small and an expensive
baseline model for the executable MVP family. The loader validates family
names, numeric ranges, model identifiers, and the recommended model.

The decision engine considers only models on the family’s Pareto frontier:
discard a model when another is at least as accurate and no more expensive.
Choose the lowest-cost frontier model whose accuracy is within
`accuracy_margin` of the best profiled accuracy. The decision records the
profile/model identifiers and never performs profiling or a live comparison.

## Backends and execution

`FakeBackend` is deterministic and used by tests, dry runs, and CI. It can
return each valid answer or malformed output without network access.

`OpenRouterBackend` is one unified adapter with a model identifier, API key,
timeout, and response parser. It supports at least two configured models from
different cost tiers. It sends the two images and the verification prompt,
parses strict structured output, measures latency, and estimates cost from
profiled token/image rates. Live execution requires explicit credentials and
is not part of CI.

## CLI and verification

The end-to-end runner performs:

```text
validate -> interpret -> decide -> execute (unless dry-run) -> record trace
```

`--dry-run` performs validation, classification, evidence checks, profile
lookup, and payload construction without a network call. Smoke verification
must cover positive and negative image pairs, missing evidence, unsupported
families, and the `true`/`false`/`unknown` result forms.

## Dependency and test boundary

Runtime and tests use only Python standard-library modules. Tests use
`unittest`; no package installation or application-code changes are implied by
this planning update. The five cards in `tasks/` are sequential and are the
complete MVP work queue.
