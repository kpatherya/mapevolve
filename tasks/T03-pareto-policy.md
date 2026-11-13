# T03 — Pareto policy

**Depends on:** T01 and T02  
**Goal:** Load static profiles and choose the cost-optimal eligible model.

## Scope

- Define and validate a stdlib JSON profile loader.
- Store per family: `recommended_model`, offline `accuracy`,
  `input_cost_usd_per_1m_tokens`, and `image_cost_usd`.
- Include at least a cheap small model and an expensive baseline for
  `same_building_verification`; retain empty/stub profiles for the other
  families.
- Compute the accuracy/cost Pareto frontier and choose the lowest-cost model
  within the configured accuracy margin of the best accuracy.
- Produce a `RoutingDecision` naming the selected backend/profile.
- Preserve `request_evidence` and unsupported-family abstentions without
  consulting or invoking a backend.

## Acceptance

- Dominated models are excluded.
- The configured margin is applied numerically (default example: `0.02`).
- Choice is deterministic and traceable to profile/version data.
- Missing, malformed, or unsupported profiles fail explicitly.
