# T04 — Backends

**Depends on:** T01  
**Goal:** Execute verification through fake and OpenRouter adapters.

## Scope

- Define one backend interface returning `BackendResult`.
- Implement `FakeBackend` for dry-run and CI, including scripted
  `true`/`false`/`unknown` and malformed-output cases.
- Implement one `OpenRouterBackend` adapter supporting at least two model
  identifiers from different cost tiers.
- Build requests from the two images and query, parse strict structured
  output, measure latency, and estimate cost from profile rates.
- Map malformed output to `invalid_output`, timeouts to `timeout`, and other
  transport failures to `error`; preserve `unknown` distinctly from `false`.
- Keep network access lazy and injectable for tests.

## Acceptance

- Fake tests cover all answer states and failure states without network calls.
- OpenRouter payload tests include both images and no folder-derived labels.
- No backend executes unless selected by a later pipeline decision.
