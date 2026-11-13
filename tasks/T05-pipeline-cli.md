# T05 — Pipeline and CLI

**Depends on:** T03 and T04  
**Goal:** Deliver the runnable vertical slice and smoke verification.

## Scope

- Implement the ordered runner:
  `validate -> interpret -> decide -> execute -> record trace`.
- Invoke exactly the backend named by an `execute` decision; never fall back
  to another model.
- Add `--dry-run`, which reaches profile lookup and payload construction but
  performs no execution or network call.
- Emit a structured trace containing request, interpretation, decision,
  backend result, latency, and estimated cost without image bytes.
- Add stdlib `unittest` coverage for positive and negative pairs,
  `request_evidence`, unsupported-family abstention, and
  `true`/`false`/`unknown` outputs.

## Acceptance

- End-to-end fake execution is runnable in CI.
- Positive and negative smoke pairs reach the selected backend and preserve
  their parsed answer.
- Missing evidence and all unsupported families have no execution result.
- Dry-run has no network call and records the selected profile/backend.
- The repository remains stdlib-only and no live OpenRouter call is required.
