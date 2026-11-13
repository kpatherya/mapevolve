# Agent working rules

- Implement only the assigned task; plan.md is background, not authorization
  to implement future stages.
- Read the task file and relevant repository-map entries first.
- Inspect exact relevant files before proposing edits.
- Reuse existing code and dependencies when suitable.
- Do not perform unrelated cleanup or introduce speculative abstractions.
- Do not delete data, weights, licenses, or unique experiment artifacts.
- Do not install dependencies, download models, invoke paid services, or
  launch benchmarks without explicit approval.
- Do not change shared contracts without explicit approval.
- Preserve user changes. Do not commit, push, or reset unless requested.
- Run focused tests first, then the task's required checks.
- After two unsuccessful attempts at the same failure, stop and report
  evidence, attempted fixes, and the smallest next investigation.
- Final response: changed files, checks and results, limitations.
- Keep final reports concise; never print full files or large logs.