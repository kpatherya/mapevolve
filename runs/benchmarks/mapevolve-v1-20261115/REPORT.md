# MapEvolve frozen benchmark results

Strategy rows: 1; graded: 0. Known VLM + judge cost: $0.0000. TypeSafe costs may be additional.

Exploratory results against AI-drafted references; human review pending. GPT-5.1 grades all answers, including its own. 34 cases overlap earlier images.

| Strategy | Cases | Graded | Strict accuracy | Mean score (0–4) | Errors | Truncated | Attributed VLM cost | Mean VLM latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cheapest_eligible | 1 | 0 | 0.0% | pending | 0 | 0 | $0.0000 | 4.49s |

Accuracy uses all completed strategy rows as the denominator; API errors, missing grades and truncations cannot count as strictly correct. Partial-run results are provisional.

See comparison.csv for individual answers and scores; summary.csv/json for type, difficulty, track, location, Jev-only and overlap-not-detected breakdowns. Costs attributed to strategies reuse shared calls and must not be added to estimate actual spend. cost_ledger.json lists unique answer and judge records.

Full requests (including encoded images), complete SDK replies, exceptions, prompt text, all classification distributions, action/routing candidates, answer checks, grading rationale and timings are stored under cases/. Credentials are redacted; SDK-hidden retries and internal reasoning are not exposed. No benchmark outputs are ingested into router training.
