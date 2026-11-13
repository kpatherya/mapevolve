# MapEvolve Repository Map

This map records repository facts verified during the T00 audit. It does not
define future package structure.

## Canonical paths and ownership

| Path | Verified responsibility |
| --- | --- |
| `README.md` | Documents the smoke benchmark commands, required environment variables, and output behavior. |
| `benchmark/smoke_runner.py` | Dependency-free Python entry point. Loads the four-case manifest, validates image paths, builds OpenRouter vision requests, calls the API, validates returned JSON, and writes results. |
| `benchmark/smoke_cases.json` | Four smoke cases, their image paths, queries, answerability, and reference expectations. |
| `data/` | Source image collection organized by building directory. No verified metadata or coordinate files were found. |
| `assets/MapEvolve proposal.md` | Project proposal and intended workflows/evaluation; not an executable interface. |
| `plan.md` | Background implementation plan; it is not a current package or entry point. |
| `.gitignore` | Ignores `.keys`, `runs/`, and `__pycache__/`. |

No canonical Python package, `src/` tree, test suite, dependency manifest, or
configuration file was found. Jev is not currently implemented.

## Existing interfaces

- `load_cases(path=CASES_PATH)` in `benchmark/smoke_runner.py:50-68`
  validates four cases and confirms each referenced image exists.
- `request_payload(case, model, max_tokens, reasoning_effort)` in
  `benchmark/smoke_runner.py:75-105` creates an OpenRouter chat-completions
  request containing text and base64 image data.
- `call_openrouter(payload, api_key, timeout, retries)` in
  `benchmark/smoke_runner.py:122-153` performs the external request with
  bounded retries.
- `validate_response(response)` in `benchmark/smoke_runner.py:172-195`
  validates the provider's first `choices` item against the required answer
  JSON shape. This is provider-response parsing, not a Jev routing choice.
- The requested model list is supplied through `OPENROUTER_MODELS` or
  `--models`; the API key is supplied through `OPENROUTER_API_KEY` or
  `--api-key`.

There is no routing decision, backend registry, retrieval interface, metadata
lookup interface, evaluation scorer, or execution-trace contract.

## Canonical commands

The following commands are documented in `README.md` but were **not verified**
during the audit:

```bash
OPENROUTER_MODELS=provider/model-a,provider/model-b \
  python3 benchmark/smoke_runner.py --dry-run
```

```bash
source .keys
export OPENROUTER_MODELS=provider/model-a,provider/model-b
python3 benchmark/smoke_runner.py
```

The first command still creates the configured output directory in the current
implementation. The second performs live OpenRouter requests and requires a
valid API key and network access. No formal test command is documented or
configured.
