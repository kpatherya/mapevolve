# Interactive benchmark report

Open `mapevolve-results.html` in a browser. It is a single offline file with
100 cases, 300 strategy results, and all 51 benchmark images embedded.

The overview charts drill into the case explorer. Filters, score thresholds,
side-by-side answers, reference evidence, grading explanations, Jev signals,
image enlargement, dark/light themes, and CSV/JSON exports work locally.

Rebuild from the completed run:

```bash
.venv/bin/python -m scripts.build_results_dashboard
```

`dashboard-template.html` is the editable source. `dashboard-qa.json` records
browser checks; the PNG files are desktop and mobile previews. QA uses
Playwright and the installed Google Chrome browser:

```bash
.venv/bin/python -m scripts.check_results_dashboard
```

The source benchmark outputs and frozen configuration are unchanged. No model
calls are made to build or view this report.
