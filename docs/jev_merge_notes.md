# Jev branch reconciliation (T06)

Checked with `git merge-tree --write-tree HEAD <branch>` (read-only):

| Branch | Result vs `jev-routing` HEAD |
| --- | --- |
| `origin/jev-adaptive-routing` | Single content conflict: `README.md`. `plan.md` also differs (it rewrote the file) and will conflict once the local `plan.md` edit is committed. |
| `origin/benchmark` | `.gitignore` add/add conflict; `.DS_Store` modify/delete. |
| `origin/main` | Ancestor of both; no conflict. |

Resolution policy when merging to `main`:

- `README.md`: keep the MVP quick start; link to this file for Jev options. Do not import the Jev results tables.
- `plan.md`: keep the MVP plan; Jev's rewrite is not adopted.
- `.gitignore`: union of both (done here: `.DS_Store`, `.env`, `.venv/`, `*.pyc`, `results/live_runs/`).
- Not imported: `jev_router/` package, `app.py`, SIFT/learned-quality modules, `results/`, `benchmark/artifacts/`, `benchmark/runs/`, demo images.
- Ported ideas (re-implemented stdlib-first in `mapevolve/`): Jev question set and leakage guard (`jev_classifier.py`), taxonomy mapping, cross-view prompt guidance (`backends.py`), optional image downscale (needs Pillow).
