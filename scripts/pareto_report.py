"""Pareto frontiers of the VLMs per query cell, plus the router's own curve.

    python -m scripts.pareto_report                 # heuristic classifications unless TYPESAFE_API_KEY is set
    python -m scripts.pareto_report --n-boot 500

Outputs (results/pareto/):
  frontier.json   per-cell points, objectives, frontier, bootstrap membership
                  (read by the policy when pareto.prune_dominated is on)
  frontier.csv    one row per cell x model
  router_curve.csv  quality/cost of the router under swept modes, budgets, weights
  index.html      tables and quality-vs-cost charts (open in a browser)
  report.md       the same tables in Markdown
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import sys
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv

from jev_router.classify_cache import resolve_image_aware
from jev_router.dataset import ROOT, load_all
from jev_router.graded import load_graded, row_cost
from jev_router.offline import deep_merge, fixed_choice, load_offline, route_all, score_choices
from jev_router.pareto import Obs, analyze_cell, cells_for, complete_cases, live_reliability, quality_cost_front
from jev_router.registry import ModelRegistry, load_routing_config

OUT = ROOT / "results" / "pareto"
BUDGETS = [0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.05]
WEIGHTS = [(0.4, 0.4, 0.2), (0.6, 0.25, 0.15), (0.8, 0.1, 0.1), (0.9, 0.05, 0.05)]


def build_obs(registry: ModelRegistry) -> tuple[list[Obs], dict]:
    cases = {c.case_id: c for c in load_all()}
    obs = []
    for g in load_graded(keep_repeats=True):
        c = cases.get(g.case_id)
        if c is None:
            continue
        cost, _ = row_cost(g, registry)
        obs.append(Obs(g.case_id, g.model_key, g.score, g.unsupported, cost, g.latency_seconds, c.answerable, g.run_id))
    return complete_cases(obs), cases


def router_curve(classifier: str, image_aware: bool) -> list[dict]:
    data = load_offline(classifier, image_aware)
    rows = []
    for key in data.pool:
        rows.append({"strategy": f"always {key}", "kind": "model", **score_choices(data, fixed_choice(data, key))})
    for mode in ("economy", "balanced", "accuracy"):
        for budget in BUDGETS:
            weight_sets = WEIGHTS if mode == "balanced" else [None]
            for w in weight_sets:
                cfg = data.cfg if w is None else deep_merge(
                    data.cfg, {"modes": {"balanced": {"weights": {"quality": w[0], "cost": w[1], "latency": w[2]}}}})
                label = f"{mode} budget=${budget}" + (f" w={w[0]}/{w[1]}/{w[2]}" if w else "")
                rows.append({"strategy": label, "kind": "router", "mode": mode, "budget_usd": budget,
                             **score_choices(data, route_all(data, cfg, mode, budget))})
    return [r for r in rows if r.get("n_cases")]


# ------------------------------------------------------------------ chart
THEMES = {
    "light": {"fg": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df", "s1": "#2a78d6", "s2": "#eb6834", "dot": "#9b9a95", "ring": "#fcfcfb"},
    "dark": {"fg": "#f0efec", "muted": "#c3c2b7", "grid": "#2e2e2c", "s1": "#3987e5", "s2": "#d95926", "dot": "#7d7c77", "ring": "#1a1a19"},
}


def _embedded_style(theme: str) -> str:
    """Self-contained colors for an SVG shown as an image (no page CSS reaches it)."""
    t = THEMES[theme]
    return (f"<style>.grid{{stroke:{t['grid']};stroke-width:1}}.tick{{fill:{t['muted']};font:11px system-ui,sans-serif}}"
            f".axis{{fill:{t['muted']};font:12px system-ui,sans-serif}}.lbl{{fill:{t['fg']};font:11.5px system-ui,sans-serif}}"
            f".fpt{{fill:{t['s1']};stroke:{t['ring']};stroke-width:2}}.dpt{{fill:{t['dot']};stroke:{t['ring']};stroke-width:2}}"
            f".rpt{{fill:{t['s2']};stroke:{t['ring']};stroke-width:2}}.fline{{fill:none;stroke:{t['s1']};stroke-width:2}}"
            f".rline{{fill:none;stroke:{t['s2']};stroke-width:2;stroke-dasharray:5 4}}</style>")


def svg_chart(title: str, models: list[tuple[str, float, float, bool]], router: list[tuple[float, float]],
              theme: Optional[str] = None) -> str:
    """Quality (y) vs cost per query (x, log). models: (name, quality, cost, on_frontier).

    theme=None styles via the host page's CSS; "light"/"dark" embeds colors so
    the SVG can be shown as a standalone image (e.g. st.image in the app)."""
    W, H, L, R, T, B = 720, 400, 64, 150, 34, 46
    pts = [(c, q) for _, q, c, _ in models if c] + [(c, q) for q, c in router if c]
    if not pts:
        return ""
    lo, hi = min(c for c, _ in pts), max(c for c, _ in pts)
    lx0, lx1 = math.log10(lo) - 0.2, math.log10(hi) + 0.2
    qs = [q for _, q in pts]
    y0, y1 = max(0.0, math.floor(min(qs) * 10) / 10 - 0.05), min(1.0, math.ceil(max(qs) * 10) / 10 + 0.02)
    X = lambda c: L + (math.log10(c) - lx0) / (lx1 - lx0) * (W - L - R)
    Y = lambda q: T + (1 - (q - y0) / (y1 - y0)) * (H - T - B)
    e = html.escape
    out = [f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="{e(title)}" class="chart" '
           f'xmlns="http://www.w3.org/2000/svg">' + (_embedded_style(theme) if theme else "")]
    # grid + y ticks
    step = 0.1 if y1 - y0 <= 0.6 else 0.2
    q = math.ceil(y0 / step) * step
    while q <= y1 + 1e-9:
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(q):.1f}" y2="{Y(q):.1f}" class="grid"/>'
                   f'<text x="{L - 8}" y="{Y(q) + 4:.1f}" text-anchor="end" class="tick">{q:.1f}</text>')
        q += step
    # x ticks at 1-2-5 per decade
    for d in range(math.floor(lx0), math.ceil(lx1) + 1):
        for m in (1, 2, 5):
            c = m * 10 ** d
            if lx0 <= math.log10(c) <= lx1:
                lab = f"${c:.{max(0, -d)}f}"
                out.append(f'<line x1="{X(c):.1f}" x2="{X(c):.1f}" y1="{T}" y2="{H - B}" class="grid"/>'
                           f'<text x="{X(c):.1f}" y="{H - B + 16}" text-anchor="middle" class="tick">{lab}</text>')
    out.append(f'<text x="{(L + W - R) / 2}" y="{H - 8}" text-anchor="middle" class="axis">cost per query (log scale)</text>'
               f'<text x="14" y="{(T + H - B) / 2}" text-anchor="middle" class="axis" transform="rotate(-90 14 {(T + H - B) / 2})">quality (mean score ÷ 4)</text>')
    # router frontier
    rf = quality_cost_front([("r", q, c) for q, c in router if c])
    if len(rf) > 1:
        d = " ".join(f"{'M' if i == 0 else 'L'}{X(c):.1f},{Y(q):.1f}" for i, (_, q, c) in enumerate(rf))
        out.append(f'<path d="{d}" class="rline"/>')
    for q, c in router:
        if c:
            out.append(f'<rect x="{X(c) - 4:.1f}" y="{Y(q) - 4:.1f}" width="8" height="8" class="rpt">'
                       f'<title>router setting: quality {q:.3f}, ${c:.5f}/query</title></rect>')
    # The line is the 2-D quality-vs-cost frontier. Blue dots are the full
    # multi-objective frontier, so a blue dot off the line is there for another
    # axis (e.g. fewest unsupported claims), which this chart doesn't show.
    mf = quality_cost_front([(n, q, c) for n, q, c, _ in models if c])
    if len(mf) > 1:
        d = " ".join(f"{'M' if i == 0 else 'L'}{X(c):.1f},{Y(q):.1f}" for i, (_, q, c) in enumerate(mf))
        out.append(f'<path d="{d}" class="fline"/>')
    for name, q, c, on in sorted(models, key=lambda m: m[3]):
        if not c:
            continue
        out.append(f'<circle cx="{X(c):.1f}" cy="{Y(q):.1f}" r="5.5" class="{"fpt" if on else "dpt"}">'
                   f'<title>{e(name)}: quality {q:.3f}, ${c:.5f}/query, {"on frontier" if on else "dominated"}</title></circle>'
                   f'<text x="{X(c) + 9:.1f}" y="{Y(q) + 4:.1f}" class="lbl">{e(name)}</text>')
    # legend
    lx = L
    items = [("fpt", "on frontier (all objectives)"), ("dpt", "dominated model")]
    if any(c for _, c in router):
        items.append(("rpt", "router setting"))
    for cls, text in items:
        shape = (f'<rect x="{lx}" y="10" width="9" height="9" class="{cls}"/>' if cls == "rpt"
                 else f'<circle cx="{lx + 5}" cy="15" r="5" class="{cls}"/>')
        out.append(shape + f'<text x="{lx + 15}" y="19" class="tick">{text}</text>')
        lx += 190
    out.append("</svg>")
    return "".join(out)


CSS = """
:root{--bg:#fcfcfb;--fg:#0b0b0b;--muted:#52514e;--grid:#e4e3df;--line:#d6d5d0;--s1:#2a78d6;--s2:#eb6834;--dot:#9b9a95}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#1a1a19;--fg:#f0efec;--muted:#c3c2b7;--grid:#2e2e2c;--line:#3a3a37;--s1:#3987e5;--s2:#d95926;--dot:#7d7c77;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#1a1a19;--fg:#f0efec;--muted:#c3c2b7;--grid:#2e2e2c;--line:#3a3a37;--s1:#3987e5;--s2:#d95926;--dot:#7d7c77;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;margin:0;padding:24px max(16px,env(safe-area-inset-left,0px))}
main{max-width:980px;margin:0 auto}h1{font-size:1.6rem;margin:0 0 6px}h2{font-size:1.2rem;margin:36px 0 8px;overflow-wrap:anywhere}
p{max-width:70ch;color:var(--muted)}.chart{width:100%;max-width:760px;height:auto;display:block}
.grid{stroke:var(--grid);stroke-width:1}.tick{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}.axis{fill:var(--muted);font-size:12px}
.lbl{fill:var(--fg);font-size:11.5px}.fpt{fill:var(--s1);stroke:var(--bg);stroke-width:2}.dpt{fill:var(--dot);stroke:var(--bg);stroke-width:2}
.rpt{fill:var(--s2);stroke:var(--bg);stroke-width:2}.fline{fill:none;stroke:var(--s1);stroke-width:2}.rline{fill:none;stroke:var(--s2);stroke-width:2;stroke-dasharray:5 4}
.tbl{overflow-x:auto}table{border-collapse:collapse;font-size:13px;margin:8px 0 4px}th,td{padding:5px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}
th:first-child,td:first-child{text-align:left}th{color:var(--muted);font-weight:600}tr.on td:first-child{font-weight:600}.badge{color:var(--s1)}
"""


def fmt(v, kind="f"):
    if v is None:
        return "—"
    return f"${v:.5f}" if kind == "$" else (f"{v:.0%}" if kind == "%" else f"{v:.3f}")


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classifier", choices=("auto", "jev", "heuristic"), default="auto")
    parser.add_argument("--image-aware", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--n-boot", type=int, default=300)
    parser.add_argument("--min-cell-cases", type=int, default=2, help="Skip cells with fewer cases")
    args = parser.parse_args()
    mode = args.classifier if args.classifier != "auto" else ("jev" if os.environ.get("TYPESAFE_API_KEY") else "heuristic")
    args.image_aware = resolve_image_aware(args.image_aware, mode)

    cfg = load_routing_config()
    registry = ModelRegistry.load(routing=cfg)
    obs, cases = build_obs(registry)
    rel = live_reliability()
    cells = {k: v for k, v in cells_for(obs, cases).items() if len({o.case_id for o in v}) >= args.min_cell_cases}
    table = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "n_boot": args.n_boot, "score_sources": "see results/graded and benchmark/artifacts",
             "cells": {k: analyze_cell(v, n_boot=args.n_boot, reliability=rel) for k, v in sorted(cells.items())}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "frontier.json").write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")

    with (OUT / "frontier.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        metric_names = list(next(iter(table["cells"].values()))["points"].values())[0]["metrics"].keys()
        w.writerow(["cell", "n_cases", "model", *metric_names, "on_frontier", "membership", "dominated_by"])
        for cell, d in table["cells"].items():
            for m, p in d["points"].items():
                w.writerow([cell, d["n_cases"], m, *[p["metrics"][k] for k in metric_names], p["on_frontier"],
                            p["membership"], "|".join(p["dominated_by"])])

    curve = router_curve(mode, args.image_aware)
    with (OUT / "router_curve.csv").open("w", newline="", encoding="utf-8") as fh:
        keys = list(dict.fromkeys(k for r in curve for k in r))
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(curve)

    # ---------------- HTML + Markdown
    md = [f"# VLM Pareto report ({table['generated_at']})", "",
          f"Graded cases with every model: {table['cells']['all']['n_cases']}. Classifier for the router curve: {mode}. "
          f"Membership = share of {args.n_boot} case bootstraps in which the model is on the frontier.", ""]
    body = [f"<h1>VLM Pareto report</h1><p>Generated {table['generated_at']} from "
            f"{table['cells']['all']['n_cases']} cases graded for every model. A model is on the frontier when no "
            f"other model is at least as good on every objective and better on one. <b>Membership</b> is the share of "
            f"{args.n_boot} case-level bootstrap resamples in which it stays on the frontier; with few cases, read "
            f"membership rather than the single frontier. In the charts the blue line is the frontier on quality and cost alone; a blue dot away from it is on the frontier because of another objective, such as fewer unsupported claims.</p>"]
    for cell, d in table["cells"].items():
        objs = ", ".join(f"{n} ({'max' if s == 'max' else 'min'})" for n, s in d["objectives"])
        pts = d["points"]
        ranked = sorted(pts.items(), key=lambda kv: (-(kv[1]["membership"] or 0), -kv[1]["metrics"]["quality"]))
        rows_html = "".join(
            f'<tr class="{"on" if p["on_frontier"] else ""}"><td>{html.escape(m)}{" <span class=badge>●</span>" if p["on_frontier"] else ""}</td>'
            f'<td>{fmt(p["metrics"]["quality"])}</td><td>{fmt(p["metrics"]["unsupported_rate"], "%")}</td>'
            f'<td>{fmt(p["metrics"]["cost_usd"], "$")}</td><td>{fmt(p["metrics"]["latency_s"])}</td>'
            f'<td>{fmt(p["membership"], "%")}</td><td>{html.escape(", ".join(p["dominated_by"]) or "—")}</td></tr>'
            for m, p in ranked)
        chart = ""
        if cell == "all":
            chart = svg_chart("Quality vs cost, all graded cases",
                              [(m, p["metrics"]["quality"], p["metrics"]["cost_usd"], p["on_frontier"]) for m, p in pts.items()],
                              [(r["quality"], r["avg_cost_usd"]) for r in curve if r["kind"] == "router"])
        else:
            chart = svg_chart(f"Quality vs cost, {cell}",
                              [(m, p["metrics"]["quality"], p["metrics"]["cost_usd"], p["on_frontier"]) for m, p in pts.items()], [])
        body.append(f"<h2>{html.escape(cell)} <small>({d['n_cases']} cases)</small></h2><p>Objectives: {html.escape(objs)}.</p>"
                    f"{chart}<div class='tbl'><table><thead><tr><th>Model</th><th>Quality</th><th>Unsupported</th>"
                    f"<th>Cost/query</th><th>Latency s</th><th>Membership</th><th>Dominated by</th></tr></thead>"
                    f"<tbody>{rows_html}</tbody></table></div>")
        md += [f"## {cell} ({d['n_cases']} cases)", "", f"Objectives: {objs}", "",
               "| Model | Quality | Unsupported | Cost/query | Latency s | On frontier | Membership | Dominated by |",
               "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        md += [f"| {m} | {fmt(p['metrics']['quality'])} | {fmt(p['metrics']['unsupported_rate'], '%')} | "
               f"{fmt(p['metrics']['cost_usd'], '$')} | {fmt(p['metrics']['latency_s'])} | "
               f"{'yes' if p['on_frontier'] else ''} | {fmt(p['membership'], '%')} | {', '.join(p['dominated_by']) or '—'} |"
               for m, p in ranked] + [""]
    rf = quality_cost_front([(r["strategy"], r["quality"], r["avg_cost_usd"]) for r in curve
                             if r["kind"] == "router" and r["avg_cost_usd"]])
    body.append("<h2>Router curve</h2><p>Each router setting (mode × budget × Balanced weights) replayed "
                "leave-one-case-out on the recorded answers. Orange squares in the first chart; the dashed line is "
                "the router's own frontier. These settings sit on the router frontier:</p><div class='tbl'><table>"
                "<thead><tr><th>Setting</th><th>Quality</th><th>Cost/query</th></tr></thead><tbody>"
                + "".join(f"<tr><td>{html.escape(n)}</td><td>{q:.3f}</td><td>${c:.5f}</td></tr>" for n, q, c in rf)
                + "</tbody></table></div>")
    md += ["## Router frontier (mode × budget × weights, leave-one-case-out)", "", "| Setting | Quality | Cost/query |",
           "| --- | --- | --- |"] + [f"| {n} | {q:.3f} | ${c:.5f} |" for n, q, c in rf] + [""]
    (OUT / "report.md").write_text("\n".join(md), encoding="utf-8")
    (OUT / "index.html").write_text(
        f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>VLM Pareto Report</title><style>{CSS}</style></head><body><main>{''.join(body)}</main></body></html>",
        encoding="utf-8")
    print("\n".join(md[:40]))
    print(f"\nwrote {OUT.relative_to(ROOT)}/ (frontier.json, frontier.csv, router_curve.csv, report.md, index.html)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
