"""Streamlit demo: Jev-based adaptive VLM routing for MapEvolve.

    streamlit run app.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


from jev_router.dataset import ROOT, load_all  # noqa: E402
from jev_router.offline import deep_merge  # noqa: E402
from jev_router.pipeline import AdaptiveRouter  # noqa: E402
from jev_router.registry import load_routing_config  # noqa: E402
from jev_router.schemas import MODES  # noqa: E402

FEEDBACK = ROOT / "runs" / "feedback" / "feedback.jsonl"

st.set_page_config(page_title="MapEvolve · Jev Router", layout="wide")


FRONTIER = ROOT / "results" / "pareto" / "frontier.json"


@st.cache_resource
def get_router(use_learned: bool, prune: bool) -> AdaptiveRouter:
    cfg = deep_merge(load_routing_config(), {"pareto": {"prune_dominated": prune}})
    return AdaptiveRouter(config=cfg, use_quality_model=True if use_learned else None)


@st.cache_data
def demo_cases() -> list[dict]:
    return [c.__dict__ for c in load_all()]


keys_present = {k: bool(os.environ.get(k)) for k in ("TYPESAFE_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY")}

st.title("MapEvolve · Jev adaptive VLM routing")
st.caption("Jev classifies the query, a deterministic policy picks an action and a VLM, and the answer is "
           "checked. Model scores are preliminary (6 graded pilot cases).")

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Preference", MODES, index=1, format_func=str.capitalize)
    budget = st.number_input("Budget per query (USD)", min_value=0.0001, max_value=1.0, value=0.05,
                             step=0.005, format="%.4f")
    image_aware = st.checkbox("Send images to Jev (experimental)", value=False)
    cascade = st.checkbox("Check answer and escalate once if risky", value=False,
                          help="Off by default: with the heuristic checker it hurt on the pilot.")
    use_learned = st.checkbox("Force learned quality predictor", value=False,
                              help="Normally used only once it has >= 30 graded cases.")
    prune = st.checkbox("Skip Pareto-dominated models", value=False,
                        help="Uses results/pareto/frontier.json; only cells with >= 5 graded cases are used.")
    compare = st.checkbox("Compare cheapest / strongest / adaptive", value=False)
    execute = st.checkbox("Call the VLM", value=keys_present["OPENROUTER_API_KEY"] or keys_present["OPENAI_API_KEY"])
    st.divider()
    st.subheader("API keys")
    for name, ok in keys_present.items():
        st.write(("✅ " if ok else "⚠️ ") + name)
    if not keys_present["TYPESAFE_API_KEY"]:
        st.info("No TypeSafe key: classification and answer checks use the labelled heuristic fallback.")

router = get_router(use_learned, prune)
st.sidebar.caption(f"Quality predictor: {router.policy.quality_model_status}")

source = st.radio("Input", ["Demo case", "Upload"], horizontal=True)
images, question, evidence, case = [], "", "", None
if source == "Demo case":
    cases = demo_cases()
    labels = {f"{c['case_id']} · {c['query_type']} · {c['difficulty']}": c for c in cases}
    case = labels[st.selectbox("Case", list(labels))]
    images = [ROOT / p for p in case["images"]]
    cols = st.columns(len(images))
    for col, p, lab in zip(cols, images, ["Image A", "Image B"] if len(images) == 2 else ["Image"]):
        col.image(str(p), caption=lab, width="stretch")
    question = st.text_area("Visual question", value=case["question"])
    evidence = st.text_input("Supplied evidence (optional)", value=case.get("supplied_evidence") or "")
    with st.expander("Reference answer (label data — never sent to Jev or the router)"):
        st.write(case["reference"]["answer"])
        st.caption(f"Expected action: {case['expected_action']} · labels: {case['labels'].get('author')}, "
                   f"reviewed={case['labels'].get('reviewed')}")
else:
    col_a, col_b = st.columns(2)
    with col_a:
        file_a = st.file_uploader("Image A", type=["jpg", "jpeg", "png", "webp"], key="a")
        if file_a:
            st.image(file_a, width="stretch")
    with col_b:
        file_b = st.file_uploader("Image B (optional)", type=["jpg", "jpeg", "png", "webp"], key="b")
        if file_b:
            st.image(file_b, width="stretch")
    images = [f.getvalue() for f in (file_a, file_b) if f]
    question = st.text_area("Visual question", value="What changed between these two images?")
    evidence = st.text_input("Supplied evidence (optional, e.g. coordinates)", value="")

if st.button("Submit", type="primary", disabled=not (images and question.strip())):
    with st.spinner("Classifying, routing and answering…"):
        try:
            st.session_state["trace"] = router.answer(
                question.strip(), images, mode=mode, budget_usd=budget, image_aware=image_aware,
                execute=execute, compare=compare, evidence=evidence.strip() or None, cascade=cascade)
        except Exception as exc:  # surface, never crash the app
            st.error(f"Request failed: {type(exc).__name__}: {exc}")
            st.stop()

trace = st.session_state.get("trace")
if trace:
    cls, action, routing, result = trace["classification"], trace["action"], trace["routing"], trace["result"]

    st.subheader("1 · Jev classification")
    st.caption(f"source: **{cls['source']}**" + (f" · fallback reason: {cls['error']}" if cls.get("error") else ""))
    c1, c2, c3 = st.columns(3)
    for col, key in zip((c1, c2, c3), ("query_type", "difficulty", "visual_expertise")):
        dist = cls[key]
        col.metric(key.replace("_", " ").title(), dist["label"],
                   help=f"confidence (distribution concentration): {dist['confidence']}")
        col.bar_chart(pd.Series(dist["probabilities"]).sort_values(ascending=False))
    flags = [{"flag": k, "probability": round(cls[k], 3), "triggered": v} for k, v in cls["flags"].items()]
    st.table(pd.DataFrame(flags))
    pair = (cls.get("input_features") or {}).get("pair")
    if pair:
        st.caption(f"Pair features: alignment **{pair['viewpoint_alignment']}** "
                   f"({pair['keypoints']['inliers']} keypoint inliers), histogram similarity "
                   f"{pair['histogram_similarity']}, brightness gap {pair['brightness_gap']}, "
                   f"monochrome {pair['any_monochrome']}")

    st.subheader("2 · Action")
    st.info(f"**{action['action']}**" + ("" if action["implemented"] else " (component not implemented yet)"))
    for r in action["reasons"]:
        st.write("• " + r)
    if action.get("tool_result"):
        st.code(action["tool_result"], language=None)

    st.subheader("3 · Routing decision")
    if routing["selected_key"]:
        st.success(f"Selected **{routing['selected_key']}** · `{routing['selected_model_id']}` · "
                   f"fallbacks: {routing['fallback_keys']}")
    else:
        st.error("No eligible model.")
    for line in routing["rationale"]:
        st.write("• " + line)
    for w in routing["warnings"]:
        st.warning(w)
    with st.expander("All candidates"):
        st.dataframe(pd.DataFrame(routing["candidates"]).drop(columns=["score_sources"]), width="stretch")

    st.subheader("4 · Answer")
    if not execute:
        st.info("VLM call disabled; classification, action and routing only.")
    elif result is None:
        st.error("Every model in the failover chain failed.")
        st.json(trace["attempts"])
    else:
        st.write(result["answer"])
        s = result.get("structured") or {}
        if s.get("evidence"):
            st.markdown("**Evidence:** " + "; ".join(map(str, s["evidence"])))
        if s.get("cannot_determine"):
            st.markdown("**Cannot determine:** " + "; ".join(map(str, s["cannot_determine"])))
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Model", result["key"])
        m2.metric("Latency", f"{result['latency_seconds']:.1f}s")
        m3.metric("Tokens (in/out)", f"{result['input_tokens']}/{result['output_tokens']}")
        m4.metric("Cost", f"${result['cost_usd']:.5f}" if result["cost_usd"] is not None else "n/a",
                  help=result.get("cost_source") or "")
        chk = trace.get("answer_check")
        if chk:
            st.caption(f"Answer check ({chk['source']}): risk {chk['risk']}")
        if trace.get("cascade"):
            cz = trace["cascade"]
            st.warning(f"Escalated: {cz['first_key']} (risk {cz['first_risk']}) → {cz.get('escalated_key')} "
                       f"(risk {cz.get('escalated_risk')}); kept {cz.get('kept', cz['first_key'])}.")

    if trace.get("comparison"):
        st.subheader("5 · Comparison")
        rows = []
        for row in trace["comparison"]:
            r = row["result"] or {}
            rows.append({"strategy": row["strategy"], "model": row["key"], "status": r.get("status"),
                         "latency_s": r.get("latency_seconds"), "in_tok": r.get("input_tokens"),
                         "out_tok": r.get("output_tokens"), "cost_usd": r.get("cost_usd"),
                         "answer": r.get("answer") or r.get("error")})
        st.dataframe(pd.DataFrame(rows), width="stretch")

    st.subheader("Feedback")
    fb_note = st.text_input("Optional note", key=f"note-{trace['trace_id']}")
    f1, f2, _ = st.columns([1, 1, 6])
    for col, label, value in ((f1, "👍 Good", 1), (f2, "👎 Bad", -1)):
        if col.button(label, key=f"fb{value}-{trace['trace_id']}"):
            FEEDBACK.parent.mkdir(parents=True, exist_ok=True)
            with FEEDBACK.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"trace_id": trace["trace_id"], "rating": value, "note": fb_note,
                                     "model": (result or {}).get("key"), "case_id": (case or {}).get("case_id"),
                                     "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}) + "\n")
            st.success("Saved with the trace ID; it can be imported as a human grade later.")
    st.caption(f"Trace {trace['trace_id']} saved under runs/routing_traces/.")


st.divider()
with st.expander("Model frontier (Pareto)", expanded=False):
    if not FRONTIER.exists():
        st.info("No frontier yet. Run `python -m scripts.pareto_report`.")
    else:
        from scripts.pareto_report import svg_chart

        table = json.loads(FRONTIER.read_text(encoding="utf-8"))
        cell = st.selectbox("Cell", list(table["cells"]), key="pareto-cell")
        d = table["cells"][cell]
        st.caption(f"{d['n_cases']} graded cases · objectives: "
                   + ", ".join(f"{n} ({s})" for n, s in d["objectives"])
                   + f" · generated {table['generated_at']}")
        pts = d["points"]
        theme = "dark" if (st.context.theme.type or "light") == "dark" else "light"
        svg = svg_chart(cell, [(m, p["metrics"]["quality"], p["metrics"]["cost_usd"], p["on_frontier"])
                               for m, p in pts.items()], [], theme=theme)
        st.image(svg, width="stretch")
        st.caption("Blue dots: on the frontier across all objectives. Line: frontier on quality and cost alone. "
                   "Grey: dominated.")
        st.dataframe(pd.DataFrame([{"model": m, "on frontier": p["on_frontier"], "membership": p["membership"],
                                    **{k: v for k, v in p["metrics"].items() if v is not None},
                                    "dominated by": ", ".join(p["dominated_by"])} for m, p in pts.items()])
                     .sort_values(["on frontier", "membership"], ascending=False), width="stretch")
