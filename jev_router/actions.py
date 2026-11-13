"""Choose *what to do* before choosing a model.

Actions (same vocabulary as the manifest's expected_action):
  answer               plain VLM answer
  verify_then_answer   run the lightweight pair check, pass its result to the VLM
  retrieve_then_answer retrieval is not implemented: answer with an explicit limitation
  use_tools            run an available tool (coordinate calculator) or answer
                       only the visible part and name the record/tool needed
  decline              answer is not obtainable from the inputs or any tool: no VLM call

The decision is a deterministic function of Jev's flags; thresholds live in
config/routing.yaml under `actions`.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from jev_router.schemas import QueryClassification

_COORD = re.compile(r"(-?\d{1,3}(?:\.\d+)?)\s*°?\s*([NS])[,;\s]+(-?\d{1,3}(?:\.\d+)?)\s*°?\s*([EW])", re.I)
_LABEL = re.compile(r"(Image\s+[AB]|Landmark\s+in\s+Image\s+[AB])", re.I)


@dataclass
class ActionDecision:
    action: str
    reasons: list[str]
    tool_result: Optional[str] = None
    pair_check: Optional[dict[str, Any]] = None
    prompt_note: Optional[str] = None
    implemented: bool = True
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_coordinates(text: str) -> list[tuple[str, float, float]]:
    """Find 'label: lat N/S, lon E/W' pairs in supplied evidence."""
    out = []
    for m in _COORD.finditer(text or ""):
        lat = float(m.group(1)) * (-1 if m.group(2).upper() == "S" else 1)
        lon = float(m.group(3)) * (-1 if m.group(4).upper() == "W" else 1)
        before = text[: m.start()]
        labels = _LABEL.findall(before)
        label = labels[-1].split()[-2] + " " + labels[-1].split()[-1] if labels else f"point {len(out) + 1}"
        out.append((label, lat, lon))
    return out


def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    r = 6_371_000.0
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp, dl = p2 - p1, math.radians(b[1] - a[1])
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def coordinate_tool(evidence: Optional[str]) -> Optional[str]:
    pts = parse_coordinates(evidence or "")
    if len(pts) < 2:
        return None
    (la, a_lat, a_lon), (lb, b_lat, b_lon) = pts[:2]
    d = haversine_m((a_lat, a_lon), (b_lat, b_lon))
    north = la if a_lat > b_lat else lb
    east = la if a_lon > b_lon else lb
    dist = f"{d:,.0f} m" if d < 10_000 else f"{d / 1000:,.1f} km"
    return (f"Coordinate tool (computed from the supplied evidence): {la} at {a_lat:.4f}, {a_lon:.4f}; "
            f"{lb} at {b_lat:.4f}, {b_lon:.4f}. Great-circle distance {dist}. "
            f"Farther north: {north}. Farther east: {east}.")


def action_thresholds(cfg: dict[str, Any], source: Optional[str] = None) -> dict[str, Any]:
    """Thresholds for one classifier source. `actions.by_source.<source>` overrides the
    defaults, because Jev and the heuristic put their probabilities on different scales."""
    t = cfg.get("actions", {})
    flag = t.get("flag_threshold", 0.5)
    out = {"decline_if_answerable_below": t.get("decline_if_answerable_below", 0.2),
           "tools_threshold": t.get("tools_threshold", flag), "retrieval_threshold": t.get("retrieval_threshold", flag),
           "verify_threshold": t.get("verify_threshold", flag), "verify_signal": t.get("verify_signal", "flag"),
           "verify_types": list(t.get("verify_types", ["cross_view_matching"]))}
    out.update((t.get("by_source") or {}).get(source or "", {}) or {})
    return out


def verify_probability(c: QueryClassification, th: dict[str, Any]) -> tuple[float, str]:
    """The probability used for the pair-verification decision, and how it was obtained."""
    if th["verify_signal"] == "query_type":
        p = sum(c.query_type.probabilities.get(t, 0.0) for t in th["verify_types"])
        return p, f"P(query type in {th['verify_types']})"
    return c.needs_pair_verification, "P(needs pair verification)"


def decide_action(c: QueryClassification, cfg: dict[str, Any], evidence: Optional[str] = None) -> ActionDecision:
    th = action_thresholds(cfg, c.source)
    decline_below = th["decline_if_answerable_below"]
    n_images = (c.input_features or {}).get("n_images", 0)
    reasons: list[str] = []

    if c.answerable_from_inputs < decline_below and c.needs_external_tools < th["tools_threshold"]:
        reasons.append(f"P(answerable from inputs)={c.answerable_from_inputs:.2f} < {decline_below} and no tool would help.")
        return ActionDecision("decline", reasons, prompt_note=None)

    if c.needs_external_tools >= th["tools_threshold"]:
        reasons.append(f"P(needs external tools)={c.needs_external_tools:.2f} >= {th['tools_threshold']}.")
        result = coordinate_tool(evidence)
        if result:
            reasons.append("Coordinates were supplied; ran the coordinate calculator.")
            return ActionDecision("use_tools", reasons, tool_result=result,
                                  prompt_note=result + " Use this tool result for any location or distance claim.")
        reasons.append("No implemented tool applies (records/search/geocoding are future components).")
        return ActionDecision(
            "use_tools", reasons, implemented=False,
            prompt_note=("The required records or tools are not available. Answer only what is directly visible, "
                         "explicitly decline any part that needs outside facts, and name the record or tool needed."))

    if c.needs_image_retrieval >= th["retrieval_threshold"]:
        reasons.append(f"P(needs image retrieval)={c.needs_image_retrieval:.2f} >= {th['retrieval_threshold']}; "
                       "retrieval not implemented.")
        return ActionDecision("retrieve_then_answer", reasons, implemented=False,
                              prompt_note="A comparable reference image could not be retrieved; state this limitation.")

    p_verify, verify_name = verify_probability(c, th)
    if n_images == 2 and p_verify >= th["verify_threshold"]:
        pair = (c.input_features or {}).get("pair") or {}
        align = pair.get("viewpoint_alignment", "unknown")
        inl = (pair.get("keypoints") or {}).get("inliers")
        reasons.append(f"{verify_name}={p_verify:.2f} >= {th['verify_threshold']}; "
                       f"lightweight geometric check: {align} ({inl} keypoint inliers).")
        note = {
            "aligned": "A geometric check found the two views closely aligned, so direct comparison is reasonable.",
            "partially_aligned": "A geometric check found the views only partly aligned; compare only regions visible in both.",
        }.get(align, "A geometric check could not align the two views; do not treat viewpoint or framing differences "
                     "as physical change, and say which parts cannot be compared.")
        return ActionDecision("verify_then_answer", reasons, pair_check=pair or None, prompt_note=note)

    reasons.append("Answerable directly from the supplied inputs.")
    return ActionDecision("answer", reasons)


def decline_message(c: QueryClassification) -> str:
    return ("I can't answer this from the supplied images and evidence: it needs information that is neither visible "
            "nor provided. Supplying the relevant record, coordinates, or a comparable reference image would make it answerable.")
