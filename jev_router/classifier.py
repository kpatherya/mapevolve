"""Query classification with TypeSafe Jev, plus a deterministic fallback.

All six routing-relevant questions are asked in a single
`TypeSafeClassifier.invoke` call. The full answer (probabilities, confidence,
usage, request ID) is preserved on the returned `QueryClassification`.

Inputs are limited to the user's question text and ground-truth-free metadata
(image count and pixel/EXIF features from `image_meta`). Images themselves are
only sent when `image_aware=True` (experimental).
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, Optional, Protocol, Sequence

from jev_router.image_meta import ImageSource, describe_image, jev_image_block
from jev_router.pair_features import pair_features
from jev_router.schemas import (
    DIFFICULTIES,
    QUERY_TYPES,
    VISUAL_EXPERTISE,
    LabelDistribution,
    QueryClassification,
)

log = logging.getLogger(__name__)

# Fields that must never reach the classifier. Checked defensively so that a
# benchmark case dict passed by mistake cannot leak answer keys.
FORBIDDEN_STATE_KEYS = frozenset(
    {
        "ground_truth",
        "ground_truth_raw",
        "rubric",
        "reference",
        "reference_answer",
        "expected_decision",
        "expected_same_building",
        "answerable",
        "difficulty",
        "case_id",
        "building",
    }
)

QUERY_TYPE_CRITERIA = {
    "landmark_identification": "Name or identify a building or landmark shown in an image.",
    "cross_view_matching": (
        "Decide whether two images show the same building or place from different "
        "viewpoints, distances, or interior versus exterior."
    ),
    "temporal_comparison": (
        "Compare images of the same place taken at different times and describe what "
        "changed or stayed the same."
    ),
    "spatial_reasoning": (
        "Reason about locations, distances, directions, or layout between places."
    ),
    "factual_retrieval": (
        "Ask for facts not visible in pixels, such as history, funding, dates, names, "
        "or records."
    ),
}

DIFFICULTY_CRITERIA = [
    "easy: near-identical viewpoint and conditions; the answer is directly visible.",
    "moderate: some viewpoint, seasonal, operational, or lighting differences must be "
    "separated from real change.",
    "hard: large time gaps, archival or monochrome photos, strong viewpoint change, "
    "subtle restoration, or the answer needs evidence beyond the images.",
]

EXPERTISE_CRITERIA = {
    "general_scene_understanding": "Ordinary scene description; no special skill.",
    "architectural_detail": "Facades, windows, materials, structural elements.",
    "historical_photo_interpretation": "Archival, sepia, or decades-apart photographs.",
    "fine_grained_change_detection": "Subtle differences between closely aligned views.",
    "spatial_geometry": "Viewpoint, alignment, distance, or map-like reasoning.",
}


class DecisionModel(Protocol):
    def invoke(self, input: dict[str, Any], config: Any = None) -> Any: ...


def build_questions() -> dict[str, Any]:
    """The typed Jev questions, built lazily so tests can run without the SDK."""
    from langchain_typesafe import Choice, Noul, NoulCriteria, Score

    return {
        "query_type": Choice(
            instructions="What kind of visual question is the user asking?",
            criteria=QUERY_TYPE_CRITERIA,
        ),
        "difficulty": Score(
            instructions=(
                "How difficult is this query for a vision-language model to answer "
                "correctly without inventing unsupported claims?"
            ),
            criteria=DIFFICULTY_CRITERIA,
        ),
        "visual_expertise": Choice(
            instructions="Which visual expertise is most required to answer well?",
            criteria=EXPERTISE_CRITERIA,
        ),
        "needs_image_retrieval": Noul(
            instructions=(
                "Does answering require retrieving additional images that the user "
                "did not supply (for example a historical or reference view)?"
            ),
            criteria=NoulCriteria(
                true="The supplied images are insufficient; another image must be found.",
                false="The supplied images are sufficient to attempt an answer.",
            ),
        ),
        "needs_pair_verification": Noul(
            instructions=(
                "Must the system first verify that two images show the same place and a "
                "comparable viewpoint before answering?"
            ),
        ),
        "needs_external_tools": Noul(
            instructions=(
                "Does the question need external records, coordinates, maps, search, "
                "or calculation tools beyond looking at the images and supplied evidence?"
            ),
        ),
        "answerable_from_inputs": Noul(
            instructions=(
                "Can the question be answered from the supplied images and any supplied "
                "evidence alone, without outside knowledge or records?"
            ),
            criteria=NoulCriteria(
                true="Everything needed is visible in the images or stated in the supplied evidence.",
                false="A correct answer needs facts that are neither visible nor supplied.",
            ),
        ),
    }


def build_state(
    question: str,
    images: Sequence[ImageSource] = (),
    metadata: Optional[dict[str, Any]] = None,
    image_aware: bool = False,
    evidence: Optional[str] = None,
    with_pair_features: bool = True,
) -> dict[str, Any]:
    """Build the Jev state from user inputs and answer-free derived features only.

    `evidence` is text the user supplied alongside the images (e.g. coordinates);
    it is a user input, not a label.
    """
    metadata = dict(metadata or {})
    leaked = FORBIDDEN_STATE_KEYS & set(metadata)
    if leaked:
        raise ValueError(f"Refusing to send answer-bearing metadata to Jev: {sorted(leaked)}")
    image_info = [describe_image(img) for img in images]
    state: dict[str, Any] = {
        "question": question,
        "num_images": len(images),
        "image_metadata": image_info,
    }
    if evidence:
        state["supplied_evidence"] = evidence
    if with_pair_features and len(images) == 2:
        state["pair_features"] = pair_features(images)
    if metadata:
        state["query_metadata"] = metadata
    if image_aware and images:
        state["images"] = [jev_image_block(img) for img in images]
    return state


def _distribution_from_choice(answer: Any) -> LabelDistribution:
    return LabelDistribution(
        label=answer.choice,
        probabilities={k: float(v) for k, v in answer.probabilities.items()},
        confidence=float(answer.confidence),
    )


def _distribution_from_score(answer: Any, labels: Sequence[str]) -> LabelDistribution:
    """Map a Jev Score (index-keyed) onto named difficulty labels."""
    probs = {labels[int(i)]: float(p) for i, p in answer.probabilities.items()}
    for label in labels:
        probs.setdefault(label, 0.0)
    label = max(probs, key=probs.get)
    return LabelDistribution(label=label, probabilities=probs, confidence=float(answer.confidence))


def parse_response(response: Any, *, image_aware: bool, latency: float) -> QueryClassification:
    """Convert a TypeSafe `ClassifierResponse` into a `QueryClassification`."""
    choices = response.choices
    scores = response.scores
    nouls = response.nouls
    usage = response.usage
    if usage is not None and not isinstance(usage, dict):
        usage = dict(usage)
    raw = response.model_dump(mode="json") if hasattr(response, "model_dump") else None
    return QueryClassification(
        query_type=_distribution_from_choice(choices["query_type"]),
        difficulty=_distribution_from_score(scores["difficulty"], DIFFICULTIES),
        visual_expertise=_distribution_from_choice(choices["visual_expertise"]),
        needs_image_retrieval=float(nouls["needs_image_retrieval"].noul),
        needs_pair_verification=float(nouls["needs_pair_verification"].noul),
        needs_external_tools=float(nouls["needs_external_tools"].noul),
        answerable_from_inputs=float(nouls["answerable_from_inputs"].noul) if "answerable_from_inputs" in nouls else 1.0,
        source="jev",
        jev_model=getattr(response, "model", None),
        request_id=getattr(response, "request_id", None),
        usage=usage,
        latency_seconds=round(latency, 3),
        image_aware=image_aware,
        raw=raw,
    )


# --------------------------------------------------------------------------- #
# Deterministic fallback used when Jev is unavailable or fails.
# --------------------------------------------------------------------------- #

_PATTERNS = {
    # Order matters: explicit change wording wins over "same building" phrasing.
    "temporal_comparison": r"\b(what changed|has changed|changed|changes|then and now|over time)\b",
    # Cross-view wording often mentions "distance" or "viewpoint", so it must
    # be checked before the spatial rule (found by scripts.evaluate_classifier).
    "cross_view_matching": r"\b(same building|same place|same location|same landmark|match|different viewpoint|inside|interior|exterior)\b",
    "factual_visible": r"\b(inscription|written|what does the sign|sign say|what does .* say|read the)\b",
    "factual_retrieval": r"\b(who|funded|founded|built in|when was|what year|which year|history|named after|architect|designed|why)\b",
    "spatial_reasoning": r"\b(closer|closest|distance|how far|far apart|farther|further|nearer|north|south|east|west|left|right|route|map)\b",
    "temporal_comparison_weak": r"\b(differences?|before|after|compare)\b",
    "landmark_identification": r"\b(what building|which building|identify|name (it|the)|what is this|landmark)\b",
}
_COORDS = re.compile(r"\d{1,3}\.\d+\s*[NS]\b.*\d{1,3}\.\d+\s*[EW]\b", re.I | re.S)
_WITHIN_IMAGE = re.compile(r"\b(left|right|above|below|in front of|behind)\b", re.I)


def _soft(label: str, labels: Sequence[str], peak: float) -> dict[str, float]:
    rest = (1.0 - peak) / (len(labels) - 1)
    return {lab: (peak if lab == label else rest) for lab in labels}


def heuristic_classify(
    question: str,
    images: Sequence[ImageSource] = (),
    image_info: Optional[list[dict[str, Any]]] = None,
    error: Optional[str] = None,
    pair: Optional[dict[str, Any]] = None,
    evidence: Optional[str] = None,
) -> QueryClassification:
    """Keyword + image-feature rules. Deliberately simple and clearly labelled."""
    text = question.lower()
    n_images = len(images) if images else len(image_info or [])
    info = image_info if image_info is not None else [describe_image(i) for i in images]
    if pair is None and images and len(images) == 2:
        pair = pair_features(images)

    hits = [t for t, pat in _PATTERNS.items() if re.search(pat, text)]
    raw_type = hits[0] if hits else ("temporal_comparison" if n_images >= 2 else "landmark_identification")
    query_type = {"factual_visible": "factual_retrieval", "temporal_comparison_weak": "temporal_comparison"}.get(raw_type, raw_type)

    monochrome = any(i.get("likely_monochrome") for i in info)
    alignment = (pair or {}).get("viewpoint_alignment")
    has_coords = bool(evidence and _COORDS.search(evidence))
    tools, answerable, verify, retrieval = 0.1, 0.85, 0.1, 0.1

    if raw_type == "factual_visible":
        difficulty = "easy"
    elif query_type == "factual_retrieval":
        difficulty, tools, answerable = "hard", 0.85, 0.2
    elif query_type == "spatial_reasoning":
        if has_coords:
            difficulty, tools, answerable = "moderate", 0.6, 0.85
        elif n_images == 1 and _WITHIN_IMAGE.search(text):
            difficulty = "easy"
        else:
            difficulty, tools, answerable = "hard", 0.85, 0.15
    elif n_images == 2:
        if monochrome:
            difficulty = "hard"
        elif alignment == "aligned":
            difficulty = "easy"
        elif alignment == "partially_aligned":
            difficulty = "moderate"
        else:
            difficulty = "hard" if query_type == "cross_view_matching" else "moderate"
        verify = 0.8 if query_type == "cross_view_matching" else (0.2 if alignment == "aligned" else 0.6)
    else:  # single-image identification
        difficulty = "hard" if monochrome else "moderate"
        retrieval = 0.7 if query_type == "temporal_comparison" else 0.1

    expertise = {
        "temporal_comparison": "historical_photo_interpretation" if monochrome else "fine_grained_change_detection",
        "cross_view_matching": "architectural_detail",
        "spatial_reasoning": "spatial_geometry",
        "landmark_identification": "historical_photo_interpretation" if monochrome else "architectural_detail",
        "factual_retrieval": "general_scene_understanding",
    }[query_type]

    return QueryClassification(
        query_type=LabelDistribution(query_type, _soft(query_type, QUERY_TYPES, 0.6)),
        difficulty=LabelDistribution(difficulty, _soft(difficulty, DIFFICULTIES, 0.5)),
        visual_expertise=LabelDistribution(expertise, _soft(expertise, VISUAL_EXPERTISE, 0.5)),
        needs_image_retrieval=retrieval,
        needs_pair_verification=verify,
        needs_external_tools=tools,
        answerable_from_inputs=answerable,
        source="heuristic_fallback",
        error=error,
    )


class JevQueryClassifier:
    """Classify a visual query with Jev; fall back to heuristics on any failure."""

    def __init__(
        self,
        decision_model: Optional[DecisionModel] = None,
        model: str = "jev-latest",
        timeout: float = 20.0,
        image_aware: bool = False,
        use_jev: bool = True,
    ) -> None:
        self._model_name = model
        self._timeout = timeout
        self.image_aware = image_aware
        self._decision_model = decision_model
        self._init_error: Optional[str] = None if use_jev else "heuristic requested"

    def _get_model(self) -> Optional[DecisionModel]:
        if self._decision_model is None and self._init_error is None:
            if not os.environ.get("TYPESAFE_API_KEY"):
                self._init_error = "TYPESAFE_API_KEY is not set"
                return None
            try:
                self._decision_model = typesafe_classifier(model=self._model_name, timeout=self._timeout)
            except Exception as exc:  # missing key, missing package
                self._init_error = f"{type(exc).__name__}: {exc}"
        return self._decision_model

    def classify(
        self,
        question: str,
        images: Sequence[ImageSource] = (),
        metadata: Optional[dict[str, Any]] = None,
        image_aware: Optional[bool] = None,
        evidence: Optional[str] = None,
    ) -> QueryClassification:
        use_images = self.image_aware if image_aware is None else image_aware
        state = build_state(question, images, metadata, image_aware=use_images, evidence=evidence)
        fallback = dict(image_info=state["image_metadata"], pair=state.get("pair_features"), evidence=evidence)
        features = input_features(state)
        model = self._get_model()
        if model is None:
            result = heuristic_classify(question, error=self._init_error, **fallback)
        else:
            started = time.perf_counter()
            try:
                response = model.invoke({"state": state, "questions": build_questions()})
                result = parse_response(response, image_aware=use_images, latency=time.perf_counter() - started)
            except Exception as exc:
                log.warning("Jev classification failed; using heuristic fallback: %s", exc)
                result = heuristic_classify(question, error=f"{type(exc).__name__}: {exc}", **fallback)
        result.input_features = features
        return result


# TypeSafe rejects image inputs (400 api_usage_error) unless this opt-in header is sent.
EXPERIMENTAL_IMAGES_HEADER = {"TypeSafe-Danger-Zone-Images": "true"}


def typesafe_classifier(**kwargs: Any) -> Any:
    """A TypeSafeClassifier whose client opts in to TypeSafe's experimental image inputs.
    The header is harmless on text-only calls, so one client serves both modes."""
    import httpx2
    from langchain_typesafe import TypeSafeClassifier

    timeout = kwargs.get("timeout")
    return TypeSafeClassifier(client=httpx2.Client(headers=EXPERIMENTAL_IMAGES_HEADER, timeout=timeout), **kwargs)


def input_features(state: dict[str, Any]) -> dict[str, Any]:
    """The answer-free derived features kept alongside a classification."""
    return {
        "n_images": state["num_images"],
        "has_evidence": bool(state.get("supplied_evidence")),
        "pair": state.get("pair_features"),
        "any_monochrome": any(m.get("likely_monochrome") for m in state["image_metadata"]),
    }
