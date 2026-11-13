"""Post-answer risk check used to decide whether to escalate (cascade).

The dominant failure in the pilot is an *unsupported visual claim*: a model
asserts a physical change that isn't there ("a smokestack was removed").
Checking that needs the images, so:

* With TYPESAFE_API_KEY: an image-aware Jev check asks typed questions about
  the answer *with the images attached*.
* Otherwise: a labelled heuristic combining change-assertion language with
  image-pair risk (unaligned views, archival photos), plus refusal/hedge cues.

Both return a risk in [0, 1]; the pipeline escalates when risk >= threshold.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Sequence

from jev_router.image_meta import ImageSource, jev_image_block
from jev_router.schemas import QueryClassification

log = logging.getLogger(__name__)

_ASSERT_CHANGE = re.compile(
    r"\b(added|addition of|removed|removal|demolished|replaced|no longer|new(ly)? (built|added|constructed)|"
    r"has been (altered|changed|modified|rebuilt|renovated|painted)|was (altered|changed|modified|renovated)|"
    r"(present|visible) in (image )?[ab] but (not|absent)|absent in|missing in|appears in image [ab] but|"
    r"most significant (structural )?(change|difference)|structural change is)\b",
    re.I,
)
_UNVERIFIABLE = re.compile(r"\b(because|due to|in (18|19|20)\d\d|built in|renovation|restoration (work|project)|fire)\b", re.I)
_HEDGE = re.compile(r"\b(cannot (be )?(determine|establish|confirm)|not (visible|established|clear)|unclear|uncertain|may|might|appears? to)\b", re.I)
_NO_CHANGE = re.compile(r"\b(no (meaningful |visible |obvious )?(structural |physical )?(change|difference)s?|unchanged|remains? the same)\b", re.I)


@dataclass
class CheckResult:
    risk: float
    source: str                     # "jev_answer_check" | "heuristic_answer_check"
    signals: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def heuristic_check(answer: str, question: str, classification: Optional[QueryClassification] = None,
                    evidence: Optional[str] = None) -> CheckResult:
    text = answer or ""
    n_assert = len(_ASSERT_CHANGE.findall(text))
    unverifiable = [m.group(0) for m in _UNVERIFIABLE.finditer(text)
                    if m.group(0).lower() not in (question + " " + (evidence or "")).lower()]
    hedged = bool(_HEDGE.search(text))
    no_change = bool(_NO_CHANGE.search(text))

    feats = (classification.input_features or {}) if classification else {}
    pair = feats.get("pair") or {}
    risky_pair = pair.get("viewpoint_alignment") == "unaligned" or feats.get("any_monochrome", False)
    p_hard = classification.difficulty.probabilities.get("hard", 0.0) if classification else 0.0
    tools_needed = classification.needs_external_tools if classification else 0.0

    risk = 0.1
    risk += min(n_assert, 3) * 0.15
    risk += 0.2 if unverifiable else 0.0
    risk += 0.15 if (risky_pair and n_assert) else 0.0
    risk += 0.1 * p_hard
    risk += 0.2 if (tools_needed >= 0.5 and not hedged) else 0.0
    risk -= 0.15 if (no_change and not n_assert) else 0.0
    risk -= 0.05 if hedged else 0.0
    return CheckResult(
        risk=round(max(0.0, min(1.0, risk)), 3),
        source="heuristic_answer_check",
        signals={"change_assertions": n_assert, "unverifiable_terms": unverifiable,
                 "hedged": hedged, "states_no_change": no_change, "risky_pair": risky_pair,
                 "p_hard": round(p_hard, 3), "tools_needed": round(tools_needed, 3)},
    )


def _jev_questions() -> dict[str, Any]:
    from langchain_typesafe import Noul

    return {
        "unsupported_visual_claim": Noul(instructions=(
            "Looking at the attached images, does the answer claim a physical or structural feature or change "
            "that is not clearly visible in them?")),
        "unsupported_fact": Noul(instructions=(
            "Does the answer state dates, causes, names or other facts that are neither visible in the images "
            "nor given in the question or supplied evidence?")),
        "answers_question": Noul(instructions="Does the answer actually address the question that was asked?"),
    }


class AnswerChecker:
    def __init__(self, decision_model: Any = None, use_jev: bool = True, timeout: float = 20.0):
        self._model = decision_model
        self._use_jev = use_jev
        self._timeout = timeout
        self._init_error: Optional[str] = None

    def _get(self) -> Any:
        import os

        if self._model is None and self._use_jev and self._init_error is None:
            if not os.environ.get("TYPESAFE_API_KEY"):
                self._init_error = "TYPESAFE_API_KEY is not set"
            else:
                try:
                    from jev_router.classifier import typesafe_classifier

                    self._model = typesafe_classifier(timeout=self._timeout)
                except Exception as exc:
                    self._init_error = f"{type(exc).__name__}: {exc}"
        return self._model

    def check(self, answer: str, question: str, images: Sequence[ImageSource] = (),
              classification: Optional[QueryClassification] = None, evidence: Optional[str] = None) -> CheckResult:
        model = self._get()
        if model is None:
            res = heuristic_check(answer, question, classification, evidence)
            res.error = self._init_error
            return res
        state = {"question": question, "answer": answer, "supplied_evidence": evidence,
                 "images": [jev_image_block(i) for i in images]}
        try:
            r = model.invoke({"state": state, "questions": _jev_questions()})
            n = r.nouls
            visual, fact, answers = (float(n[k].noul) for k in ("unsupported_visual_claim", "unsupported_fact", "answers_question"))
            risk = max(visual, fact, 1.0 - answers)
            return CheckResult(round(risk, 3), "jev_answer_check",
                               {"unsupported_visual_claim": visual, "unsupported_fact": fact, "answers_question": answers,
                                "request_id": getattr(r, "request_id", None)})
        except Exception as exc:
            log.warning("Jev answer check failed; using heuristic: %s", exc)
            res = heuristic_check(answer, question, classification, evidence)
            res.error = f"{type(exc).__name__}: {exc}"
            return res
