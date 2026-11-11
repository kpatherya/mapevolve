"""Learned per-model quality predictor with bootstrap uncertainty.

Replaces the registry's shrunk lookup table when enough graded data exists.

    score_hat(model, query) = w · phi(query features, model features)

* Query features come only from routing inputs: classifier probabilities
  (query type, difficulty), pair features, monochrome flag, evidence flag.
* Model features are a model one-hot plus price attributes (log input and
  output price), so an unseen model still gets a prediction from attributes.
* Interactions (model x difficulty, model x pair features) let the best model
  differ by query instead of being one global ranking.

Fitting is closed-form ridge regression on score / 4 and a ridge-penalized
logistic regression for unsupported-claim risk. Uncertainty comes from a
case-level bootstrap: resampling whole cases (not rows) so the eight answers to
one case don't count as eight independent samples.

Pure numpy; coefficients are stored as JSON (no pickles).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from jev_router.schemas import DIFFICULTIES, QUERY_TYPES, QueryClassification

PAIR_KEYS = ("hist_sim", "dhash_sim", "log_inliers", "inlier_ratio", "brightness_gap")
INTERACT = ("p_easy", "p_moderate", "p_hard", "monochrome", "log_inliers")


@dataclass
class TrainingRow:
    case_id: str
    model_key: str
    score: float                 # 0..4
    unsupported: bool
    classification: QueryClassification


def query_features(c: QueryClassification) -> dict[str, float]:
    f: dict[str, float] = {}
    for t in QUERY_TYPES:
        f[f"qt_{t}"] = float(c.query_type.probabilities.get(t, 0.0))
    for d in DIFFICULTIES:
        f[f"p_{d}"] = float(c.difficulty.probabilities.get(d, 0.0))
    feats = c.input_features or {}
    pair = feats.get("pair") or {}
    kp = pair.get("keypoints") or {}
    f["two_images"] = float(feats.get("n_images", 1) == 2)
    f["hist_sim"] = float(pair.get("histogram_similarity", 0.0))
    f["dhash_sim"] = float(pair.get("dhash_similarity", 0.0))
    f["log_inliers"] = math.log1p(kp.get("inliers", 0)) / 5.0  # ~[0, 1]
    f["inlier_ratio"] = float(kp.get("inlier_ratio", 0.0))
    f["brightness_gap"] = float(pair.get("brightness_gap", 0.0))
    f["monochrome"] = float(feats.get("any_monochrome", False))
    f["has_evidence"] = float(feats.get("has_evidence", False))
    f["p_tools"] = float(c.needs_external_tools)
    f["p_answerable"] = float(c.answerable_from_inputs)
    return f


class FeatureSpace:
    def __init__(self, model_keys: list[str], model_attrs: dict[str, dict[str, float]]):
        self.model_keys = list(model_keys)
        self.model_attrs = model_attrs

    def names(self) -> list[str]:
        base = list(query_features(_blank()).keys())
        names = ["bias", *base, "attr_log_price_in", "attr_log_price_out"]
        names += [f"attr_{a}_x_{d}" for a in ("log_price_in", "log_price_out") for d in ("p_easy", "p_moderate", "p_hard")]
        for m in self.model_keys:
            names.append(f"m_{m}")
            names += [f"m_{m}_x_{i}" for i in INTERACT]
        return names

    def vector(self, qf: dict[str, float], model_key: str, attrs: Optional[dict[str, float]] = None) -> np.ndarray:
        attrs = attrs or self.model_attrs.get(model_key) or {"log_price_in": 0.0, "log_price_out": 0.0}
        v = [1.0, *qf.values(), attrs["log_price_in"], attrs["log_price_out"]]
        v += [attrs[a] * qf[d] for a in ("log_price_in", "log_price_out") for d in ("p_easy", "p_moderate", "p_hard")]
        for m in self.model_keys:
            on = 1.0 if m == model_key else 0.0
            v.append(on)
            v += [on * qf[i] for i in INTERACT]
        return np.asarray(v, dtype=float)


def _blank() -> QueryClassification:
    from jev_router.schemas import LabelDistribution

    return QueryClassification(
        LabelDistribution("x", {}), LabelDistribution("x", {}), LabelDistribution("x", {}), 0, 0, 0, "blank"
    )


def model_attributes(registry_models: Iterable[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Log-scaled prices as capability proxies; centred later by the ridge bias."""
    out = {}
    for m in registry_models:
        p = m.get("pricing") or {}
        pin, pout = p.get("input_usd_per_mtok"), p.get("output_usd_per_mtok")
        if pin is None:  # unlisted model: fall back to observed cost scale
            obs = (m.get("observed_cost") or {}).get("mean_usd_per_query")
            pin = pout = (obs * 1e6 / 4000) if obs else 1.0
        out[m["key"]] = {"log_price_in": math.log10(max(pin, 1e-3)), "log_price_out": math.log10(max(pout, 1e-3))}
    return out


def _ridge(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    pen = lam * np.eye(X.shape[1])
    pen[0, 0] = 0.0  # don't shrink the bias
    return np.linalg.solve(X.T @ X + pen, X.T @ y)


def _logistic(X: np.ndarray, y: np.ndarray, lam: float, iters: int = 200) -> np.ndarray:
    w = np.zeros(X.shape[1])
    pen = lam * np.ones(X.shape[1])
    pen[0] = 0.0
    for _ in range(iters):  # Newton / IRLS
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        g = X.T @ (p - y) + pen * w
        H = (X.T * (p * (1 - p))) @ X + np.diag(pen) + 1e-6 * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-6:
            break
    return w


@dataclass
class Prediction:
    quality: float           # normalized 0..1 (mean over bootstrap)
    quality_std: float
    unsupported: float       # probability
    source: str


class QualityModel:
    def __init__(self, space: FeatureSpace, quality_w: np.ndarray, unsup_w: np.ndarray, meta: dict[str, Any]):
        self.space = space
        self.quality_w = quality_w   # shape (B, d)
        self.unsup_w = unsup_w       # shape (B, d)
        self.meta = meta

    # -------------------------------------------------------------- fitting
    @classmethod
    def fit(
        cls,
        rows: list[TrainingRow],
        model_attrs: dict[str, dict[str, float]],
        lam: float = 3.0,
        lam_logit: float = 3.0,
        n_boot: int = 40,
        seed: int = 0,
    ) -> "QualityModel":
        keys = sorted({r.model_key for r in rows})
        space = FeatureSpace(keys, model_attrs)
        X = np.stack([space.vector(query_features(r.classification), r.model_key) for r in rows])
        y = np.array([r.score / 4.0 for r in rows])
        u = np.array([float(r.unsupported) for r in rows])
        cases = sorted({r.case_id for r in rows})
        idx_by_case = {c: [i for i, r in enumerate(rows) if r.case_id == c] for c in cases}
        rng = np.random.default_rng(seed)
        qw, uw = [], []
        for b in range(n_boot):
            sample = cases if b == 0 else list(rng.choice(cases, size=len(cases), replace=True))
            idx = [i for c in sample for i in idx_by_case[c]]
            qw.append(_ridge(X[idx], y[idx], lam))
            uw.append(_logistic(X[idx], u[idx], lam_logit))
        meta = {"n_rows": len(rows), "n_cases": len(cases), "models": keys, "lambda": lam,
                "lambda_logit": lam_logit, "n_boot": n_boot, "feature_names": space.names()}
        return cls(space, np.stack(qw), np.stack(uw), meta)

    # ----------------------------------------------------------- inference
    def predict(self, classification: QueryClassification, model_key: str,
                attrs: Optional[dict[str, float]] = None) -> Prediction:
        x = self.space.vector(query_features(classification), model_key, attrs)
        q = np.clip(self.quality_w @ x, 0.0, 1.0)
        p = 1 / (1 + np.exp(-np.clip(self.unsup_w @ x, -30, 30)))
        seen = model_key in self.space.model_keys
        src = (f"learned predictor (n={self.meta['n_rows']} rows / {self.meta['n_cases']} cases"
               f"{'' if seen else ', unseen model: attributes only'})")
        return Prediction(float(q[0]), float(q[1:].std()) if len(q) > 1 else 0.0, float(p.mean()), src)

    def knows(self, model_key: str) -> bool:
        return model_key in self.space.model_keys

    # -------------------------------------------------------- persistence
    def to_json(self) -> dict[str, Any]:
        return {"meta": self.meta, "model_keys": self.space.model_keys, "model_attrs": self.space.model_attrs,
                "quality_w": self.quality_w.round(6).tolist(), "unsup_w": self.unsup_w.round(6).tolist()}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "QualityModel":
        space = FeatureSpace(data["model_keys"], data["model_attrs"])
        return cls(space, np.array(data["quality_w"]), np.array(data["unsup_w"]), data["meta"])

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_json()) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Optional["QualityModel"]:
        return cls.from_json(json.loads(path.read_text(encoding="utf-8"))) if path.exists() else None
