"""Answer-free features of an image pair.

Difficulty in visual queries mostly lives in the images, not the question
text ("What changed?" is the same for an easy and a hard pair). These cheap
features give Jev and the router that signal:

* global appearance similarity (HSV histogram correlation, difference hash)
* viewpoint alignment from SIFT keypoints + RANSAC homography inliers
* brightness gap (day vs night), monochrome/archival flags, EXIF year gap

They are a lightweight signal, not a pair-verification pipeline: a high
inlier count says "these views are geometrically alignable", nothing more.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np
from PIL import Image

from jev_router.image_meta import ImageSource, describe_image

_WORK_EDGE = 800


def _load_gray_and_bgr(source: ImageSource) -> tuple[np.ndarray, np.ndarray]:
    data = Path(source).read_bytes() if isinstance(source, (str, Path)) else source
    with Image.open(io.BytesIO(data)) as img:
        rgb = img.convert("RGB")
        rgb.thumbnail((_WORK_EDGE, _WORK_EDGE))
        arr = np.asarray(rgb)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), bgr


def _hist_similarity(a: np.ndarray, b: np.ndarray) -> float:
    hsv_a, hsv_b = cv2.cvtColor(a, cv2.COLOR_BGR2HSV), cv2.cvtColor(b, cv2.COLOR_BGR2HSV)
    hist = lambda h: cv2.normalize(cv2.calcHist([h], [0, 1], None, [30, 32], [0, 180, 0, 256]), None).flatten()
    return float(max(0.0, cv2.compareHist(hist(hsv_a), hist(hsv_b), cv2.HISTCMP_CORREL)))


def _dhash(gray: np.ndarray, size: int = 16) -> np.ndarray:
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).flatten()


def _keypoint_alignment(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    # SIFT rather than ORB: ORB found 0 inliers on a same-viewpoint pair that
    # differed only in zoom (GT-E1); SIFT found 20.
    sift = cv2.SIFT_create(nfeatures=3000)
    ka, da = sift.detectAndCompute(a, None)
    kb, db = sift.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return {"keypoints": [len(ka or []), len(kb or [])], "good_matches": 0, "inliers": 0, "inlier_ratio": 0.0}
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [m for m, *rest in pairs if rest and m.distance < 0.75 * rest[0].distance]
    inliers = 0
    if len(good) >= 8:
        src = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 6.0)
        inliers = int(mask.sum()) if mask is not None else 0
    return {
        "keypoints": [len(ka), len(kb)],
        "good_matches": len(good),
        "inliers": inliers,
        "inlier_ratio": round(inliers / len(good), 3) if good else 0.0,
    }


def alignment_label(inliers: int, inlier_ratio: float) -> str:
    # Provisional thresholds, set from the 11 Georgia Tech pairs.
    if inliers >= 20 and inlier_ratio >= 0.4:
        return "aligned"
    if inliers >= 10:
        return "partially_aligned"
    return "unaligned"


def pair_features(images: Sequence[ImageSource]) -> Optional[dict[str, Any]]:
    """Features for a two-image query; None for single-image queries."""
    if len(images) != 2:
        return None
    (ga, ca), (gb, cb) = (_load_gray_and_bgr(i) for i in images)
    meta = [describe_image(i) for i in images]
    kp = _keypoint_alignment(ga, gb)
    years = [m["exif_year"] for m in meta]
    return {
        "histogram_similarity": round(_hist_similarity(ca, cb), 3),
        "dhash_similarity": round(float(1 - np.mean(_dhash(ga) != _dhash(gb))), 3),
        "keypoints": kp,
        "viewpoint_alignment": alignment_label(kp["inliers"], kp["inlier_ratio"]),
        "brightness_gap": round(abs(float(ga.mean()) - float(gb.mean())) / 255.0, 3),
        "any_monochrome": any(m["likely_monochrome"] for m in meta),
        "exif_year_gap": abs(years[0] - years[1]) if all(years) else None,
    }


def feature_vector(features: Optional[dict[str, Any]], n_images: int) -> dict[str, float]:
    """Flat numeric view used by the learned quality predictor."""
    if not features:
        return {"two_images": float(n_images == 2), "hist_sim": 0.0, "dhash_sim": 0.0, "log_inliers": 0.0,
                "inlier_ratio": 0.0, "brightness_gap": 0.0, "monochrome": 0.0}
    return {
        "two_images": 1.0,
        "hist_sim": features["histogram_similarity"],
        "dhash_sim": features["dhash_similarity"],
        "log_inliers": float(np.log1p(features["keypoints"]["inliers"])),
        "inlier_ratio": features["keypoints"]["inlier_ratio"],
        "brightness_gap": features["brightness_gap"],
        "monochrome": float(features["any_monochrome"]),
    }
