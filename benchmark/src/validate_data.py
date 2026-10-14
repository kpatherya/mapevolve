from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

from benchmark.src.paths import (
    BENCHMARK_ROOT,
    CASE_IDS,
    CASES_PATH,
    CONTACT_SHEET_PATH,
    PAIRING_VALIDATION_PATH,
    PROCESSED_DIR,
    VLM_PROMPT_PATH,
)
from benchmark.src.prepare_data import load_cases, prompt_contains_forbidden_metadata, sha256_file, to_rgb
from benchmark.src.prompting import render_vlm_prompt

def _ratio(size: tuple[int, int]) -> float:
    width, height = size
    return width / float(height)

def validate() -> list[str]:
    errors: list[str] = []
    if not CASES_PATH.exists():
        return ["cases.json is missing. Run python -m benchmark.src.prepare_data first."]
    try:
        cases = load_cases()
    except Exception as exc:
        return [f"cases.json could not be read: {exc}"]

    if len(cases) != 6:
        errors.append(f"expected 6 cases, found {len(cases)}")
    ids = [case.get("case_id") for case in cases]
    if ids != list(CASE_IDS):
        errors.append(f"case order/ids {ids} != {list(CASE_IDS)}")

    expected_names = {f"{case_id}_{side}.jpg" for case_id in CASE_IDS for side in ("A", "B")}
    found_names = {path.name for path in PROCESSED_DIR.glob("*") if path.is_file()}
    if found_names != expected_names:
        errors.append(f"processed filenames {sorted(found_names)} != {sorted(expected_names)}")

    if not VLM_PROMPT_PATH.exists():
        errors.append("vlm prompt template is missing")
    if not PAIRING_VALIDATION_PATH.exists():
        errors.append("pairing validation report is missing")
    if not CONTACT_SHEET_PATH.exists():
        errors.append("human contact sheet is missing")

    for case in cases:
        case_id = case.get("case_id")
        for field in (
            "difficulty",
            "image_a",
            "image_b",
            "question",
            "ground_truth_raw",
            "rubric",
            "image_a_sha256",
            "image_b_sha256",
            "original_image_a_sha256",
            "original_image_b_sha256",
        ):
            if field not in case:
                errors.append(f"{case_id} missing {field}")
        if case.get("difficulty") not in ("easy", "medium", "hard"):
            errors.append(f"{case_id} difficulty {case.get('difficulty')} is not easy/medium/hard")
        rubric = case.get("rubric") or {}
        for key in ("core_expectations", "transient_or_confounder_expectations", "unsupported_claims_to_avoid"):
            if not isinstance(rubric.get(key), list) or not rubric.get(key):
                errors.append(f"{case_id} rubric missing {key}")
        if VLM_PROMPT_PATH.exists() and case.get("question"):
            prompt = render_vlm_prompt(case["question"])
            problems = prompt_contains_forbidden_metadata(prompt, case)
            if problems:
                errors.append(f"{case_id} prompt exposes {problems}")
            if case["question"] not in prompt:
                errors.append(f"{case_id} prompt does not contain its question")

        for side, image_key, hash_key, original_key, original_hash_key in (
            ("A", "image_a", "image_a_sha256", "original_image_a", "original_image_a_sha256"),
            ("B", "image_b", "image_b_sha256", "original_image_b", "original_image_b_sha256"),
        ):
            rel = case.get(image_key)
            if not rel:
                continue
            path = BENCHMARK_ROOT / rel
            if not path.exists():
                errors.append(f"{case_id} {side} missing file {rel}")
                continue
            digest = sha256_file(path)
            if digest != case.get(hash_key):
                errors.append(f"{case_id} {side} processed hash mismatch")
            original_rel = case.get(original_key)
            if original_rel:
                original_path = BENCHMARK_ROOT / original_rel
                if not original_path.exists():
                    errors.append(f"{case_id} {side} raw file missing")
                elif sha256_file(original_path) != case.get(original_hash_key):
                    errors.append(f"{case_id} {side} raw hash mismatch")
                else:
                    _check_geometry(errors, case_id, side, original_path, path)
            _check_processed_image(errors, case_id, side, path)

        text_rel = case.get("source_text")
        if text_rel:
            text_path = BENCHMARK_ROOT / "data" / "raw" / text_rel
            if not text_path.exists():
                errors.append(f"{case_id} ground-truth source missing")
            else:
                raw = text_path.read_bytes()
                if raw.decode("utf-8") != case.get("ground_truth_raw"):
                    errors.append(f"{case_id} ground_truth_raw does not match the source file bytes")

    return errors

def _check_processed_image(errors: list[str], case_id: str, side: str, path: Path) -> None:
    with Image.open(path) as image:
        image.load()
        if image.format != "JPEG":
            errors.append(f"{case_id} {side} format is {image.format}, expected JPEG")
        if image.mode != "RGB":
            errors.append(f"{case_id} {side} mode is {image.mode}, expected RGB")
        if max(image.size) > 2048:
            errors.append(f"{case_id} {side} long edge {max(image.size)} exceeds 2048")
        orientation = image.getexif().get(274)
        if orientation not in (None, 1):
            errors.append(f"{case_id} {side} still has EXIF orientation {orientation}")

def _check_geometry(errors: list[str], case_id: str, side: str, original: Path, processed: Path) -> None:
    with Image.open(original) as source, Image.open(processed) as output:
        source.load()
        output.load()
        oriented = to_rgb(source)
        if max(output.size) > max(oriented.size):
            errors.append(f"{case_id} {side} was upscaled")
        if abs(_ratio(output.size) - _ratio(oriented.size)) > 0.02:
            errors.append(
                f"{case_id} {side} aspect ratio changed from {oriented.size} to {output.size}"
            )
        if max(oriented.size) <= 2048 and output.size != oriented.size:
            errors.append(f"{case_id} {side} changed size {oriented.size} -> {output.size} without needing a resize")
        if max(oriented.size) > 2048 and max(output.size) != 2048:
            errors.append(f"{case_id} {side} long edge is {max(output.size)}, expected 2048")

def main() -> None:
    errors = validate()
    if errors:
        print("VALIDATION FAILED")
        for error in errors:
            print(f"- {error}")
        print("Stopped before any model API call.")
        sys.exit(2)
    print("Validation passed: 6 cases, 12 processed images, 6 ground-truth files.")

if __name__ == "__main__":
    main()
