from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from benchmark.src.models import pick_reserve, resolve_openrouter_slot
from benchmark.src.paths import CASE_IDS
from benchmark.src.prepare_data import (
    extract_difficulty,
    extract_question,
    find_image_zip,
    group_archive,
    pairing_validation_report,
    prompt_contains_forbidden_metadata,
)
from benchmark.src.prompting import render_judge_prompt, render_vlm_prompt
from benchmark.src.providers.openai_provider import build_openai_content
from benchmark.src.providers.openrouter_provider import build_openrouter_content
from benchmark.src.records import append_jsonl, successful_pairs

REQUIRED_PROMPT = (
    "Identify the most important differences that are directly visible in the two\n"
    "images. Distinguish persistent physical or structural changes from temporary\n"
    "or environmental differences such as people, weather, snow, vegetation,\n"
    "lighting, framing, or image quality when relevant."
)

def _jpeg(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (8, 6), color).save(path, format="JPEG")

def _vision(model_id: str, image: bool = True) -> dict:
    return {
        "id": model_id,
        "architecture": {
            "input_modalities": ["text", "image"] if image else ["text"],
            "output_modalities": ["text"],
        },
        "context_length": 128000,
    }

def test_zip_grouping_ignores_macos_metadata_and_requires_two_images(tmp_path: Path) -> None:
    root = tmp_path / "pack"
    easy = root / "easy"
    easy.mkdir(parents=True)
    _jpeg(easy / "E1.jpg", (1, 2, 3))
    _jpeg(easy / "E1 2.jpg", (4, 5, 6))
    (easy / "E1_notes.txt").write_text(
        "Difficulty: Easy\nQuestion: “What changed between these two images?”\nGround truth: none\n",
        encoding="utf-8",
    )
    _jpeg(easy / "E2.jpg", (1, 1, 1))
    macos = root / "__MACOSX" / "easy"
    macos.mkdir(parents=True)
    (macos / "._E1.jpg").write_bytes(b"junk")
    (easy / ".DS_Store").write_bytes(b"store")
    (easy / "._E2.jpg").write_bytes(b"appledouble")
    archive_path = tmp_path / "IMAGE PAIRS.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for path in root.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(root.parent).as_posix())
    with pytest.raises(ValueError, match="E2"):
        group_archive(archive_path)

    _jpeg(easy / "E2_2.jpg", (2, 2, 2))
    (easy / "E2_gtruth_q.txt").write_text(
        "Difficulty: Easy\nQuestion: “What changed?”\nGround truth: unchanged\n",
        encoding="utf-8",
    )
    for case_id, folder in (("M1", "medium"), ("M2", "medium"), ("H1", "hard"), ("H2", "hard")):
        directory = root / folder
        directory.mkdir(exist_ok=True)
        _jpeg(directory / f"{case_id}.jpg", (3, 3, 3))
        _jpeg(directory / f"{case_id}_2.jpg", (4, 4, 4))
        (directory / f"{case_id}_gtruth_q.txt").write_text(
            f"{case_id} — Example, Medium, note\nQuestion: “What changed between these two images?”\n"
            if case_id.startswith("M")
            else f"{case_id} — Example, Hard, note\nQuestion:\u2028“Compare these two images of the same building. What changed?”\n",
            encoding="utf-8",
        )
    archive_path.unlink()
    with zipfile.ZipFile(archive_path, "w") as archive:
        for path in root.rglob("*"):
            if path.is_file() and not path.name.startswith("._") and path.name != ".DS_Store" and "__MACOSX" not in path.parts:
                archive.write(path, path.relative_to(root.parent).as_posix())
        archive.writestr("__MACOSX/._E1.jpg", b"junk")
        archive.writestr("pack/easy/.DS_Store", b"store")
    grouped = group_archive(archive_path)
    assert set(grouped) == set(CASE_IDS)
    for case_id, item in grouped.items():
        assert set(item["images"]) == {"A", "B"}
        assert len(item["texts"]) == 1
    assert Path(grouped["E1"]["images"]["B"]).name == "E1 2.jpg"
    assert Path(grouped["M1"]["images"]["A"]).name == "M1.jpg"
    assert Path(grouped["M1"]["images"]["B"]).name == "M1_2.jpg"
    report = pairing_validation_report(grouped)
    assert report["counts"] == {"cases": 6, "images": 12, "ground_truth_files": 6}

def test_case_id_parsing_and_question_extraction() -> None:
    text = "Difficulty: Easy\nQuestion: “What changed between these two images?”\nGround truth: Keep Kandeda spelling.\n"
    assert extract_question(text) == "What changed between these two images?"
    assert extract_difficulty(text) == "easy"
    hard = "H2 — Carnegie Building, Hard, Subtle Restoration\nQuestion:\u2028“Compare these two images of the same building. What changed?”\n"
    assert extract_difficulty(hard) == "hard"
    assert extract_question(hard) == "Compare these two images of the same building. What changed?"
    medium = "M1 — Klaus Stairs, Medium, Real Change\nQuestion: “What changed between these two images?”\n"
    assert extract_difficulty(medium) == "medium"

def test_real_archive_has_six_pairs() -> None:
    archive = find_image_zip()
    grouped = group_archive(archive)
    assert set(grouped) == set(CASE_IDS)
    report = pairing_validation_report(grouped)
    assert report["counts"]["images"] == 12
    assert report["underscore_marker_count"] == 6
    assert report["space_marker_assumption_holds"] is False
    import zipfile as zipfile_module

    with zipfile_module.ZipFile(archive) as handle:
        for case_id, item in grouped.items():
            raw = handle.read(item["texts"][0]).decode("utf-8")
            question = extract_question(raw)
            difficulty = extract_difficulty(raw)
            assert question
            assert difficulty in {"easy", "medium", "hard"}
            assert Path(item["images"]["B"]).name.startswith(f"{case_id}_2") or " 2" in Path(item["images"]["B"]).name

def test_prompt_hides_answer_key_and_uses_fixed_wording() -> None:
    template = render_vlm_prompt("What changed between these two images?")
    assert REQUIRED_PROMPT in template
    assert template.startswith("Image A and Image B show the same location at different times.")
    case = {
        "case_id": "E1",
        "difficulty": "easy",
        "question": "What changed between these two images?",
        "ground_truth_raw": "Ground truth: Both images show Tech Tower. No structural change is visible.",
        "source_filename_a": "Tech Tower easy.jpg",
        "source_filename_b": "Tech Tower easy 2.jpg",
    }
    assert prompt_contains_forbidden_metadata(template, case) == []
    judge = render_judge_prompt(case["question"], case["ground_truth_raw"], {"core_expectations": ["none"]}, "No structural change.")
    assert "OPENAI_FRONTIER_VISION" not in judge
    assert "Tech Tower" in judge
    assert "logical_model" not in judge

def test_message_order_and_high_detail() -> None:
    openai_content = build_openai_content("PROMPT", "data:image/jpeg;base64,AA", "data:image/jpeg;base64,BB")
    assert [item.get("text") for item in openai_content if item["type"] == "input_text"] == ["Image A:", "Image B:", "PROMPT"]
    assert openai_content[1]["detail"] == "high"
    assert openai_content[3]["detail"] == "high"
    openrouter_content = build_openrouter_content("PROMPT", "data:image/jpeg;base64,AA", "data:image/jpeg;base64,BB")
    assert openrouter_content[0]["text"] == "Image A:"
    assert openrouter_content[1]["image_url"]["detail"] == "high"
    assert openrouter_content[2]["text"] == "Image B:"
    assert openrouter_content[4]["text"] == "PROMPT"

def test_resume_skips_only_latest_success(tmp_path: Path) -> None:
    path = tmp_path / "raw_responses.jsonl"
    append_jsonl(path, {"logical_model_name": "GEMINI", "case_id": "E1", "status": "success"})
    append_jsonl(path, {"logical_model_name": "GEMINI", "case_id": "E2", "status": "error"})
    assert successful_pairs(path) == {("GEMINI", "E1")}
    append_jsonl(path, {"logical_model_name": "GEMINI", "case_id": "E1", "status": "error"})
    assert ("GEMINI", "E1") not in successful_pairs(path)
    append_jsonl(path, {"logical_model_name": "GEMINI", "case_id": "E1", "status": "success"})
    assert ("GEMINI", "E1") in successful_pairs(path)

def test_exact_dated_alias_and_family_resolution() -> None:
    exact = resolve_openrouter_slot(
        "GEMINI",
        "google/gemini-3.1-pro-preview",
        [_vision("google/gemini-3.1-pro-preview"), _vision("google/gemini-2.5-flash")],
    )
    assert exact["resolved_model_id"] == "google/gemini-3.1-pro-preview"
    assert exact["substitution_reason"] == ""

    alias = resolve_openrouter_slot(
        "MISTRAL",
        "mistralai/mistral-small-3.2-24b-instruct",
        [
            _vision("mistralai/mistral-small-3.2-24b-instruct-2503"),
            _vision("mistralai/mistral-small-3.2-24b-instruct-2506"),
        ],
    )
    assert alias["resolved_model_id"] == "mistralai/mistral-small-3.2-24b-instruct-2506"
    assert "dated alias" in alias["substitution_reason"]

    family = resolve_openrouter_slot(
        "GEMINI",
        "google/gemini-3.1-pro-preview",
        [_vision("google/gemini-2.5-pro"), _vision("google/gemini-2.5-flash"), _vision("google/gemini-2.5-pro:online")],
    )
    assert family["resolved_model_id"] == "google/gemini-2.5-pro"
    assert "closest current image-capable" in family["substitution_reason"]

    text_only = resolve_openrouter_slot(
        "GEMINI",
        "google/gemini-3.1-pro-preview",
        [_vision("google/gemini-2.5-pro", image=False)],
    )
    assert text_only["resolved_model_id"] is None
    reserve, reason = pick_reserve([_vision("qwen/qwen2.5-vl-72b-instruct")], set(), set())
    assert reserve is not None
    assert reserve["id"] == "qwen/qwen2.5-vl-72b-instruct"
    assert reason.startswith("RESERVE SUBSTITUTION")

def test_prompt_file_is_json_free_of_ground_truth_keys() -> None:
    rendered = render_vlm_prompt("What changed between these two images?")
    assert "Ground truth" not in rendered
    assert "Kandeda" not in rendered
    payload = {"question": rendered}
    json.dumps(payload)
