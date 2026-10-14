from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Optional

from PIL import Image, ImageDraw, ImageFont, ImageOps

from benchmark.src.paths import (
    ARTIFACTS_DIR,
    CASE_IDS,
    CASES_PATH,
    CONTACT_SHEET_PATH,
    PAIRING_VALIDATION_PATH,
    PREPROCESS_LOG_PATH,
    PROCESSED_DIR,
    PROJECT_ROOT,
    RAW_DIR,
)

CASE_PATTERN = "|".join(CASE_IDS)
IMAGE_NAME_RE = re.compile(rf"^({CASE_PATTERN})(_2| 2)?(\.[A-Za-z0-9.]+)$")
TEXT_NAME_RE = re.compile(rf"^({CASE_PATTERN}).*\.txt$", re.IGNORECASE)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".jpeg.webp"}

RUBRICS: dict[str, dict[str, list[str]]] = {
    "E1": {
        "core_expectations": [
            "Recognize that no meaningful structural change to the building is visible.",
        ],
        "transient_or_confounder_expectations": [
            "Framing/zoom differs.",
            "Image B contains more surrounding area.",
            "Temporary scene content such as a person may differ.",
        ],
        "unsupported_claims_to_avoid": [
            "Any invented architectural/facade/window/roof/entrance change.",
        ],
    },
    "E2": {
        "core_expectations": [
            "Recognize that no meaningful structural change is visible.",
        ],
        "transient_or_confounder_expectations": [
            "Illumination differs.",
            "Sky/weather differs.",
            "Minor environmental/temporary differences may be present.",
        ],
        "unsupported_claims_to_avoid": [
            "Invented architectural changes.",
        ],
    },
    "M1": {
        "core_expectations": [
            "The staircase is painted in multiple colors in Image B.",
            "Main building entrance/structure appears unchanged.",
        ],
        "transient_or_confounder_expectations": [
            "Image B contains snow/winter conditions.",
        ],
        "unsupported_claims_to_avoid": [
            "Claiming that the entrance/facade/building was structurally rebuilt or substantially redesigned without visual support.",
        ],
    },
    "M2": {
        "core_expectations": [
            "The fountain is filled/operating with water jets in Image A.",
            "The fountain is empty or inactive in Image B.",
            "No clear structural change to the fountain/Campanile is visible.",
        ],
        "transient_or_confounder_expectations": [
            "Image B contains a much larger crowd.",
        ],
        "unsupported_claims_to_avoid": [
            "Invented structural modification to the Campanile/fountain.",
        ],
    },
    "H1": {
        "core_expectations": [
            "The central historic building remains recognizable.",
            "The surrounding campus environment changes substantially.",
            "Vegetation is much more mature in the modern image.",
            "Landscaping/path layout differs.",
            "Some surrounding structures disappear and/or newer elements appear.",
        ],
        "transient_or_confounder_expectations": [
            "The old image is monochrome/sepia.",
            "Color differences must NOT be interpreted as the building physically changing from gray/brown to red.",
        ],
        "unsupported_claims_to_avoid": [
            "The building changed color from gray to red.",
            "Specific renovations that are not visually supported.",
            "Unsupported claims about construction dates or historical events.",
        ],
    },
    "H2": {
        "core_expectations": [
            "Overall structure and architectural form remain substantially the same.",
            "Visible differences include some window/door/entrance details.",
            "Exterior/masonry appears cleaner or restored.",
        ],
        "transient_or_confounder_expectations": [
            "Be conservative about differences that may come from photography, lighting, or image quality.",
        ],
        "unsupported_claims_to_avoid": [
            "Claiming major changes to columns, roofline, or building footprint.",
            "Inventing specific historical events not visible in the pair.",
        ],
    },
}

BUILDING_NAMES = (
    "Tech Tower",
    "Kandeda",
    "Kendeda",
    "Klaus",
    "Campanile",
    "Carnegie",
    "Kessler",
)

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def find_image_zip(search_root: Optional[Path] = None) -> Path:
    root = search_root or PROJECT_ROOT
    candidates = sorted(root.glob("IMAGE PAIRS*.zip"))
    exact = [path for path in candidates if path.name == "IMAGE PAIRS.zip"]
    if exact:
        return exact[0]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No IMAGE PAIRS*.zip file found in {root}")
    names = ", ".join(path.name for path in candidates)
    raise FileNotFoundError(f"Multiple image-pair archives found: {names}")

def _ignored_zip_name(filename: str) -> bool:
    if "__MACOSX" in filename.split("/"):
        return True
    name = Path(filename).name
    if not name or name == ".DS_Store" or name.startswith("._") or name.startswith("."):
        return True
    if any(part == ".." for part in Path(filename).parts):
        return True
    return False

def parse_image_filename(filename: str) -> Optional[tuple[str, str]]:
    name = Path(filename).name
    match = IMAGE_NAME_RE.match(name)
    if not match:
        return None
    case_id, marker, ext = match.group(1), match.group(2), match.group(3).lower()
    if ext not in IMAGE_SUFFIXES and not ext.endswith((".jpg", ".jpeg", ".png", ".webp")):
        return None
    side = "B" if marker else "A"
    return case_id, side

def parse_text_filename(filename: str) -> Optional[str]:
    name = Path(filename).name
    match = TEXT_NAME_RE.match(name)
    if not match:
        return None
    return match.group(1)

def iter_archive_files(zip_path: Path) -> list[dict[str, Any]]:
    records = []
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir() or _ignored_zip_name(info.filename):
                continue
            records.append(
                {
                    "filename": info.filename,
                    "name": Path(info.filename).name,
                    "size": info.file_size,
                }
            )
    return records

def group_archive(zip_path: Path) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {
        case_id: {"images": {}, "texts": [], "unparsed_images": []} for case_id in CASE_IDS
    }
    seen_names: list[str] = []
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir() or _ignored_zip_name(info.filename):
                continue
            name = Path(info.filename).name
            seen_names.append(name)
            image_parsed = parse_image_filename(name)
            text_case = parse_text_filename(name)
            if image_parsed and text_case:
                raise ValueError(f"Member matches both image and text rules: {name}")
            if image_parsed:
                case_id, side = image_parsed
                if side in grouped[case_id]["images"]:
                    raise ValueError(f"Duplicate image {side} for {case_id}: {name}")
                grouped[case_id]["images"][side] = info.filename
            elif text_case:
                grouped[text_case]["texts"].append(info.filename)
            else:
                suffix = Path(name).suffix.lower()
                if suffix in {".jpg", ".jpeg", ".png", ".webp"} or name.lower().endswith(".jpeg.webp"):
                    raise ValueError(f"Image could not be assigned to a case: {name}")
    errors = []
    for case_id in CASE_IDS:
        images = grouped[case_id]["images"]
        texts = grouped[case_id]["texts"]
        if set(images) != {"A", "B"}:
            errors.append(f"{case_id} images={sorted(images)} expected A and B")
        if len(texts) != 1:
            errors.append(f"{case_id} ground-truth files={len(texts)} expected 1")
    image_count = sum(len(item["images"]) for item in grouped.values())
    text_count = sum(len(item["texts"]) for item in grouped.values())
    if errors or image_count != 12 or text_count != 6:
        detail = "; ".join(errors) if errors else "count mismatch"
        raise ValueError(
            "Archive validation failed before any model call. "
            f"cases={len(CASE_IDS)} images={image_count} ground_truth_files={text_count}. {detail}"
        )
    return grouped

def pairing_validation_report(grouped: dict[str, dict[str, Any]]) -> dict[str, Any]:
    space_markers = 0
    underscore_markers = 0
    for case_id in CASE_IDS:
        image_b = Path(grouped[case_id]["images"]["B"]).name
        if " 2" in image_b:
            space_markers += 1
        if "_2" in image_b:
            underscore_markers += 1
    space_assumption = space_markers == 6 and underscore_markers == 0
    return {
        "expected_b_marker": 'trailing " 2" before the extension',
        "observed_b_marker": "_2" if underscore_markers == 6 else "mixed",
        "space_marker_count": space_markers,
        "underscore_marker_count": underscore_markers,
        "space_marker_assumption_holds": space_assumption,
        "counts": {"cases": 6, "images": 12, "ground_truth_files": 6},
        "decision": (
            "The archive uses '_2' before the extension rather than a trailing space and '2'. "
            "Each case still has exactly one unmarked image (Image A) and one '_2' image (Image B). "
            "Counts are 6 cases, 12 images, and 6 ground-truth files, so preprocessing continues."
            if not space_assumption and underscore_markers == 6
            else "Pairing marker matched the requested trailing-' 2' rule."
        ),
    }

def extract_question(text: str) -> str:
    index = text.find("Question:")
    if index < 0:
        raise ValueError("Ground-truth file has no Question field")
    rest = text[index + len("Question:") :]
    rest = rest.lstrip(" \t\r\n\u2028\u2029")
    if not rest:
        raise ValueError("Question field is empty")
    opener = rest[0]
    closers = {"“": "”", '"': '"', "‘": "’"}
    if opener in closers:
        end = rest.find(closers[opener], 1)
        if end < 0:
            raise ValueError("Question quotation is not closed")
        question = rest[1:end].strip()
    else:
        question = rest.splitlines()[0].strip()
    if not question:
        raise ValueError("Extracted question is empty")
    return question

def extract_difficulty(text: str) -> str:
    match = re.search(r"Difficulty:\s*(Easy|Medium|Hard)\b", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).lower()
    first = text.splitlines()[0] if text.splitlines() else text
    for label in ("Easy", "Medium", "Hard"):
        if re.search(rf"\b{label}\b", first, flags=re.IGNORECASE):
            return label.lower()
    raise ValueError("Could not read difficulty from the ground-truth file")

def _resample():
    resampling = getattr(Image, "Resampling", Image)
    return resampling.LANCZOS

def to_rgb(image: Image.Image) -> Image.Image:
    oriented = ImageOps.exif_transpose(image)
    if oriented.mode == "RGB":
        return oriented
    if oriented.mode in ("RGBA", "LA") or (oriented.mode == "P" and "transparency" in oriented.info):
        rgba = oriented.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return oriented.convert("RGB")

def resize_long_edge(image: Image.Image, max_edge: int = 2048) -> Image.Image:
    width, height = image.size
    long_edge = max(width, height)
    if long_edge <= max_edge:
        return image
    if width >= height:
        new_width = max_edge
        new_height = max(1, int(round(height * (max_edge / float(width)))))
    else:
        new_height = max_edge
        new_width = max(1, int(round(width * (max_edge / float(height)))))
    return image.resize((new_width, new_height), _resample())

def strip_metadata(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        raise ValueError(f"Expected RGB before metadata strip, got {image.mode}")
    return Image.frombytes("RGB", image.size, image.tobytes())

def preprocess_image(image: Image.Image, max_edge: int = 2048) -> tuple[Image.Image, dict[str, Any]]:
    source_size = image.size
    orientation = None
    try:
        orientation = image.getexif().get(274)
    except Exception:
        orientation = None
    rgb = to_rgb(image)
    oriented_size = rgb.size
    resized = resize_long_edge(rgb, max_edge=max_edge)
    clean = strip_metadata(resized)
    info = {
        "source_size": list(source_size),
        "oriented_size": list(oriented_size),
        "processed_size": list(clean.size),
        "source_mode": image.mode,
        "exif_orientation": orientation,
        "resized": list(resized.size) != list(oriented_size),
        "upscaled": False,
        "operations": [
            "exif_transpose",
            "convert_rgb",
            "resize_long_edge_if_needed",
            "strip_metadata",
            "jpeg_q95",
        ],
    }
    return clean, info

def save_jpeg(image: Image.Image, path: Path, quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG", quality=quality, optimize=True)

def _read_zip_bytes(archive: zipfile.ZipFile, name: str) -> bytes:
    return archive.read(name)

def build_contact_sheet(processed: dict[str, dict[str, Path]], dest: Path) -> None:
    thumb_w, thumb_h = 520, 360
    label_h = 28
    margin = 16
    columns = 2
    rows = len(CASE_IDS)
    width = margin + columns * (thumb_w + margin)
    height = margin + rows * (label_h + thumb_h + margin)
    sheet = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for row, case_id in enumerate(CASE_IDS):
        for col, side in enumerate(("A", "B")):
            x = margin + col * (thumb_w + margin)
            y = margin + row * (label_h + thumb_h + margin)
            draw.text((x, y), f"{case_id} {side}", fill=(0, 0, 0), font=font)
            with Image.open(processed[case_id][side]) as image:
                frame = image.convert("RGB")
                frame.thumbnail((thumb_w, thumb_h), _resample())
                pad = Image.new("RGB", (thumb_w, thumb_h), (245, 245, 245))
                pad.paste(frame, ((thumb_w - frame.size[0]) // 2, (thumb_h - frame.size[1]) // 2))
                sheet.paste(pad, (x, y + label_h))
    dest.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dest, format="JPEG", quality=90, optimize=True)

def load_cases(path: Path = CASES_PATH) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        cases = payload.get("cases")
    else:
        cases = payload
    if not isinstance(cases, list):
        raise ValueError("cases.json must be a list of case objects")
    return cases

def prompt_contains_forbidden_metadata(prompt: str, case: dict[str, Any]) -> list[str]:
    problems = []
    if case["ground_truth_raw"] in prompt:
        problems.append("full ground truth")
    for token in ("Ground truth:", "Expected answer:", "Expected behavior:", "Expected model behavior:"):
        if token in prompt:
            problems.append(token)
    if case["case_id"] in prompt:
        problems.append("case id")
    for name in BUILDING_NAMES:
        if name in prompt and name not in case["question"]:
            problems.append(name)
    for key in ("source_filename_a", "source_filename_b"):
        filename = case.get(key)
        if filename and filename in prompt:
            problems.append(filename)
    difficulty = case.get("difficulty", "")
    if difficulty and re.search(rf"\b{re.escape(difficulty)}\b", prompt, flags=re.IGNORECASE):
        problems.append("difficulty label")
    return problems

def prepare(zip_path: Optional[Path] = None, max_edge: int = 2048, quality: int = 95) -> list[dict[str, Any]]:
    archive_path = zip_path or find_image_zip()
    grouped = group_archive(archive_path)
    validation = pairing_validation_report(grouped)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    PAIRING_VALIDATION_PATH.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    if validation["counts"]["cases"] != 6 or validation["counts"]["images"] != 12:
        raise ValueError(validation["decision"])

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    preprocess_log = []
    processed_paths: dict[str, dict[str, Path]] = {}

    with zipfile.ZipFile(archive_path) as archive:
        for case_id in CASE_IDS:
            item = grouped[case_id]
            text_name = item["texts"][0]
            text_bytes = _read_zip_bytes(archive, text_name)
            text_path = RAW_DIR / text_name
            text_path.parent.mkdir(parents=True, exist_ok=True)
            text_path.write_bytes(text_bytes)
            if sha256_file(text_path) != sha256_bytes(text_bytes):
                raise RuntimeError(f"Ground-truth copy hash mismatch for {text_name}")
            text = text_bytes.decode("utf-8")
            question = extract_question(text)
            difficulty = extract_difficulty(text)
            case_record: dict[str, Any] = {
                "case_id": case_id,
                "difficulty": difficulty,
                "question": question,
                "ground_truth_raw": text,
                "ground_truth_sha256": sha256_bytes(text_bytes),
                "rubric": RUBRICS[case_id],
                "source_text": text_name,
                "source_zip": archive_path.name,
            }
            processed_paths[case_id] = {}
            for side, label in (("A", "image_a"), ("B", "image_b")):
                source_name = item["images"][side]
                source_bytes = _read_zip_bytes(archive, source_name)
                raw_rel = Path("data") / "raw" / source_name
                raw_path = RAW_DIR.parent.parent / raw_rel

                raw_path = RAW_DIR / source_name
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path.write_bytes(source_bytes)
                written_hash = sha256_file(raw_path)
                original_hash = sha256_bytes(source_bytes)
                if written_hash != original_hash:
                    raise RuntimeError(f"Raw copy hash mismatch for {source_name}")
                with Image.open(raw_path) as image:
                    image.load()
                    clean, info = preprocess_image(image, max_edge=max_edge)
                processed_name = f"{case_id}_{side}.jpg"
                processed_path = PROCESSED_DIR / processed_name
                save_jpeg(clean, processed_path, quality=quality)
                with Image.open(processed_path) as check:
                    check.load()
                    if check.mode != "RGB":
                        raise RuntimeError(f"{processed_name} is {check.mode}, expected RGB")
                    if max(check.size) > max_edge:
                        raise RuntimeError(f"{processed_name} long edge {max(check.size)} exceeds {max_edge}")
                    if check.getexif().get(274) not in (None, 1):
                        raise RuntimeError(f"{processed_name} still carries EXIF orientation")
                processed_rel = f"data/processed/{processed_name}"
                case_record[label] = processed_rel
                case_record[f"{label}_sha256"] = sha256_file(processed_path)
                case_record[f"original_{label}"] = str(Path("data/raw") / source_name)
                case_record[f"original_{label}_sha256"] = original_hash
                case_record[f"source_filename_{side.lower()}"] = Path(source_name).name
                processed_paths[case_id][side] = processed_path
                preprocess_log.append(
                    {
                        "case_id": case_id,
                        "side": side,
                        "source_filename": Path(source_name).name,
                        "raw_path": str(Path("data/raw") / source_name),
                        "processed_path": processed_rel,
                        "original_sha256": original_hash,
                        "processed_sha256": case_record[f"{label}_sha256"],
                        **info,
                    }
                )
            cases.append(case_record)

    CASES_PATH.write_text(json.dumps(cases, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    PREPROCESS_LOG_PATH.write_text(json.dumps(preprocess_log, indent=2) + "\n", encoding="utf-8")
    build_contact_sheet(processed_paths, CONTACT_SHEET_PATH)
    return cases

def main() -> None:
    cases = prepare()
    print(f"Prepared {len(cases)} cases")
    for case in cases:
        print(
            f"{case['case_id']} {case['difficulty']} "
            f"A={case['source_filename_a']} B={case['source_filename_b']} "
            f"Q={case['question']}"
        )
    print(f"Wrote {CASES_PATH}")
    print(f"Contact sheet (human inspection only): {CONTACT_SHEET_PATH}")

if __name__ == "__main__":
    main()
