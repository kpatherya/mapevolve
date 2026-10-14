from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

import httpx
import yaml
from PIL import Image

from benchmark.src.models import (
    image_support,
    load_model_config,
    pick_reserve,
    relevant_catalog_entries,
    resolve_openrouter_slot,
)
from benchmark.src.paths import (
    ARTIFACTS_DIR,
    AVAILABILITY_CSV,
    AVAILABILITY_JSON,
    BENCHMARK_YAML,
    RESOLVED_MODELS_PATH,
)
from benchmark.src.providers.common import ProviderError
from benchmark.src.providers.openai_provider import generate as openai_generate
from benchmark.src.providers.openrouter_provider import generate as openrouter_generate
from benchmark.src.redact import sanitize_text

AVAILABILITY_COLUMNS = [
    "logical_name",
    "requested_model_id",
    "resolved_model_id",
    "provider/backend",
    "image_input_supported",
    "two_image_smoke_test",
    "status",
    "substitution_reason",
]

def load_settings() -> dict[str, Any]:
    return yaml.safe_load(BENCHMARK_YAML.read_text(encoding="utf-8"))

def require_env(names: list[str]) -> None:
    missing = [name for name in names if not os.environ.get(name)]
    if not missing:
        return
    print("Required API keys are missing. Export them before paid execution:")
    for name in missing:
        print(f'export {name}="..."')
    raise SystemExit(2)

def fetch_openrouter_catalog(api_key: str) -> list[dict[str, Any]]:
    response = httpx.get(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=60.0,
    )
    if response.status_code == 401:
        raise ProviderError("OpenRouter rejected the API key", status_code=401, auth=True)
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data", [])
    if not isinstance(data, list):
        raise RuntimeError("OpenRouter model list did not return a list")
    return data

def _smoke_settings(settings: dict[str, Any]) -> dict[str, Any]:
    smoke = dict(settings)
    smoke["max_output_tokens"] = 64
    smoke["empty_output_retry_max_tokens"] = 256
    return smoke

def smoke_two_images(
    backend: str,
    model_id: str,
    image_a: Path,
    image_b: Path,
    settings: dict[str, Any],
) -> dict[str, Any]:
    prompt = "Reply with the single word OK."
    try:
        if backend == "openai":
            result = openai_generate(model_id, prompt, image_a, image_b, _smoke_settings(settings))
        elif backend == "openrouter":
            result = openrouter_generate(model_id, prompt, image_a, image_b, _smoke_settings(settings))
        else:
            raise ValueError(backend)
    except ProviderError as exc:
        if exc.auth:
            raise
        return {
            "ok": False,
            "reject": exc.reject_model,
            "error": sanitize_text(str(exc)),
            "result": None,
        }
    return {"ok": True, "reject": False, "error": "", "result": result}

def _tiny_images() -> tuple[Path, Path, tempfile.TemporaryDirectory]:
    directory = tempfile.TemporaryDirectory(prefix="mapevolve-smoke-")
    root = Path(directory.name)
    image_a = root / "a.jpg"
    image_b = root / "b.jpg"
    Image.new("RGB", (64, 64), (30, 60, 180)).save(image_a, format="JPEG", quality=90)
    Image.new("RGB", (64, 64), (180, 60, 30)).save(image_b, format="JPEG", quality=90)
    return image_a, image_b, directory

def _row_from_success(
    logical_name: str,
    requested: str,
    model_id: str,
    backend: str,
    status: str,
    reason: str,
    outcome: dict[str, Any],
    metadata_support: Optional[bool],
) -> dict[str, Any]:
    result = outcome.get("result") or {}
    image_ok = True if outcome.get("ok") else metadata_support is True
    resolved_id = result.get("resolved_model_id") or model_id
    if outcome.get("ok") and resolved_id != model_id:
        note = f"API returned snapshot id {resolved_id} for requested {model_id}."
        reason = f"{reason} | {note}".strip(" |") if reason else note
    return {
        "logical_name": logical_name,
        "requested_model_id": requested,
        "resolved_model_id": result.get("resolved_model_id") or model_id,
        "backend": backend,
        "provider_if_available": result.get("provider_name"),
        "image_input_supported": bool(image_ok),
        "two_image_smoke_test": "pass" if outcome.get("ok") else "fail",
        "status": status,
        "substitution_reason": reason,
        "smoke_latency_seconds": result.get("latency_seconds"),
        "smoke_reported_cost_usd": result.get("reported_cost_usd"),
        "smoke_adjustments": result.get("actual_parameter_adjustments") or [],
        "smoke_error": outcome.get("error") or None,
        "smoke_response_excerpt": (result.get("response_text") or "")[:200],
    }

def _unavailable(logical_name: str, requested: str, backend: str, reason: str) -> dict[str, Any]:
    return {
        "logical_name": logical_name,
        "requested_model_id": requested,
        "resolved_model_id": None,
        "backend": backend,
        "provider_if_available": None,
        "image_input_supported": False,
        "two_image_smoke_test": "fail",
        "status": "unavailable",
        "substitution_reason": reason,
        "smoke_latency_seconds": None,
        "smoke_reported_cost_usd": None,
        "smoke_adjustments": [],
        "smoke_error": reason,
        "smoke_response_excerpt": "",
    }

def _accept_outcome(outcome: dict[str, Any]) -> bool:
    return bool(outcome.get("ok") or not outcome.get("reject"))

def resolve_slots(settings: Optional[dict[str, Any]] = None, catalog: Optional[list[dict[str, Any]]] = None) -> list[dict[str, Any]]:
    settings = settings or load_settings()
    config = load_model_config()
    need_openrouter = any(slot["backend"] == "openrouter" for slot in config["slots"])
    names = ["OPENAI_API_KEY"]
    if need_openrouter:
        names.append("OPENROUTER_API_KEY")
    require_env(names)
    if catalog is None and need_openrouter:
        catalog = fetch_openrouter_catalog(os.environ["OPENROUTER_API_KEY"])
    catalog = catalog or []
    preferred_ids = {
        slot.get("preferred_model_id")
        for slot in config["slots"]
        if slot.get("preferred_model_id")
    }
    preferred_ids.update(config.get("reserves") or [])
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    relevant_path = ARTIFACTS_DIR / "openrouter_relevant_models.json"
    relevant_path.write_text(
        json.dumps(relevant_catalog_entries(catalog, {item for item in preferred_ids if item}), indent=2) + "\n",
        encoding="utf-8",
    )

    image_a, image_b, temp_dir = _tiny_images()
    resolved: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    try:
        for slot in config["slots"]:
            logical_name = slot["logical_name"]
            print(f"Preflight {logical_name}", flush=True)
            if slot["backend"] == "openai":
                row = _resolve_openai_slot(slot, settings, image_a, image_b, catalog, used_ids)
            else:
                row = _resolve_openrouter_slot(slot, settings, image_a, image_b, catalog, used_ids)
            if row.get("resolved_model_id"):
                used_ids.add(row["resolved_model_id"])
            resolved.append(row)
            print(
                f"  {row['status']} {row.get('resolved_model_id')} smoke={row['two_image_smoke_test']}",
                flush=True,
            )
    finally:
        temp_dir.cleanup()

    _write_availability(resolved)
    RESOLVED_MODELS_PATH.write_text(json.dumps(resolved, indent=2) + "\n", encoding="utf-8")
    return resolved

def _resolve_openai_slot(slot, settings, image_a, image_b, catalog, used_ids) -> dict[str, Any]:
    env_name = slot.get("preferred_model_env") or "OPENAI_VLM_MODEL"
    env_model = (os.environ.get(env_name) or "").strip() or None
    fallbacks = list(slot.get("fallback_model_ids") or [])
    requested = env_model or (fallbacks[0] if fallbacks else "")
    candidates = [env_model] if env_model else fallbacks
    notes: list[str] = []
    for model_id in candidates:
        if "chatgpt" in model_id.lower():
            notes.append(f"{model_id} skipped because it is a ChatGPT-only model id")
            continue
        outcome = smoke_two_images("openai", model_id, image_a, image_b, settings)
        if _accept_outcome(outcome) and (outcome.get("ok") or not outcome.get("reject")):
            if outcome.get("ok") or not outcome.get("reject"):
                status = "ready" if model_id == requested and not notes else "substituted"
                reason = ""
                if notes or model_id != requested:
                    status = "substituted"
                    reason = "OpenAI candidate search: " + " | ".join(notes + [f"using {model_id}"])
                if outcome.get("ok"):
                    return _row_from_success(
                        slot["logical_name"], requested, model_id, "openai", status, reason, outcome, True
                    )

                row = _row_from_success(
                    slot["logical_name"], requested, model_id, "openai", status, reason, outcome, None
                )
                row["two_image_smoke_test"] = "fail"
                row["smoke_error"] = outcome.get("error")
                row["substitution_reason"] = (reason + " | " if reason else "") + (
                    f"two-image smoke did not succeed: {outcome.get('error')}"
                )
                return row
        notes.append(f"{model_id} rejected: {outcome.get('error')}")
    return _fill_reserve(
        slot["logical_name"],
        requested,
        "openai",
        catalog,
        used_ids,
        set(),
        " | ".join(notes) or "no OpenAI vision candidate accepted two images",
        settings,
        image_a,
        image_b,
    )

def _resolve_openrouter_slot(slot, settings, image_a, image_b, catalog, used_ids) -> dict[str, Any]:
    logical_name = slot["logical_name"]
    requested = slot["preferred_model_id"]
    failed: set[str] = set()
    notes: list[str] = []
    for _ in range(4):
        selection = resolve_openrouter_slot(
            logical_name,
            requested,
            catalog,
            failed_ids=failed,
            used_ids=used_ids,
        )
        model_id = selection.get("resolved_model_id")
        if not model_id:
            notes.append(selection.get("substitution_reason") or "family unresolved")
            break
        outcome = smoke_two_images("openrouter", model_id, image_a, image_b, settings)
        base_reason = selection.get("substitution_reason") or ""
        if notes:
            base_reason = " | ".join(note for note in [base_reason, *notes] if note)
        if outcome.get("ok"):
            status = "ready" if selection["status"] == "selected" and model_id == requested else "substituted"
            return _row_from_success(
                logical_name,
                requested,
                model_id,
                "openrouter",
                status,
                "" if status == "ready" else base_reason,
                outcome,
                image_support(selection.get("catalog_entry") or {}),
            )
        if not outcome.get("reject"):
            row = _row_from_success(
                logical_name,
                requested,
                model_id,
                "openrouter",
                "ready" if model_id == requested else "substituted",
                base_reason,
                outcome,
                image_support(selection.get("catalog_entry") or {}),
            )
            row["two_image_smoke_test"] = "fail"
            row["smoke_error"] = outcome.get("error")
            return row
        failed.add(model_id)
        notes.append(f"{model_id} rejected: {outcome.get('error')}")
    return _fill_reserve(
        logical_name,
        requested,
        "openrouter",
        catalog,
        used_ids,
        failed,
        " | ".join(notes),
        settings,
        image_a,
        image_b,
    )

def _fill_reserve(logical_name, requested, backend, catalog, used_ids, failed, prior_notes, settings, image_a, image_b):
    entry, reason = pick_reserve(catalog, used_ids, failed)
    full_reason = " | ".join(part for part in (prior_notes, reason) if part)
    if entry is None:
        return _unavailable(logical_name, requested, backend, full_reason)
    model_id = entry["id"]
    outcome = smoke_two_images("openrouter", model_id, image_a, image_b, settings)
    if outcome.get("reject") and not outcome.get("ok"):
        failed.add(model_id)
        entry2, reason2 = pick_reserve(catalog, used_ids, failed)
        if entry2 is None:
            return _unavailable(logical_name, requested, "openrouter", full_reason + " | " + outcome.get("error", ""))
        model_id = entry2["id"]
        full_reason = full_reason + " | " + reason2
        outcome = smoke_two_images("openrouter", model_id, image_a, image_b, settings)
    if not outcome.get("ok") and outcome.get("reject"):
        return _unavailable(
            logical_name,
            requested,
            "openrouter",
            full_reason + " | reserve smoke rejected: " + outcome.get("error", ""),
        )
    row = _row_from_success(
        logical_name,
        requested,
        model_id,
        "openrouter",
        "reserve_substitution",
        full_reason,
        outcome,
        image_support(entry),
    )
    if not outcome.get("ok"):
        row["two_image_smoke_test"] = "fail"
        row["smoke_error"] = outcome.get("error")
    row["status"] = "reserve_substitution"
    if not str(row["substitution_reason"]).startswith("RESERVE"):
        row["substitution_reason"] = "RESERVE SUBSTITUTION: " + row["substitution_reason"]
    return row

def _write_availability(rows: list[dict[str, Any]]) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    AVAILABILITY_JSON.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with AVAILABILITY_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AVAILABILITY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "logical_name": row["logical_name"],
                    "requested_model_id": row["requested_model_id"],
                    "resolved_model_id": row.get("resolved_model_id") or "",
                    "provider/backend": row["backend"],
                    "image_input_supported": row["image_input_supported"],
                    "two_image_smoke_test": row["two_image_smoke_test"],
                    "status": row["status"],
                    "substitution_reason": row.get("substitution_reason") or "",
                }
            )

def main() -> None:
    rows = resolve_slots()
    ready = sum(1 for row in rows if row.get("resolved_model_id"))
    print(f"Resolved {ready}/8 slots. Wrote {AVAILABILITY_CSV}")

if __name__ == "__main__":
    main()
