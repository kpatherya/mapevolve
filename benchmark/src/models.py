from __future__ import annotations

from typing import Any, Optional

import yaml

from benchmark.src.paths import LOGICAL_MODEL_ORDER, MODELS_YAML

RESERVE_IDS = (
    "qwen/qwen2.5-vl-72b-instruct",
    "openai/gpt-4.1",
)

FAMILY_RULES: dict[str, dict[str, tuple[str, ...]]] = {
    "GEMINI": {
        "prefixes": ("google/gemini",),
        "prefer": ("gemini-3.1-pro", "gemini-3-pro", "gemini-2.5-pro", "gemini-pro"),
        "deprioritize": ("flash", "image-preview", "tts", "embed", "aqa"),
    },
    "CLAUDE": {
        "prefixes": ("anthropic/claude",),
        "prefer": ("claude-sonnet-4", "sonnet-4", "claude-sonnet", "sonnet"),
        "deprioritize": ("haiku",),
    },
    "QWEN": {
        "prefixes": ("qwen/",),
        "prefer": ("qwen3-vl-235b-a22b-instruct", "qwen3-vl", "qwen2.5-vl-72b", "qwen2.5-vl", "-vl-"),
        "deprioritize": ("30b", "coder", "embed", "ocr"),
    },
    "GEMMA": {
        "prefixes": ("google/gemma",),
        "prefer": ("gemma-3-27b-it", "gemma-3-27b", "gemma-3-12b", "gemma-3"),
        "deprioritize": ("embed",),
    },
    "LLAMA": {
        "prefixes": ("meta-llama/",),
        "prefer": ("llama-4-maverick", "llama-4", "maverick", "scout"),
        "deprioritize": ("instruct-8b", "text"),
    },
    "MISTRAL": {
        "prefixes": ("mistralai/",),
        "prefer": (
            "mistral-small-3.2-24b-instruct",
            "mistral-small-3.2",
            "mistral-small-3",
            "pixtral",
        ),
        "deprioritize": ("embed", "codestral", "nemo", "ocr"),
    },
    "KIMI": {
        "prefixes": ("moonshotai/",),
        "prefer": ("kimi-vl-a3b-thinking", "kimi-vl", "kimi-k2.5", "vl"),
        "deprioritize": ("embed",),
    },
}

def load_model_config(path=MODELS_YAML) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    slots = data.get("slots") or []
    names = [slot["logical_name"] for slot in slots]
    if tuple(names) != LOGICAL_MODEL_ORDER:
        raise ValueError(f"models.yaml slot order {names} does not match the required eight slots")
    return data

def image_support(entry: dict[str, Any]) -> Optional[bool]:
    architecture = entry.get("architecture") or {}
    inputs = architecture.get("input_modalities")
    if isinstance(inputs, list) and inputs:
        return "image" in {str(item).lower() for item in inputs}
    modality = str(architecture.get("modality") or entry.get("modality") or "")
    if not modality:
        return None
    lowered = modality.lower()
    if "image" in lowered:
        return True
    if "text" in lowered:
        return False
    return None

def text_output_supported(entry: dict[str, Any]) -> bool:
    architecture = entry.get("architecture") or {}
    outputs = architecture.get("output_modalities")
    if isinstance(outputs, list) and outputs:
        return "text" in {str(item).lower() for item in outputs}
    modality = str(architecture.get("modality") or "")
    if "->" in modality:
        return "text" in modality.split("->", 1)[1].lower()
    return True

def usable_model_id(model_id: str) -> bool:
    lowered = model_id.lower()
    if "chatgpt" in lowered:
        return False
    if ":online" in lowered or ":free" in lowered:
        return False
    return True

def _context_length(entry: dict[str, Any]) -> int:
    try:
        return int(entry.get("context_length") or 0)
    except (TypeError, ValueError):
        return 0

def _rank_family(entry: dict[str, Any], rule: dict[str, tuple[str, ...]]) -> tuple:
    model_id = entry["id"].lower()
    prefer_index = 100
    for index, token in enumerate(rule["prefer"]):
        if token in model_id:
            prefer_index = index
            break
    penalty = 0
    for token in rule.get("deprioritize", ()):
        if token in model_id:
            penalty += 50
    return (penalty, prefer_index, -_context_length(entry), entry["id"])

def _accepts_image_metadata(entry: dict[str, Any], allow_unknown: bool) -> bool:
    if not text_output_supported(entry):
        return False
    support = image_support(entry)
    if support is False:
        return False
    if support is None and not allow_unknown:
        return False
    return True

def resolve_openrouter_slot(
    logical_name: str,
    preferred_id: str,
    catalog: list[dict[str, Any]],
    failed_ids: Optional[set[str]] = None,
    used_ids: Optional[set[str]] = None,
) -> dict[str, Any]:
    failed = failed_ids or set()
    used = used_ids or set()
    rule = FAMILY_RULES[logical_name]
    by_id = {entry["id"]: entry for entry in catalog if entry.get("id")}

    def blocked(model_id: str) -> bool:
        return model_id in failed or model_id in used or not usable_model_id(model_id)

    exact = by_id.get(preferred_id)
    if exact and not blocked(preferred_id) and _accepts_image_metadata(exact, allow_unknown=True):
        return _selection(logical_name, preferred_id, exact, "", "selected")

    alias_pool = []
    prefix = preferred_id + "-"
    for entry in catalog:
        model_id = entry.get("id") or ""
        if not model_id.startswith(prefix) or blocked(model_id):
            continue
        if _accepts_image_metadata(entry, allow_unknown=True):
            alias_pool.append(entry)
    if alias_pool:
        chosen = sorted(alias_pool, key=lambda item: item["id"])[-1]
        reason = (
            f"exact id {preferred_id} is unavailable; "
            f"using dated alias {chosen['id']} from the same model family"
        )
        return _selection(logical_name, preferred_id, chosen, reason, "substituted")

    family_pool = []
    for entry in catalog:
        model_id = entry.get("id") or ""
        if blocked(model_id) or not any(model_id.startswith(prefix) for prefix in rule["prefixes"]):
            continue
        support = image_support(entry)
        prefer_hit = any(token in model_id.lower() for token in rule["prefer"])
        if not text_output_supported(entry):
            continue
        if support is False:
            continue
        if support is None and not prefer_hit:
            continue
        family_pool.append(entry)
    if family_pool:
        chosen = sorted(family_pool, key=lambda item: _rank_family(item, rule))[0]
        reason = (
            f"exact id {preferred_id} is unavailable or not image-capable; "
            f"using closest current image-capable {logical_name} family model {chosen['id']}"
        )
        return _selection(logical_name, preferred_id, chosen, reason, "substituted")

    return {
        "logical_name": logical_name,
        "requested_model_id": preferred_id,
        "resolved_model_id": None,
        "backend": "openrouter",
        "status": "unresolved",
        "substitution_reason": f"no image-capable {logical_name} family model found for {preferred_id}",
        "image_input_supported_metadata": None,
        "catalog_entry": None,
    }

def pick_reserve(
    catalog: list[dict[str, Any]],
    used_ids: set[str],
    failed_ids: set[str],
) -> tuple[Optional[dict[str, Any]], str]:
    by_id = {entry["id"]: entry for entry in catalog if entry.get("id")}
    for reserve_id in RESERVE_IDS:
        entry = by_id.get(reserve_id)
        if entry is None or reserve_id in used_ids or reserve_id in failed_ids:
            continue
        if not usable_model_id(reserve_id) or not _accepts_image_metadata(entry, allow_unknown=True):
            continue
        return entry, (
            "RESERVE SUBSTITUTION: no image-capable model remained in the requested family; "
            f"using reserve model {reserve_id}"
        )

    quality_tokens = (
        "gemini-2.5-pro",
        "gemini-3",
        "claude-sonnet",
        "gpt-4.1",
        "gpt-5",
        "qwen2.5-vl-72b",
        "qwen3-vl",
        "llama-4",
    )
    ranked = []
    for entry in catalog:
        model_id = entry.get("id") or ""
        if not model_id or model_id in used_ids or model_id in failed_ids or not usable_model_id(model_id):
            continue
        if image_support(entry) is not True or not text_output_supported(entry):
            continue
        lowered = model_id.lower()
        if any(token in lowered for token in ("embed", "tts", "whisper", "flux", "image-generation")):
            continue
        quality = 0 if any(token in lowered for token in quality_tokens) else 1
        ranked.append((quality, -_context_length(entry), model_id, entry))
    if not ranked:
        return None, "RESERVE SUBSTITUTION FAILED: no remaining image-capable OpenRouter model"
    ranked.sort()
    chosen = ranked[0][3]
    return chosen, (
        "RESERVE SUBSTITUTION: no image-capable model remained in the requested family; "
        f"using another image-capable OpenRouter model {chosen['id']}"
    )

def _selection(
    logical_name: str,
    preferred_id: str,
    entry: dict[str, Any],
    reason: str,
    status: str,
) -> dict[str, Any]:
    return {
        "logical_name": logical_name,
        "requested_model_id": preferred_id,
        "resolved_model_id": entry["id"],
        "backend": "openrouter",
        "status": status,
        "substitution_reason": reason,
        "image_input_supported_metadata": image_support(entry),
        "catalog_entry": entry,
    }

def relevant_catalog_entries(
    catalog: list[dict[str, Any]],
    preferred_ids: set[str],
) -> list[dict[str, Any]]:
    prefixes = tuple(prefix for rule in FAMILY_RULES.values() for prefix in rule["prefixes"])
    family_tokens = ("gemini", "claude", "qwen", "gemma", "llama", "mistral", "kimi", "gpt-4", "gpt-5")
    selected = []
    for entry in catalog:
        model_id = entry.get("id") or ""
        keep = model_id in preferred_ids or any(model_id.startswith(prefix) for prefix in prefixes)
        if not keep and image_support(entry) is True and any(token in model_id for token in family_tokens):
            keep = True
        if not keep:
            continue
        description = entry.get("description") or ""
        selected.append(
            {
                "id": model_id,
                "name": entry.get("name"),
                "architecture": entry.get("architecture"),
                "context_length": entry.get("context_length"),
                "pricing": entry.get("pricing"),
                "description": description[:400],
            }
        )
    selected.sort(key=lambda item: item["id"])
    return selected
