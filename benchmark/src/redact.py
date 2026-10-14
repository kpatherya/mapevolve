from __future__ import annotations

import re
from typing import Any

_SECRET_RE = re.compile(
    r"(sk-[A-Za-z0-9_\-]{8,}|Bearer\s+[A-Za-z0-9_\-\.]+|AQ\.[A-Za-z0-9_\-]{10,})"
)
_DATA_URL_RE = re.compile(
    r"data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=\s]+",
    re.DOTALL,
)

def sanitize_text(text: str, limit: int = 4000) -> str:
    if not text:
        return ""
    cleaned = _DATA_URL_RE.sub("[redacted-image-data]", text)
    cleaned = _SECRET_RE.sub("[redacted-secret]", cleaned)
    if len(cleaned) > limit:
        cleaned = cleaned[:limit] + "...[truncated]"
    return cleaned

def sanitize_obj(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): sanitize_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_obj(v) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize_obj(v) for v in obj]
    if isinstance(obj, str):
        if (
            obj.startswith("data:image")
            or "base64," in obj[:80]
            or obj.startswith("sk-")
            or obj.startswith("AQ.")
        ):
            return sanitize_text(obj, limit=200)
        if "sk-" in obj or "data:image" in obj or "Bearer " in obj:
            return sanitize_text(obj, limit=8000)
        return obj
    if isinstance(obj, (int, float, bool)) or obj is None:
        return obj
    return sanitize_text(str(obj))
