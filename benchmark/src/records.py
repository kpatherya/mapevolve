from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("pilot-%Y%m%dT%H%M%SZ")

def append_jsonl(path: Path, record: dict[str, Any], lock: Optional[Any] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"

    def _write() -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    if lock is None:
        _write()
    else:
        with lock:
            _write()

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at {path}:{line_number}") from exc
    return rows

def latest_by(rows: list[dict[str, Any]], key_fn: Callable[[dict[str, Any]], Any]) -> dict[Any, dict[str, Any]]:
    latest: dict[Any, dict[str, Any]] = {}
    for row in rows:
        latest[key_fn(row)] = row
    return latest

def response_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("logical_model_name")), str(row.get("case_id")))

def successful_pairs(path: Path) -> set[tuple[str, str]]:
    latest = latest_by(read_jsonl(path), response_key)
    return {key for key, row in latest.items() if row.get("status") == "success"}
