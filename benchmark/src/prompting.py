from __future__ import annotations

import json
from typing import Any, Optional

from benchmark.src.paths import JUDGE_PROMPT_PATH, VLM_PROMPT_PATH

def load_vlm_template() -> str:
    return VLM_PROMPT_PATH.read_text(encoding="utf-8")

def load_judge_template() -> str:
    return JUDGE_PROMPT_PATH.read_text(encoding="utf-8")

def render_vlm_prompt(question: str, template: Optional[str] = None) -> str:
    text = load_vlm_template() if template is None else template
    if "{QUESTION}" not in text:
        raise ValueError("VLM prompt template is missing {QUESTION}")
    return text.replace("{QUESTION}", question)

def render_judge_prompt(
    question: str,
    ground_truth: str,
    rubric: dict[str, Any],
    response: str,
    template: Optional[str] = None,
) -> str:
    text = load_judge_template() if template is None else template
    rubric_json = json.dumps(rubric, indent=2, ensure_ascii=False)
    for token, value in (
        ("<<<QUESTION>>>", question),
        ("<<<GROUND_TRUTH>>>", ground_truth),
        ("<<<RUBRIC_JSON>>>", rubric_json),
        ("<<<RESPONSE>>>", response),
    ):
        if token not in text:
            raise ValueError(f"Judge prompt template is missing {token}")
        text = text.replace(token, value)
    return text
