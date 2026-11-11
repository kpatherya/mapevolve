"""VLM execution through ChatOpenRouter (OpenRouter) and ChatOpenAI (direct).

Every model receives the identical system prompt, user content layout,
temperature, and timeout. Only `max_output_tokens` can differ, because some
models spend hidden reasoning tokens from the same budget (see
config/models.yaml); the visible answer length is constrained in the prompt.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from PIL import Image

from jev_router.image_meta import ImageSource
from jev_router.registry import ROOT, ModelRegistry
from jev_router.schemas import VLMResult

log = logging.getLogger(__name__)

_SUPPORTED = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp", "GIF": "image/gif"}
MAX_LONG_EDGE = 2048  # matches benchmark/config/benchmark.yaml


def image_data_url(source: ImageSource) -> str:
    """Encode an image as a data URL, downscaling to the benchmark's max edge."""
    raw = Path(source).read_bytes() if isinstance(source, (str, Path)) else source
    with Image.open(io.BytesIO(raw)) as img:
        fmt = img.format
        if fmt in _SUPPORTED and max(img.size) <= MAX_LONG_EDGE:
            return f"data:{_SUPPORTED[fmt]};base64,{base64.b64encode(raw).decode('ascii')}"
        rgb = img.convert("RGB")
        rgb.thumbnail((MAX_LONG_EDGE, MAX_LONG_EDGE))
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=95)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def build_messages(system_prompt: str, question: str, images: Sequence[ImageSource],
                   evidence: Optional[str] = None, note: Optional[str] = None) -> list[Any]:
    """System prompt, then labelled images, supplied evidence, routing note, question."""
    from langchain_core.messages import HumanMessage, SystemMessage

    if not 1 <= len(images) <= 2:
        raise ValueError("Supply one or two images.")
    content: list[dict[str, Any]] = []
    labels = ["Image A", "Image B"] if len(images) == 2 else ["Image"]
    for label, img in zip(labels, images):
        content.append({"type": "text", "text": f"{label}:"})
        content.append({"type": "image_url", "image_url": {"url": image_data_url(img), "detail": "high"}})
    if evidence:
        content.append({"type": "text", "text": f"Supplied evidence: {evidence}"})
    if note:
        content.append({"type": "text", "text": f"Routing note: {note}"})
    content.append({"type": "text", "text": f"Question: {question}"})
    return [SystemMessage(content=system_prompt), HumanMessage(content=content)]


def parse_structured(text: str) -> Optional[dict[str, Any]]:
    """Leniently extract the JSON answer object; None if the model ignored the format."""
    import json
    import re

    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    for candidate in (cleaned, cleaned[cleaned.find("{"): cleaned.rfind("}") + 1] if "{" in cleaned else ""):
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(data, dict) and isinstance(data.get("answer"), str):
            return data
    return None


def default_chat_factory(model: dict[str, Any], generation: dict[str, Any]) -> Any:
    """Create the LangChain chat model for a registry entry."""
    max_tokens = model.get("max_output_tokens") or generation["max_output_tokens"]
    common = dict(
        temperature=generation["temperature"],
        max_tokens=max_tokens,
        timeout=generation["timeout_seconds"],
        max_retries=generation["max_retries"],
        **(model.get("chat_kwargs") or {}),
    )
    if model["backend"] == "openrouter":
        from langchain_openrouter import ChatOpenRouter

        return ChatOpenRouter(model=model["model_id"], **common)
    if model["backend"] == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model["model_id"], **common)
    raise ValueError(f"Unknown backend {model['backend']!r}")


class VLMExecutor:
    def __init__(
        self,
        registry: ModelRegistry,
        generation: dict[str, Any],
        chat_factory: Callable[[dict[str, Any], dict[str, Any]], Any] = default_chat_factory,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.registry = registry
        self.generation = generation
        self.chat_factory = chat_factory
        prompt_path = ROOT / generation.get("system_prompt_path", "config/prompts/vlm_system_v1.txt")
        self.system_prompt = system_prompt or prompt_path.read_text(encoding="utf-8")
        self.task_dir = ROOT / generation["task_prompt_dir"] if generation.get("task_prompt_dir") else None
        self.structured = bool(generation.get("structured_output"))

    def system_for(self, task: Optional[str]) -> str:
        """Same prompt for every model; the task block depends only on Jev's query type."""
        if task and self.task_dir and (self.task_dir / f"{task}.txt").exists():
            return self.system_prompt.rstrip() + "\n\n" + (self.task_dir / f"{task}.txt").read_text(encoding="utf-8")
        return self.system_prompt

    def _estimate_cost(self, key: str, in_tok: Optional[int], out_tok: Optional[int]) -> Optional[float]:
        pricing = self.registry.get(key).get("pricing")
        if not pricing or pricing.get("input_usd_per_mtok") is None or in_tok is None:
            return None
        return round(
            in_tok * pricing["input_usd_per_mtok"] / 1e6 + (out_tok or 0) * (pricing.get("output_usd_per_mtok") or 0) / 1e6,
            6,
        )

    def run(self, key: str, question: str, images: Sequence[ImageSource], task: Optional[str] = None,
            evidence: Optional[str] = None, note: Optional[str] = None) -> VLMResult:
        model = self.registry.get(key)
        result = VLMResult(key=key, model_id=model["model_id"], backend=model["backend"], status="error")
        started = time.perf_counter()
        try:
            chat = self.chat_factory(model, self.generation)
            message = chat.invoke(build_messages(self.system_for(task), question, images, evidence, note))
        except Exception as exc:
            result.latency_seconds = round(time.perf_counter() - started, 3)
            result.error = f"{type(exc).__name__}: {exc}"[:2000]
            log.warning("VLM %s failed: %s", key, result.error)
            return result
        result.latency_seconds = round(time.perf_counter() - started, 3)
        content = message.content
        if isinstance(content, list):  # some providers return content blocks
            content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
        meta = getattr(message, "response_metadata", {}) or {}
        usage = getattr(message, "usage_metadata", None) or {}
        result.input_tokens = usage.get("input_tokens")
        result.output_tokens = usage.get("output_tokens")
        result.total_tokens = usage.get("total_tokens")
        result.provider = meta.get("provider") or meta.get("model_provider")
        result.finish_reason = meta.get("finish_reason")
        if meta.get("cost") is not None:
            result.cost_usd, result.cost_source = float(meta["cost"]), "provider_reported"
        else:
            result.cost_usd = self._estimate_cost(key, result.input_tokens, result.output_tokens)
            result.cost_source = "estimated_from_registry_pricing" if result.cost_usd is not None else None
        text = (content or "").strip()
        if not text:
            result.error = f"empty answer (finish_reason={result.finish_reason})"
            return result
        result.raw_text = text
        parsed = parse_structured(text) if self.structured else None
        result.structured = parsed
        result.answer = parsed["answer"].strip() if parsed else text
        result.status = "ok" if result.finish_reason != "length" else "truncated"
        return result

    def run_with_failover(
        self, keys: Sequence[str], question: str, images: Sequence[ImageSource], **kwargs: Any
    ) -> tuple[Optional[VLMResult], list[VLMResult]]:
        """Try models in order; return the first usable answer and all attempts."""
        attempts = []
        for key in keys:
            res = self.run(key, question, images, **kwargs)
            attempts.append(res)
            if res.status == "ok":
                return res, attempts
        usable = [a for a in attempts if a.answer]
        return (usable[0] if usable else None), attempts
