from __future__ import annotations

from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BENCHMARK_ROOT.parent

CASES_PATH = BENCHMARK_ROOT / "cases.json"
RAW_DIR = BENCHMARK_ROOT / "data" / "raw"
PROCESSED_DIR = BENCHMARK_ROOT / "data" / "processed"
ARTIFACTS_DIR = BENCHMARK_ROOT / "artifacts"
RUNS_DIR = BENCHMARK_ROOT / "runs"
RAW_API_DIR = RUNS_DIR / "raw_api"
PROMPTS_DIR = BENCHMARK_ROOT / "prompts"
CONFIG_DIR = BENCHMARK_ROOT / "config"
VLM_PROMPT_PATH = PROMPTS_DIR / "vlm_prompt_v1.txt"
JUDGE_PROMPT_PATH = PROMPTS_DIR / "judge_prompt_v1.txt"
MODELS_YAML = CONFIG_DIR / "models.yaml"
BENCHMARK_YAML = CONFIG_DIR / "benchmark.yaml"
RESPONSES_PATH = RUNS_DIR / "raw_responses.jsonl"
JUDGE_RESULTS_PATH = RUNS_DIR / "judge_results.jsonl"
MANIFEST_PATH = RUNS_DIR / "run_manifest.json"
RESOLVED_MODELS_PATH = ARTIFACTS_DIR / "resolved_models.json"
AVAILABILITY_JSON = ARTIFACTS_DIR / "model_availability.json"
AVAILABILITY_CSV = ARTIFACTS_DIR / "model_availability.csv"
SMOKE_MD = ARTIFACTS_DIR / "smoke_test_results.md"
PAIRING_VALIDATION_PATH = ARTIFACTS_DIR / "pairing_validation.json"
PREPROCESS_LOG_PATH = ARTIFACTS_DIR / "preprocessing_log.json"
CONTACT_SHEET_PATH = ARTIFACTS_DIR / "preflight_contact_sheet.jpg"
JOINED_RESULTS_PATH = ARTIFACTS_DIR / "joined_results.json"
FIGURE_DATA_PATH = ARTIFACTS_DIR / "figure_data.json"
INTEGRITY_PATH = ARTIFACTS_DIR / "integrity_audit.json"

CASE_IDS = ("E1", "E2", "M1", "M2", "H1", "H2")
DIFFICULTIES = ("easy", "medium", "hard")
LOGICAL_MODEL_ORDER = (
    "OPENAI_FRONTIER_VISION",
    "GEMINI",
    "CLAUDE",
    "QWEN",
    "GEMMA",
    "LLAMA",
    "MISTRAL",
    "KIMI",
)
