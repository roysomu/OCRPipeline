"""Pinned/configurable inference configuration and independently auditable pricing."""
from dataclasses import dataclass, asdict
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true/false, 1/0, yes/no, or on/off")

REPO_ID = "ServiceNow/repliqa"
ALLOWED_SPLITS = ("repliqa_0", "repliqa_1", "repliqa_2", "repliqa_3")
TOPICS = (
    "Company Policies", "Cybersecurity News", "Local Technology and Innovation",
    "Local Environmental Issues", "Regional Folklore and Myths",
    "Local Politics and Governance", "News Stories", "Local Economy and Market",
    "Local Education Systems", "Local Arts and Culture", "Local News",
    "Small and Medium Enterprises", "Incident Report", "Regional Cuisine and Recipes",
    "Neighborhood Stories", "Local Sports and Activities", "Local Health and Wellness",
)

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / ".cache" / "v2"
RESULTS_DIR = ROOT / "results"

@dataclass(frozen=True)
class Settings:
    # Prefer release identifiers to mutable 'latest' aliases; verify availability in account.
    ocr_model: str = os.getenv("MISTRAL_OCR_MODEL", "mistral-ocr-4-1")
    topic_model: str = os.getenv("MISTRAL_TOPIC_MODEL", "mistral-small-2603")
    # Programmatic per-route override: Small supports none/high; Large supports low/medium/high.
    topic_reasoning_effort: str = "none"
    topic_refinement_model: str = os.getenv(
        "MISTRAL_TOPIC_REFINEMENT_MODEL", "mistral-large-4")
    # None preserves the historical runner-up request; Tab 2 explicitly uses medium.
    topic_refinement_reasoning_effort: str | None = None
    chat_model: str = os.getenv("MISTRAL_CHAT_MODEL", "mistral-small-2603")
    judge_model: str = os.getenv("MISTRAL_JUDGE_MODEL", "mistral-large-latest")
    embedding_model: str = os.getenv("MISTRAL_EMBEDDING_MODEL", "mistral-embed")
    # off restores the pre-embedding classifier exactly; shadow measures candidate
    # recall without changing the prompt; advisory adds the top candidates as hints.
    topic_embedding_mode: str = os.getenv("TOPIC_EMBEDDING_MODE", "off").strip().lower()
    topic_embedding_top_k: int = int(os.getenv("TOPIC_EMBEDDING_TOP_K", "3"))
    topic_embedding_max_chars: int = int(os.getenv("TOPIC_EMBEDDING_MAX_CHARS", "12000"))
    # Benchmark-only oracle experiment. It can be removed cleanly by setting this
    # false; disabled mode preserves the prior topic cache fingerprint and behavior.
    topic_runner_up_large_enabled: bool = _env_flag(
        "TOPIC_RUNNER_UP_LARGE_EXPERIMENT", True)
    dataset_revision: str = os.getenv("HF_DATASET_REVISION", "main")
    # Prices from Mistral public API pricing as checked Nov 3, 2026; editable overrides.
    price_as_of: str = os.getenv("PRICE_AS_OF", "2026-11-03")
    ocr_per_1000_pages: float = float(os.getenv("OCR_USD_PER_1000_PAGES", "4"))
    topic_input_per_million: float = float(os.getenv("TOPIC_INPUT_USD_PER_M_TOKENS", "0.15"))
    topic_output_per_million: float = float(os.getenv("TOPIC_OUTPUT_USD_PER_M_TOKENS", "0.60"))
    chat_input_per_million: float = float(os.getenv("CHAT_INPUT_USD_PER_M_TOKENS", "0.15"))
    chat_output_per_million: float = float(os.getenv("CHAT_OUTPUT_USD_PER_M_TOKENS", "0.60"))
    judge_input_per_million: float = float(os.getenv("JUDGE_INPUT_USD_PER_M_TOKENS", "0.50"))
    judge_output_per_million: float = float(os.getenv("JUDGE_OUTPUT_USD_PER_M_TOKENS", "1.50"))
    embedding_input_per_million: float = float(os.getenv("EMBEDDING_USD_PER_M_TOKENS", "0.10"))
    incumbent_per_page: float = float(os.getenv("INCUMBENT_USD_PER_PAGE", "0.75"))
    max_document_chars: int = int(os.getenv("MAX_DOCUMENT_CHARS", "90000"))
    api_base: str = os.getenv("MISTRAL_API_BASE", "https://api.mistral.ai/v1")

    def fingerprint_fields(self):
        fields = {"ocr_model": self.ocr_model, "topic_model": self.topic_model,
                "chat_model": self.chat_model,
                "judge_model": self.judge_model, "dataset_revision": self.dataset_revision}
        if self.topic_reasoning_effort != "none":
            fields["topic_reasoning_effort"] = self.topic_reasoning_effort
        if self.topic_runner_up_large_enabled:
            refinement = {
                "enabled": True,
                "model": self.topic_refinement_model,
                "trigger": "small_primary_wrong_and_small_runner_up_equals_reference",
            }
            if self.topic_refinement_reasoning_effort is not None:
                refinement["reasoning_effort"] = self.topic_refinement_reasoning_effort
            fields["topic_runner_up_large_experiment"] = refinement
        if self.topic_embedding_mode != "off":
            fields["topic_embedding"] = {
                "mode": self.topic_embedding_mode,
                "model": self.embedding_model,
                "top_k": self.topic_embedding_top_k,
                "max_chars": self.topic_embedding_max_chars,
            }
        return fields

    def prices(self):
        return {"as_of": self.price_as_of, "ocr_per_page": self.ocr_per_1000_pages / 1000,
                "topic_input_per_token": self.topic_input_per_million / 1_000_000,
                "topic_output_per_token": self.topic_output_per_million / 1_000_000,
                "chat_input_per_token": self.chat_input_per_million / 1_000_000,
                "chat_output_per_token": self.chat_output_per_million / 1_000_000,
                "judge_input_per_token": self.judge_input_per_million / 1_000_000,
                "judge_output_per_token": self.judge_output_per_million / 1_000_000,
                "embedding_input_per_token": self.embedding_input_per_million / 1_000_000}

    def to_dict(self):
        return asdict(self)
