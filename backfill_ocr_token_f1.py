"""Backfill OCR Token F1 into recent legacy evaluation artifacts without API calls."""
from collections import Counter
import json
import os
from pathlib import Path
import re

import pandas as pd
from rapidfuzz.distance import Levenshtein

from config import CACHE_DIR, RESULTS_DIR
from metrics import (METRICS_VERSION, aggregate, cluster_bootstrap_ci,
                     markdown_visible_text, normalise, ocr_markdown_text)
from pipeline import atomic_json

COMPATIBLE_SOURCE_VERSION = "2026-11-07-partial-judge-coverage-v11"


def _reference_map(split: str) -> dict[str, str]:
    pattern = (Path.home() / ".cache" / "huggingface" / "hub" /
               "datasets--ServiceNow--repliqa" / "snapshots" / "*" /
               "data" / f"{split}-00000-of-00001.parquet")
    matches = sorted(pattern.parent.parent.parent.glob(
        f"*/data/{split}-00000-of-00001.parquet"))
    if not matches:
        raise FileNotFoundError(f"No local RepLiQA parquet cache for {split}")
    frame = pd.read_parquet(matches[-1], columns=["document_id", "document_extracted"])
    return {
        str(row.document_id): str(row.document_extracted or "")
        for row in frame.drop_duplicates("document_id").itertuples(index=False)
    }


def _token_fields(reference_text: str, ocr: dict) -> dict:
    reference = normalise(reference_text)
    prediction = normalise(markdown_visible_text(ocr_markdown_text(ocr)))
    reference_words = reference.split()
    prediction_words = prediction.split()
    overlap = sum((Counter(reference_words) & Counter(prediction_words)).values())
    return {
        "ocr_reference_words": len(reference_words),
        "ocr_prediction_words": len(prediction_words),
        "ocr_token_overlap": overlap,
        "ocr_proxy_token_precision": (
            overlap / len(prediction_words) if prediction_words else None),
        "ocr_proxy_token_recall": (
            overlap / len(reference_words) if reference_words else None),
        "ocr_proxy_token_f1": (
            2 * overlap / (len(prediction_words) + len(reference_words))
            if prediction_words or reference_words else 1.0),
        "candidate_wer": (
            Levenshtein.distance(reference_words, prediction_words) / len(reference_words)
            if reference_words else None),
        "candidate_cer": (
            Levenshtein.distance(reference, prediction) / len(reference)
            if reference else None),
    }


def _matching_ocr(split: str, document_id: str, reference_text: str,
                  scored: dict) -> tuple[dict, dict]:
    candidates = sorted((CACHE_DIR / "ocr" / split).glob(f"{document_id}-*.json"))
    matches = []
    for path in candidates:
        with path.open(encoding="utf-8") as handle:
            ocr = json.load(handle)
        fields = _token_fields(reference_text, ocr)
        wer_matches = (fields["candidate_wer"] is None
                       if scored.get("ocr_proxy_wer") is None else
                       abs(fields["candidate_wer"] - float(scored["ocr_proxy_wer"])) < 1e-12)
        cer_matches = (fields["candidate_cer"] is None
                       if scored.get("ocr_proxy_cer") is None else
                       abs(fields["candidate_cer"] - float(scored["ocr_proxy_cer"])) < 1e-12)
        if wer_matches and cer_matches:
            matches.append((ocr, fields))
    if not matches:
        raise ValueError(
            f"No cached OCR candidate reproduces stored WER/CER for {split}/{document_id}")
    return matches[0]


def backfill_run(run_dir: Path) -> float | None:
    summary_path = run_dir / "summary.json"
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("ocr_proxy_micro_token_f1") is not None:
        return float(summary["ocr_proxy_micro_token_f1"])

    split = str(summary["split"])
    references = _reference_map(split)
    updates = []
    for document_path in sorted((run_dir / "documents").glob("*.json")):
        with document_path.open(encoding="utf-8") as handle:
            scored = json.load(handle)
        document_id = str(scored["document_id"])
        _, fields = _matching_ocr(split, document_id, references[document_id], scored)
        fields.pop("candidate_wer")
        fields.pop("candidate_cer")
        scored.update(fields)
        updates.append((document_path, scored))

    if not updates:
        raise ValueError(f"No document checkpoints in {run_dir}")

    documents = [scored for _, scored in updates]
    recalculated = aggregate(documents)
    f1_keys = (
        "ocr_proxy_micro_token_precision", "ocr_proxy_micro_token_recall",
        "ocr_proxy_micro_token_f1", "ocr_proxy_document_mean_token_f1",
    )
    summary.update({key: recalculated[key] for key in f1_keys})

    confidence = summary.setdefault("confidence_intervals", {})
    replications = int(confidence.get("replications") or 1000)
    run_config_path = run_dir / "run_config.json"
    with run_config_path.open(encoding="utf-8") as handle:
        run_config = json.load(handle)
    seed = int(run_config.get("seed") or 42)
    if len(documents) >= 5 and replications >= 20:
        f1_interval = cluster_bootstrap_ci(
            documents, metrics=("ocr_proxy_micro_token_f1",),
            n_boot=replications, seed=seed)
        confidence.setdefault("intervals", {}).update(f1_interval["intervals"])

    previous_version = summary.get("metrics_version")
    summary["metrics_version"] = METRICS_VERSION
    summary["ocr_token_f1_backfill"] = {
        "source_metrics_version": previous_version,
        "method": "local_reference_and_cached_ocr_matched_by_stored_wer_cer",
        "api_calls": 0,
    }
    run_config["metrics_version"] = METRICS_VERSION
    run_config["ocr_token_f1_backfill"] = summary["ocr_token_f1_backfill"]

    summary_times = (summary_path.stat().st_atime, summary_path.stat().st_mtime)
    for document_path, scored in updates:
        atomic_json(document_path, scored)
    atomic_json(summary_path, summary)
    atomic_json(run_config_path, run_config)
    os.utime(summary_path, summary_times)
    return float(summary["ocr_proxy_micro_token_f1"])


def backfill_recent(limit: int = 6) -> list[tuple[str, float | None]]:
    runs = sorted(
        ((path.stat().st_mtime, path.parent)
         for path in RESULTS_DIR.glob("*/summary.json")),
        reverse=True,
    )[:limit]
    return [(run_dir.name, backfill_run(run_dir)) for _, run_dir in runs]


def backfill_compatible_completed_runs() -> list[tuple[str, float]]:
    """Upgrade only the immediately preceding compatible metrics schema."""
    changed = []
    for summary_path in sorted(RESULTS_DIR.glob("*/summary.json")):
        with summary_path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
        if (summary.get("status") != "COMPLETE"
                or summary.get("metrics_version") != COMPATIBLE_SOURCE_VERSION):
            continue
        score = backfill_run(summary_path.parent)
        if score is not None:
            changed.append((summary_path.parent.name, score))
    return changed


def backfill_tab1_runs() -> list[tuple[str, float]]:
    references = {}
    for split in ("repliqa_0", "repliqa_1", "repliqa_2", "repliqa_3"):
        for document_id, reference_text in _reference_map(split).items():
            references[document_id] = (split, reference_text)

    inserted_after_exact_span = re.compile(
        r'(<div class="exec-card"><div class="exec-label">Exact Span Recall</div>'
        r'<div class="exec-value [^"]+">.*?</div>'
        r'<div class="exec-detail">.*?</div></div>)',
        flags=re.DOTALL,
    )
    changed = []
    for path in sorted((RESULTS_DIR / "tab1_runs").glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            stored = json.load(handle)
        if not stored.get("topic_correct") or "OCR Token F1" in stored.get("summary_html", ""):
            continue

        document_id = str(stored["document_id"])
        _, reference_text = references[document_id]
        ocr = {"markdown_pages": stored["page_state"].get("ocr_pages") or []}
        fields = _token_fields(reference_text, ocr)
        fields.pop("candidate_wer")
        fields.pop("candidate_cer")

        metrics = stored["raw_details"]["metrics"]
        metrics["document"].update(fields)
        metrics["aggregate"].update({
            "ocr_proxy_micro_token_precision": fields["ocr_proxy_token_precision"],
            "ocr_proxy_micro_token_recall": fields["ocr_proxy_token_recall"],
            "ocr_proxy_micro_token_f1": fields["ocr_proxy_token_f1"],
            "ocr_proxy_document_mean_token_f1": fields["ocr_proxy_token_f1"],
        })

        value = f'{100 * float(fields["ocr_proxy_token_f1"]):.1f}%'
        card = (
            '<div class="exec-card"><div class="exec-label">OCR Token F1</div>'
            f'<div class="exec-value exec-good">{value}</div>'
            '<div class="exec-detail">Whole-document token overlap; '
            'order-insensitive reference-text proxy</div></div>'
        )
        summary_html, replacements = inserted_after_exact_span.subn(
            lambda match: match.group(1) + card, stored["summary_html"], count=1)
        if replacements != 1:
            raise ValueError(f"Could not locate Exact Span Recall card in {path}")
        stored["summary_html"] = summary_html
        stored["ocr_token_f1_backfill"] = {
            "method": "local_reference_and_stored_ocr_pages",
            "api_calls": 0,
        }
        original_times = (path.stat().st_atime, path.stat().st_mtime)
        atomic_json(path, stored)
        os.utime(path, original_times)
        changed.append((path.name, float(fields["ocr_proxy_token_f1"])))
    return changed


if __name__ == "__main__":
    for name, score in backfill_recent():
        print(f"{name}: {score:.6f}" if score is not None else f"{name}: n.a.")
    for name, score in backfill_compatible_completed_runs():
        print(f"aggregate/{name}: {score:.6f}")
    for name, score in backfill_tab1_runs():
        print(f"tab1/{name}: {score:.6f}")
