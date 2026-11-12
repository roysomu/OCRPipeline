"""RepLiQA loading; reproducible PDF-level proportional stratification.

Gold labels remain on the caller's side and are never passed into inference prompts.
"""
from collections import defaultdict
import hashlib
import json
import math
import os
import random
from pathlib import Path
from functools import lru_cache
import pandas as pd
from huggingface_hub import hf_hub_download
from config import ALLOWED_SPLITS, REPO_ID, Settings
from reference_labels import normalise_reference_answer


def _hf_download(**kwargs):
    """Download with an optional Hugging Face token loaded from the environment.

    Keep the credential out of ``Settings`` because settings are exported to run
    metadata. When HF_TOKEN is unset, huggingface_hub retains its normal behavior,
    including use of a token previously saved by ``hf auth login``.
    """
    token = os.getenv("HF_TOKEN")
    if token:
        kwargs["token"] = token
    return hf_hub_download(**kwargs)


def group_records(records, split: str) -> list[dict]:
    if split not in ALLOWED_SPLITS:
        raise ValueError("Invalid split")
    groups = {}
    for r in records:
        did = str(r["document_id"])
        if did not in groups:
            groups[did] = {
                "document_id": did, "split": split,
                "topic": str(r["document_topic"]),
                "pdf_path": str(r["document_path"]),
                "reference_text": str(r["document_extracted"] or ""),
                "questions": [],
            }
        raw_answer = str(r["answer"] or "")
        raw_long_answer = str(r.get("long_answer") or "")
        answer = normalise_reference_answer(raw_answer, raw_long_answer)
        groups[did]["questions"].append({
            "question_id": str(r["question_id"]),
            "question": str(r["question"]),
            "answer": answer,
            "long_answer": "" if answer == "UNANSWERABLE" else raw_long_answer,
            "answer_raw": raw_answer,
            "long_answer_raw": raw_long_answer,
        })
    return [groups[k] for k in sorted(groups)]


def proportional_sample(groups: list[dict], n_docs: int, seed: int) -> list[dict]:
    """Largest-remainder allocation across topics, then seeded selection within each.

    This preserves category proportions without selecting 'first N' dataset rows.
    Optional one-per-class minimum for sufficiently large samples ensures coverage;
    the resulting slight deviation should be disclosed when interpreting results.
    """
    if not 1 <= n_docs <= len(groups):
        raise ValueError(f"n_docs must be within 1..{len(groups)}")
    buckets = defaultdict(list)
    for g in groups:
        buckets[g["topic"]].append(g)
    names = sorted(buckets)
    total = len(groups)
    if n_docs == total:
        return list(groups)
    target = {t: n_docs * len(buckets[t]) / total for t in names}
    quotas = {t: int(math.floor(target[t])) for t in names}
    if n_docs >= len(names):
        for t in names:
            quotas[t] = max(1, quotas[t])
    while sum(quotas.values()) > n_docs:
        t = min((t for t in names if quotas[t] > (1 if n_docs >= len(names) else 0)),
                key=lambda t: (target[t] - quotas[t], t))
        quotas[t] -= 1
    while sum(quotas.values()) < n_docs:
        t = max((t for t in names if quotas[t] < len(buckets[t])),
                key=lambda t: (target[t] - quotas[t], t))
        quotas[t] += 1
    rng = random.Random(seed)
    selected = []
    for t in names:
        candidates = sorted(buckets[t], key=lambda x: x["document_id"])
        selected.extend(rng.sample(candidates, quotas[t]))
    rng.shuffle(selected)
    assert len(selected) == n_docs
    return selected


def manifest(split: str, n_docs: int, seed: int = 42, settings: Settings | None = None) -> list[dict]:
    if split not in ALLOWED_SPLITS:
        raise ValueError("Only repliqa_0 through repliqa_3 are allowed")
    settings = settings or Settings()
    path = _hf_download(repo_id=REPO_ID, repo_type="dataset",
        filename=f"data/{split}-00000-of-00001.parquet", revision=settings.dataset_revision)
    df = pd.read_parquet(path)
    groups = group_records(df.to_dict("records"), split)
    # hf_hub_download resolves 'main' to a content-addressed snapshot; pin the
    # selected PDF downloads to the SAME commit even if upstream main changes.
    parts = Path(path).parts
    resolved_revision = (parts[parts.index("snapshots") + 1]
                         if "snapshots" in parts and parts.index("snapshots") + 1 < len(parts)
                         else settings.dataset_revision)
    for group in groups:
        group["dataset_revision"] = resolved_revision
    return proportional_sample(groups, n_docs, seed)


def pdf_download(group: dict, settings: Settings | None = None) -> Path:
    settings = settings or Settings()
    path = str(group["pdf_path"])
    # An untrusted row should never point outside the expected repository directory.
    if not path.startswith(f"pdfs/{group['split']}/") or not path.endswith(".pdf"):
        raise ValueError("Unexpected PDF path")
    if ".." in Path(path).parts or "\\" in path:
        raise ValueError("Unsafe relative PDF path")
    return Path(_hf_download(repo_id=REPO_ID, repo_type="dataset",
                             filename=path, revision=group.get("dataset_revision", settings.dataset_revision)))


def selection_digest(groups: list[dict], seed: int, settings: Settings | None = None) -> str:
    settings = settings or Settings()
    doc_ids = sorted(g["document_id"] for g in groups)
    actual_revisions = sorted({g.get("dataset_revision", settings.dataset_revision) for g in groups})
    payload = {"doc_ids": doc_ids, "seed": seed, "revision": actual_revisions}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
