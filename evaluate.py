"""Evaluation runner with durable per-document checkpointing for every RepLiQA split.

python evaluate.py --split repliqa_3 --docs 500 --seed 42 --bootstrap 1000
python evaluate.py --split repliqa_0 --docs 10 --seed 42 --judge
"""
import argparse
import csv
import json
from pathlib import Path
import sys

from config import RESULTS_DIR, Settings
from dataset import manifest, selection_digest
from metrics import (METRICS_VERSION, score_document, aggregate, topic_breakdown,
                     cluster_bootstrap_ci, assignment_accuracy_metrics)
from pipeline import (process_document, evaluate_semantic, stable_hash,
                      TOPIC_SYSTEM_PROMPT, QA_SYSTEM_PROMPT, JUDGE_SYSTEM_PROMPT,
                      topic_request_config, atomic_json, read_json,
                      topic_embedding_config, topic_runner_up_large_config)
from schemas import DocumentAnalysis, QuestionAnswers, TopicClassification


def _csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_signature(groups, split, seed, settings, use_judge):
    effective_topic_request = topic_request_config(settings)
    payload = {"split": split, "selection": selection_digest(groups, seed, settings),
        "metrics_version": METRICS_VERSION,
        "judge": use_judge, "topic": settings.topic_model,
        "topic_request_config": effective_topic_request,
        "chat": settings.chat_model, "ocr": settings.ocr_model,
        "judge_model": settings.judge_model if use_judge else None,
        "schemas": {"topic": TopicClassification.model_json_schema(),
                    "qa": QuestionAnswers.model_json_schema(),
                    "combined": DocumentAnalysis.model_json_schema()},
        "prompts": {"topic": TOPIC_SYSTEM_PROMPT, "qa": QA_SYSTEM_PROMPT},
        "judge_prompt": JUDGE_SYSTEM_PROMPT if use_judge else None,
        "dataset_revision": settings.dataset_revision}
    embedding = topic_embedding_config(settings)
    if embedding["mode"] != "off":
        payload["topic_embedding"] = embedding
    refinement = topic_runner_up_large_config(settings)
    if refinement["enabled"]:
        payload["topic_runner_up_large_experiment"] = refinement
    return stable_hash(payload)


def iterate_evaluation(split="repliqa_0", n_docs=10, seed=42, use_judge=False,
                       bootstrap=1000, settings=None, force=False):
    """Yield status updates, including a final summary and CSV paths.

    A cached scored JSON is only reused for this exact run signature. Interrupted
    runs resume from already checkpointed documents; failures are explicitly reported.
    """
    settings = settings or Settings()
    groups = manifest(split, n_docs, seed, settings)
    signature = run_signature(groups, split, seed, settings, use_judge)
    resolved_revisions = sorted({g.get("dataset_revision", settings.dataset_revision) for g in groups})
    run_dir = RESULTS_DIR / f"{split}_{len(groups)}docs_seed{seed}_{signature}"
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(run_dir / "run_config.json", {
        "dataset": split, "requested_documents": n_docs, "seed": seed,
        "sampling": "proportional stratification by document topic",
        "selection_digest": selection_digest(groups, seed, settings),
        "resolved_dataset_revision": resolved_revisions,
        "document_ids": [x["document_id"] for x in groups],
        "settings": settings.to_dict(), "run_signature": signature,
        "metrics_version": METRICS_VERSION,
        "judge_enabled": use_judge,
        "notice": "Split selection and evaluation governance are user-managed. API usage is estimated, not invoiced."})
    docs = []
    errors = []
    judge_errors = []
    for i, group in enumerate(groups, 1):
        dpath = run_dir / "documents" / f"{group['document_id']}.json"
        try:
            scored = None if force else read_json(dpath)
            if scored is not None:
                # A reused checkpoint did not spend API credits on this execution.
                scored["this_run_fresh_api_estimated_usd"] = 0.0
                scored["judge_fresh_estimated_usd"] = 0.0
                scored["ocr_cached"] = True
                scored["topic_cached"] = True
                scored["qa_cached"] = True
                if use_judge:
                    scored["judge_cached"] = not bool(scored.get("judge_error"))
            if scored is None:
                result = process_document(group, settings=settings, force_ocr=force, force_qa=force)
                judged = None
                judge_error = None
                if use_judge:
                    try:
                        judged = evaluate_semantic(group, result, settings=settings, force=force)
                    except Exception as exc:
                        judge_error = f"{type(exc).__name__}: {str(exc)[:350]}"
                scored = score_document(group, result, judged)
                scored["judge_error"] = judge_error
                atomic_json(dpath, scored)
            if scored.get("judge_error"):
                judge_errors.append({"document_id": group["document_id"],
                                     "error": scored["judge_error"]})
            docs.append(scored)
            suffix = " (optional judge unavailable)" if scored.get("judge_error") else ""
            msg = f"{i}/{len(groups)}: {group['document_id']} scored{suffix}"
        except Exception as exc:
            errors.append({"document_id": group["document_id"], "error": str(exc)[:400]})
            msg = f"{i}/{len(groups)}: {group['document_id']} FAILED ({type(exc).__name__})"
        if i == 1 or i % 10 == 0 or i == len(groups):
            partial = aggregate(docs, incumbent_per_page=settings.incumbent_per_page)
            partial.update({"split": split, "requested_documents": len(groups),
                            "evaluated_documents": len(docs), "failures": errors,
                            "judge_failures": judge_errors,
                            "status": "PARTIAL" if i < len(groups) or errors else "SCORING_COMPLETE"})
            atomic_json(run_dir / "summary.json", partial)
            atomic_json(run_dir / "failures.json", errors)
            atomic_json(run_dir / "judge_failures.json", judge_errors)
        yield {"stage": "progress", "message": msg,
               "completed": i, "total": len(groups), "scored": len(docs),
               "failures": len(errors), "judge_failures": len(judge_errors),
               "dir": str(run_dir)}
    summary = aggregate(docs, incumbent_per_page=settings.incumbent_per_page)
    ci = cluster_bootstrap_ci(docs, n_boot=bootstrap, seed=seed) if len(docs) >= 5 and bootstrap >= 20 else {}
    summary.update({"split": split, "requested_documents": len(groups),
        "evaluated_documents": len(docs), "failed_documents": len(errors),
        "judge_failed_documents": len(judge_errors),
        "status": "COMPLETE" if not errors and len(docs) == len(groups) else "INCOMPLETE",
        "confidence_intervals": ci, "selection_digest": selection_digest(groups, seed, settings),
        "dataset_revision": resolved_revisions, "judge_enabled": use_judge,
        "metrics_version": METRICS_VERSION,
        "model_config": settings.fingerprint_fields(),
        "sampling_note": "Proportionally stratified documents; all five questions kept together; bootstrap resamples PDF clusters."})
    summary.update(assignment_accuracy_metrics(summary))
    atomic_json(run_dir / "summary.json", summary)
    atomic_json(run_dir / "failures.json", errors)
    atomic_json(run_dir / "judge_failures.json", judge_errors)
    _csv(run_dir / "per_topic.csv", topic_breakdown(docs))
    document_rows = [{k: v for k, v in d.items() if k != "questions"} for d in docs]
    _csv(run_dir / "per_document.csv", document_rows)
    question_rows = [{"document_id": d["document_id"],
        "topic_gold": d["topic_gold"], "topic_pred": d["topic_pred"], **q}
        for d in docs for q in d["questions"]]
    _csv(run_dir / "per_question.csv", question_rows)
    yield {"stage": "complete", "message": summary["status"], "summary": summary,
        "dir": str(run_dir), "summary_path": str(run_dir / "summary.json"),
        "question_csv": str(run_dir / "per_question.csv") if question_rows else None,
        "topic_csv": str(run_dir / "per_topic.csv") if docs else None,
        "errors": errors, "judge_errors": judge_errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["repliqa_0", "repliqa_1", "repliqa_2", "repliqa_3"], default="repliqa_0")
    parser.add_argument("--docs", type=int, default=10, help="Unique PDFs (1..3591)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--judge", action="store_true", help="Use separately charged optional Large reference-alignment judge")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--force", action="store_true", help="Rerun API calls (may incur charges)")
    args = parser.parse_args()
    complete = None
    for event in iterate_evaluation(args.split, args.docs, args.seed, args.judge, args.bootstrap,
                                    force=args.force):
        if event["stage"] == "progress":
            print(event["message"], flush=True)
        else:
            complete = event
    if complete:
        print(json.dumps({"results_path": complete["dir"], "summary": complete["summary"]}, indent=2))
    if not complete or complete["summary"]["status"] != "COMPLETE":
        sys.exit(1)

if __name__ == "__main__":
    main()
