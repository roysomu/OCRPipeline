"""Reproducible RepLiQA metrics, per-topic analysis and document-clustered 95% CI.

No API calls. OCR WER/CER only compares visible OCR page Markdown with the
RepLiQA auto-extracted text proxy. WER is not capped at 100%.
"""
import difflib
from collections import Counter
import html
import math
import re
from statistics import mean

import numpy as np
from rapidfuzz.distance import Levenshtein
from reference_labels import is_unanswerable_reference


METRICS_VERSION = "2026-11-09-ocr-token-f1-v12"


def normalise(text):
    return " ".join(re.findall(r"[a-z0-9]+", str(text or "").lower()))


def markdown_visible_text(markdown):
    """Strip non-visible Markdown syntax, URL targets and OCR image references."""
    text = str(markdown or "")
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"<img\b[^>]*>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"(?m)^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*$", " ", text)
    text = re.sub(r"\[PAGE\s+\d+\]", " ", text)
    return text


def ocr_markdown_text(ocr):
    if "markdown_text" in ocr:
        return str(ocr["markdown_text"] or "")
    if "markdown_pages" in ocr:
        return "\n\n".join(ocr["markdown_pages"])
    return re.sub(r"\[PAGE\s+\d+\]", " ", str(ocr.get("text") or ""))


def token_f1(pred, truth):
    p, t = normalise(pred).split(), normalise(truth).split()
    if not p and not t:
        return 1.0
    if not p or not t:
        return 0.0
    overlap = sum((Counter(p) & Counter(t)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(t)
    return 2 * precision * recall / (precision + recall)


def exact_normalized_span_present(gold_span, extracted_text):
    """Return whether a non-empty normalized gold span occurs contiguously.

    Padding both strings with spaces prevents a one-token gold span such as
    ``art`` from matching inside an unrelated OCR token such as ``article``.
    ``None`` marks an empty span as ineligible rather than as an OCR failure.
    """
    gold = normalise(gold_span)
    if not gold:
        return None
    extracted = normalise(extracted_text)
    return f" {gold} " in f" {extracted} "


def score_document(group, result, judged=None):
    reference = normalise(group.get("reference_text", ""))
    ocr_visible_text = markdown_visible_text(ocr_markdown_text(result["ocr"]))
    prediction = normalise(ocr_visible_text)
    word_gold, word_pred = reference.split(), prediction.split()
    word_overlap = sum((Counter(word_gold) & Counter(word_pred)).values())
    word_errors = Levenshtein.distance(word_gold, word_pred)
    char_errors = Levenshtein.distance(reference, prediction)
    pred_map = {x["question_id"]: x for x in result["analysis"]["answers"]}
    judge_map = {x["question_id"]: x for x in (judged or {}).get("assessments", [])}
    questions = []
    for row in group["questions"]:
        pred = pred_map[row["question_id"]]
        truth = row["answer"]
        gold_abstain = is_unanswerable_reference(truth, row.get("long_answer"))
        pred_abstain = not pred["is_answerable"]
        judged_item = judge_map.get(row["question_id"])
        answerable = not gold_abstain
        gold_long_answer = row.get("long_answer", "")
        exact_span_recovered = (
            exact_normalized_span_present(gold_long_answer, ocr_visible_text)
            if answerable else None
        )
        words = len(pred["long_answer"].split())
        questions.append({
            "question_id": row["question_id"],
            "question": row["question"],
            "gold_answer": truth,
            "predicted_short": pred["short_answer"],
            "predicted_long": pred["long_answer"],
            "gold_unanswerable": gold_abstain,
            "predicted_unanswerable": pred_abstain,
            "correct_answerability": int(gold_abstain == pred_abstain),
            "answerable_em": int(normalise(truth) == normalise(pred["short_answer"])) if answerable else None,
            "answerable_token_f1": token_f1(pred["short_answer"], truth) if answerable else None,
            "gold_long_answer_word_count": (len(normalise(gold_long_answer).split())
                                              if exact_span_recovered is not None else None),
            "gold_long_answer_span_recovered": exact_span_recovered,
            "evidence_quote": pred["evidence_quote"],
            "evidence_page": pred["evidence_page"],
            "evidence_quote_verified": bool(pred.get("evidence_verified", False)) if not pred_abstain else None,
            "long_answer_word_count": words,
            "long_answer_length_ok": 20 <= words <= 55 if not pred_abstain else None,
            "judge_semantic_alignment_1_to_5": judged_item["semantic_alignment_score"] if judged_item else None,
            "judge_reference_supported": judged_item["reference_supported"] if judged_item else None,
            "judge_rationale": judged_item["rationale"] if judged_item else None,
        })
    full_cost = result["cost"]["full_inference_estimated_usd"]
    fresh_cost = result["cost"]["this_run_fresh_api_estimated_usd"]
    judge_full = (judged or {}).get("estimated_full_judge_cost_usd", 0.0)
    judge_fresh = (judged or {}).get("this_run_fresh_judge_cost_usd", 0.0)
    topic_details = result.get("topic_classification") or {}
    small_topic_details = result.get("topic_small_classification") or topic_details
    refinement = result.get("topic_refinement") or {}
    embedding = result.get("topic_embedding") or {}
    embedding_candidates = [item.get("label") for item in embedding.get("candidates", [])
                            if item.get("label")]
    span_results = [q["gold_long_answer_span_recovered"] for q in questions
                    if q["gold_long_answer_span_recovered"] is not None]
    return {
        "document_id": group["document_id"], "topic_gold": group["topic"],
        "topic_pred": result["analysis"]["topic"],
        "topic_correct": int(group["topic"] == result["analysis"]["topic"]),
        "topic_small_pred": small_topic_details.get("topic"),
        "topic_small_runner_up": small_topic_details.get("runner_up"),
        "topic_runner_up_large_status": refinement.get("status"),
        "topic_runner_up_large_triggered": bool(refinement.get("oracle_triggered")),
        "topic_runner_up_large_fallback": bool(refinement.get("fallback_used")),
        "topic_final_source": refinement.get("final_source", "small"),
        "topic_document_type": topic_details.get("document_type"),
        "topic_primary_subject": topic_details.get("primary_subject"),
        "topic_runner_up": topic_details.get("runner_up"),
        "topic_decisive_reason": topic_details.get("decisive_reason"),
        "topic_embedding_mode": embedding.get("mode"),
        "topic_embedding_candidates": embedding_candidates or None,
        "topic_embedding_gold_in_top_k": (int(group["topic"] in embedding_candidates)
                                           if embedding_candidates else None),
        "ocr_word_errors": word_errors, "ocr_reference_words": len(word_gold),
        "ocr_prediction_words": len(word_pred), "ocr_token_overlap": word_overlap,
        "ocr_proxy_token_precision": (word_overlap / len(word_pred) if word_pred else None),
        "ocr_proxy_token_recall": (word_overlap / len(word_gold) if word_gold else None),
        "ocr_proxy_token_f1": (
            2 * word_overlap / (len(word_pred) + len(word_gold))
            if word_pred or word_gold else 1.0),
        "ocr_char_errors": char_errors, "ocr_reference_chars": len(reference),
        "ocr_proxy_wer": word_errors / len(word_gold) if word_gold else None,
        "ocr_proxy_cer": char_errors / len(reference) if reference else None,
        "ocr_wer_input": "visible pages[].markdown only (no blocks / page markers / answer text)",
        "exact_span_recovered_questions": sum(bool(value) for value in span_results),
        "exact_span_eligible_questions": len(span_results),
        "exact_span_recall": _mean(span_results),
        "pages": result["ocr"]["pages"],
        "ocr_processing_seconds": result["ocr"].get("processing_seconds"),
        "ocr_processing_seconds_per_page": result["ocr"].get(
            "processing_seconds_per_page"),
        "ocr_processing_timing_scope": result["ocr"].get("processing_timing_scope"),
        "full_inference_estimated_usd": full_cost,
        "ocr_estimated_usd": result["cost"].get("ocr_usd", 0.0),
        "topic_refinement_estimated_usd": result["cost"].get("topic_refinement_usd", 0.0),
        "this_run_fresh_api_estimated_usd": fresh_cost,
        "judge_full_estimated_usd": judge_full,
        "judge_fresh_estimated_usd": judge_fresh,
        "latency_seconds": result.get("seconds"),
        "ocr_cached": result.get("ocr_cached", False),
        "topic_cached": result.get("topic_cached", False),
        "qa_cached": result.get("qa_cached", False),
        "judge_cached": (judged or {}).get("cached"),
        "questions": questions,
    }


def _ratio(num, den):
    return num / den if den else None


def _mean(items):
    return sum(items) / len(items) if items else None


def assignment_accuracy_metrics(summary, requested_documents=None):
    """Return the declared three-pillar assignment score and auditable components.

    Text, strict topic classification, and long-form answering receive equal weight.
    Processing success multiplies the result so failed PDFs are not silently excluded.
    At least one successful optional-judge assessment is required because semantic
    alignment carries most of the long-form component. Failed judge assessments are
    excluded from the judge mean and reported through an explicit coverage rate.
    """
    def bounded(value):
        return max(0.0, min(1.0, float(value)))

    requested = int(requested_documents or summary.get("requested_documents")
                    or summary.get("documents") or 0)
    evaluated = int(summary.get("evaluated_documents")
                    if summary.get("evaluated_documents") is not None
                    else summary.get("documents") or 0)
    questions = int(summary.get("questions") or 0)
    judge_scored = int(summary.get("judge_scored_questions") or 0)
    base = {
        "assignment_accuracy": None,
        "assignment_text_extraction_score": None,
        "assignment_topic_score": None,
        "assignment_long_form_score": None,
        "assignment_processing_success_rate": (evaluated / requested if requested else None),
        "assignment_judge_coverage_rate": (judge_scored / questions if questions else None),
        "assignment_judge_scored_questions": judge_scored,
        "assignment_judge_total_questions": questions,
        "assignment_accuracy_status": "unavailable",
        "assignment_accuracy_formula": (
            "processing_success * mean(text, topic, long_form); "
            "text=.60*(1-WER)+.40*(1-CER); "
            "topic=strict_primary_label_accuracy; "
            "long_form=.70*normalized_successful_judges+.15*answerability+"
            ".10*evidence+.05*length; failed_judges_excluded"
        ),
    }
    required = (summary.get("ocr_proxy_micro_wer"), summary.get("ocr_proxy_micro_cer"),
                summary.get("topic_accuracy"),
                summary.get("judge_semantic_alignment_mean_1_to_5"),
                summary.get("answerability_accuracy"))
    if not requested or not questions or any(value is None for value in required):
        base["assignment_accuracy_status"] = (
            "requires_core_metrics_and_at_least_one_successful_optional_judge")
        return base
    word_accuracy = bounded(1.0 - float(summary["ocr_proxy_micro_wer"]))
    character_accuracy = bounded(1.0 - float(summary["ocr_proxy_micro_cer"]))
    text_score = 0.60 * word_accuracy + 0.40 * character_accuracy
    topic_score = bounded(summary["topic_accuracy"])
    normalized_judge = bounded(
        (float(summary["judge_semantic_alignment_mean_1_to_5"]) - 1.0) / 4.0)
    # Evidence and length are not applicable when every prediction correctly abstains.
    evidence = (1.0 if summary.get("evidence_quote_location_rate") is None
                else bounded(summary["evidence_quote_location_rate"]))
    length = (1.0 if summary.get("long_answer_target_length_rate") is None
              else bounded(summary["long_answer_target_length_rate"]))
    long_form_score = (0.70 * normalized_judge
                       + 0.15 * bounded(summary["answerability_accuracy"])
                       + 0.10 * evidence + 0.05 * length)
    success_rate = bounded(evaluated / requested)
    base.update({
        "assignment_accuracy": success_rate * (text_score + topic_score + long_form_score) / 3,
        "assignment_text_extraction_score": text_score,
        "assignment_topic_score": topic_score,
        "assignment_long_form_score": long_form_score,
        "assignment_processing_success_rate": success_rate,
        "assignment_accuracy_status": (
            "available" if judge_scored == questions
            else "available_partial_optional_judge_coverage"),
    })
    return base


def _percentile(values, quantile):
    a = sorted(float(x) for x in values if x is not None)
    if not a:
        return None
    pos = (len(a) - 1) * quantile
    l = int(pos)
    return round(a[l] + (pos - l) * (a[min(l + 1, len(a) - 1)] - a[l]), 3)


def aggregate(documents, incumbent_per_page=0.75):
    """Micro OCR WER, question-level QA and one topic decision per PDF."""
    if not documents:
        return {"documents": 0, "questions": 0}
    q = [item for d in documents for item in d["questions"]]
    valid_q = [x for x in q if not x["gold_unanswerable"]]
    topic_names = sorted({d["topic_gold"] for d in documents})
    f1s = []
    for name in topic_names:
        tp = sum(d["topic_gold"] == name and d["topic_pred"] == name for d in documents)
        fp = sum(d["topic_gold"] != name and d["topic_pred"] == name for d in documents)
        fn = sum(d["topic_gold"] == name and d["topic_pred"] != name for d in documents)
        p = _ratio(tp, tp + fp) or 0
        r = _ratio(tp, tp + fn) or 0
        f1s.append(2*p*r / (p+r) if p+r else 0)
    abst_tp = sum(x["gold_unanswerable"] and x["predicted_unanswerable"] for x in q)
    abst_fp = sum(not x["gold_unanswerable"] and x["predicted_unanswerable"] for x in q)
    abst_fn = sum(x["gold_unanswerable"] and not x["predicted_unanswerable"] for x in q)
    evidence = [x["evidence_quote_verified"] for x in q if x["evidence_quote_verified"] is not None]
    lengths = [x["long_answer_length_ok"] for x in q if x["long_answer_length_ok"] is not None]
    exact_spans = [x.get("gold_long_answer_span_recovered") for x in q
                   if x.get("gold_long_answer_span_recovered") is not None]
    judge_scores = [x["judge_semantic_alignment_1_to_5"] for x in q if x["judge_semantic_alignment_1_to_5"] is not None]
    judge_failed_documents = sum(bool(d.get("judge_error")) for d in documents)
    runner_up_correct = sum(
        d["topic_pred"] != d["topic_gold"]
        and d.get("topic_runner_up") == d["topic_gold"]
        for d in documents
    )
    refinement_triggered = sum(bool(d.get("topic_runner_up_large_triggered"))
                               for d in documents)
    refinement_fallbacks = sum(bool(d.get("topic_runner_up_large_fallback"))
                               for d in documents)
    refinement_large_final = sum(d.get("topic_final_source") == "large"
                                 for d in documents)
    refinement_improved = sum(
        bool(d.get("topic_runner_up_large_triggered"))
        and d.get("topic_small_pred") != d["topic_gold"]
        and d["topic_pred"] == d["topic_gold"]
        for d in documents
    )
    embedding_docs = [d for d in documents
                      if d.get("topic_embedding_gold_in_top_k") is not None]
    embedding_modes = sorted({d.get("topic_embedding_mode") for d in embedding_docs
                              if d.get("topic_embedding_mode")})
    embedding_top_ks = sorted({len(d.get("topic_embedding_candidates") or [])
                               for d in embedding_docs})
    pages = sum(d["pages"] for d in documents)
    timed_ocr_docs = [d for d in documents
                      if d.get("ocr_processing_seconds") is not None]
    timed_ocr_pages = sum(d["pages"] for d in timed_ocr_docs)
    timed_ocr_seconds = sum(float(d["ocr_processing_seconds"])
                            for d in timed_ocr_docs)
    total = sum(d["full_inference_estimated_usd"] for d in documents)
    ocr_total = sum(float(d.get("ocr_estimated_usd") or 0) for d in documents)
    judge_total = sum(d["judge_full_estimated_usd"] for d in documents)
    token_overlap = sum(int(d.get("ocr_token_overlap") or 0) for d in documents)
    prediction_words = sum(int(d.get("ocr_prediction_words") or 0) for d in documents)
    reference_words = sum(int(d.get("ocr_reference_words") or 0) for d in documents)
    return {
        "documents": len(documents), "questions": len(q), "pages": pages,
        "ocr_timed_documents": len(timed_ocr_docs),
        "ocr_timed_pages": timed_ocr_pages,
        "ocr_processing_seconds_total": timed_ocr_seconds if timed_ocr_docs else None,
        "ocr_processing_seconds_per_page": (
            timed_ocr_seconds / timed_ocr_pages if timed_ocr_pages else None),
        "topic_accuracy": _mean([d["topic_correct"] for d in documents]),
        "topic_correct_in_runner_up_count": runner_up_correct,
        "topic_correct_in_runner_up_rate": _ratio(runner_up_correct, len(documents)),
        "topic_runner_up_large_triggered_count": refinement_triggered,
        "topic_runner_up_large_fallback_count": refinement_fallbacks,
        "topic_runner_up_large_final_count": refinement_large_final,
        "topic_runner_up_large_improved_count": refinement_improved,
        "topic_embedding_mode": ", ".join(embedding_modes) if embedding_modes else None,
        "topic_embedding_documents": len(embedding_docs),
        "topic_embedding_top_k": embedding_top_ks[0] if len(embedding_top_ks) == 1 else None,
        "topic_embedding_top_k_recall": _mean(
            [d["topic_embedding_gold_in_top_k"] for d in embedding_docs]),
        "topic_macro_f1_observed_classes": _mean(f1s),
        "topic_classes_observed": len(topic_names),
        "qa_answerable_em": _mean([x["answerable_em"] for x in valid_q]),
        "qa_answerable_token_f1": _mean([x["answerable_token_f1"] for x in valid_q]),
        "answerability_accuracy": _mean([x["correct_answerability"] for x in q]),
        "abstention_precision": _ratio(abst_tp, abst_tp + abst_fp),
        "abstention_recall": _ratio(abst_tp, abst_tp + abst_fn),
        "evidence_quote_location_rate": _mean(evidence),
        "long_answer_target_length_rate": _mean(lengths),
        "exact_span_recovered_questions": sum(bool(value) for value in exact_spans),
        "exact_span_eligible_questions": len(exact_spans),
        "exact_span_recall": _mean(exact_spans),
        "ocr_proxy_micro_wer": _ratio(sum(d["ocr_word_errors"] for d in documents),
                                      sum(d["ocr_reference_words"] for d in documents)),
        "ocr_proxy_micro_cer": _ratio(sum(d["ocr_char_errors"] for d in documents),
                                      sum(d["ocr_reference_chars"] for d in documents)),
        "ocr_proxy_micro_token_precision": _ratio(token_overlap, prediction_words),
        "ocr_proxy_micro_token_recall": _ratio(token_overlap, reference_words),
        "ocr_proxy_micro_token_f1": (
            2 * token_overlap / (prediction_words + reference_words)
            if prediction_words or reference_words else 1.0),
        "ocr_proxy_document_mean_token_f1": _mean([
            d["ocr_proxy_token_f1"] for d in documents
            if d.get("ocr_proxy_token_f1") is not None]),
        "ocr_proxy_document_mean_wer": _mean([d["ocr_proxy_wer"] for d in documents if d["ocr_proxy_wer"] is not None]),
        "judge_semantic_alignment_mean_1_to_5": _mean(judge_scores),
        "judge_semantic_alignment_pass_rate_ge_4": _mean([score >= 4 for score in judge_scores]),
        "judge_scored_questions": len(judge_scores),
        "judge_failed_documents": judge_failed_documents,
        "latency_p50_sec_uncached": _percentile([d["latency_seconds"] for d in documents
             if not d["ocr_cached"] or not d.get("topic_cached", False) or not d["qa_cached"]], 0.5),
        "latency_p95_sec_uncached": _percentile([d["latency_seconds"] for d in documents
             if not d["ocr_cached"] or not d.get("topic_cached", False) or not d["qa_cached"]], 0.95),
        "cached_document_results": sum(d["ocr_cached"] and d.get("topic_cached", False)
                                       and d["qa_cached"] for d in documents),
        "full_run_api_cost_estimated_usd": round(total, 6),
        "ocr_api_cost_estimated_usd": round(ocr_total, 6),
        "fresh_api_cost_in_current_execution_estimated_usd": round(sum(d["this_run_fresh_api_estimated_usd"] for d in documents), 6),
        "judge_api_cost_estimated_usd": round(judge_total, 6),
        "judge_fresh_cost_estimated_usd": round(sum(d["judge_fresh_estimated_usd"] for d in documents), 6),
        "full_run_cost_per_page_estimated_usd": _ratio(total, pages),
        "incumbent_reported_api_cost_for_same_pages_usd": round(pages * incumbent_per_page, 2),
        "modeled_api_only_cost_reduction_pct": (100 * (1 - total / (pages * incumbent_per_page))) if pages and incumbent_per_page else None,
        "interpretation": "OCR is proxy WER/CER against auto-extracted text. Judge is 1-5 alignment, NOT accuracy. Incumbent's ~85% is undefined and not directly comparable. Cost is modeled from returned tokens and configured prices, NOT billed spend."
    }


def topic_breakdown(documents):
    by_topic = {}
    for doc in documents:
        by_topic.setdefault(doc["topic_gold"], []).append(doc)
    rows = []
    for topic, docs in sorted(by_topic.items()):
        a = aggregate(docs)
        rows.append({"topic": topic, "documents": a["documents"],
                    "questions": a["questions"], "topic_accuracy": a["topic_accuracy"],
                    "qa_answerable_token_f1": a["qa_answerable_token_f1"],
                    "answerability_accuracy": a["answerability_accuracy"],
                    "exact_span_recall": a["exact_span_recall"],
                    "ocr_proxy_micro_wer": a["ocr_proxy_micro_wer"]})
    return rows


def cluster_bootstrap_ci(documents, metrics=None, n_boot=1000, seed=42, stratify=True):
    """Resample whole PDFs, retaining all five questions. Stratify by gold topic.

    No finite population correction; CI estimates variability across sampled
    documents, not uncertainty in the deterministic complete fixed 3,591-PDF set.
    """
    if not documents:
        return {}
    if n_boot < 20:
        raise ValueError("Use at least 20 bootstrap replicates")
    metrics = metrics or ("topic_accuracy", "qa_answerable_token_f1", "qa_answerable_em",
                          "answerability_accuracy", "exact_span_recall",
                          "ocr_proxy_micro_wer", "ocr_proxy_micro_cer",
                          "ocr_proxy_micro_token_f1")
    rng = np.random.default_rng(seed)
    if stratify:
        by_topic = {}
        for index, doc in enumerate(documents):
            by_topic.setdefault(doc["topic_gold"], []).append(index)
        clusters = [np.asarray(indices) for indices in by_topic.values()]
    else:
        clusters = [np.arange(len(documents))]
    estimates = {name: [] for name in metrics}
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(c, size=len(c), replace=True) for c in clusters])
        scored = aggregate([documents[int(i)] for i in idx])
        for key in metrics:
            if scored.get(key) is not None:
                estimates[key].append(scored[key])
    ci = {}
    original = aggregate(documents)
    for key in metrics:
        if not estimates[key]:
            continue
        lo, hi = np.percentile(estimates[key], [2.5, 97.5])
        ci[key] = {"estimate": original[key], "lower_95": float(lo),
                   "upper_95": float(hi)}
    return {"method": "document-clustered stratified bootstrap" if stratify else "document-clustered bootstrap",
            "replications": n_boot, "seed": seed, "sampled_documents": len(documents),
            "notes": "Within-topic PDF resampling; correlated questions kept together. No finite population correction.",
            "intervals": ci}


def ocr_diff_html(reference_text, ocr_markdown, max_words=2000):
    """Safe side-by-side token diff: red deletes, green inserts. UI only, not scoring."""
    ref = normalise(markdown_visible_text(reference_text)).split()[:max_words]
    pred = normalise(markdown_visible_text(ocr_markdown)).split()[:max_words]
    sm = difflib.SequenceMatcher(a=ref, b=pred, autojunk=False)
    left, right = [], []
    for operation, a0, a1, b0, b1 in sm.get_opcodes():
        a = html.escape(" ".join(ref[a0:a1]))
        b = html.escape(" ".join(pred[b0:b1]))
        if operation == "equal":
            left.append(a); right.append(b)
        else:
            if a: left.append(f'<mark style="background:#fee2e2">{a}</mark>')
            if b: right.append(f'<mark style="background:#dcfce7">{b}</mark>')
    return ("<div style='display:grid;grid-template-columns:1fr 1fr;gap:14px'>"
            "<div><b>Reference text</b><div style='white-space:pre-wrap;overflow:auto;max-height:480px'>" + " ".join(left) + "</div></div>"
            "<div><b>OCR Markdown — visible text</b><div style='white-space:pre-wrap;overflow:auto;max-height:480px'>" + " ".join(right) + "</div></div></div>")
