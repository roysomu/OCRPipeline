import math
import pytest
from metrics import (ocr_markdown_text, markdown_visible_text, score_document,
                     aggregate, assignment_accuracy_metrics,
                     cluster_bootstrap_ci, exact_normalized_span_present,
                     ocr_diff_html)
from config import TOPICS


def fixture_doc(did="a", topic="Company Policies", prediction=None):
    group = {"document_id": did, "split": "repliqa_0", "topic": topic,
             "reference_text": "Company Policies Employees receive 20 days leave Item Qty Pen 2 See policy",
             "questions": [{"question_id": f"{did}-q1", "question": "Leave?", "answer": "20 days",
                            "long_answer": "Employees receive 20 days leave"},
                           {"question_id": f"{did}-q2", "question": "CEO?", "answer": "UNANSWERABLE",
                            "long_answer": ""}]}
    ocr_md = ("# Company Policies\nEmployees receive 20 days leave\n"
              "| Item | Qty |\n| --- | --- |\n| Pen | 2 |\n"
              "[See policy](https://extra.example.com/url) ![image](image-0.png)")
    result = {"ocr": {"text": "[PAGE 1] OUT OF SCOPE TEXT", "markdown_text": ocr_md,
                       "markdown_pages": [ocr_md], "pages": 1},
              "analysis": {"topic": topic, "answers": [
                  {"question_id": f"{did}-q1", "is_answerable": True,
                   "short_answer": prediction or "20 days", "long_answer": "The handbook says twenty days are provided to eligible employees, as described within its policy.",
                   "evidence_quote": "20 days leave", "evidence_page": 1, "evidence_verified": True},
                  {"question_id": f"{did}-q2", "is_answerable": False,
                   "short_answer": "UNANSWERABLE", "long_answer": "The document does not provide enough information to answer this question.",
                   "evidence_quote": "", "evidence_page": None, "evidence_verified": True}]},
              "cost": {"ocr_usd": 0.004,
                       "full_inference_estimated_usd": 0.0045,
                       "this_run_fresh_api_estimated_usd": 0.0045},
              "seconds": 0.8, "ocr_cached": False, "topic_cached": False,
              "qa_cached": False}
    return group, result


def test_markdown_only_wer_ignores_structural_fields_and_urls():
    group, result = fixture_doc()
    scored = score_document(group, result)
    assert scored["ocr_proxy_wer"] == 0.0
    assert scored["ocr_proxy_cer"] == 0.0
    assert scored["ocr_proxy_token_f1"] == 1.0
    assert aggregate([scored])["qa_answerable_em"] == 1.0
    assert aggregate([scored])["abstention_recall"] == 1.0
    assert scored["questions"][0]["gold_long_answer_span_recovered"] is True
    assert aggregate([scored])["exact_span_recall"] == 1.0
    assert aggregate([scored])["ocr_proxy_micro_token_f1"] == 1.0
    assert aggregate([scored])["ocr_api_cost_estimated_usd"] == 0.004


def test_exact_span_recall_normalizes_formatting_but_requires_contiguous_tokens():
    assert exact_normalized_span_present(
        "Employees receive 20 days' leave.",
        "EMPLOYEES\nreceive 20 days leave") is True
    assert exact_normalized_span_present("art", "An article") is False
    assert exact_normalized_span_present("", "Any OCR text") is None

    group, result = fixture_doc()
    result["ocr"]["markdown_text"] = "Employees receive leave after 20 days"
    scored = score_document(group, result)

    assert scored["questions"][0]["gold_long_answer_span_recovered"] is False
    assert scored["exact_span_recall"] == 0.0
    assert aggregate([scored])["exact_span_recall"] == 0.0


def test_ocr_processing_time_is_aggregated_per_timed_page():
    group_a, result_a = fixture_doc("a")
    result_a["ocr"].update({"pages": 2, "processing_seconds": 1.0,
                            "processing_seconds_per_page": 0.5,
                            "processing_timing_scope": "ocr_post_request_elapsed"})
    group_b, result_b = fixture_doc("b")
    result_b["ocr"].update({"pages": 3, "processing_seconds": 3.0,
                            "processing_seconds_per_page": 1.0,
                            "processing_timing_scope": "ocr_post_request_elapsed"})

    summary = aggregate([score_document(group_a, result_a),
                         score_document(group_b, result_b)])

    assert summary["ocr_timed_documents"] == 2
    assert summary["ocr_timed_pages"] == 5
    assert summary["ocr_processing_seconds_total"] == 4.0
    assert summary["ocr_processing_seconds_per_page"] == 0.8


def test_wer_can_exceed_one_not_capped():
    group, result = fixture_doc()
    group["reference_text"] = "One"
    result["ocr"]["markdown_text"] = "one two three four five six"
    scored = score_document(group, result)
    assert scored["ocr_proxy_wer"] > 1
    assert aggregate([scored])["ocr_proxy_micro_wer"] > 1


def test_ocr_token_f1_is_micro_aggregated_from_whole_document_token_counts():
    group_a, result_a = fixture_doc("a")
    group_a["reference_text"] = "one two"
    result_a["ocr"]["markdown_text"] = "two one extra"
    group_b, result_b = fixture_doc("b")
    group_b["reference_text"] = "three"
    result_b["ocr"]["markdown_text"] = "missing"

    first = score_document(group_a, result_a)
    second = score_document(group_b, result_b)
    summary = aggregate([first, second])

    assert first["ocr_proxy_token_precision"] == 2 / 3
    assert first["ocr_proxy_token_recall"] == 1.0
    assert first["ocr_proxy_token_f1"] == 0.8
    assert summary["ocr_proxy_micro_token_precision"] == 0.5
    assert summary["ocr_proxy_micro_token_recall"] == 2 / 3
    assert summary["ocr_proxy_micro_token_f1"] == pytest.approx(4 / 7)


def test_dataset_unanswerable_sentence_is_scored_as_abstention():
    group, result = fixture_doc()
    group["questions"] = [{"question_id": "a-q2", "question": "CEO?",
                           "answer": "The answer is not found in the document.",
                           "long_answer": "NA"}]
    result["analysis"]["answers"] = [result["analysis"]["answers"][1]]

    scored = score_document(group, result)
    summary = aggregate([scored])

    assert scored["questions"][0]["gold_unanswerable"] is True
    assert scored["questions"][0]["correct_answerability"] == 1
    assert summary["abstention_precision"] == 1.0
    assert summary["abstention_recall"] == 1.0
    assert summary["qa_answerable_em"] is None
    assert summary["exact_span_recall"] is None


def test_document_clustered_ci_reproducible():
    docs = []
    for i in range(20):
        group, result = fixture_doc(str(i), TOPICS[i % 2], "wrong" if i % 3 else "20 days")
        docs.append(score_document(group, result))
    first = cluster_bootstrap_ci(docs, n_boot=200, seed=42)
    second = cluster_bootstrap_ci(docs, n_boot=200, seed=42)
    assert first == second
    assert first["sampled_documents"] == 20
    assert first["intervals"]["qa_answerable_em"]["lower_95"] <= aggregate(docs)["qa_answerable_em"] <= first["intervals"]["qa_answerable_em"]["upper_95"]
    assert first["intervals"]["exact_span_recall"]["estimate"] == 1.0
    assert first["intervals"]["ocr_proxy_micro_token_f1"]["estimate"] == 1.0


def test_runner_up_gold_is_reported_without_changing_primary_topic_accuracy():
    group, result = fixture_doc(topic=TOPICS[0])
    result["analysis"]["topic"] = TOPICS[1]
    result["topic_classification"] = {"runner_up": TOPICS[0]}

    scored = score_document(group, result)
    summary = aggregate([scored])

    assert scored["topic_correct"] == 0
    assert summary["topic_accuracy"] == 0.0
    assert summary["topic_correct_in_runner_up_count"] == 1
    assert summary["topic_correct_in_runner_up_rate"] == 1.0


def test_embedding_top_k_recall_is_reported_separately_from_topic_accuracy():
    group, result = fixture_doc(topic=TOPICS[0])
    result["topic_embedding"] = {
        "mode": "advisory",
        "candidates": [{"label": TOPICS[1], "score": 0.9},
                       {"label": TOPICS[0], "score": 0.8},
                       {"label": TOPICS[2], "score": 0.7}],
    }
    scored = score_document(group, result)
    summary = aggregate([scored])
    assert summary["topic_accuracy"] == 1.0
    assert summary["topic_embedding_top_k_recall"] == 1.0
    assert summary["topic_embedding_top_k"] == 3
    assert summary["topic_embedding_mode"] == "advisory"


def test_judge_failure_count_does_not_remove_document_from_core_metrics():
    group, result = fixture_doc()
    scored = score_document(group, result)
    scored["judge_error"] = "ValueError: malformed optional judge response"
    summary = aggregate([scored])
    assert summary["documents"] == 1
    assert summary["topic_accuracy"] == 1.0
    assert summary["judge_failed_documents"] == 1
    assert summary["judge_scored_questions"] == 0


def test_assignment_accuracy_uses_three_pillars_and_success_penalty():
    summary = {
        "documents": 98, "evaluated_documents": 98, "requested_documents": 100,
        "questions": 490, "judge_scored_questions": 490,
        "ocr_proxy_micro_wer": 0.00138, "ocr_proxy_micro_cer": 0.00011,
        "topic_accuracy": 0.78,
        "judge_semantic_alignment_mean_1_to_5": 4.5,
        "answerability_accuracy": 0.90, "evidence_quote_location_rate": 0.95,
        "long_answer_target_length_rate": 0.80,
    }
    result = assignment_accuracy_metrics(summary)
    assert math.isclose(result["assignment_text_extraction_score"], 0.999128)
    assert math.isclose(result["assignment_topic_score"], 0.78)
    assert math.isclose(result["assignment_long_form_score"], 0.8825)
    expected = 0.98 * (0.999128 + 0.78 + 0.8825) / 3
    assert math.isclose(result["assignment_accuracy"], expected)
    assert result["assignment_accuracy_status"] == "available"


def test_assignment_accuracy_uses_successful_partial_judge_coverage():
    summary = {
        "documents": 1, "questions": 5, "judge_scored_questions": 4,
        "ocr_proxy_micro_wer": 0.01, "ocr_proxy_micro_cer": 0.005,
        "topic_accuracy": 1.0, "topic_macro_f1_observed_classes": 1.0,
        "judge_semantic_alignment_mean_1_to_5": 5.0,
        "answerability_accuracy": 1.0, "evidence_quote_location_rate": 1.0,
        "long_answer_target_length_rate": 1.0,
    }
    result = assignment_accuracy_metrics(summary)
    assert result["assignment_accuracy"] is not None
    assert result["assignment_long_form_score"] == 1.0
    assert result["assignment_judge_coverage_rate"] == 0.8
    assert result["assignment_accuracy_status"] == "available_partial_optional_judge_coverage"


def test_assignment_accuracy_requires_at_least_one_successful_judge():
    summary = {
        "documents": 1, "questions": 5, "judge_scored_questions": 0,
        "ocr_proxy_micro_wer": 0.01, "ocr_proxy_micro_cer": 0.005,
        "topic_accuracy": 1.0,
        "judge_semantic_alignment_mean_1_to_5": None,
        "answerability_accuracy": 1.0, "evidence_quote_location_rate": 1.0,
        "long_answer_target_length_rate": 1.0,
    }
    result = assignment_accuracy_metrics(summary)
    assert result["assignment_accuracy"] is None
    assert result["assignment_judge_coverage_rate"] == 0.0
    assert result["assignment_accuracy_status"] == (
        "requires_core_metrics_and_at_least_one_successful_optional_judge")


def test_diff_is_html_escaped():
    value = ocr_diff_html("One <script>alert('x')</script>", "One normal")
    assert "<script>" not in value
    assert "&lt;script&gt;" not in value  # punctuation normalized and stripped
