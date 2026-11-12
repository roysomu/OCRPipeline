import evaluate
from config import Settings


def test_run_signature_tracks_topic_reasoning_effort():
    groups = [{"document_id": "d1", "dataset_revision": "revision-1"}]
    none = Settings(topic_reasoning_effort="none")
    medium = Settings(topic_reasoning_effort="medium")

    assert evaluate.run_signature(groups, "repliqa_0", 42, none, False) != \
        evaluate.run_signature(groups, "repliqa_0", 42, medium, False)


def test_iterative_batch_checkpoints_and_resume(tmp_path, monkeypatch):
    from config import TOPICS
    monkeypatch.setattr(evaluate, "RESULTS_DIR", tmp_path)
    groups = [{"document_id": f"d{i}", "split": "repliqa_0", "topic": TOPICS[0],
               "questions": [{"question_id": f"q{i}", "question": "Why?", "answer": "Yes"}]}
              for i in range(3)]
    monkeypatch.setattr(evaluate, "manifest", lambda *args: groups)
    calls = []
    def fake_process(group, **kwargs):
        calls.append(group["document_id"])
        return {"placeholder": group["document_id"]}
    def fake_score(group, result, judged):
        return {"document_id": group["document_id"], "topic_gold": TOPICS[0],
                "topic_pred": TOPICS[0], "topic_correct": 1, "ocr_word_errors": 0,
                "ocr_reference_words": 1, "ocr_char_errors": 0, "ocr_reference_chars": 4,
                "ocr_proxy_wer": 0.0, "ocr_proxy_cer": 0.0, "pages": 1,
                "full_inference_estimated_usd": .01, "this_run_fresh_api_estimated_usd": .01,
                "judge_full_estimated_usd": 0, "judge_fresh_estimated_usd": 0,
                "latency_seconds": 1, "ocr_cached": False, "topic_cached": False,
                "qa_cached": False,
                "questions": [{"question_id": f'q{group["document_id"][1:]}',
                "question": "Why?", "gold_answer": "Yes", "predicted_short": "Yes",
                "predicted_long": "Because it says so", "gold_unanswerable": False,
                "predicted_unanswerable": False, "correct_answerability": 1,
                "answerable_em": 1, "answerable_token_f1": 1,
                "evidence_quote": "Yes", "evidence_page": 1,
                "evidence_quote_verified": True, "long_answer_word_count": 4,
                "long_answer_length_ok": False, "judge_semantic_alignment_1_to_5": None,
                "judge_reference_supported": None, "judge_rationale": None}]}
    monkeypatch.setattr(evaluate, "process_document", fake_process)
    monkeypatch.setattr(evaluate, "score_document", fake_score)
    events = list(evaluate.iterate_evaluation("repliqa_0", 3, 42, False, 0))
    assert events[-1]["summary"]["status"] == "COMPLETE"
    assert calls == ["d0", "d1", "d2"]
    events = list(evaluate.iterate_evaluation("repliqa_0", 3, 42, False, 0))
    assert calls == ["d0", "d1", "d2"]
    assert events[-1]["summary"]["fresh_api_cost_in_current_execution_estimated_usd"] == 0


def test_optional_judge_failure_does_not_fail_core_document(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "RESULTS_DIR", tmp_path)
    group = {"document_id": "d1", "split": "repliqa_0", "topic": "Company Policies",
             "questions": [{"question_id": "d1-q1", "question": "Why?", "answer": "Yes"}]}
    monkeypatch.setattr(evaluate, "manifest", lambda *args: [group])
    monkeypatch.setattr(evaluate, "process_document", lambda *args, **kwargs: {"core": "ok"})
    monkeypatch.setattr(evaluate, "evaluate_semantic",
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            ValueError("Judge returned missing, extra or duplicate IDs")))
    monkeypatch.setattr(evaluate, "score_document", lambda group, result, judged: {
        "document_id": group["document_id"], "questions": [], "core": result["core"]})
    monkeypatch.setattr(evaluate, "aggregate", lambda docs, **kwargs: {
        "documents": len(docs), "questions": 0,
        "judge_failed_documents": sum(bool(d.get("judge_error")) for d in docs)})
    monkeypatch.setattr(evaluate, "topic_breakdown", lambda docs: [])

    events = list(evaluate.iterate_evaluation(
        "repliqa_0", 1, 42, True, 0))
    completed = events[-1]
    run_dir = tmp_path / completed["dir"].split("/")[-1]

    assert completed["summary"]["status"] == "COMPLETE"
    assert completed["summary"]["failed_documents"] == 0
    assert completed["summary"]["judge_failed_documents"] == 1
    assert completed["errors"] == []
    judge_failures = evaluate.read_json(run_dir / "judge_failures.json")
    assert judge_failures[0]["document_id"] == "d1"
    assert "missing, extra or duplicate IDs" in judge_failures[0]["error"]
