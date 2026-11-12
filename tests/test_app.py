import os
import json
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mistral-psa-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/mistral-psa-xdg-cache")

import pymupdf
import pytest

import app
from config import Settings, TOPICS


def _tab1_result(topic, model, topic_cost, full_cost, fresh_cost):
    classification = {"topic": topic, "runner_up": TOPICS[1]}
    topic_call = {"prompt_tokens": 20, "completion_tokens": 5, "model": model}
    return {
        "analysis": {"topic": topic},
        "topic_classification": classification,
        "topic_small_classification": classification,
        "topic_cached": False,
        "usage": {"prompt_tokens": 120, "completion_tokens": 25,
                  "topic_call": topic_call},
        "cost": {"topic_usd": topic_cost, "chat_usd": full_cost - 0.01,
                 "full_inference_estimated_usd": full_cost,
                 "this_run_fresh_api_estimated_usd": fresh_cost},
        "model_config": {"topic_model": model},
    }


def test_tab1_uses_small_first_and_stops_when_it_matches(monkeypatch):
    configured = Settings(topic_model="small-test", topic_refinement_model="large-test",
                          topic_runner_up_large_enabled=True,
                          topic_input_per_million=0.15, topic_output_per_million=0.60,
                          judge_input_per_million=0.50, judge_output_per_million=1.50)
    group = {"topic": TOPICS[0]}
    calls = []

    def fake_process(_group, settings):
        calls.append(settings)
        return _tab1_result(TOPICS[0], settings.topic_model, 0.02, 0.10, 0.05)

    monkeypatch.setattr(app, "process_document", fake_process)
    result = app.process_tab1_document(group, configured)

    assert len(calls) == 1
    assert calls[0].topic_model == "small-test"
    assert calls[0].topic_runner_up_large_enabled is False
    assert calls[0].topic_reasoning_effort == "none"
    assert calls[0].topic_input_per_million == configured.topic_input_per_million
    assert result["topic_tab1_strategy"]["final_source"] == "small_primary"
    assert result["topic_tab1_strategy"]["final_matches_reference"] is True
    assert result["topic_small_classification"]["topic"] == TOPICS[0]
    assert result["topic_large_classification"] is None


def test_tab2_settings_do_not_change_tab1_sequence():
    tab2 = app.settings_for_tab2_embedding_mode("advisory")
    tab1_small, tab1_large = app._tab1_topic_settings(
        app.settings_for_embedding_mode("advisory"))

    assert tab2.topic_embedding_mode == "advisory"
    assert tab2.topic_reasoning_effort == "none"
    assert tab2.topic_refinement_reasoning_effort == "low"
    assert tab1_large.topic_reasoning_effort == "low"
    assert tab1_small.topic_reasoning_effort == "none"


def test_tab1_uses_large_backup_after_small_miss_and_keeps_both_audits(monkeypatch):
    configured = Settings(topic_model="small-test", topic_refinement_model="large-test",
                          topic_runner_up_large_enabled=True)
    group = {"topic": TOPICS[0]}
    calls = []

    def fake_process(_group, settings):
        calls.append(settings.topic_model)
        if settings.topic_model == "small-test":
            return _tab1_result(TOPICS[2], "small-test", 0.01, 0.10, 0.04)
        return _tab1_result(TOPICS[0], "large-test", 0.02, 0.11, 0.05)

    monkeypatch.setattr(app, "process_document", fake_process)
    result = app.process_tab1_document(group, configured)

    assert calls == ["small-test", "large-test"]
    small_settings, large_settings = app._tab1_topic_settings(configured)
    assert large_settings.topic_reasoning_effort == "low"
    assert small_settings.topic_reasoning_effort == "none"
    assert result["analysis"]["topic"] == TOPICS[0]
    assert result["topic_small_classification"]["topic"] == TOPICS[2]
    assert result["topic_large_classification"]["topic"] == TOPICS[0]
    assert result["topic_tab1_strategy"]["primary_status"] == "missed_reference"
    assert result["topic_tab1_strategy"]["final_source"] == "large_backup"
    assert result["cost"]["full_inference_estimated_usd"] == pytest.approx(0.12)
    assert result["cost"]["this_run_fresh_api_estimated_usd"] == pytest.approx(0.09)


def test_tab1_uses_large_backup_after_small_error(monkeypatch):
    configured = Settings(topic_model="small-test", topic_refinement_model="large-test")
    group = {"topic": TOPICS[0]}
    calls = []

    def fake_process(_group, settings):
        calls.append(settings.topic_model)
        if settings.topic_model == "small-test":
            raise RuntimeError("temporary Small outage")
        return _tab1_result(TOPICS[0], "large-test", 0.02, 0.11, 0.05)

    monkeypatch.setattr(app, "process_document", fake_process)
    result = app.process_tab1_document(group, configured)

    assert calls == ["small-test", "large-test"]
    assert result["topic_tab1_strategy"]["primary_status"] == "error"
    assert result["topic_tab1_strategy"]["final_source"] == "large_backup"
    assert "temporary Small outage" in result["topic_tab1_strategy"]["primary_error"]


def test_tab1_retains_small_result_if_large_backup_errors(monkeypatch):
    configured = Settings(topic_model="small-test", topic_refinement_model="large-test")
    group = {"topic": TOPICS[0]}

    def fake_process(_group, settings):
        if settings.topic_model == "large-test":
            raise RuntimeError("temporary Large outage")
        return _tab1_result(TOPICS[2], "small-test", 0.01, 0.10, 0.04)

    monkeypatch.setattr(app, "process_document", fake_process)
    result = app.process_tab1_document(group, configured)

    assert result["analysis"]["topic"] == TOPICS[2]
    assert result["topic_tab1_strategy"]["final_source"] == "small_primary_retained"
    assert result["topic_tab1_strategy"]["backup_status"] == "error"
    assert "temporary Large outage" in result["topic_tab1_strategy"]["backup_error"]


def test_document_inspector_is_hidden_before_processing():
    config = app.demo.get_config_file()
    inspector = next(
        component for component in config["components"]
        if component.get("props", {}).get("elem_id") == "document-inspector-shell"
    )
    assert inspector["props"]["visible"] is False


def test_tab1_omits_ocr_reference_comparison_section():
    config = app.demo.get_config_file()
    labels = {
        component.get("props", {}).get("label")
        for component in config["components"]
    }

    assert "OCR visible Markdown versus auto-extracted reference" not in labels
    assert "Mistral OCR: Markdown only" not in labels
    assert "RepLiQA document_extracted (proxy reference)" not in labels


def test_aggregated_runs_tab_follows_tco_tab():
    config = app.demo.get_config_file()
    tabs = [
        component.get("props", {}).get("label")
        for component in config["components"]
        if component.get("type") == "tabitem"
    ]

    assert tabs[-2:] == ["3 · Customer TCO scenario", "4 · Aggregated Runs"]


def test_aggregated_runs_snapshot_adds_new_runs_and_keeps_newest_duplicate(
        tmp_path, monkeypatch):
    monkeypatch.setattr(app, "RESULTS_DIR", tmp_path)

    def fake_aggregate(documents, **_kwargs):
        return {
            "documents": len(documents), "questions": 0,
            "pages": sum(document["pages"] for document in documents),
            "full_run_api_cost_estimated_usd": 0,
            "judge_api_cost_estimated_usd": 0,
        }

    monkeypatch.setattr(app, "aggregate", fake_aggregate)
    monkeypatch.setattr(
        app, "assignment_accuracy_metrics",
        lambda _summary: {"assignment_accuracy": 1.0})

    def add_run(name, modified, documents):
        run = tmp_path / name
        document_dir = run / "documents"
        document_dir.mkdir(parents=True)
        summary_path = run / "summary.json"
        summary_path.write_text(json.dumps({
            "status": "COMPLETE", "metrics_version": app.METRICS_VERSION,
            "split": "repliqa_0",
        }))
        os.utime(summary_path, (modified, modified))
        for document in documents:
            (document_dir / f'{document["document_id"]}.json').write_text(
                json.dumps(document))

    add_run("run-1", 100, [{
        "document_id": "d1", "topic_gold": "Company Policies", "pages": 1,
    }])
    first, _ = app.aggregated_runs_snapshot()
    assert first["documents"] == 1
    assert first["pages"] == 1

    add_run("run-2", 200, [
        {"document_id": "d1", "topic_gold": "Company Policies", "pages": 9},
        {"document_id": "d2", "topic_gold": "Company Policies", "pages": 2},
    ])
    refreshed, per_topic = app.aggregated_runs_snapshot()

    assert refreshed["documents"] == 2
    assert refreshed["pages"] == 11
    assert per_topic.iloc[0]["Documents"] == 2


def test_tab2_recent_runs_lists_only_six_newest(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "RESULTS_DIR", tmp_path)

    for index in range(7):
        run = tmp_path / f"run-{index}"
        run.mkdir()
        summary_path = run / "summary.json"
        summary_path.write_text(json.dumps({
            "split": "repliqa_1", "requested_documents": index + 1,
            "evaluated_documents": index + 1, "status": "COMPLETE",
        }))
        (run / "run_config.json").write_text(json.dumps({
            "dataset": "repliqa_1", "requested_documents": index + 1,
            "seed": 40 + index,
        }))
        os.utime(summary_path, (100 + index, 100 + index))

    choices = app.recent_evaluation_runs()

    assert len(choices) == 6
    assert [value for _, value in choices] == [
        "run-6", "run-5", "run-4", "run-3", "run-2", "run-1"]
    assert choices[0][0].startswith("TS - ")
    assert "repliqa_1" not in choices[0][0]


def test_tab2_recent_run_selection_reads_stored_outputs_without_api(
        tmp_path, monkeypatch):
    monkeypatch.setattr(app, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(
        app, "iterate_evaluation",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("stored selection must not enter the API workflow")))
    run = tmp_path / "latest-run"
    run.mkdir()
    summary = {
        "split": "repliqa_2", "requested_documents": 2,
        "evaluated_documents": 2, "failed_documents": 0,
        "status": "COMPLETE", "pages": 4,
    }
    (run / "summary.json").write_text(json.dumps(summary))
    (run / "run_config.json").write_text(json.dumps({
        "dataset": "repliqa_2", "requested_documents": 2, "seed": 42,
    }))
    (run / "per_topic.csv").write_text("Topic,Documents\nCompany Policies,2\n")
    (run / "per_question.csv").write_text("document_id,question\nd1,Q1\n")

    status, rendered, raw, table, summary_file, question_file = (
        app.load_stored_evaluation("latest-run"))

    assert "Stored result · COMPLETE" in status
    assert "No API request was made" in status
    assert "Evaluation summary · COMPLETE" in rendered
    assert raw == summary
    assert table.iloc[0]["Documents"] == 2
    assert summary_file == str(run / "summary.json")
    assert question_file == str(run / "per_question.csv")


def test_tab2_cleared_or_stale_recent_run_is_a_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "RESULTS_DIR", tmp_path)

    for run_name in (None, "stale-run"):
        outputs = app.load_stored_evaluation(run_name)
        assert len(outputs) == 6
        assert all(output.get("__type__") == "update" for output in outputs)


def test_tab2_run_button_enters_evaluation_workflow(monkeypatch):
    calls = []

    def fake_iterate(*args, **kwargs):
        calls.append((args, kwargs))
        yield {
            "stage": "progress", "completed": 1, "total": 1,
            "scored": 1, "failures": 0, "judge_failures": 0,
        }

    monkeypatch.setattr(app, "iterate_evaluation", fake_iterate)
    updates = app.run_evaluation_ui("repliqa_3", 1, 42, False, "advisory")

    started = next(updates)
    assert len(started) == 7
    assert started[-1]["value"] is None
    assert len(next(updates)) == 7
    assert len(calls) == 1
    assert calls[0][0][:5] == ("repliqa_3", 1, 42, False, app.UI_BOOTSTRAP_REPLICATIONS)


def test_tab2_exposes_recent_run_dropdown():
    config = app.demo.get_config_file()
    dropdown = next(
        component for component in config["components"]
        if "six most recent stored runs, newest first" in
        str(component.get("props", {}).get("info") or "")
    )
    stored_handler = next(
        dependency for dependency in config["dependencies"]
        if dependency.get("api_name") == "load_stored_evaluation"
    )
    run_handler = next(
        dependency for dependency in config["dependencies"]
        if dependency.get("api_name") == "run_evaluation_ui"
    )

    assert "six most recent stored runs, newest first" in dropdown["props"]["info"]
    assert stored_handler["targets"] == [(dropdown["id"], "input")]
    assert dropdown["id"] not in run_handler["inputs"]


def test_tab1_recent_runs_keeps_six_newest_correct_topics(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "RESULTS_DIR", tmp_path)
    run_dir = tmp_path / "tab1_runs"
    run_dir.mkdir()

    for index in range(8):
        run_path = run_dir / f"correct-{index}.json"
        run_path.write_text(json.dumps({
            "topic_correct": True,
            "created_at": f"2026-11-09T00:00:0{index}-08:00",
        }))
        # Deliberately oppose filesystem time to prove created_at controls chronology.
        os.utime(run_path, (1000 - index, 1000 - index))
    incorrect = run_dir / "incorrect-newest.json"
    incorrect.write_text(json.dumps({"topic_correct": False}))
    os.utime(incorrect, (1000, 1000))

    choices = app.recent_tab1_runs()

    assert len(choices) == 6
    assert [value for _, value in choices] == [
        "correct-7.json", "correct-6.json", "correct-5.json",
        "correct-4.json", "correct-3.json", "correct-2.json",
    ]
    assert all(label.startswith("TS - ") for label, _ in choices)
    assert "incorrect-newest.json" not in {value for _, value in choices}


def test_tab1_recent_run_selection_reads_snapshot_without_api(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(
        app, "process_document",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("stored selection must not enter the API workflow")))
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"stored-image")
    run_dir = tmp_path / "tab1_runs"
    run_dir.mkdir()
    run_path = run_dir / "correct.json"
    run_path.write_text(json.dumps({
        "topic_correct": True,
        "summary_html": "<div>Stored document</div>",
        "quality_header_html": "<div>Stored quality</div>",
        "question_rows": [{"Question": "Q1"}],
        "raw_details": {"stored": True},
        "page_state": {
            "document_id": "d1", "images": [str(image_path)],
            "ocr_pages": ["# Stored OCR"], "page_details": [{}], "page": 1,
        },
    }))

    outputs = app.load_stored_tab1_run("correct.json")

    assert len(outputs) == 10
    assert outputs[0] == "<div>Stored document</div>"
    assert outputs[1] == "<div>Stored quality</div>"
    assert outputs[2].iloc[0]["Question"] == "Q1"
    assert outputs[3] == {"stored": True}
    assert "data:image/png;base64," in outputs[5]
    assert "Stored OCR" in outputs[6]


def test_tab1_cleared_or_stale_recent_run_is_a_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "RESULTS_DIR", tmp_path)

    for run_name in (None, "stale-run.json"):
        outputs = app.load_stored_tab1_run(run_name)
        assert len(outputs) == 10
        assert all(output.get("__type__") == "update" for output in outputs)


def test_tab1_exposes_filtered_history_and_process_has_precedence():
    config = app.demo.get_config_file()
    dropdown = next(
        component for component in config["components"]
        if "six most recent stored runs with a correct topic, newest first" in
        str(component.get("props", {}).get("info") or "")
    )
    stored_handler = next(
        dependency for dependency in config["dependencies"]
        if dependency.get("api_name") == "load_stored_tab1_run"
    )
    process_handler = next(
        dependency for dependency in config["dependencies"]
        if dependency.get("api_name") == "inspect_document"
    )
    process_button = next(
        component for component in config["components"]
        if component.get("props", {}).get("value") == "Process selected PDF"
    )
    clear_handler = next(
        dependency for dependency in config["dependencies"]
        if dependency.get("targets") == [(process_button["id"], "click")]
    )

    assert stored_handler["targets"] == [(dropdown["id"], "input")]
    assert dropdown["id"] not in process_handler["inputs"]
    assert dropdown["id"] in process_handler["outputs"]
    assert dropdown["id"] in clear_handler["outputs"]


def test_tab1_cost_per_page_card_is_ocr_only():
    group = {"document_id": "doc-1", "split": "repliqa_0", "topic": TOPICS[0]}
    result = {
        "analysis": {"topic": TOPICS[0]},
        "cost": {"ocr_usd": 0.012, "full_inference_estimated_usd": 0.15},
        "ocr_cached": False, "topic_cached": False, "qa_cached": False,
        "topic_tab1_strategy": {"final_source": "small_primary"},
    }
    scored = {
        "pages": 3, "questions": [], "judge_error": None,
        "ocr_proxy_wer": 0.0, "ocr_proxy_cer": 0.0,
        "ocr_proxy_token_f1": 0.96, "topic_correct": 1,
        "ocr_processing_seconds": 0.9,
        "ocr_processing_seconds_per_page": 0.3,
    }

    rendered = app.executive_metrics_html(group, result, scored, {})
    core_card = rendered[rendered.index("Core API cost per page"):rendered.index("Total cost per page")]
    ocr_card = rendered[rendered.index("Total cost per page"):rendered.index("Answerability")]

    assert "Exact Span Recall" in rendered
    assert rendered.index("Exact Span Recall") < rendered.index("OCR Token F1") < rendered.index("Topic match") < rendered.index("End-to-end assignment accuracy")
    assert "Long-answer length" not in rendered
    exact_span_card = rendered[rendered.index("Exact Span Recall"):rendered.index("Topic match")]
    assert "exec-good" in exact_span_card
    assert "$0.05000" in core_card
    assert "$0.00400" in ocr_card
    assert "OCR API only; excludes embedding, topic, QA and optional judge" in ocr_card
    assert "$0.05000" not in ocr_card


def test_tab2_cost_per_page_card_is_ocr_only():
    summary = {
        "pages": 4,
        "evaluated_documents": 2,
        "requested_documents": 2,
        "failed_documents": 0,
        "full_run_api_cost_estimated_usd": 1.25,
        "judge_api_cost_estimated_usd": 2.0,
        "ocr_api_cost_estimated_usd": 0.016,
        "ocr_proxy_micro_token_f1": 0.95,
    }

    rendered = app.evaluation_metrics_html(summary)
    core_card = rendered[rendered.index("Core API cost per page"):rendered.index("Total cost per page")]
    ocr_card = rendered[rendered.index("Total cost per page"):rendered.index("Answerability")]
    total_card = rendered[rendered.index("Total estimated API cost per page"):]

    assert "Exact Span Recall" in rendered
    assert rendered.index("Exact Span Recall") < rendered.index("OCR Token F1") < rendered.index("Topic accuracy") < rendered.index("End-to-end assignment accuracy")
    assert "Pipeline latency p50" not in rendered
    assert "Pipeline latency p95" not in rendered
    assert "Long-answer length" not in rendered
    exact_span_card = rendered[rendered.index("Exact Span Recall"):rendered.index("Topic accuracy")]
    assert "exec-good" in exact_span_card
    assert "$0.31250" in core_card
    assert "$0.00400" in ocr_card
    assert "OCR API only; excludes embedding, topic, QA and optional judge" in ocr_card
    assert "$0.81250" not in ocr_card
    assert "$0.81250" in total_card


def test_aggregated_runs_card_places_ocr_token_f1_after_exact_span():
    rendered = app.aggregated_runs_metrics_html({
        "documents": 3,
        "pages": 6,
        "exact_span_recall": 0.9,
        "exact_span_recovered_questions": 9,
        "exact_span_eligible_questions": 10,
        "ocr_proxy_micro_token_f1": 0.98,
    })

    assert (rendered.index("Exact Span Recall")
            < rendered.index("OCR Token F1")
            < rendered.index("Topic accuracy"))
    assert "98.0%" in rendered


def test_pdf_page_renderer_and_synchronized_navigation(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "CACHE_DIR", tmp_path / "cache")
    pdf = tmp_path / "two-pages.pdf"
    document = pymupdf.open()
    for number in (1, 2):
        page = document.new_page()
        page.insert_text((72, 72), f"Source page {number}", fontsize=18)
    document.save(pdf)
    document.close()

    images = app.render_pdf_pages(pdf, 2)
    assert len(images) == 2
    assert all(Path(path).is_file() and Path(path).stat().st_size > 0
               for path in images)

    page_detail = {
        "dimensions": {"width": 612, "height": 792},
        "confidence_scores": {"average_page_confidence_score": 0.94,
                              "minimum_page_confidence_score": 0.81},
        "blocks": [{"type": "title", "top_left_x": 60, "top_left_y": 45,
                    "bottom_right_x": 500, "bottom_right_y": 90,
                    "content": "OCR two",
                    "confidence_scores": {
                        "average_content_confidence_score": 0.91,
                        "minimum_content_confidence_score": 0.84,
                        "block_type_confidence_score": 0.98}}],
    }
    state = {"images": images, "ocr_pages": ["# OCR one", "# OCR two"],
             "page_details": [{}, page_detail], "page": 1}
    updated, source_html, ocr_html, status, slider = app.select_document_page(state, 9)

    assert updated["page"] == 2
    assert "data:image/png;base64," in source_html
    assert "ocr-source-box" in source_html
    assert "Title" in source_html and "91%" in source_html
    assert "ocr-block-card" in ocr_html
    assert "OCR two" in ocr_html and "91%" in ocr_html
    assert "<strong>2</strong>" in status
    assert slider["value"] == 2
    assert slider["maximum"] == 2

    _, _, clean_html, _, _ = app.select_document_page(
        state, 2, view_mode="Clean Markdown")
    assert "<h1>OCR two</h1>" in clean_html

    _, no_label_source, no_label_ocr, _, _ = app.select_document_page(
        state, 2, show_labels=False)
    assert "ocr-box-label" not in no_label_source
    assert "ocr-confidence-badge" not in no_label_ocr
    assert "ocr-block-meta" not in no_label_ocr
