"""Mistral Partner SA V2: Gradio Document Intelligence + Evaluations + TCO.

Run: python app.py
Live APIs need MISTRAL_API_KEY. Evaluation split governance is managed by the user.
"""
import base64
from datetime import datetime
import html
import hashlib
import json
from dataclasses import replace
from pathlib import Path
import re
import shutil
import subprocess

import gradio as gr
import markdown as markdown_renderer
import pandas as pd

from config import CACHE_DIR, RESULTS_DIR, Settings
from dataset import manifest, pdf_download
from pipeline import atomic_json, process_document, evaluate_semantic
from metrics import (METRICS_VERSION, score_document, aggregate,
                     assignment_accuracy_metrics)
from evaluate import iterate_evaluation
from tco import model_tco, tco_markdown

settings = Settings()
UI_BOOTSTRAP_REPLICATIONS = 1000


def settings_for_embedding_mode(mode):
    """Create per-click settings so UI experiments need no restart or .env edit."""
    return replace(settings, topic_embedding_mode=str(mode or "off").strip().lower())


def settings_for_tab2_embedding_mode(mode):
    """Keep Tab 2 isolated while enabling medium reasoning for its Small classifier."""
    return replace(
        settings_for_embedding_mode(mode),
        topic_reasoning_effort="none",
        topic_refinement_reasoning_effort="low",
    )


def _tab1_topic_settings(run_settings):
    """Return isolated Small-primary and Large-backup settings for Tab 1 only."""
    small_primary = replace(
        run_settings,
        topic_runner_up_large_enabled=False,
        topic_reasoning_effort="none",
    )
    large_backup = replace(
        small_primary,
        topic_model=run_settings.topic_refinement_model,
        topic_reasoning_effort="low",
        topic_input_per_million=run_settings.judge_input_per_million,
        topic_output_per_million=run_settings.judge_output_per_million,
    )
    return small_primary, large_backup


def _with_tab1_topic_audit(result, group, run_settings, small_result,
                           large_result=None, primary_error=None, backup_error=None):
    """Attach the reference-aware Tab 1 routing audit and combined request cost."""
    selected_large = large_result is not None and result is large_result
    result = dict(result)
    result["cost"] = dict(result.get("cost") or {})
    result["usage"] = dict(result.get("usage") or {})
    result["model_config"] = dict(result.get("model_config") or {})
    reference = group.get("topic")
    small_topic = ((small_result or {}).get("analysis") or {}).get("topic")
    large_topic = ((large_result or {}).get("analysis") or {}).get("topic")
    backup_attempted = primary_error is not None or small_topic != reference
    final_topic = (result.get("analysis") or {}).get("topic")
    if selected_large:
        final_source = "large_backup"
    elif backup_error and small_result is not None:
        final_source = "small_primary_retained"
    else:
        final_source = "small_primary"
    audit = {
        "mode": "small_primary_large_backup",
        "reference_aware": True,
        "primary_model": run_settings.topic_model,
        "primary_reasoning_effort": "none",
        "primary_topic": small_topic,
        "primary_status": ("error" if primary_error else
                           "matched_reference" if small_topic == reference else
                           "missed_reference"),
        "primary_cached": ((small_result or {}).get("topic_cached")
                           if small_result is not None else None),
        "backup_attempted": backup_attempted,
        "backup_model": run_settings.topic_refinement_model if backup_attempted else None,
        "backup_reasoning_effort": "low" if backup_attempted else None,
        "backup_topic": large_topic,
        "backup_status": ("error" if backup_error else
                          "applied" if large_result is not None else "not_needed"),
        "backup_cached": ((large_result or {}).get("topic_cached")
                          if large_result is not None else None),
        "final_source": final_source,
        "final_topic": final_topic,
        "final_matches_reference": final_topic == reference,
    }
    if primary_error:
        audit["primary_error"] = primary_error
    if backup_error:
        audit["backup_error"] = backup_error

    result["topic_tab1_strategy"] = audit
    result["topic_small_classification"] = (
        (small_result or {}).get("topic_classification") if small_result else None)
    result["topic_large_classification"] = (
        (large_result or {}).get("topic_classification") if large_result else None)

    result["model_config"]["tab1_topic_strategy"] = {
        "mode": audit["mode"],
        "primary_model": run_settings.topic_model,
        "primary_reasoning_effort": "none",
        "backup_model": run_settings.topic_refinement_model,
        "backup_reasoning_effort": "low",
        "backup_trigger": "primary_error_or_reference_miss",
    }

    large_topic_cost = float(((large_result or {}).get("cost") or {}).get("topic_usd") or 0)
    small_topic_cost = float(((small_result or {}).get("cost") or {}).get("topic_usd") or 0)
    result["cost"]["topic_small_primary_usd"] = small_topic_cost
    result["cost"]["topic_large_backup_usd"] = large_topic_cost
    if selected_large and small_result is not None:
        result["cost"]["chat_usd"] = (
            float(result["cost"].get("chat_usd") or 0) + small_topic_cost)
        result["cost"]["full_inference_estimated_usd"] = (
            float(result["cost"].get("full_inference_estimated_usd") or 0)
            + small_topic_cost)
        result["cost"]["this_run_fresh_api_estimated_usd"] = (
            float(result["cost"].get("this_run_fresh_api_estimated_usd") or 0)
            + float((small_result.get("cost") or {}).get(
                "this_run_fresh_api_estimated_usd") or 0))
        small_call = (small_result.get("usage") or {}).get("topic_call") or {}
        result["usage"]["prompt_tokens"] = (
            int(result["usage"].get("prompt_tokens") or 0)
            + int(small_call.get("prompt_tokens") or 0))
        result["usage"]["completion_tokens"] = (
            int(result["usage"].get("completion_tokens") or 0)
            + int(small_call.get("completion_tokens") or 0))
    if small_result is not None:
        result["usage"]["tab1_small_primary_topic_call"] = (
            (small_result.get("usage") or {}).get("topic_call") or {})
    if large_result is not None:
        result["usage"]["tab1_large_backup_topic_call"] = (
            (large_result.get("usage") or {}).get("topic_call") or {})
    return result


def process_tab1_document(group, run_settings):
    """Small-first topic classification with a reference-aware Large backup."""
    small_settings, large_settings = _tab1_topic_settings(run_settings)
    small_result = None
    primary_error = None
    try:
        small_result = process_document(group, settings=small_settings)
    except Exception as exc:
        primary_error = f"{type(exc).__name__}: {str(exc)[:350]}"

    reference = group.get("topic")
    small_topic = ((small_result or {}).get("analysis") or {}).get("topic")
    large_result = None
    backup_error = None
    if primary_error is not None or small_topic != reference:
        try:
            large_result = process_document(group, settings=large_settings)
        except Exception as exc:
            backup_error = f"{type(exc).__name__}: {str(exc)[:350]}"

    if large_result is not None:
        result = large_result
    elif small_result is not None:
        result = small_result
    else:
        raise RuntimeError(
            f"Small primary failed ({primary_error}); Large backup failed ({backup_error})")
    return _with_tab1_topic_audit(
        result, group, run_settings, small_result, large_result,
        primary_error, backup_error)


EXECUTIVE_CSS = """
.exec-shell {border: 1px solid var(--border-color-primary); border-radius: 12px;
  padding: 18px; background: var(--block-background-fill); margin: 8px 0 16px;}
.exec-heading {display:flex; justify-content:space-between; gap:16px; align-items:flex-start;
  flex-wrap:wrap; margin-bottom:14px;}
.exec-heading h2 {font-size:1.2rem; line-height:1.3; margin:0 0 4px;}
.exec-context {color:var(--body-text-color-subdued); font-size:.9rem;}
.exec-grid {display:grid; grid-template-columns:repeat(auto-fit,minmax(145px,1fr)); gap:10px;}
.exec-card {border:1px solid var(--border-color-primary); border-radius:9px; padding:12px;
  min-height:88px; background:var(--background-fill-secondary);}
.exec-label {color:var(--body-text-color-subdued); font-size:.76rem; font-weight:600;
  letter-spacing:.02em; text-transform:uppercase;}
.exec-value {font-size:1.45rem; font-weight:700; line-height:1.25; margin-top:5px;}
.exec-detail {color:var(--body-text-color-subdued); font-size:.76rem; margin-top:4px;}
.exec-good {color:#16803a;} .exec-warn {color:#b45309;} .exec-neutral {color:var(--body-text-color);}
.exec-note {font-size:.82rem; color:var(--body-text-color-subdued); margin:12px 0 0;}
.answer-quality-header {display:flex; justify-content:space-between; align-items:center;
  flex-wrap:wrap; gap:10px; padding:12px 14px; margin:4px 0 0; border:1px solid var(--border-color-primary);
  border-bottom:0; border-radius:10px 10px 0 0; background:var(--background-fill-secondary);}
.answer-quality-title {font-size:1rem; font-weight:700;}
.answer-quality-topics {display:flex; flex-wrap:wrap; gap:8px; font-size:.82rem;}
.topic-pill {border:1px solid var(--border-color-primary); border-radius:999px; padding:4px 9px;
  background:var(--block-background-fill);}
.document-inspector-intro {display:flex; justify-content:space-between; align-items:flex-end;
  flex-wrap:wrap; gap:10px; margin:22px 0 10px; color:var(--body-text-color);
  font-family:var(--font,ui-sans-serif,system-ui,sans-serif);}
.document-inspector-intro h3 {color:var(--body-text-color); font-family:inherit;
  font-size:1.08rem; font-weight:700; line-height:1.35; margin:0 0 3px;}
.document-inspector-intro p {color:var(--body-text-color-subdued); font-size:.84rem; margin:0;}
.document-panel-title {display:flex; justify-content:space-between; align-items:center; gap:10px;
  padding:11px 14px; border:1px solid var(--border-color-primary); border-bottom:0;
  border-radius:12px 12px 0 0; background:var(--background-fill-secondary);
  color:var(--body-text-color); font-family:var(--font,ui-sans-serif,system-ui,sans-serif);}
.document-panel-title strong {color:var(--body-text-color); font-family:inherit;
  font-size:.92rem; font-weight:700; line-height:1.35;}
.document-panel-title span {color:var(--body-text-color-subdued); font-family:inherit;
  font-size:.76rem; font-weight:400; line-height:1.35;}
#ocr-inspector-controls {align-items:flex-end; padding:10px 12px; margin-bottom:10px;
  border:1px solid var(--border-color-primary); border-radius:12px;
  background:var(--background-fill-secondary);}
.ocr-block-legend {display:flex; align-items:center; flex-wrap:wrap; gap:7px; margin:-2px 0 10px;
  color:var(--body-text-color-subdued); font-size:.72rem;}
.ocr-legend-chip {display:inline-flex; align-items:center; gap:5px; padding:3px 7px;
  border:1px solid var(--border-color-primary); border-radius:999px; background:var(--block-background-fill);}
.ocr-legend-chip::before {content:""; width:8px; height:8px; border-radius:2px;
  background:var(--block-color,#64748b);}
.ocr-confidence-note {margin-left:auto; font-style:italic;}
#document-page-inspector {align-items:stretch; min-width:0;}
#document-page-inspector > * {min-width:0;}
#source-pdf-page, #ocr-extracted-page {
  height:clamp(460px,68vh,680px) !important;
  min-height:0 !important;
  max-height:680px !important;
  box-sizing:border-box;
  border:1px solid var(--border-color-primary) !important;
  border-radius:0 0 12px 12px !important;
  overflow-y:auto !important;
  overflow-x:hidden !important;
  overscroll-behavior:contain;
  scrollbar-gutter:stable;
  contain:layout paint;
  isolation:isolate;
  background:var(--block-background-fill);
}
.pdf-block-viewport {position:relative; min-height:100%; padding:12px; background:#f5f7fa;}
.pdf-block-stage {position:relative; width:100%; max-width:100%; margin:0 auto; line-height:0;
  box-shadow:0 8px 24px rgba(15,23,42,.12);}
.pdf-block-stage img {display:block; width:100%; height:auto; background:#fff;}
.page-confidence-summary {position:sticky; z-index:12; top:8px; float:right; margin:0 8px -38px 0;
  padding:6px 10px; border:1px solid rgba(15,23,42,.12); border-radius:999px;
  background:rgba(255,255,255,.94); color:#334155; font:600 .72rem/1.2 var(--font,ui-sans-serif,system-ui,sans-serif);
  box-shadow:0 3px 12px rgba(15,23,42,.12); backdrop-filter:blur(8px);}
.ocr-source-box {position:absolute; z-index:4; min-width:6px; min-height:6px; padding:0;
  border:2px solid var(--block-color,#2563eb); border-radius:4px; background:transparent;
  box-shadow:0 0 0 1px rgba(255,255,255,.55) inset; cursor:pointer; line-height:1;}
.ocr-box-label {position:absolute; top:-24px; left:-2px; display:flex; gap:5px; align-items:center;
  max-width:190px; padding:4px 7px; border-radius:6px 6px 6px 0;
  background:var(--block-color,#2563eb); color:white; font:700 .65rem/1 var(--font,ui-sans-serif,system-ui,sans-serif);
  letter-spacing:.025em; text-transform:uppercase; white-space:nowrap; box-shadow:0 2px 7px rgba(15,23,42,.2);}
.ocr-box-label em {font-style:normal; opacity:.9;}
#ocr-extracted-page {padding:24px 28px !important;
  color:var(--body-text-color); font-family:var(--font,ui-sans-serif,system-ui,sans-serif);}
#ocr-extracted-page .prose {max-width:none; color:var(--body-text-color);
  font-family:inherit; line-height:1.62; min-width:0; overflow-wrap:anywhere;}
#ocr-extracted-page .prose :is(h1,h2,h3,h4,h5,h6,p,li,th,td) {
  color:var(--body-text-color); font-family:inherit;}
.ocr-block-list {display:flex; width:100%; min-width:0; flex-direction:column; gap:10px;}
.ocr-block-card {display:block; width:100%; padding:12px 14px; text-align:left;
  min-width:0; max-width:100%; box-sizing:border-box; overflow:hidden;
  border:1px solid color-mix(in srgb,var(--block-color,#2563eb) 45%,var(--border-color-primary));
  border-left:4px solid var(--block-color,#2563eb); border-radius:9px;
  background:var(--background-fill-secondary); color:var(--body-text-color); cursor:pointer;
  font-family:inherit; transition:transform .12s ease,box-shadow .12s ease,background .12s ease;}
.ocr-block-card-header {display:flex; align-items:center; justify-content:space-between; gap:10px;
  margin-bottom:7px;}
.ocr-block-type {color:var(--block-color,#2563eb); font-size:.68rem; font-weight:800;
  letter-spacing:.06em; text-transform:uppercase;}
.ocr-confidence-badge {padding:3px 7px; border-radius:999px; font-size:.68rem; font-weight:800;}
.confidence-high .ocr-confidence-badge {background:#dcfce7; color:#166534;}
.confidence-medium .ocr-confidence-badge {background:#fef3c7; color:#92400e;}
.confidence-low .ocr-confidence-badge {background:#fee2e2; color:#991b1b;}
.confidence-unknown .ocr-confidence-badge {background:#e2e8f0; color:#475569;}
.ocr-block-content {display:block; color:var(--body-text-color); font-size:.9rem; line-height:1.5;
  white-space:pre-wrap; overflow-wrap:anywhere; word-break:break-word;}
.ocr-block-meta {display:block; margin-top:7px; color:var(--body-text-color-subdued); font-size:.7rem;}
.ocr-empty-state {padding:48px 24px; color:var(--body-text-color-subdued); text-align:center;}
.ocr-clean-markdown {color:var(--body-text-color); line-height:1.62;}
.ocr-clean-markdown :is(h1,h2,h3,h4,h5,h6) {color:var(--body-text-color);}
.type-title {--block-color:#7c3aed;} .type-text {--block-color:#2563eb;}
.type-list {--block-color:#d97706;} .type-table {--block-color:#0f766e;}
.type-equation,.type-code {--block-color:#be123c;}
.type-image,.type-caption {--block-color:#64748b;}
.type-header,.type-footer,.type-aside-text,.type-references,.type-signature {--block-color:#475569;}
.ocr-source-box:focus,.ocr-block-card:focus {outline:3px solid color-mix(in srgb,var(--block-color,#2563eb) 35%,transparent); outline-offset:2px;}
#document-page-navigation {align-items:center; max-width:820px; margin:12px auto 20px; gap:10px;}
#document-page-navigation button {border-radius:999px;}
#document-page-slider {padding-top:4px;}
.document-page-status {display:flex; align-items:baseline; justify-content:center; gap:6px;
  min-width:96px; color:var(--body-text-color-subdued); font-size:.8rem;}
.document-page-status strong {color:var(--body-text-color); font-size:1.05rem;}
@media (max-width:900px) {
  #source-pdf-page, #ocr-extracted-page {
    height:clamp(420px,62vh,560px) !important;
    max-height:560px !important;
  }
  #ocr-extracted-page {padding:18px 20px !important;}
}
"""


def _metric_percent(value, digits=1):
    return "n.a." if value is None else f"{100 * float(value):.{digits}f}%"


def _display_decimal(value, digits=4):
    return None if value is None else round(float(value), digits)


def _rendered_page_number(path: Path) -> int:
    match = re.search(r"-(\d+)$", path.stem)
    return int(match.group(1)) if match else 0


def render_pdf_pages(pdf: Path, expected_pages: int) -> list[str]:
    """Render stable, layout-faithful PNGs for the synchronized document inspector."""
    digest = hashlib.sha256(pdf.read_bytes()).hexdigest()[:24]
    output_dir = CACHE_DIR / "pdf_pages" / digest
    output_dir.mkdir(parents=True, exist_ok=True)
    pages = sorted(output_dir.glob("page-*.png"), key=_rendered_page_number)
    renders_complete = (len(pages) == expected_pages
                        and all(path.stat().st_size > 0 for path in pages))
    if not renders_complete:
        try:
            import pymupdf
        except ImportError:
            pymupdf = None
        if pymupdf is not None:
            with pymupdf.open(pdf) as document:
                if document.page_count != expected_pages:
                    raise RuntimeError(
                        f"PDF contains {document.page_count} pages but OCR returned {expected_pages}")
                scale = 160 / 72
                matrix = pymupdf.Matrix(scale, scale)
                for index, page in enumerate(document, 1):
                    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                    pixmap.save(output_dir / f"page-{index}.png")
        else:
            renderer = shutil.which("pdftoppm")
            if not renderer:
                raise RuntimeError(
                    "PDF rendering requires the pymupdf package or Poppler's pdftoppm")
            completed = subprocess.run(
                [renderer, "-png", "-r", "160", str(pdf), str(output_dir / "page")],
                check=False, capture_output=True, text=True, timeout=180)
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "unknown render error").strip()
                raise RuntimeError(f"Could not render PDF pages: {detail[:300]}")
        pages = sorted(output_dir.glob("page-*.png"), key=_rendered_page_number)
    if (len(pages) != expected_pages
            or any(path.stat().st_size <= 0 for path in pages)):
        raise RuntimeError(
            f"Rendered {len(pages)} PDF pages but OCR returned {expected_pages}")
    return [str(path) for path in pages]


def _document_page_status(page: int, total: int) -> str:
    return (f'<div class="document-page-status"><span>Page</span>'
            f'<strong>{page}</strong><span>of {total}</span></div>')


def _confidence(value):
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, score))


def _confidence_text(score) -> str:
    return "n.a." if score is None else f"{100 * score:.0f}%"


def _confidence_tone(score) -> str:
    if score is None:
        return "confidence-unknown"
    if score >= 0.95:
        return "confidence-high"
    if score >= 0.80:
        return "confidence-medium"
    return "confidence-low"


def _block_score(block: dict):
    scores = block.get("confidence_scores") or {}
    return _confidence(scores.get("average_content_confidence_score"))


def _block_type(block: dict) -> tuple[str, str]:
    raw = str(block.get("type") or "text").strip().lower()
    safe = re.sub(r"[^a-z0-9-]+", "-", raw.replace("_", "-")) or "text"
    return safe, raw.replace("_", " ").title()


def _page_blocks(page_detail):
    visible = []
    for index, block in enumerate((page_detail or {}).get("blocks") or []):
        if not isinstance(block, dict):
            continue
        score = _block_score(block)
        visible.append((index, block, score))
    return visible


def _block_geometry(block: dict, page_detail: dict):
    dimensions = (page_detail or {}).get("dimensions") or {}
    try:
        width, height = float(dimensions["width"]), float(dimensions["height"])
        x1, y1 = float(block["top_left_x"]), float(block["top_left_y"])
        x2, y2 = float(block["bottom_right_x"]), float(block["bottom_right_y"])
    except (KeyError, TypeError, ValueError):
        return None
    if width <= 0 or height <= 0 or x2 <= x1 or y2 <= y1:
        return None
    left = max(0.0, min(100.0, 100 * x1 / width))
    top = max(0.0, min(100.0, 100 * y1 / height))
    right = max(left, min(100.0, 100 * x2 / width))
    bottom = max(top, min(100.0, 100 * y2 / height))
    if right <= left or bottom <= top:
        return None
    return left, top, right - left, bottom - top


def _block_tooltip(block: dict, score) -> str:
    scores = block.get("confidence_scores") or {}
    _, label = _block_type(block)
    minimum = _confidence(scores.get("minimum_content_confidence_score"))
    type_score = _confidence(scores.get("block_type_confidence_score"))
    return (f"{label} | average content {_confidence_text(score)} | "
            f"minimum content {_confidence_text(minimum)} | "
            f"block type {_confidence_text(type_score)}")


def _source_block_overlay(image_path, page_detail, show_outlines, show_labels):
    encoded = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
    page_scores = (page_detail or {}).get("confidence_scores") or {}
    page_average = _confidence(page_scores.get("average_page_confidence_score"))
    page_minimum = _confidence(page_scores.get("minimum_page_confidence_score"))
    page_summary = (f"Page confidence · Avg {_confidence_text(page_average)} · "
                    f"Min {_confidence_text(page_minimum)}")
    boxes = []
    focus_rules = []
    if show_outlines:
        for index, block, score in _page_blocks(page_detail):
            geometry = _block_geometry(block, page_detail or {})
            if geometry is None:
                continue
            left, top, width, height = geometry
            type_class, type_label = _block_type(block)
            tooltip = html.escape(_block_tooltip(block, score), quote=True)
            label = (f'<span class="ocr-box-label">{html.escape(type_label)}'
                     f'<em>{_confidence_text(score)}</em></span>') if show_labels else ""
            link_class = f"block-link-{index}"
            boxes.append(
                f'<button type="button" class="ocr-source-box {link_class} '
                f'type-{type_class} {_confidence_tone(score)}" '
                f'style="left:{left:.4f}%;top:{top:.4f}%;width:{width:.4f}%;height:{height:.4f}%" '
                f'title="{tooltip}" aria-label="{tooltip}">{label}</button>')
            focus_rules.append(
                f'#document-page-inspector:has(.{link_class}:hover) .{link_class},'
                f'#document-page-inspector:has(.{link_class}:focus) .{link_class}'
                '{box-shadow:0 0 0 3px color-mix(in srgb,var(--block-color,#2563eb) 35%,transparent);'
                'background:color-mix(in srgb,var(--block-color,#2563eb) 13%,transparent);z-index:10;}')
    return (
        f'<style>{"".join(focus_rules)}</style>'
        '<div class="pdf-block-viewport">'
        f'<div class="page-confidence-summary">{html.escape(page_summary)}</div>'
        '<div class="pdf-block-stage">'
        f'<img src="data:image/png;base64,{encoded}" alt="Rendered original PDF page">'
        f'{"".join(boxes)}</div></div>')


def _clean_markdown_html(markdown_text: str) -> str:
    # Escape raw HTML from the untrusted OCR document before applying Markdown syntax.
    safe_markdown = html.escape(str(markdown_text or ""))
    rendered = markdown_renderer.markdown(
        safe_markdown, extensions=["tables", "fenced_code", "sane_lists"])
    return f'<div class="ocr-clean-markdown">{rendered}</div>'


def _ocr_block_html(markdown_text, page_detail, view_mode, show_labels):
    if view_mode == "Clean Markdown":
        return _clean_markdown_html(markdown_text)
    blocks = _page_blocks(page_detail)
    if not blocks:
        reason = "No structural block metadata was returned for this page."
        return (f'<div class="ocr-empty-state">{html.escape(reason)}</div>'
                + _clean_markdown_html(markdown_text))
    cards = []
    for index, block, score in blocks:
        type_class, type_label = _block_type(block)
        scores = block.get("confidence_scores") or {}
        minimum = _confidence(scores.get("minimum_content_confidence_score"))
        type_score = _confidence(scores.get("block_type_confidence_score"))
        content = html.escape(str(block.get("content") or "No textual content"))
        tooltip = html.escape(_block_tooltip(block, score), quote=True)
        badge = (f'<span class="ocr-confidence-badge">'
                 f'{_confidence_text(score)}</span>') if show_labels else ""
        metadata = (
            f'<span class="ocr-block-meta">Minimum {_confidence_text(minimum)} · '
            f'Type {_confidence_text(type_score)}</span>' if show_labels else "")
        cards.append(
            f'<button type="button" class="ocr-block-card block-link-{index} '
            f'type-{type_class} {_confidence_tone(score)}" title="{tooltip}">'
            '<span class="ocr-block-card-header">'
            f'<span class="ocr-block-type">{html.escape(type_label)}</span>'
            f'{badge}</span>'
            f'<span class="ocr-block-content">{content}</span>'
            f'{metadata}</button>')
    return f'<div class="ocr-block-list">{"".join(cards)}</div>'


def select_document_page(state, requested_page, view_mode="Block inspector",
                         show_outlines=True, show_labels=True):
    """Return synchronized source and OCR block views and clamp navigation."""
    state = dict(state or {})
    images = list(state.get("images") or [])
    ocr_pages = list(state.get("ocr_pages") or [])
    total = min(len(images), len(ocr_pages))
    if not total:
        return (state, '<div class="ocr-empty-state">No PDF page loaded.</div>',
                '<div class="ocr-empty-state">No OCR page loaded.</div>',
                _document_page_status(0, 0),
                gr.update(value=1, minimum=1, maximum=2, interactive=False))
    try:
        page = int(round(float(requested_page)))
    except (TypeError, ValueError):
        page = int(state.get("page") or 1)
    page = max(1, min(total, page))
    state["page"] = page
    markdown = ocr_pages[page - 1].strip()
    page_details = list(state.get("page_details") or [])
    page_detail = page_details[page - 1] if page <= len(page_details) else {}
    source_html = _source_block_overlay(
        images[page - 1], page_detail, bool(show_outlines), bool(show_labels))
    ocr_html = _ocr_block_html(
        markdown, page_detail, view_mode, bool(show_labels))
    return (state, source_html, ocr_html, _document_page_status(page, total),
            gr.update(value=page, minimum=1, maximum=max(2, total),
                      interactive=total > 1))


def step_document_page(state, delta, view_mode="Block inspector",
                       show_outlines=True, show_labels=True):
    current = int((state or {}).get("page") or 1)
    return select_document_page(
        state, current + int(delta), view_mode, show_outlines, show_labels)


def refresh_document_page(state, view_mode, show_outlines, show_labels):
    return select_document_page(
        state, (state or {}).get("page", 1), view_mode, show_outlines,
        show_labels)


def _assignment_accuracy_display(summary):
    assignment = assignment_accuracy_metrics(summary)
    score = assignment["assignment_accuracy"]
    if score is None:
        return "n.a.", "Requires at least one successful optional-judge assessment", "warn"
    judge_scored = assignment["assignment_judge_scored_questions"]
    judge_total = assignment["assignment_judge_total_questions"]
    detail = (
        f'Text {_metric_percent(assignment["assignment_text_extraction_score"])} · '
        f'Topic {_metric_percent(assignment["assignment_topic_score"])} · '
        f'Long form {_metric_percent(assignment["assignment_long_form_score"])} · '
        f'Success {_metric_percent(assignment["assignment_processing_success_rate"])} · '
        f'Judge coverage {judge_scored}/{judge_total} '
        f'({_metric_percent(assignment["assignment_judge_coverage_rate"])})'
    )
    tone = ("warn" if assignment["assignment_accuracy_status"] ==
            "available_partial_optional_judge_coverage" else "neutral")
    return _metric_percent(score, 2), detail, tone


def executive_metrics_html(group, result, scored, summary):
    """Render the single-document metrics as a compact, escaped executive scorecard."""
    judge_scores = [q["judge_semantic_alignment_1_to_5"] for q in scored["questions"]
                    if q["judge_semantic_alignment_1_to_5"] is not None]
    judge_mean = sum(judge_scores) / len(judge_scores) if judge_scores else None
    judge_pass = (sum(score >= 4 for score in judge_scores) / len(judge_scores)
                  if judge_scores else None)
    judge_error = scored.get("judge_error")
    judge_detail = ("Optional judge failed; core result retained" if judge_error
                    else "Optional model-rated semantic alignment")
    core_cost = float(result["cost"]["full_inference_estimated_usd"])
    core_cost_per_page = core_cost / scored["pages"] if scored["pages"] else None
    ocr_cost = float(result["cost"].get("ocr_usd") or 0)
    ocr_cost_per_page = ocr_cost / scored["pages"] if scored["pages"] else None
    ocr_seconds = scored.get("ocr_processing_seconds")
    ocr_seconds_per_page = scored.get("ocr_processing_seconds_per_page")
    ocr_timing_detail = (
        "Unavailable for legacy cached OCR"
        if ocr_seconds is None else
        f'{float(ocr_seconds):.3f}s raw OCR request · '
        f'{"original measurement; OCR cached now" if result["ocr_cached"] else "fresh OCR call"}'
    )
    assignment_value, assignment_detail, assignment_tone = _assignment_accuracy_display(summary)

    def card(label, value, detail, tone="neutral"):
        return (f'<div class="exec-card"><div class="exec-label">{html.escape(label)}</div>'
                f'<div class="exec-value exec-{tone}">{html.escape(str(value))}</div>'
                f'<div class="exec-detail">{html.escape(detail)}</div></div>')

    topic_strategy = result.get("topic_tab1_strategy") or {}
    topic_source = topic_strategy.get("final_source")
    if topic_source == "large_backup":
        topic_detail = "Large backup used after Small " + (
            "error" if topic_strategy.get("primary_status") == "error" else "reference miss")
    elif topic_source == "small_primary":
        topic_detail = "Mistral Small primary matched the reference"
    elif topic_source == "small_primary_retained":
        topic_detail = "Small result retained after the Large backup failed"
    else:
        topic_detail = f'Reference: {group["topic"]}'

    cards = [
        card("Pages", scored["pages"], "Pages returned by OCR", "good"),
        card("OCR processing time/page",
             "n.a." if ocr_seconds_per_page is None
             else f'{float(ocr_seconds_per_page):.3f}s',
             ocr_timing_detail, "good"),
        card("OCR word error rate", _metric_percent(scored["ocr_proxy_wer"], 3),
             "Lower is better; reference-text proxy", "good"),
        card("OCR character error rate", _metric_percent(scored["ocr_proxy_cer"], 3),
             "Lower is better; reference-text proxy", "good"),
        card("Exact Span Recall", _metric_percent(summary.get("exact_span_recall")),
             f'{summary.get("exact_span_recovered_questions", 0)}/'
             f'{summary.get("exact_span_eligible_questions", 0)} answerable gold spans found in OCR text',
             "good"),
        card("OCR Token F1", _metric_percent(scored.get("ocr_proxy_token_f1")),
             "Whole-document token overlap; order-insensitive reference-text proxy", "good"),
        card("Topic match", "Correct" if scored["topic_correct"] else "Incorrect",
             topic_detail, "good" if scored["topic_correct"] else "warn"),
        card("End-to-end assignment accuracy", assignment_value,
             assignment_detail, assignment_tone),
        card("Core API cost per page", "n.a." if core_cost_per_page is None
             else f"${core_cost_per_page:.5f}",
             "OCR + embedding + topic + QA, divided by document pages"),
        card("Total cost per page", "n.a." if ocr_cost_per_page is None
             else f"${ocr_cost_per_page:.5f}",
             "OCR API only; excludes embedding, topic, QA and optional judge"),
        card("Answerability", _metric_percent(summary.get("answerability_accuracy")),
             "Correct answer-versus-abstain decisions"),
        card("Exact match", _metric_percent(summary.get("qa_answerable_em")),
             "Answerable questions only"),
        card("Long-form judge mean", "Not run" if judge_mean is None else f"{judge_mean:.2f}/5",
             judge_detail, "warn" if judge_error else "neutral"),
        card("Judge pass rate", "Not run" if judge_pass is None else _metric_percent(judge_pass),
             "Judge unavailable" if judge_error else "Individual scores of 4 or 5",
             "warn" if judge_error else "neutral"),
        card("Evidence located", _metric_percent(summary.get("evidence_quote_location_rate")),
             "Quote location only; not entailment"),
    ]
    title = html.escape(str(group["document_id"]))
    topic = html.escape(str(result["analysis"]["topic"]))
    context = (f'{html.escape(str(group["split"]))} · {scored["pages"]} pages · '
               f'Predicted topic: {topic}')
    cache_state = (f'OCR {"cached" if result["ocr_cached"] else "fresh"}, '
                   f'topic {"cached" if result["topic_cached"] else "fresh"}, '
                   f'QA {"cached" if result["qa_cached"] else "fresh"}')
    return (f'<div class="exec-shell"><div class="exec-heading"><div><h2>{title}</h2>'
            f'<div class="exec-context">{context}</div></div>'
            f'<div class="exec-context">{cache_state}</div>'
            f'</div><div class="exec-grid">{"".join(cards)}</div>'
            '<p class="exec-note">Exact match and token F1 exclude reference-unanswerable questions. '
            'The optional judge evaluates semantic alignment of generated short and long answers; '
            'it is not a measured production accuracy rate.</p></div>')


def answer_quality_header_html(group, result):
    predicted = html.escape(str(result["analysis"]["topic"]))
    reference = html.escape(str(group["topic"]))
    strategy = result.get("topic_tab1_strategy") or {}
    source = strategy.get("final_source")
    source_label = ("Mistral Small primary" if source == "small_primary" else
                    "Mistral Large backup" if source == "large_backup" else
                    "Mistral Small retained" if source == "small_primary_retained" else None)
    source_pill = (f'<span class="topic-pill"><strong>Topic source:</strong> '
                   f'{html.escape(source_label)}</span>') if source_label else ""
    return ('<div class="answer-quality-header">'
            '<div class="answer-quality-title">Individual answer quality</div>'
            '<div class="answer-quality-topics">'
            f'<span class="topic-pill"><strong>Predicted topic:</strong> {predicted}</span>'
            f'<span class="topic-pill"><strong>Reference topic:</strong> {reference}</span>'
            f'{source_pill}'
            '</div></div>')


def _tab1_runs_dir():
    return RESULTS_DIR / "tab1_runs"


def recent_tab1_runs(limit=6):
    """Return the newest stored Tab 1 runs with correct final topic selection."""
    run_dir = _tab1_runs_dir()
    if not run_dir.exists():
        return []

    runs = []
    for run_path in run_dir.glob("*.json"):
        try:
            with run_path.open(encoding="utf-8") as handle:
                stored = json.load(handle)
            if not isinstance(stored, dict) or not bool(stored.get("topic_correct")):
                continue
            modified = run_path.stat().st_mtime
            try:
                created = datetime.fromisoformat(str(stored.get("created_at") or ""))
                chronological_key = created.timestamp()
                display_time = created.astimezone()
            except ValueError:
                chronological_key = modified
                display_time = datetime.fromtimestamp(modified).astimezone()
            timestamp = display_time.strftime("%b %d, %Y %I:%M %p %Z")
            runs.append((chronological_key, f"TS - {timestamp}", run_path.name))
        except (OSError, ValueError, TypeError):
            continue

    runs.sort(key=lambda item: item[0], reverse=True)
    return [(label, value) for _, label, value in runs[:limit]]


def _store_tab1_run(group, scored, summary_html, quality_header_html,
                    question_rows, raw_details, page_state):
    """Persist one inspection snapshot; incorrect topics remain audit-only."""
    created = datetime.now().astimezone()
    safe_document_id = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(group["document_id"]))
    run_name = f'{created.strftime("%Y%m%dT%H%M%S%f%z")}-{safe_document_id}.json'
    atomic_json(_tab1_runs_dir() / run_name, {
        "schema_version": 1,
        "created_at": created.isoformat(),
        "document_id": group["document_id"],
        "topic_correct": bool(scored.get("topic_correct")),
        "summary_html": summary_html,
        "quality_header_html": quality_header_html,
        "question_rows": question_rows,
        "raw_details": raw_details,
        "page_state": page_state,
    })
    return run_name


def load_stored_tab1_run(run_name, view_mode="Block inspector",
                         show_outlines=True, show_labels=True):
    """Render one eligible Tab 1 snapshot without invoking document APIs."""
    allowed_names = {value for _, value in recent_tab1_runs()}
    if not run_name or run_name not in allowed_names:
        return tuple(gr.skip() for _ in range(10))
    run_path = _tab1_runs_dir() / run_name
    try:
        with run_path.open(encoding="utf-8") as handle:
            stored = json.load(handle)
        if not bool(stored.get("topic_correct")):
            raise ValueError("The stored run has an incorrect topic selection")
        page_state = stored["page_state"]
        synchronized_page = select_document_page(
            page_state, 1, view_mode, show_outlines, show_labels)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise gr.Error(f"Could not load stored Tab 1 run: {type(exc).__name__}: {exc}") from exc
    return (stored["summary_html"], stored["quality_header_html"],
            pd.DataFrame(stored.get("question_rows") or []), stored["raw_details"],
            *synchronized_page, gr.update(visible=True))


def evaluation_metrics_html(summary):
    """Render aggregate benchmark results without hiding the raw summary JSON."""
    if not summary:
        return ""

    def card(label, value, detail="", tone="neutral"):
        return (f'<div class="exec-card"><div class="exec-label">{html.escape(label)}</div>'
                f'<div class="exec-value exec-{tone}">{html.escape(str(value))}</div>'
                f'<div class="exec-detail">{html.escape(detail)}</div></div>')

    intervals = summary.get("confidence_intervals", {}).get("intervals", {})

    def interval_detail(metric, fallback):
        ci = intervals.get(metric) or {}
        if ci.get("lower_95") is None or ci.get("upper_95") is None:
            return fallback
        return (f'95% CI {_metric_percent(ci["lower_95"])}–'
                f'{_metric_percent(ci["upper_95"])}')

    pages = int(summary.get("pages") or 0)
    core_cost = float(summary.get("full_run_api_cost_estimated_usd") or 0)
    judge_cost = float(summary.get("judge_api_cost_estimated_usd") or 0)
    total_cost = core_cost + judge_cost
    core_cost_per_page = core_cost / pages if pages else None
    total_cost_per_page = total_cost / pages if pages else None
    ocr_cost = summary.get("ocr_api_cost_estimated_usd")
    ocr_cost_per_page = (float(ocr_cost) / pages
                         if ocr_cost is not None and pages else None)
    failed = int(summary.get("failed_documents") or 0)
    judge_failed = int(summary.get("judge_failed_documents") or 0)
    topic_detail = interval_detail("topic_accuracy", "Exact 17-label primary match")
    refinement_count = int(summary.get("topic_runner_up_large_triggered_count") or 0)
    refinement_fallbacks = int(summary.get("topic_runner_up_large_fallback_count") or 0)
    refinement_improved = int(summary.get("topic_runner_up_large_improved_count") or 0)
    if refinement_count:
        topic_detail += (f' · Large runner-up experiment triggered {refinement_count} times; '
                         f'corrected {refinement_improved}; '
                         f'{refinement_fallbacks} fell back to Small')
    assignment_value, assignment_detail, assignment_tone = _assignment_accuracy_display(summary)

    cards = [
        card("Documents evaluated", f'{summary.get("evaluated_documents", 0)}/'
             f'{summary.get("requested_documents", 0)}',
             f'{failed} failed', "good" if failed == 0 else "warn"),
        card("Pages", pages, "Across all evaluated PDFs", "good"),
        card("OCR processing time/page",
             "n.a." if summary.get("ocr_processing_seconds_per_page") is None
             else f'{float(summary["ocr_processing_seconds_per_page"]):.3f}s',
             ("No timed OCR calls in these results"
              if summary.get("ocr_processing_seconds_total") is None else
              f'{float(summary["ocr_processing_seconds_total"]):.3f}s across '
              f'{summary.get("ocr_timed_pages", 0)} timed pages'),
             "good"),
        card("OCR word error rate", _metric_percent(summary.get("ocr_proxy_micro_wer"), 3),
             interval_detail("ocr_proxy_micro_wer", "Micro WER; lower is better"), "good"),
        card("OCR character error rate", _metric_percent(summary.get("ocr_proxy_micro_cer"), 3),
             interval_detail("ocr_proxy_micro_cer", "Micro CER; lower is better"), "good"),
        card("Exact Span Recall", _metric_percent(summary.get("exact_span_recall")),
             interval_detail(
                 "exact_span_recall",
                 f'{summary.get("exact_span_recovered_questions", 0)}/'
                 f'{summary.get("exact_span_eligible_questions", 0)} answerable gold spans found in OCR text'),
             "good"),
        card("OCR Token F1", _metric_percent(summary.get("ocr_proxy_micro_token_f1")),
             interval_detail(
                 "ocr_proxy_micro_token_f1",
                 "Micro whole-document token overlap; order-insensitive reference-text proxy"),
             "good"),
        card("Topic accuracy", _metric_percent(summary.get("topic_accuracy")),
             topic_detail),
        card("End-to-end assignment accuracy", assignment_value,
             assignment_detail, assignment_tone),
        card("Core API cost per page", "n.a." if core_cost_per_page is None
             else f'${core_cost_per_page:.5f}',
             "OCR + embedding + topic + QA across evaluated PDFs"),
        card("Total cost per page", "n.a." if ocr_cost_per_page is None
             else f'${ocr_cost_per_page:.5f}',
             "OCR API only; excludes embedding, topic, QA and optional judge"),
        card("Answerability", _metric_percent(summary.get("answerability_accuracy")),
             interval_detail("answerability_accuracy", "Answer-versus-abstain decisions")),
        card("Long-form judge mean", "Not run" if summary.get("judge_semantic_alignment_mean_1_to_5") is None
             else f'{float(summary["judge_semantic_alignment_mean_1_to_5"]):.2f}/5',
             f'{summary.get("judge_scored_questions", 0)} questions scored · '
             f'{judge_failed} document failures', "warn" if judge_failed else "neutral"),
        card("Judge pass rate", _metric_percent(
             summary.get("judge_semantic_alignment_pass_rate_ge_4")),
             "Individual scores of 4 or 5"),
        card("Evidence located", _metric_percent(summary.get("evidence_quote_location_rate")),
             "Quote location only; not entailment"),
        card("Total estimated API cost per page", "n.a." if total_cost_per_page is None
             else f'${total_cost_per_page:.5f}',
             "Core pipeline plus optional judge, divided by evaluated pages"),
    ]
    status = html.escape(str(summary.get("status", "UNKNOWN")))
    split = html.escape(str(summary.get("split", "")))
    model_config = summary.get("model_config") or {}
    model = html.escape(str(model_config.get("topic_model", "")))
    refinement = model_config.get("topic_runner_up_large_experiment") or {}
    refinement_model = html.escape(str(refinement.get("model", "off")))
    embedding_mode = html.escape(str((model_config.get("topic_embedding") or {}).get("mode", "off")))
    ci_method = html.escape(str((summary.get("confidence_intervals") or {}).get("method",
                                "No confidence interval")))
    return (f'<div class="exec-shell"><div class="exec-heading"><div>'
            f'<h2>Evaluation summary · {status}</h2>'
            f'<div class="exec-context">{split} · topic model: {model} · '
            f'runner-up refiner: {refinement_model} · embedding: {embedding_mode}</div></div>'
            f'<div class="exec-context">{ci_method}</div></div>'
            f'<div class="exec-grid">{"".join(cards)}</div>'
            '<p class="exec-note">OCR WER/CER use the dataset extracted-text proxy. Judge scores '
            'measure reference alignment, not production accuracy. Cost values are modeled from '
            'reported token and page usage.</p></div>')


def recent_evaluation_runs(limit=6):
    """Return stored evaluation runs in descending completion chronology."""
    if not RESULTS_DIR.exists():
        return []

    runs = []
    for summary_path in RESULTS_DIR.glob("*/summary.json"):
        try:
            with summary_path.open(encoding="utf-8") as handle:
                summary = json.load(handle)
            if not isinstance(summary, dict):
                continue
            modified = summary_path.stat().st_mtime
            timestamp = datetime.fromtimestamp(modified).astimezone().strftime("%b %d, %Y %I:%M %p %Z")
            label = f"TS - {timestamp}"
            runs.append((modified, label, summary_path.parent.name))
        except (OSError, ValueError, TypeError):
            continue

    runs.sort(key=lambda item: item[0], reverse=True)
    return [(label, value) for _, label, value in runs[:limit]]


def _stored_evaluation_outputs(run_name):
    """Load tab 2 outputs from one of the six recent run directories."""
    choices = recent_evaluation_runs()
    allowed_names = {value for _, value in choices}
    if not run_name or run_name not in allowed_names:
        return tuple(gr.skip() for _ in range(6))

    run_dir = RESULTS_DIR / run_name
    summary_path = run_dir / "summary.json"
    topic_path = run_dir / "per_topic.csv"
    question_path = run_dir / "per_question.csv"
    try:
        with summary_path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
        table = pd.read_csv(topic_path) if topic_path.exists() else pd.DataFrame()
    except (OSError, ValueError, TypeError) as exc:
        raise gr.Error(f"Could not load stored evaluation: {type(exc).__name__}: {exc}") from exc

    status = str(summary.get("status") or "UNKNOWN")
    evaluated = int(summary.get("evaluated_documents") or 0)
    requested = int(summary.get("requested_documents") or 0)
    progress = (f"**Stored result · {status}** — {evaluated}/{requested} PDFs. "
                "No API request was made. Click **Run or resume evaluation** to execute "
                "the configured evaluation workflow.")
    return (progress, evaluation_metrics_html(summary), summary, table,
            str(summary_path), str(question_path) if question_path.exists() else None)


def load_stored_evaluation(run_name):
    """Dropdown handler: render a stored run without entering the API workflow."""
    return _stored_evaluation_outputs(run_name)


def initial_recent_evaluation_view():
    """Populate recent choices and show the latest stored run when the app opens."""
    choices = recent_evaluation_runs()
    if not choices:
        return (gr.update(choices=[], value=None),
                "No stored evaluations are available.", "", {}, pd.DataFrame(), None, None)
    latest = choices[0][1]
    return (gr.update(choices=choices, value=latest),
            *_stored_evaluation_outputs(latest))


def aggregated_runs_snapshot():
    """Return a read-only, deduplicated snapshot of compatible completed runs."""
    latest = {}
    if not RESULTS_DIR.exists():
        return {}, pd.DataFrame()

    for summary_path in RESULTS_DIR.glob("*/summary.json"):
        try:
            with summary_path.open(encoding="utf-8") as handle:
                run_summary = json.load(handle)
            if (run_summary.get("status") != "COMPLETE"
                    or run_summary.get("metrics_version") != METRICS_VERSION):
                continue
            split = str(run_summary.get("split") or "")
            modified = summary_path.stat().st_mtime
            for document_path in (summary_path.parent / "documents").glob("*.json"):
                with document_path.open(encoding="utf-8") as handle:
                    document = json.load(handle)
                key = (split, str(document["document_id"]))
                if key not in latest or modified > latest[key][0]:
                    latest[key] = (modified, document)
        except (OSError, ValueError, KeyError, TypeError):
            continue

    documents = [item[1] for item in latest.values()]
    if not documents:
        return {}, pd.DataFrame()

    summary = aggregate(documents, incumbent_per_page=settings.incumbent_per_page)
    summary.update({"requested_documents": len(documents),
                    "evaluated_documents": len(documents),
                    "failed_documents": 0})
    summary.update(assignment_accuracy_metrics(summary))

    by_topic = {}
    for document in documents:
        by_topic.setdefault(document["topic_gold"], []).append(document)
    topic_rows = []
    for topic, topic_documents in sorted(by_topic.items()):
        topic_summary = aggregate(
            topic_documents, incumbent_per_page=settings.incumbent_per_page)
        topic_summary.update({"requested_documents": len(topic_documents),
                              "evaluated_documents": len(topic_documents)})
        topic_summary.update(assignment_accuracy_metrics(topic_summary))
        pages = int(topic_summary.get("pages") or 0)
        core_cost = float(topic_summary.get("full_run_api_cost_estimated_usd") or 0)
        judge_cost = float(topic_summary.get("judge_api_cost_estimated_usd") or 0)
        topic_rows.append({
            "Topic": topic,
            "Documents": topic_summary["documents"],
            "Pages": pages,
            "OCR WER": _metric_percent(topic_summary.get("ocr_proxy_micro_wer"), 3),
            "OCR CER": _metric_percent(topic_summary.get("ocr_proxy_micro_cer"), 3),
            "Exact Span Recall": _metric_percent(topic_summary.get("exact_span_recall")),
            "OCR Token F1": _metric_percent(
                topic_summary.get("ocr_proxy_micro_token_f1")),
            "Topic Accuracy": _metric_percent(topic_summary.get("topic_accuracy")),
            "Answerability": _metric_percent(topic_summary.get("answerability_accuracy")),
            "Token F1": _metric_percent(topic_summary.get("qa_answerable_token_f1")),
            "Judge Mean": ("n.a." if topic_summary.get(
                "judge_semantic_alignment_mean_1_to_5") is None else
                f'{float(topic_summary["judge_semantic_alignment_mean_1_to_5"]):.2f}/5'),
            "Evidence Located": _metric_percent(
                topic_summary.get("evidence_quote_location_rate")),
            "End-to-End Accuracy": _metric_percent(
                topic_summary.get("assignment_accuracy"), 2),
            "Core Cost/Page": "n.a." if not pages else f'${core_cost / pages:.5f}',
            "Total Cost/Page": ("n.a." if not pages else
                                f'${(core_cost + judge_cost) / pages:.5f}'),
        })
    return summary, pd.DataFrame(topic_rows)


def aggregated_runs_metrics_html(summary):
    """Render the display-only aggregate scorecard."""
    if not summary:
        return ('<div class="exec-shell"><div class="ocr-empty-state">'
                'No compatible completed runs are available.</div></div>')

    def card(label, value, detail="", tone="neutral"):
        return (f'<div class="exec-card"><div class="exec-label">{html.escape(label)}</div>'
                f'<div class="exec-value exec-{tone}">{html.escape(str(value))}</div>'
                f'<div class="exec-detail">{html.escape(detail)}</div></div>')

    pages = int(summary.get("pages") or 0)
    core_cost = float(summary.get("full_run_api_cost_estimated_usd") or 0)
    judge_cost = float(summary.get("judge_api_cost_estimated_usd") or 0)
    ocr_cost = float(summary.get("ocr_api_cost_estimated_usd") or 0)
    core_cost_per_page = core_cost / pages if pages else None
    ocr_cost_per_page = ocr_cost / pages if pages else None
    total_cost_per_page = (core_cost + judge_cost) / pages if pages else None
    assignment_value, assignment_detail, assignment_tone = (
        _assignment_accuracy_display(summary))
    judge_failed = int(summary.get("judge_failed_documents") or 0)

    cards = [
        card("Documents evaluated", summary.get("documents", 0),
             "Completed document results", "good"),
        card("Pages", pages, "Across aggregated runs", "good"),
        card("OCR processing time/page",
             "n.a." if summary.get("ocr_processing_seconds_per_page") is None
             else f'{float(summary["ocr_processing_seconds_per_page"]):.3f}s',
             "Raw OCR request time", "good"),
        card("OCR word error rate", _metric_percent(
            summary.get("ocr_proxy_micro_wer"), 3), "Micro WER; lower is better", "good"),
        card("OCR character error rate", _metric_percent(
            summary.get("ocr_proxy_micro_cer"), 3), "Micro CER; lower is better", "good"),
        card("Exact Span Recall", _metric_percent(summary.get("exact_span_recall")),
             f'{summary.get("exact_span_recovered_questions", 0)}/'
             f'{summary.get("exact_span_eligible_questions", 0)} answerable gold spans',
             "good"),
        card("OCR Token F1", _metric_percent(summary.get("ocr_proxy_micro_token_f1")),
             "Micro whole-document token overlap; order-insensitive reference-text proxy",
             "good"),
        card("Topic accuracy", _metric_percent(summary.get("topic_accuracy")),
             "Exact primary-label match"),
        card("End-to-end assignment accuracy", assignment_value,
             assignment_detail, assignment_tone),
        card("Core API cost per page", "n.a." if core_cost_per_page is None
             else f'${core_cost_per_page:.5f}', "OCR + embedding + topic + QA"),
        card("Total cost per page", "n.a." if ocr_cost_per_page is None
             else f'${ocr_cost_per_page:.5f}', "OCR API only"),
        card("Answerability", _metric_percent(summary.get("answerability_accuracy")),
             "Correct answer-versus-abstain decisions"),
        card("Long-form judge mean", "Not run" if summary.get(
            "judge_semantic_alignment_mean_1_to_5") is None else
            f'{float(summary["judge_semantic_alignment_mean_1_to_5"]):.2f}/5',
            f'{summary.get("judge_scored_questions", 0)} questions scored · '
            f'{judge_failed} document failures', "warn" if judge_failed else "neutral"),
        card("Judge pass rate", _metric_percent(
            summary.get("judge_semantic_alignment_pass_rate_ge_4")),
            "Individual scores of 4 or 5"),
        card("Evidence located", _metric_percent(
            summary.get("evidence_quote_location_rate")),
            "Quote location only; not entailment"),
        card("Total estimated API cost per page",
             "n.a." if total_cost_per_page is None else f'${total_cost_per_page:.5f}',
             "Core pipeline plus optional judge"),
    ]
    return (f'<div class="exec-shell"><div class="exec-grid">'
            f'{"".join(cards)}</div></div>')


def aggregated_runs_view():
    """Refresh the display whenever the aggregate tab is opened."""
    summary, per_topic = aggregated_runs_snapshot()
    return aggregated_runs_metrics_html(summary), per_topic


def inspect_document(split, docs_to_sample, document_number, include_judge, embedding_mode,
                     view_mode="Block inspector", show_outlines=True, show_labels=True):
    run_settings = settings_for_embedding_mode(embedding_mode)
    groups = manifest(split, int(docs_to_sample), seed=42, settings=run_settings)
    index = max(0, min(len(groups) - 1, int(document_number) - 1))
    group = groups[index]
    try:
        pdf = pdf_download(group, run_settings)
        result = process_tab1_document(group, run_settings)
        page_images = render_pdf_pages(pdf, int(result["ocr"]["pages"]))
    except Exception as exc:
        raise gr.Error(f"Document inspection failed: {type(exc).__name__}: {exc}") from exc
    judged = None
    judge_error = None
    if include_judge:
        try:
            judged = evaluate_semantic(group, result, settings=run_settings)
        except Exception as exc:
            judge_error = f"{type(exc).__name__}: {str(exc)[:350]}"
    scored = score_document(group, result, judged)
    scored["judge_error"] = judge_error
    a = aggregate([scored], incumbent_per_page=run_settings.incumbent_per_page)
    a.update(assignment_accuracy_metrics(a, requested_documents=1))
    qrows = [{"Question": q["question"],
              "Predicted short answer": q["predicted_short"],
              "Reference answer": q["gold_answer"],
              "Exact match": q["answerable_em"],
              "Token F1": _display_decimal(q["answerable_token_f1"]),
              "Long-form answer": q["predicted_long"],
              "Judge rating (1–5)": q["judge_semantic_alignment_1_to_5"],
              "Judge reference-supported": q["judge_reference_supported"],
              "Answerability correct": q["correct_answerability"],
              "Evidence page": q["evidence_page"],
              "Evidence located": q["evidence_quote_verified"]} for q in scored["questions"]]
    summary_html = executive_metrics_html(group, result, scored, a)
    quality_header_html = answer_quality_header_html(group, result)
    document_metrics = {k: v for k, v in scored.items() if k != "questions"}
    raw_details = {"metrics": {"aggregate": a, "document": document_metrics,
                               "questions": scored["questions"]},
                   "model": result["model_config"], "usage": result["usage"],
                   "cost": result["cost"], "topic_classification": result["topic_classification"],
                   "topic_tab1_strategy": result.get("topic_tab1_strategy"),
                   "topic_large_classification": result.get("topic_large_classification"),
                   "topic_small_classification": result.get("topic_small_classification"),
                   "topic_refinement": result.get("topic_refinement"),
                   "topic_embedding": result.get("topic_embedding"),
                   "judge": judged if judged is not None else
                            ({"error": judge_error, "status": "unavailable"}
                             if judge_error else None)}
    page_state = {
        "document_id": group["document_id"],
        "images": page_images,
        "ocr_pages": result["ocr"]["markdown_pages"],
        "page_details": result["ocr"].get("page_details") or [],
        "page": 1,
    }
    synchronized_page = select_document_page(
        page_state, 1, view_mode, show_outlines, show_labels)
    try:
        _store_tab1_run(
            group, scored, summary_html, quality_header_html, qrows,
            raw_details, page_state)
    except (OSError, ValueError, TypeError) as exc:
        raw_details["history"] = {
            "status": "not_saved",
            "error": f"{type(exc).__name__}: {str(exc)[:350]}",
        }
    recent_choices = recent_tab1_runs()
    return (summary_html, quality_header_html, pd.DataFrame(qrows), raw_details,
            *synchronized_page, gr.update(visible=True),
            gr.update(choices=recent_choices, value=None))


def run_evaluation_ui(split, n_docs, seed, use_judge, embedding_mode):
    n_docs, seed = int(n_docs), int(seed)
    run_settings = settings_for_tab2_embedding_mode(embedding_mode)
    yield ("Evaluation started. API requests may be billed; partial results are checkpointed.",
           "", {}, pd.DataFrame(), None, None, gr.update(value=None))
    try:
        for event in iterate_evaluation(split, n_docs, seed, use_judge,
                                        UI_BOOTSTRAP_REPLICATIONS,
                                        settings=run_settings):
            if event["stage"] == "progress":
                progress = event["completed"] / event["total"]
                yield (f"**Progress:** {event['completed']}/{event['total']} PDFs ({progress:.0%}) — "
                       f"{event['scored']} scored, {event['failures']} failures, "
                       f"{event.get('judge_failures', 0)} optional judge failures.  ",
                       "", {}, pd.DataFrame(), None, None, gr.skip())
            else:
                summary = event["summary"]
                csv = Path(event["topic_csv"]) if event["topic_csv"] else None
                table = pd.read_csv(csv) if csv and csv.exists() else pd.DataFrame()
                status = f"**{summary['status']}** — {summary['evaluated_documents']}/{summary['requested_documents']} PDFs. "
                if summary["failed_documents"]:
                    status += f"**{summary['failed_documents']} failed.** Re-run the same configuration to resume. "
                if summary.get("judge_failed_documents"):
                    status += (f"**{summary['judge_failed_documents']} optional judge failures; "
                               "core results retained.** ")
                status += "Intervals resample entire PDF clusters, not individual questions."
                choices = recent_evaluation_runs()
                yield (status, evaluation_metrics_html(summary), summary, table,
                       event["summary_path"], event["question_csv"],
                       gr.update(choices=choices, value=None))
    except Exception as exc:
        raise gr.Error(f"Bulk evaluation failed: {type(exc).__name__}: {exc}") from exc


def calculate_tco(pages, pages_per_doc, questions, batch, incumbent_review, mistral_review, review_cost,
                  tokens_per_page, output_per_question):
    try:
        data = model_tco(monthly_pages=float(pages), avg_pages_per_doc=float(pages_per_doc),
            questions_per_doc=int(questions), batch_mode=bool(batch),
            incumbent_review_rate_pct=float(incumbent_review),
            mistral_review_rate_pct=float(mistral_review),
            review_cost_per_page=float(review_cost), tokens_per_page=int(tokens_per_page),
            output_tokens_per_question=int(output_per_question), settings=settings)
    except Exception as exc:
        raise gr.Error(str(exc)) from exc
    return tco_markdown(data), data


with gr.Blocks(title="Mistral Document Intelligence | Partner SA V2") as demo:
    gr.Markdown("""# Mistral Document Intelligence — Partner Solutions Architect V2

    **Working pipeline:** Tab 1 uses **Mistral Small topic classification first without extended reasoning**, with **Mistral Large at low reasoning as a reference-aware backup** on a Small miss or error. Tab 2 uses **Mistral Small first without extended reasoning** and the benchmark-only **Mistral Large runner-up adjudication with low reasoning** when eligible. Both use Mistral OCR and evidence-first Mistral Small five-question QA.
    """)
    with gr.Tabs():
        with gr.Tab("1 · Live document inspector"):
            gr.Markdown("""**This tab processes one PDF per click.** First define a reproducible pool, then choose
            one PDF's position within it. Example: pool size `20` and PDF position `7` processes
            only the seventh PDF in that 20-PDF pool. Use Tab 2 to process an entire sample.
            """)
            with gr.Row():
                split_dd = gr.Dropdown(
                    choices=["repliqa_0", "repliqa_1", "repliqa_2", "repliqa_3"],
                    value="repliqa_0", label="Dataset split")
                n_sample = gr.Slider(1, 50, value=10, step=1,
                                     label="Reproducible PDF pool size (does not process all)")
                doc_number = gr.Number(value=1, precision=0,
                                       label="PDF position to inspect (1 = first PDF)")
                judge_flag = gr.Checkbox(value=False, label="Optional Mistral Large semantic judge (extra API cost)")
                topic_embedding_mode = gr.Dropdown(
                    choices=["advisory", "shadow", "off"], value="advisory",
                    label="Topic embedding candidates",
                    info="Advisory affects the topic prompt; shadow measures only; off is the original path")
            with gr.Row():
                recent_tab1 = gr.Dropdown(
                    choices=recent_tab1_runs(), value=None,
                    label="Show results from a recent run",
                    info=("The six most recent stored runs with a correct topic, newest first; "
                          "selecting one makes no API call"))
                inspect_btn = gr.Button("Process selected PDF", variant="primary")
            doc_summary = gr.HTML()
            document_page_state = gr.State(
                {"document_id": None, "images": [], "ocr_pages": [], "page": 1})
            with gr.Column(visible=False, elem_id="document-inspector-shell") as document_inspector_panel:
                gr.HTML("""<div class="document-inspector-intro"><div>
                    <h3>Source document and OCR extraction</h3>
                    <p>Page-synchronized structural blocks, boundaries and confidence from Mistral OCR.</p>
                    </div><p>Use the controls below to move both panes together.</p></div>""")
                with gr.Row(elem_id="ocr-inspector-controls"):
                    block_view_mode = gr.Radio(
                        ["Block inspector", "Clean Markdown"], value="Block inspector",
                        label="OCR view", scale=2)
                    show_block_outlines = gr.Checkbox(
                        True, label="Block outlines", scale=1)
                    show_block_labels = gr.Checkbox(
                        True, label="Labels and confidence", scale=1)
                gr.HTML("""<div class="ocr-block-legend"><span>Block types</span>
                    <span class="ocr-legend-chip type-title">Title</span>
                    <span class="ocr-legend-chip type-text">Paragraph</span>
                    <span class="ocr-legend-chip type-list">List</span>
                    <span class="ocr-legend-chip type-table">Table</span>
                    <span class="ocr-legend-chip type-image">Image / caption</span>
                    <span class="ocr-confidence-note">Confidence is a review signal, not calibrated accuracy.</span>
                    </div>""")
                with gr.Row(equal_height=True, elem_id="document-page-inspector"):
                    with gr.Column(scale=1, min_width=380):
                        gr.HTML("""<div class="document-panel-title"><strong>Original PDF</strong>
                            <span>Bounding boxes over 160 DPI render</span></div>""")
                        source_pdf_page = gr.HTML(
                            '<div class="ocr-empty-state">No PDF page loaded.</div>',
                            elem_id="source-pdf-page")
                    with gr.Column(scale=1, min_width=380):
                        gr.HTML("""<div class="document-panel-title"><strong>Mistral OCR</strong>
                            <span>Structural blocks in reading order</span></div>""")
                        ocr_page_content = gr.HTML(
                            '<div class="ocr-empty-state">No OCR page loaded.</div>',
                            elem_id="ocr-extracted-page")
                with gr.Row(elem_id="document-page-navigation"):
                    previous_page = gr.Button("Previous page", size="sm", scale=0, min_width=120)
                    document_page_slider = gr.Slider(
                        minimum=1, maximum=2, value=1, step=1, precision=0,
                        show_label=False, container=False, interactive=False, scale=4,
                        elem_id="document-page-slider")
                    document_page_status = gr.HTML(
                        _document_page_status(0, 0), scale=0, min_width=105)
                    next_page = gr.Button("Next page", size="sm", scale=0, min_width=120)
            quality_header = gr.HTML()
            question_table = gr.Dataframe(
                show_label=False, interactive=False, wrap=True, line_breaks=True, max_height=650,
                column_widths=[280, 220, 220, 90, 90, 360, 110, 140, 130, 90, 110],
                elem_id="individual-answer-quality")
            with gr.Accordion("Raw metrics, usage, model and cost JSON", open=False):
                detail_out = gr.JSON(show_label=False)
            inspection_started = inspect_btn.click(
                lambda: (gr.update(visible=False), gr.update(value=None)),
                outputs=[document_inspector_panel, recent_tab1], queue=False)
            inspection_started.then(inspect_document,
                inputs=[split_dd, n_sample, doc_number, judge_flag, topic_embedding_mode,
                        block_view_mode, show_block_outlines, show_block_labels],
                outputs=[doc_summary, quality_header, question_table, detail_out,
                         document_page_state, source_pdf_page,
                         ocr_page_content, document_page_status, document_page_slider,
                         document_inspector_panel, recent_tab1])
            recent_tab1.input(
                load_stored_tab1_run,
                inputs=[recent_tab1, block_view_mode, show_block_outlines,
                        show_block_labels],
                outputs=[doc_summary, quality_header, question_table, detail_out,
                         document_page_state, source_pdf_page, ocr_page_content,
                         document_page_status, document_page_slider,
                         document_inspector_panel],
                queue=False)
            previous_page.click(
                lambda state, mode, outlines, labels:
                    step_document_page(
                        state, -1, mode, outlines, labels),
                inputs=[document_page_state, block_view_mode, show_block_outlines,
                        show_block_labels],
                outputs=[document_page_state, source_pdf_page, ocr_page_content,
                         document_page_status, document_page_slider])
            next_page.click(
                lambda state, mode, outlines, labels:
                    step_document_page(
                        state, 1, mode, outlines, labels),
                inputs=[document_page_state, block_view_mode, show_block_outlines,
                        show_block_labels],
                outputs=[document_page_state, source_pdf_page, ocr_page_content,
                         document_page_status, document_page_slider])
            document_page_slider.release(
                select_document_page,
                inputs=[document_page_state, document_page_slider, block_view_mode,
                        show_block_outlines, show_block_labels],
                outputs=[document_page_state, source_pdf_page, ocr_page_content,
                         document_page_status, document_page_slider])
            view_inputs = [document_page_state, block_view_mode, show_block_outlines,
                           show_block_labels]
            view_outputs = [document_page_state, source_pdf_page, ocr_page_content,
                            document_page_status, document_page_slider]
            for control in (block_view_mode, show_block_outlines, show_block_labels):
                control.change(refresh_document_page, inputs=view_inputs, outputs=view_outputs)

        with gr.Tab("2 · Bulk evaluator"):
            with gr.Row():
                eval_split = gr.Dropdown(choices=["repliqa_3", "repliqa_0", "repliqa_1", "repliqa_2"],
                                          value="repliqa_3", label="Evaluation split")
                eval_docs = gr.Slider(1, 500, value=3, step=1, label="Unique PDFs (up to 500 in UI)")
                eval_seed = gr.Number(value=42, precision=0, label="Random seed")
                eval_judge = gr.Checkbox(False, label="Run optional Mistral Large judge (separately metered)")
                eval_embedding_mode = gr.Dropdown(
                    choices=["advisory", "shadow", "off"], value="advisory",
                    label="Topic embedding candidates",
                    info="Select off to reproduce the original classifier immediately")
            with gr.Row():
                recent_eval = gr.Dropdown(
                    choices=recent_evaluation_runs(), value=None,
                    label="Show results from a recent run",
                    info="The six most recent stored runs, newest first; selecting one makes no API call")
                launch_eval = gr.Button("Run or resume evaluation", variant="primary")
            progress_label = gr.Markdown("No evaluation executed in this session.")
            evaluation_summary = gr.HTML()
            per_topic_table = gr.Dataframe(label="Per-topic results", interactive=False)
            with gr.Row():
                summary_download = gr.File(label="Download summary JSON")
                questions_download = gr.File(label="Download question-level CSV")
            with gr.Accordion("Raw aggregate benchmark and confidence-interval JSON", open=False):
                summary_json = gr.JSON(show_label=False)
            launch_eval.click(run_evaluation_ui,
                inputs=[eval_split, eval_docs, eval_seed, eval_judge, eval_embedding_mode],
                outputs=[progress_label, evaluation_summary, summary_json, per_topic_table,
                         summary_download, questions_download, recent_eval],
                concurrency_limit=1)
            recent_eval.input(
                load_stored_evaluation,
                inputs=[recent_eval],
                outputs=[progress_label, evaluation_summary, summary_json, per_topic_table,
                         summary_download, questions_download],
                queue=False)
            demo.load(
                initial_recent_evaluation_view,
                outputs=[recent_eval, progress_label, evaluation_summary, summary_json,
                         per_topic_table, summary_download, questions_download],
                queue=False)

        with gr.Tab("3 · Customer TCO scenario"):
            gr.Markdown("""All numbers here are **scenario estimates**, not proof of achieved accuracy.
            The 50% batch toggle models Mistral's published batch pricing; this application does not
            submit actual Batch API jobs. Review rates are entered independently, not inferred from 85%.
            """)
            with gr.Row():
                volume = gr.Slider(10000, 1000000, value=100000, step=10000, label="Pages per month")
                avg_pages = gr.Slider(1, 20, value=2, step=1, label="Average pages / PDF")
                q_count = gr.Slider(1, 10, value=5, step=1, label="Questions / PDF (one bundled QA call)")
            with gr.Row():
                batch_toggle = gr.Checkbox(value=False, label="Batch price projection (50% discount; not executed)")
                baseline_review = gr.Slider(0, 100, value=15, step=1, label="Assumed incumbent manual-review rate (%)")
                mistral_review = gr.Slider(0, 100, value=5, step=1, label="Assumed Mistral manual-review rate (%)")
                review_cost = gr.Slider(0, 10, value=1, step=0.25, label="Cost per reviewed page ($)")
            with gr.Accordion("Token assumptions", open=False):
                tok_per_page = gr.Slider(200, 3000, value=800, step=100, label="OCR Markdown input tokens/page")
                tok_out = gr.Slider(25, 250, value=95, step=5, label="Generated output tokens/question")
            tco_btn = gr.Button("Calculate scenario")
            tco_md = gr.Markdown()
            tco_json = gr.JSON(label="Auditable scenario inputs and outputs")
            inputs = [volume, avg_pages, q_count, batch_toggle, baseline_review,
                      mistral_review, review_cost, tok_per_page, tok_out]
            tco_btn.click(calculate_tco, inputs=inputs, outputs=[tco_md, tco_json])
            demo.load(calculate_tco, inputs=inputs, outputs=[tco_md, tco_json])

        with gr.Tab("4 · Aggregated Runs") as aggregated_runs_tab:
            aggregated_metrics = gr.HTML()
            aggregated_per_topic = gr.Dataframe(
                label="Per-topic results", interactive=False, wrap=True,
                max_height=720)
            aggregated_runs_tab.select(
                aggregated_runs_view,
                outputs=[aggregated_metrics, aggregated_per_topic],
                queue=False)
            demo.load(
                aggregated_runs_view,
                outputs=[aggregated_metrics, aggregated_per_topic],
                queue=False)
    gr.Markdown("""---
    **Model/pricing configuration:** `.env`. **Run evaluations from terminal:** `python evaluate.py --split repliqa_3 --docs 500 --seed 42 --bootstrap 1000`.
    The provided sample PDFs and annotations are from the public ServiceNow RepLiQA dataset.
    """)

if __name__ == "__main__":
    # Local-only by default. Do not expose this UI publicly with keys or customer PDFs.
    demo.queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", server_port=7860,
                                                      theme=gr.themes.Soft(), css=EXECUTIVE_CSS)
