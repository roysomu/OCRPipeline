# Implementation priorities — completion checklist

The priorities came from the comparison of the user-uploaded Gradio app and the previous Streamlit panel kit. "Implemented" means written, locally tested where possible, **not validated by a live Mistral API run**.

| Priority | Change | Implementation | Verification |
|---|---|---|---|
| P0 | Correct Mistral prices; avoid unsupported accuracy-lift | `config.py`, `tco.py`, `metrics.py`, Gradio copy | Verified published prices 2026-11-03; unit tests |
| P0 | One OCR, one dedicated Small topic call and one evidence-first Small five-question QA call per document | `pipeline.py` | Mocked model-routing, prompt and separate-cache tests |
| P0 | OCR WER uses Markdown only | `metrics.py` `ocr_markdown_text`/`markdown_visible_text` | Markdown image/table/URL fixture passes with WER 0 |
| P1 | Pydantic and schema validation | `schemas.py`, `MistralAPI._structured_chat` | Pydantic tests, 17-value JSON-schema topic enum |
| P1 | 500-PDF holdout evaluation | `dataset.py`, `evaluate.py` | Proportional sample tests; UI max=500, CLI supports full partition |
| P1 | 95% document-clustered CIs | `metrics.py::cluster_bootstrap_ci` | Reproducible fixed-seed test; no competitor significance claims |
| P1 | Optional semantic judge | `pipeline.py::evaluate_semantic` | Separate model/cost/cache; mocked judge schema & cache tests |
| P2 | Interactive TCO | `tco.py`, Gradio third tab | 50% projection and independent review-rate tests |
| P2 | Side-by-side OCR inspection | Gradio dev tab, `metrics.py::ocr_diff_html` | HTML escaping test, matching text normalization |
| Additional | Version-sensitive cache, safer holdout freeze | `pipeline.py`, `evaluate.py` | cache/holdout/regression tests |
| Additional | Reproducible dataset commit | `dataset.py` | Parquet commit resolved from Hugging Face snapshot and PDFs pinned to it |
| Additional | Local Gradio health check | `app.py` | Local server returned HTTP 200 |

**Not implemented and not claimed:** actual Batch Inference API job submission, production multi-tenant security, guaranteed file deletion, full token-aware retrieval for oversized PDFs, live benchmark results, real invoice comparisons, a measured incumbent 85% metric, human-validated OCR ground truth.
