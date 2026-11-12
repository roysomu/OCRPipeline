# Mistral Partner Solutions Architect — Gradio V2

**Status:** The Python project has passed offline unit/integration-style tests and Gradio import checks. **No real Mistral API call, RepLiQA network download, OCR benchmark, or production-billing validation was performed in this environment.** A valid Mistral organization API key and Internet access are required to produce actual results.

## 1. What changed from the uploaded `app.py` and the older Streamlit kit

| Area | Implemented in V2 |
|---|---|
| UI | Gradio: live document inspector, bulk evaluator, interactive TCO scenario |
| OCR | One Mistral OCR request per PDF; Markdown enters WER/CER while structural blocks, bounding boxes and block confidence power the Tab 1 inspector |
| QA / topic | Tab 1 uses Mistral Small first without extended reasoning and invokes pinned-generation `mistral-large-4` at low reasoning only after a Small error or reference mismatch. Tab 2 uses no extended reasoning for its dedicated Mistral Small topic request and `mistral-large-4` with low reasoning for benchmark-only runner-up adjudication. Both use the same definitions, tie-breakers, optional cached Mistral Embed hints, and one evidence-first Mistral Small request for all five questions per PDF. |
| Schema | Pydantic v2 topic, question-batch, combined analysis, answer and judge contracts; Mistral JSON-schema response mode + local validation |
| Evidence | Verbatim quote must be located on claimed one-based OCR page; does **not** prove semantic entailment |
| Metrics | Micro OCR proxy WER/CER, exact gold long-answer span recall, exact topic, 17-category macro F1 on observed classes, EM/token F1, abstention precision/recall, answerability, quote location, length and optional judge alignment |
| Evaluation | Supports 500 PDFs in the UI (and up to a full split in CLI); seeded **proportionally stratified** sample; all five questions clustered by PDF; 95% PDF-clustered bootstrap intervals |
| Auditing | Per-topic, per-document, per-question CSV; JSON summary, failures, manifest and selected IDs |
| Caching | Separate version-sensitive OCR / topic / QA / judge caches, hashed on PDF bytes, prompt content, schema, answers and model IDs; checkpoints support resumable CLI jobs |
| Cost | Official public list-price defaults as of Nov 3, 2026; separate original full-run modeled cost, incremental fresh-run cost and optional judge cost |
| Business case | Gradio TCO simulator with explicit manual-review rates, unit costs and optional *scenario-only* 50% batch discount |
| Reliability | Safe 429 backoff and per-PDF failure reporting; no blind retries of ambiguous 5xx POSTs |

The older user-uploaded `app.py` provided the original Gradio layout, Pydantic response idea, judge concept and commercial simulator. V2 incorporates those with a separate modular, benchmark-defensible pipeline.

## 2. Install and run (macOS / Linux)

```bash
unzip Mistral_PSA_Gradio_V2.zip
cd mistral_psa_gradio_v2
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
# Edit .env; supply MISTRAL_API_KEY and HF_TOKEN
python -m pytest -q
python app.py
```

Open **http://127.0.0.1:7860/**. The UI binds to localhost and is not a public hosted service.

**Windows PowerShell:** use `.venv\\Scripts\\Activate.ps1` instead of `source .venv/bin/activate`.

Python 3.11+ recommended. Gradio versions 5–6 supported in requirements; the project was imported successfully with Gradio 6.5.1. A valid key is not needed for tests or the scenario-only TCO calculator.

`HF_TOKEN` is used for RepLiQA Parquet and PDF downloads. Create a Hugging Face
read token at <https://huggingface.co/settings/tokens>. The token is read directly
from the environment and is never written to run configuration, caches, or results.
If `HF_TOKEN` is omitted, `huggingface_hub` may still use credentials previously
saved by `hf auth login`.

## 3. Three Gradio tabs

1. **Live document inspector:** run a selected PDF from any `repliqa_0`–`repliqa_3` split; inspect synchronized original-PDF and rendered OCR pages side by side, then review five predictions, expected answers, optional judge and per-doc costs. Its read-only history dropdown lists the six newest stored inspections whose final topic matched the reference; selecting one makes no API request, while **Process selected PDF** always enters the live processing workflow.
2. **Bulk evaluator:** select any split and up to **500 PDFs** in the UI, seed 42, and a fixed 1,000 bootstrap replications; see progress, confidence intervals, per-topic breakdown and downloadable outputs. The four most recently updated stored runs are available from a read-only dropdown; selecting one does not make an API request, while **Run or resume evaluation** always enters the live evaluation workflow. Its Small-first topic process is unchanged by Tab 1's Large-first inspection strategy. Split governance is managed manually by the user. Test runs are sequential and may take substantial time depending on API limits.
3. **Customer TCO scenario:** change monthly pages, pages/document, questions/document, manual-review assumptions and batch discount. Neither the Mistral review rate nor incumbent review rate is derived from the reported ~85% incumbent accuracy.

## 4. CLI evaluation — recommended for 500 PDFs

```bash
# Embedding experiment: advisory top-3 hints. Use `off` for the original path or
# `shadow` to measure candidate recall without changing Small's prompt.
TOPIC_EMBEDDING_MODE=advisory python evaluate.py --split repliqa_1 --docs 100 --seed 42 --bootstrap 1000

# Development smoke test (10 PDFs, approximately 50 questions)
python evaluate.py --split repliqa_0 --docs 10 --seed 42 --bootstrap 1000

# Development with separately priced model judge
python evaluate.py --split repliqa_1 --docs 20 --seed 42 --judge --bootstrap 1000

# Example repliqa_3 evaluation
python evaluate.py --split repliqa_3 --docs 500 --seed 42 --bootstrap 1000

# Optional full split, 3591 PDFs / 17955 questions, if credits/time allow
python evaluate.py --split repliqa_3 --docs 3591 --seed 42 --bootstrap 1000
```

The runner-up/Large path is an experimental oracle analysis, not a deployable routing
policy: eligibility uses the benchmark reference label, although Large is never told which
candidate is the reference. Large may choose only between Small's primary and runner-up. Any Large request,
schema, or candidate-validation failure falls back to Small's complete result. Revert to
the prior Small-only behavior and cache fingerprint with:

```bash
TOPIC_RUNNER_UP_LARGE_EXPERIMENT=false python evaluate.py --split repliqa_1 --docs 100 --seed 42 --bootstrap 1000
```

The 500-PDF run requires an API key and will **incur real API charges**, even though benchmark results are not known in advance. Every PDF is checkpointed separately in `results/<split>_<docs>docs_seed<seed>_<signature>/documents/`. Repeating the same CLI command resumes completed PDFs. `--force` deliberately reruns calls and may incur fresh charges. The application does not enforce holdout policies or freeze `repliqa_3`; split governance is managed manually by the user.

**RepLiQA source pinning:** Downloaded Parquet paths normally resolve Hugging Face `main` to an immutable snapshot commit; V2 records this revision and downloads the corresponding original PDFs using that same commit. For maximum reproducibility, set `HF_DATASET_REVISION` to an explicit repository commit.

## 5. Result files

Inside each run folder:

- `run_config.json` — seed, models, price assumptions, prompt signature, dataset revision, selected document IDs.
- `summary.json` — performance metrics and optional 95% confidence intervals; indicates COMPLETE or INCOMPLETE.
- `per_document.csv` — OCR proxy errors, topic correctness, pages, latency and cost.
- `per_question.csv` — question texts, references, predicted answers, F1/EM, abstentions and optional judge results.
- `per_topic.csv` — observed document counts and topic-specific metrics.
- `failures.json` — failed documents, never silently removed from denominator.
- `judge_failures.json` — optional judge failures; core OCR/topic/QA documents remain evaluated.
- `documents/*.json` — durable scored checkpoints.

`summary.json` uses the field **`full_run_api_cost_estimated_usd`**, which is the modeled standard price of processing all successful documents once. **`fresh_api_cost_in_current_execution_estimated_usd`** is the estimated API spending incurred by the current invocation, including cache/partial-run semantics. The optional judge is separately metered under `judge_api_cost_estimated_usd`. These are pricing estimates based on model usage responses, **not actual invoices**.

## 6. Evaluation methodology

- **OCR:** WER and CER compare the *visible words* of `pages[].markdown` to RepLiQA `document_extracted`. Exact Span Recall checks whether each normalized, answerable gold `long_answer` occurs as one contiguous token span in that same OCR text; empty and reference-unanswerable spans are excluded. Structural blocks and confidence scores are retained for the Tab 1 overlay but excluded from scoring. Markup syntax, image placeholders/URL destinations, page markers, metadata, model-generated answers and evidence are excluded. Reference text is automatically extracted and **not human-verified**. WER can legitimately exceed 1.0 due to insertions; the code does not cap it.
- **Topic:** exact canonical 17-label primary match and macro F1 across classes observed in the evaluated set. With the experimental flag enabled, a Small primary miss whose runner-up equals the reference triggers a two-candidate Large adjudication; the report preserves Small's prediction and records trigger, final-source, and fallback counts. This is an oracle upper-bound experiment, not a production classifier. Optional embedding modes are `off` (original path), `shadow` (rank and measure only), and `advisory` (give Small a non-binding, alphabetized top-k shortlist). Advisory hides similarity ranks/scores and requires contrastive comparison of every candidate. Embedding top-k recall is diagnostic and never receives topic-accuracy credit.
- **QA:** normalized exact match and token F1 of short answers on *answerable questions*, plus answerability accuracy and abstention precision/recall across all questions.
- **Answerability labels:** known RepLiQA aliases (`UNANSWERABLE`, `The answer is not found in the document.`, and `NA` evidence) are normalized to `UNANSWERABLE`; raw source labels remain available in grouped records for auditing.
- **Long form:** target word count 20–55 is scored as format compliance; quote-location verification is not semantic factuality.
- **Optional judge:** Mistral Large provides **reference-alignment 1–5**, not a calibrated accuracy percentage; ground truth is provided to this separate evaluation call only *after* primary inference. A malformed or unavailable judge response is recorded separately and does not fail the core OCR/topic/QA document. Judge expense is not part of base production inference cost.
- **End-to-end assignment accuracy:** a declared three-pillar score gives equal weight to text extraction, strict topic classification and long-form answering, then multiplies by the PDF processing-success rate. Text is `0.60×(1−WER) + 0.40×(1−CER)`; topic is strict primary-label accuracy; long form is `0.70×normalized judge + 0.15×answerability + 0.10×evidence location + 0.05×length compliance`. The judge mean uses successful optional-judge assessments only; failed assessments are excluded and the resulting judge coverage is displayed. The score is unavailable only when there are no successful judge assessments or other required metrics are missing. Component metrics remain the primary audit trail.
- **Confidence intervals:** 1,000 draws by default, resampling PDFs **within topic strata** with replacement and retaining all five correlated questions. These quantify variability of the sampled benchmark (no finite population correction). They do **not** establish significance versus a competitor without paired measurements.
- **Incumbent:** $0.75/page and ~85% are case statements. The 85% has no specified quality metric and is not directly comparable to the declared assignment score unless the incumbent is evaluated on the same PDFs with the same rubric; the application **does not report an unsupported accuracy lift**.

## 7. Pricing assumptions

Defaults (USD, checked Nov 3, 2026; reverify prior to presentation):

- OCR 4.1 standard: **$4 / 1,000 pages** (`mistral-ocr-4-1`)
- Mistral Small 4 topic classifier: **$0.15 / 1M input tokens, $0.60 / 1M output tokens**
- Mistral Small 4 five-question QA: **$0.15 / 1M input tokens, $0.60 / 1M output tokens**
- Optional Mistral Large 3 judge: **$0.50 / 1M input, $1.50 / 1M output**
- Published qualifying Batch inference: **50% discount**. **Only simulated in the calculator**; synchronous OCR is the actual inference mechanism implemented.

Sources: https://mistral.ai/pricing/api/ and https://docs.mistral.ai/models/ocr-4-1 and https://docs.mistral.ai/studio/conversations/structured-output/custom

Costs exclude cloud infrastructure, egress, retries, failed calls, human review and negotiated contracts. Distinguish *actual SDK/API usage* from *modeled unit pricing*. The TCO calculator separately models manual review as an input assumption, rather than claiming ~85% accuracy implies 15% manual handling.

## 8. Important prototype limitations

- **No credentials/test data included.** Network-backed smoke tests and measured evaluations must be run by the user.
- The 500-PDF evaluation is a sequential, resumable client orchestration. **It does not implement Mistral's actual Batch Inference API.** Large runs require credit limits, rate limits and run monitoring. 429s use bounded backoff; ambiguous 5xx responses are not automatically repeated because they might have been billed.
- Pydantic/JSON-schema validation checks answer shapes, question IDs, topic membership and coherent abstentions. It is not proof that the content is correct.
- Existing older Streamlit cache files are not automatically migrated: output schema and prompt versions changed.
- The context guard is a **character** limit, not a model-exact token budget. Oversized PDFs require token-aware chunking/retrieval before production.
- OCR temp files are best-effort deleted from Mistral's Files API. Local caches contain full OCR text, may contain confidential documents in other uses, and lack encryption/tenant isolation. The app must not be publicly exposed without access controls.
- Do not claim production availability, SLA, accuracy superiority, or human-review savings based solely on this prototype.

## 9. Files

- `app.py` — three-tab Gradio interface.
- `pipeline.py` — versioned OCR, dedicated topic classification, five-question structured QA, optional semantic judge, caching and costs.
- `schemas.py` — Pydantic response contracts.
- `dataset.py` — RepLiQA grouping, proportional sampling and source-pinned downloads.
- `metrics.py` — Markdown WER/CER, QA/topic scoring, clustered confidence intervals, HTML error diff.
- `evaluate.py` — resumable evaluation CLI and output exports.
- `tco.py` — auditable commercial calculator.
- `tests/` — self-contained tests using mocked API and synthetic PDFs.
- `PANEL_DEMO_GUIDE.md` — suggested 5-minute video and 60-minute panel narrative.
