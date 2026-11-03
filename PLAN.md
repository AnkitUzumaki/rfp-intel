# RFP Intelligence Platform — Build Plan

Scope rule: **every requirement, deliverable and bonus in `data/brief.pdf` is in scope.**
Section 3 is the source of truth — each row maps one PDF line to where it is built and an **objective** check.
Nothing ships until every row is ✅. Row IDs are stable; new rows are appended (R91+), never renumbered.

---

## 0. Grading weights (PDF §10) → priority

| Criterion | Weight | Phases that earn it |
|---|---|---|
| RAG Search Engine | 25% | 2 |
| Multi-Agent System | 25% | 3, 4 |
| Extraction Accuracy | 20% | 3, 6 |
| Robustness | 10% | 1, 6 |
| Code Quality & Architecture | 10% | 0, all |
| Documentation & Demo | 10% | 7 |

Search + agents = 50% → front-loaded. Every phase ends with its eval/test numbers, not "looks done".

## 1. Final stack

| Layer | Choice | Notes |
|---|---|---|
| Language | Python 3.10+, `uv` + `uv.lock` pinned | PDF §9 |
| PDF parse | PyMuPDF (text/layout/pages) + pdfplumber (tables) | PDF §5 |
| HTML parse | BeautifulSoup + lxml | Bid-info table → key/value + body text |
| OCR (bonus) | rapidocr-onnxruntime | pip-only, CPU; runs only on pages with no text layer |
| Chunking | Heading-aware sections ~400–600 tok, ~15% overlap; tables = atomic chunks | Size tuned by eval, justified in README |
| Embeddings | BAAI/bge-small-en-v1.5 via fastembed (local) | Configurable to API model |
| Keyword | BM25 sparse vectors (fastembed `Qdrant/bm25`) | |
| Vector store | Qdrant (embedded local mode; server mode in docker compose) | Payload filters, RRF fusion, incremental upserts |
| Reranker | BAAI/bge-reranker-base cross-encoder (local) | |
| LLM | OpenAI-compatible client → Gemini free tier (primary), Groq (fallback), Ollama (offline) | All via `.env`; temperature=0 for extraction/validation |
| Agents | LangGraph + Pydantic v2 | Typed shared state, Send fan-out, conditional retry edges |
| Observability | JSONL trace per run + optional Langfuse | Tokens, latency, cost per step and per run |
| API / CLI / UI | FastAPI, Typer, Streamlit | |
| Cache (bonus) | SQLite semantic cache (query embedding cosine ≥ threshold + same filters) | |
| Quality | ruff, mypy (standard, own modules), pytest | |
| Packaging | Makefile (`setup/test/eval/demo`), Dockerfile, docker compose, GitHub Actions CI | |

## 2. Repo structure

```
main.py                     # one-command entry: python main.py --bid ./Bid1 | ask | serve | eval
Makefile  uv.lock  pyproject.toml
config/settings.yaml        # models, chunk sizes, thresholds, retries, confidence weights, prices (no secrets)
config/company_profile.yaml # go/no-go capabilities
config/fields.yaml          # §8 field rubric (section 5) — single source for prompts, validator, scorer
.env.example                # LLM_BASE_URL, LLM_MODEL, LLM_API_KEY, FALLBACK_*, LANGFUSE_*
ingestion/   detect.py parse_pdf.py parse_html.py clean.py classify.py ocr.py chunk.py
search/      index.py retrieve.py rerank.py query_rewrite.py cache.py
agents/      state.py schemas.py graph.py orchestrator.py ingestion_agent.py retrieval_agent.py
             extraction_agents.py reconciliation_agent.py validator_agent.py report_agent.py
             qa_agent.py comparison_agent.py go_no_go_agent.py llm.py tracing.py
prompts/     one .md per agent prompt
api/         app.py
ui/          streamlit_app.py
eval/        gold.jsonl run_eval.py report.md extraction_gold.json score_extraction.py calibration.md
tests/       test_parse_*.py test_chunk.py test_search.py test_validator.py test_graph.py
             test_guardrails.py test_failures.py test_unseen_bid.py test_api.py test_deliverables.py
tests/fixtures/  blank.pdf image_only.pdf corrupt.pdf encrypted.pdf wrong_ext.pdf long_200p.pdf
                 unseen_bid/ (odd names, "Questions and Answers.pdf", empty PDF) html_only_bid/
outputs/     Bid1.json Bid2.json Bid*.md Bid*_go_no_go.md comparison.md qa_log.md
             traces/extract_Bid1.jsonl traces/retry_loop_example.jsonl
docs/        architecture.svg demo_script.md walkthrough.md
Dockerfile docker-compose.yml .github/workflows/ci.yml README.md
```

## 3. Requirement traceability matrix

### §1–3 Objective / Documents
| ID | Requirement | Built in | Verified by |
|---|---|---|---|
| R1 | Search engine indexing all HTML + PDF incl. addendums, returns cited passages | `search/` | test_search, eval |
| R2 | Multi-agent system using search as tool | `agents/` | test_graph, trace |
| R3 | Structured JSON per bid, every value cited | `report_agent` | JSON schema test on outputs/Bid*.json |
| R4 | Free-form Q&A over bids | `qa_agent` | qa_log.md |
| R5 | Each folder processed as one bid unit (bid_id = folder) | `ingestion_agent` | test |
| R6 | Not hard-coded; works on unseen bid folder | whole system | test_unseen_bid (fixtures/unseen_bid, html_only_bid) in CI |
| R7 | Detect addendum overrides; never invent facts | reconciler, validator | Bid1 due-date test; addendum rows in extraction score |

### §4 Architecture
| ID | Requirement | Built in | Verified by |
|---|---|---|---|
| R8 | Ingestion → chunk/embed → hybrid index → orchestrator → retrieval/extraction/reconciliation/validator → JSON / cited answer | `agents/graph.py` | docs/architecture.svg exists; graph node names match diagram (test_deliverables) |

### §5 Part A — Ingestion & Parsing
| ID | Requirement | Built in | Verified by |
|---|---|---|---|
| R9 | Parse HTML bid pages | parse_html.py | test_parse_html |
| R10 | Parse PDFs, keep page structure | parse_pdf.py | test_parse_pdf (page count, page numbers) |
| R11 | Extract tables correctly (spec + bid-info tables) | parse_pdf.py, parse_html.py | test: Bid2 spec table cells, BidNet table keys |
| R12 | Remove repeated headers/footers | clean.py | test_clean |
| R13 | Fix broken lines + hyphenation | clean.py | test_clean |
| R14 | Normalize whitespace | clean.py | test_clean |
| R15 | Metadata: bid_id, file_name, doc_type (bid_page/rfp/addendum/specs/affidavit/other), addendum_number, page_number, document date | classify.py | test_classify on all 9 files + "Questions and Answers.pdf" → other |
| R16 | Handle scanned / empty / unreadable files gracefully + log; also surfaced in output `processing_errors[]` | parse_pdf.py, detect.py | test_failures with blank, image_only, corrupt, encrypted fixtures |
| R17 | OCR (bonus) | ocr.py | test on image_only fixture |
| R18 | File type detection by content (magic bytes) | detect.py | wrong_ext fixture test |

### §6.1 Indexing
| ID | Requirement | Built in | Verified by |
|---|---|---|---|
| R19 | RFP-suited chunking (heading/table-aware, overlap) | chunk.py | test_chunk (tables never split mid-row, headings kept) |
| R20 | Chunk size + strategy justified in README | README | eval table comparing ≥2 chunk sizes; README section exists (test_deliverables) |
| R21 | Embeddings | index.py | test_search |
| R22 | Vector store with metadata | index.py | test_search |
| R23 | Incremental indexing — new bid w/o full re-index; changed/new file in existing bid re-indexed by content hash | index.py | test: index Bid1, add Bid2, Bid1 point count unchanged; modify file → only its chunks replaced |

### §6.2 Retrieval
| ID | Requirement | Built in | Verified by |
|---|---|---|---|
| R24 | Hybrid dense + BM25, fused (RRF) | retrieve.py | eval config table |
| R25 | Metadata filters (bid, doc_type, addendum_number, …) | retrieve.py | test "only Bid2 specs" |
| R26 | Cross-encoder re-ranking | rerank.py | eval config table |
| R27 | Query rewriting / expansion | query_rewrite.py (synonym map + LLM expansion) | test "deadline" expansion |
| R28 | Citations: file, page, chunk text on every result | retrieve.py | test |

### §6.3 Interface
| ID | Requirement | Built in | Verified by |
|---|---|---|---|
| R29 | REST API: POST /index, GET /search?q&bid_id&top_k, POST /ask (+ POST /extract, /compare, /go-no-go) | api/app.py | test_api (TestClient) |
| R30 | CLI | main.py (Typer) | test_api CLI smoke (CliRunner) |
| R31 | Web UI search + chat (bonus) | ui/streamlit_app.py | streamlit AppTest smoke + demo |

### §6.4 Retrieval evaluation
| ID | Requirement | Built in | Verified by |
|---|---|---|---|
| R32 | ≥15 question → expected passage pairs, both bids. Target 20+: coverage matrix Bid1/Bid2 × {rfp, addendum, specs table, affidavit, bid_page}; 4–5 exact-value questions (bid no, part no, model no); balanced with paraphrase/semantic questions so the set isn't rigged; split 15 tune / 5 held-out | eval/gold.jsonl (file+page+snippet, chunk-independent) | test_deliverables: ≥15 rows, both bids, every doc_type covered |
| R33 | Recall@5, Recall@10, MRR, nDCG@10, + p50 latency per config | eval/run_eval.py | eval/report.md generated |
| R34 | Compare configs: vector-only / BM25-only / hybrid / hybrid+rerank / hybrid+rerank+rewrite; tune + held-out reported separately | run_eval.py | results table in README + eval/report.md |

### §7.1 Agents
| ID | Requirement | Built in | Verified by |
|---|---|---|---|
| R35 | Orchestrator/Planner: goal → plan → route → decide done | orchestrator.py | trace |
| R36 | Ingestion Agent: detect type, parse, clean, metadata, trigger indexing | ingestion_agent.py | trace |
| R37 | Retrieval Agent: rewrite, search, filters, rerank, cited evidence | retrieval_agent.py | trace |
| R38 | Extraction specialists (Dates&Logistics, Commercial&Legal, Product&Specs, Org&Contact), evidence-only, cited, prompts generated from config/fields.yaml | extraction_agents.py | outputs + score_extraction |
| R39 | Addendum Reconciliation: base vs addenda, latest valid wins, change log | reconciliation_agent.py | Bid1 due date + addendum_changes |
| R40 | Validator/Critic: evidence support (quote ∈ chunk + LLM judge), format (field rubric), consistency (cross-field), hallucination; confidence scores; reject → re-extract | validator_agent.py | test_validator |
| R41 | Q&A/Report: cited answers, final JSON **and human-readable summary** | report_agent.py, qa_agent.py | outputs/Bid*.md |
| R42 | Explain merges/splits/additions of agents | README | README section exists (test_deliverables) |
| R91 | **§7 intro: agents use search as a tool, never whole documents in the prompt** | llm.py prompt guard | test_guardrails: every extraction/QA LLM call's context is built only from `Evidence` objects returned by retrieve(); hard cap on raw-text chars per prompt; violation raises |

### §7.2 Orchestration
| ID | Requirement | Built in | Verified by |
|---|---|---|---|
| R43 | Framework choice explained (LangGraph) | README | README section exists (test_deliverables) |
| R44 | Shared state: plan, evidence, draft fields, validation, final output (+ retries, change log, trace id) | agents/state.py | test_graph |
| R45 | Feedback loop validator → re-retrieve/re-extract, max retries | graph.py conditional edge | test: forced rejection stops at max_retries |
| R92 | **Loop shown working end-to-end**: a field rejected then re-extracted successfully | graph.py | outputs/traces/retry_loop_example.jsonl — natural rejection if one occurs; else via `--inject-fault` flag, **labeled as induced in trace + README** |
| R46 | Parallel execution (field groups concurrently) | graph.py Send fan-out | trace timestamps overlap |
| R47 | Structured outputs (Pydantic) for all agent messages | agents/schemas.py | test |
| R48 | Guardrail "Not found in documents", ≥1 citation per value | validator + schemas | test_guardrails |
| R49 | Observability: input, tool calls, output, tokens, latency per step | tracing.py (+Langfuse optional) | outputs/traces/ |
| R50 | One example trace in submission | outputs/traces/extract_Bid1.jsonl | test_deliverables: file exists, every step has input/tool_calls/output/tokens/latency |
| R51 | Errors → no crash. Four named fixtures: (a) LLM timeout → retry then fallback provider, (b) malformed JSON → one repair retry then field failed, (c) empty retrieval → query expansion retry then "Not found", (d) 429 rate limit → backoff then fallback | llm.py, graph.py | test_failures (one test per fixture, mocked LLM) |

### §7.3 Modes
| ID | Requirement | Built in | Verified by |
|---|---|---|---|
| R52 | Extraction mode: bid folder → full JSON | main.py extract | outputs/Bid1.json, Bid2.json |
| R53 | Q&A mode with citations. Orchestrator classifies intent and routes: fact → qa_agent; compare → comparison_agent; "what changed" → reconciliation change log; bid resolved by name ("Dell laptop bid"). Answering agent logged in trace | orchestrator.py, qa_agent.py | qa_log.md covers all 5 PDF example questions, each tagged with answering agent |

### §8 Fields (all 20) — semantics in section 5
| ID | Requirement | Built in | Verified by |
|---|---|---|---|
| R54 | Dates & Logistics: Due Date, Pre Bid Meeting, Delivery Date, Bid Submission Type, Installation | extraction_agents.py | score_extraction |
| R55 | Commercial & Legal: Term of Bid, Bid Bond Requirement, Payment Terms, Any Additional Documentation Required, Contract or Cooperative to use | extraction_agents.py | score_extraction |
| R56 | Product & Specs: Model_no, Part_no, Product, Product Specification, MFG for Registration | extraction_agents.py | score_extraction |
| R57 | Org & Contact: Bid Number, Title, contact_info, company_name | extraction_agents.py | score_extraction |
| R58 | Bid Summary (3–6 sentences, generated after validation, cited) | report_agent.py | validator: sentence count 3–6, ≥1 citation |
| R93 | Every §8 field description encoded as rubric (section 5) used by prompts, validator and scorer | config/fields.yaml | test: all 20 fields present in fields.yaml, prompts rendered from it |

### §8.1 Output format
| ID | Requirement | Built in | Verified by |
|---|---|---|---|
| R59 | One JSON per bid; each field: value, sources[{file,page}], confidence, notes; null + reason when absent; addendum_changes[]; validation{passed,failed,not_found}; + processing_errors[] | agents/schemas.py | JSON schema test on outputs |

### §9 Technical
| ID | Requirement | Built in | Verified by |
|---|---|---|---|
| R60 | Python 3.10+, any LLM | pyproject.toml `requires-python>=3.10` | CI runs on 3.10 |
| R61 | Keys + model names from env/config only | .env, settings.yaml | CI secret scan (gitleaks) + grep: no model-name literals outside config |
| R62 | One-command run (`python main.py --bid ./Bid1`, `docker compose up`) | main.py, compose | CI smoke: `python main.py --bid tests/fixtures/unseen_bid` with mocked LLM; compose build |
| R63 | Modular structure ingestion/ search/ agents/ api/ eval/ tests/ | repo | test_deliverables: dirs exist |
| R64 | Unit tests: parsing + search minimum (we add validator, graph, api, failures) | tests/ | CI green, coverage ≥80% on ingestion/ + search/ |
| R65 | Unseen bid works without code changes | — | = R6 test_unseen_bid |

### §10 Evaluation criteria (self-check before submit)
| ID | Requirement | Built in | Verified by |
|---|---|---|---|
| R66 | Retrieval quality, hybrid, rerank, filters, citations, API/UI | — | eval/report.md: hybrid+rerank ≥ vector-only on held-out |
| R67 | Agent roles, orchestration, state, loop, errors, observability | — | R50 + R92 traces exist; test_failures green |
| R68 | Extraction accuracy per field, both bids, scoring rules per field type (section 5); addendum-affected fields reported as a separate row | eval/extraction_gold.json + score_extraction.py | per-field table in README |
| R69 | Robustness: structures, missing fields, tables, long docs, unseen bids | tests/fixtures | test_unseen_bid (odd names, Q&A pdf, empty PDF, HTML-only bid) + long_200p.pdf (ingest + index within time budget, retrieval still finds planted fact) |
| R70 | Code quality: typed, tested, documented, configured, reproducible | — | ruff + mypy + pytest in CI; `make setup && make test` from clean clone |
| R71 | Docs & demo | — | §11 rows |
| R94 | Confidence calibration: tune section-6 weights on gold, report "confidence ≥0.8 → X% correct" (small-sample caveat stated) | eval/calibration.md | file exists, linked in README |

### §11 Deliverables
| ID | Requirement | Built in | Verified by |
|---|---|---|---|
| R72 | Git repository | repo | `git status` clean, pushed |
| R73 | README: setup, deps, how to run each mode, design decisions (chunking, embedding, retrieval, agent framework, **prompts**) | README.md, prompts/ | test_deliverables: required README headings present |
| R74 | Architecture diagram (search engine, agents, communication) | docs/architecture.svg | test_deliverables: exists + linked in README |
| R75 | JSON outputs per bid | outputs/ | schema test |
| R76 | Retrieval eval report (questions + Recall@k/MRR per config) | eval/report.md | test_deliverables: exists, all configs present |
| R77 | Sample Q&A log ≥10 cited answers | outputs/qa_log.md | test_deliverables: ≥10 entries, each with ≥1 citation |
| R78 | Full extraction agent trace | outputs/traces/extract_Bid1.jsonl | = R50 |
| R79 | Demo video 5–10 min | docs/demo_script.md (timed storyboard ordered by §10 weights: search → agents → JSON → eval numbers → unseen bid → bonuses) + `make demo` | **user records**; script exists |

### §12 Bonus (all in scope)
| ID | Requirement | Built in | Verified by |
|---|---|---|---|
| R80 | Web UI | = R31 | — |
| R81 | OCR scanned PDFs + image-based tables | = R17 (+ table OCR via rapidocr layout) | fixture test |
| R82 | Bid comparison agent → side-by-side report | comparison_agent.py, POST /compare | outputs/comparison.md |
| R83 | Go / no-go recommendation from configurable capabilities | go_no_go_agent.py + company_profile.yaml | outputs/Bid*_go_no_go.md; test: changing profile flips decision |
| R84 | Semantic caching of repeated queries | search/cache.py | test cache hit + miss on different filters |
| R85 | Cost/latency tracking per run | tracing.py run summary + price table in settings.yaml | trace footer has totals |
| R86 | Docker deployment | Dockerfile, docker-compose.yml (api, ui, qdrant) | CI compose build + healthcheck |
| R87 | CI runs tests + retrieval eval | .github/workflows/ci.yml (eval gated on Recall@5 threshold) | green CI |

### §13 Submission
| ID | Requirement | Built in | Verified by |
|---|---|---|---|
| R88 | Git link | — | user shares |
| R89 | Assumptions + known limitations listed | README (from sections 7–8) | test_deliverables: both headings present |
| R90 | Explainable by author | docs/walkthrough.md (per-module plain explanation) | exists, one section per top-level module |

---

## 4. Phases & exit criteria (every row assigned)

| Phase | Rows | Exit criteria |
|---|---|---|
| 0 Setup | R60, R61, R63, R70, R72 | repo, pyproject + uv.lock, Makefile, ruff/mypy/pytest, CI skeleton green, .env.example |
| 1 Ingestion | R5, R9–R18, R64 (parse) | All 9 files parsed, metadata correct, failure fixtures logged not crashed, tests green |
| 2 Search + eval | R1, R19–R30, R32–R34, R64 (search), R66, R84 | Eval table for 5 configs (tune + held-out); hybrid+rerank ≥ vector-only; /search works |
| 3 Agents — extraction | R2, R3, R7, R8, R35–R40, R43–R52, R54–R59, R85, R91–R93 | Bid1.json + Bid2.json valid, 20 fields, addendum changes, both traces saved |
| 4 Agents — Q&A + bonus agents | R4, R41, R53, R82, R83 | qa_log.md ≥10 incl. 5 PDF questions with answering agent; comparison + go/no-go outputs |
| 5 Interfaces + deploy | R31, R62, R86, R87 | `python main.py --bid ./Bid1` and `docker compose up` work; CI green incl. eval gate |
| 6 Robustness + scoring | R6, R65, R67, R68, R69, R94 | unseen/HTML-only/long-doc tests green; per-field accuracy + calibration reported |
| 7 Docs | R20, R42, R71, R73–R79, R88–R90 | test_deliverables green; every matrix row ✅ |

## 5. Field rubric (PDF §8 → config/fields.yaml)

Absent vs explicit-negative rule: if the documents **state** "not required"/"none", that is the value (with citation). If the documents are silent → `value: null`, `notes: "Not found in documents"`.

| Field | Group | Value shape / rule | Scoring rule |
|---|---|---|---|
| Bid Number | Org | official solicitation ID string | exact (normalized case/space) |
| Title | Org | full title | fuzzy ≥0.9 |
| Due Date | Dates | `YYYY-MM-DD HH:MM TZ`; final after addenda; notes = original date if changed | date+time+TZ normalized exact |
| Bid Submission Type | Dates | enum list: electronic_portal / sealed_paper / email / other (+ detail text) | set match |
| Term of Bid | Commercial | duration + renewal options | key-term match |
| Pre Bid Meeting | Dates | {date, time, location, mandatory: bool} or "None" | per-subfield exact |
| Installation | Dates | {required: bool, services: [installation, deployment, imaging…]} | bool exact + set |
| Bid Bond Requirement | Commercial | amount or percentage, or "Not required" (explicit) / null (silent) | exact on amount/% or category |
| Delivery Date | Dates | date or window after award (e.g. "within 30 days ARO") | normalized match |
| Payment Terms | Commercial | terms (e.g. Net 30) + invoicing conditions | key-term match |
| Any Additional Documentation Required | Commercial | list: affidavits, forms, certificates, insurance, W-9… | set match (recall + precision) |
| MFG for Registration | Product | list of manufacturers, or null ("if any") | set match |
| Contract or Cooperative to use | Commercial | contract vehicle name, or null ("if any") | fuzzy |
| Model_no | Product | list | set exact |
| Part_no | Product | list | set exact |
| Product | Product | list of {name, quantity} | name fuzzy + qty exact |
| contact_info | Org | {name, email, phone} | per-subfield exact |
| company_name | Org | issuing agency | fuzzy |
| Bid Summary | Report | 3–6 sentences, cited | sentence count + citation present (not accuracy-scored) |
| Product Specification | Product | {cpu, ram, storage, display, warranty, os, other…} | per-key keyword match |

## 6. Confidence score (computed, not LLM-self-reported)
`confidence = w1·quote_match + w2·rerank_score + w3·source_agreement + w4·judge_support − penalty(retries)`; weights in settings.yaml, tuned in R94.

## 7. Assumptions (initial — finalize in README)
- Bid folder = one bid; folder name = bid_id.
- Addendum precedence: higher addendum number wins; ties → later document date.
- Times without TZ are reported with the TZ stated elsewhere in the bid, else flagged in notes.
- Free-tier LLMs: rate limits handled by backoff + fallback provider.

## 8. Known limitations (initial — finalize in README)
- Free-tier LLM quotas cap throughput; large batches need a paid tier.
- OCR (rapidocr) accuracy on dense image tables is lower than native text.
- Time-zone ambiguity when a document omits TZ.
- Gold sets are small (~20 retrieval Qs, ~40 fields); held-out split and calibration are indicative, not statistically strong.
- Reconciliation assumes addenda are explicitly numbered or dated; unnumbered/undated revisions are flagged, not auto-ordered.
- Local CPU models (embeddings, reranker, Ollama) are slower than GPU/API alternatives.
