# Experiment Log

## 2026-06-26: Project Setup

### Dataset Preparation
- Downloaded 5 reasoning datasets to `data/raw/`:
  - GSM8K (test: 1319), AIME 2024 (30), CommonsenseQA (dev: 1221), HotpotQA (dev: 7405), 2WikiMultiHopQA (dev: 12576)
- Sources: GSM8K from OpenAI GitHub, AIME24 from HuggingFace AI-MO, QA datasets from FlashRAG HuggingFace

### Braille Translation System
- Installed liblouis 3.32.0 from source (`/Users/jinghanz/local/`)
- Built translation pipeline: `src/translator.py` (liblouis wrapper) + `src/latex_preprocess.py` (rule-based + Opus fallback)
- Supports 3 output formats: Braille ASCII, Unicode Braille, Dot Notation
- Supports Grade 1 (uncontracted) and Grade 2 (contracted) UEB

### BrailleLLM Paper Analysis
- Paper uses Grade 2 Braille ASCII (BRF format) via liblouis
- No Unicode Braille dots in their data
- Reference code cloned to `../reference/BrailleLLM/`

---

## 2026-06-26 ~ 2026-06-29: Braille Translation

### Full Dataset Translation
- Script: `src/run_translate_all.py`
- Output: `data/braille/<dataset>/<grade>/<format>.jsonl` (30 files total)
- All records include translated question, answer, choices (where applicable)
- AIME24 additionally cleaned with Claude Sonnet 4.6 for residual LaTeX (`data/braille/aime24_opus_cleaned.jsonl`)

### Translation Quality Check
- Round-trip accuracy (translate → back-translate → compare):
  - GSM8K: G1=95.9%, G2=95.2%
  - CommonsenseQA: G1=100%, G2=99.8%
  - HotpotQA: G1=98.8%, G2=96.4%
  - 2WikiMultiHopQA: G1=97.9%, G2=90.3%
  - AIME24: G1=73.3%, G2=63.3% (Grade 2 variable/contraction ambiguity is expected)
- Grade 2 compression ratio: saves 14-23% characters vs Grade 1
- Remaining G2 "errors" are inherent Grade 2 ambiguity (b=but, c=can, f=from, etc.) — this is the core phenomenon the benchmark tests

---

## 2026-06-26 ~ 2026-07-10: Baseline Evaluation (EN→EN)

### Setup
- Script: `src/evaluate.py`, orchestrator: `src/run_full_eval.py`
- Metrics: math-verify (OpenCompass/SymPy) for math, EM+F1 (FlashRAG normalize_answer) for QA
- Prompt: `\boxed{}` for math, "The answer is:" for QA
- Temperature: 0, zero-shot, closed-book

### Models Evaluated
- Claude Opus 4.8, Claude Haiku 4.5, Llama 4 Maverick, Llama 3.1 8B, Qwen3 Next 80B, Qwen3 32B
- Results: `data/results/baseline_full/<model>/<model>_<dataset>_details.jsonl`
- Summary: `data/results/baseline_full/baseline_results.md`

### Key Results (math_verify / EM for QA)
| Model | GSM8K | AIME24 | CSQA | HotpotQA | 2Wiki |
|-------|-------|--------|------|----------|-------|
| Claude Opus 4.8 | 95.3% | 96.7% | 67.2% | 39.9% | 45.9% |
| Claude Haiku 4.5 | 95.7% | 53.3% | 17.7% | 14.3% | 19.3% |
| Llama 4 Maverick | 13.3% | 36.7% | 13.5% | 10.3% | 40.4% |
| Llama 3.1 8B | 83.2% | 6.7% | 10.3% | 21.3% | 0.7% |
| Qwen3 Next 80B | 94.8% | 0.0% | 22.4% | 28.1% | 32.3% |
| Qwen3 32B | 94.0% | 13.3% | 18.6% | 21.3% | 28.1% |

### Issues & Decisions
- Llama 4 Maverick: broken output format on most tasks → **dropped**
- Qwen3 Next 80B: 17s/call (45% of total time), AIME24=0% → **dropped**
- Credential expiration (~12h) caused multiple re-runs over ~7 days
- Total compute: ~59h pure, ~7 days wall-clock with interruptions

---

## 2026-07-10: Model List Update

### Final Model Selection (Bedrock)
| Model | Role | Speed |
|-------|------|-------|
| Claude Opus 4.8 | Flagship/ceiling | ~5s/call |
| Claude Haiku 4.5 | Fast Claude | ~1.5s/call |
| Llama 3.3 70B | Best open dense (NEW) | ~2s/call |
| Llama 3.1 8B | Small open | ~1s/call |
| Qwen3 32B | Thinking model | ~12s/call |

### Local GPU Model
| Model | Role | Script |
|-------|------|--------|
| Qwen3 1.5B | Tiny baseline | `src/model_client_local.py`, `src/run_local_eval.py` |

---

## 2026-07-12 ~ 2026-07-17: Braille Evaluation (G1-EN, G2-EN)

### Setup
- Script: `src/evaluate_braille.py`, orchestrator: `src/run_braille_eval.py`
- Config G1-EN: Grade 1 Braille ASCII input → English output
- Config G2-EN: Grade 2 Braille ASCII input → English output
- Braille text directly inserted into prompt (no intermediate translation at eval time)
- Checkpoint/resume: each record written incrementally, survives credential expiry

### Code Fixes During This Phase
- `model_client.py`: handle empty content response (was crashing with `list index out of range`)
- `evaluate_braille.py`: incremental write with resume from checkpoint
- `run_braille_eval.py`: abort after 3 consecutive failures; correct dataset limits (hotpotqa/2wiki = 1500)
- Empty model responses marked as `[MODEL RETURNED EMPTY RESPONSE]`, counted as EM=0

### Key Finding: Claude Opus 4.8 Refusal on Grade 2 Braille

**Only Opus shows massive refusal rates on G2-EN, other models do not:**

| Model | G1-EN Refusal | G2-EN Refusal |
|-------|:-:|:-:|
| Claude Opus 4.8 | 0-1.5% | **12-72%** |
| Claude Haiku 4.5 | 0-3.2% | 0% |
| Llama 3.3 70B | 0-0.7% | 0-2.0% |
| Llama 3.1 8B | 0-0.3% | 0-1.4% |
| Qwen3 32B | 0-0.2% | 0-0.6% |

Opus G2-EN refusal by dataset: GSM8K 12% → CSQA 55% → HotpotQA 65% → 2Wiki 72%

**Hypothesis**: Grade 2 Braille ASCII uses characters like `>`, `<`, `#`, `$`, `\` which may trigger Opus's safety filters. Grade 1 uses mostly plain letters and doesn't trigger this. Other models (Haiku, Llama, Qwen) have less aggressive content filtering.

**Implication**: The strongest model (Opus) performs worst on G2-EN not just from comprehension difficulty but also from safety-filter interference. This is a significant finding for the paper — accessibility tools may be inadvertently penalized by safety mechanisms.

### Progress (as of 2026-07-17)

| Config | Complete | Status |
|--------|:--------:|--------|
| G1-EN | 24/25 | opus 2wiki in progress (1189/1500) |
| G2-EN | 20/25 | haiku gsm8k in progress (504/1319) |

### Preliminary Results Summary

#### G1-EN (Grade 1 → English)
| Model | GSM8K (MV) | AIME24 (MV) | CSQA (EM) | HotpotQA (EM) | 2Wiki (EM) |
|-------|:-:|:-:|:-:|:-:|:-:|
| Claude Opus 4.8 | 85.9% | 90.0% | 86.8% | 38.8% | pending |
| Claude Haiku 4.5 | 35.3% | 3.3% | 29.7% | 25.3% | 22.8% |
| Llama 3.3 70B | 7.2% | 0.0% | 8.9% | 18.6% | 17.2% |
| Llama 3.1 8B | 4.1% | 0.0% | 0.2% | 10.9% | 7.1% |
| Qwen3 32B | 12.6% | 0.0% | 57.1% | 14.1% | 20.3% |

#### G2-EN (Grade 2 → English)
| Model | GSM8K (MV) | AIME24 (MV) | CSQA (EM) | HotpotQA (EM) | 2Wiki (EM) |
|-------|:-:|:-:|:-:|:-:|:-:|
| Claude Opus 4.8 | 65.6% | 76.7% | 34.6%* | 8.8%* | 5.5%* |
| Claude Haiku 4.5 | pending | - | - | - | - |
| Llama 3.3 70B | 3.6% | 0.0% | 4.1% | 3.0% | 0.8% |
| Llama 3.1 8B | 1.4% | 0.0% | 0.5% | 0.7% | 0.5% |
| Qwen3 32B | 7.7% | 3.3% | 17.4% | 1.3% | 0.9% |

*Opus G2-EN scores heavily impacted by refusal rate (54-72% empty responses counted as 0)

#### Baseline vs Braille Drop (GSM8K math_verify)
| Model | EN→EN | G1→EN | G2→EN | G1 drop | G2 drop |
|-------|:-:|:-:|:-:|:-:|:-:|
| Claude Opus 4.8 | 95.3% | 85.9% | 65.6% | -9.4% | -29.7% |
| Claude Haiku 4.5 | 95.7% | 35.3% | pending | -60.4% | - |
| Llama 3.3 70B | 94.4% | 7.2% | 3.6% | -87.2% | -90.8% |
| Llama 3.1 8B | 83.2% | 4.1% | 1.4% | -79.1% | -81.8% |
| Qwen3 32B | 94.0% | 12.6% | 7.7% | -81.4% | -86.3% |

---

## 2026-07-17 ~ 2026-07-18: Completion & Bug Fixes

### Baseline Fix
- CommonsenseQA baseline: all models had inflated EM failure due to answer extraction including choice letter prefix ("D) lots of attention" vs gold "lots of attention"). Fixed by stripping `^[A-E]\)\s*` in `extract_answer()`. Scores re-computed without re-running API.
- Opus 2wiki baseline: was corrupted (983/1500 credential errors). Fully re-run. ✅ Complete.

### Haiku 2Wiki Refusal Finding
- Haiku refuses 50.4% of 2Wiki baseline questions ("Unable to determine")
- In G1-EN Braille input, refusal drops to 26.2% → EM appears higher (22.8% vs 19.3%)
- Interpretation: Braille input degrades model's ability to gauge question difficulty → less conservative → sometimes guesses correctly
- Other models don't show this pattern (their baseline refusal <1%)

### Updated Baseline Scores (CommonsenseQA after fix)
| Model | Old EM | New EM |
|-------|:------:|:------:|
| Claude Opus 4.8 | 67.2% | 88.5% |
| Claude Haiku 4.5 | 17.7% | 81.7% |
| Llama 3.3 70B | 16.3% | 80.3% |
| Llama 3.1 8B | 10.3% | 43.3% |
| Qwen3 32B | 18.6% | 83.5% |

### Partial File Fix
- 82 credential-error records in 3 braille eval files re-run via `fix_failed_records.py` (API calls only for failed records, others preserved)

### Completion Status
- Baseline EN→EN: ✅ **25/25 complete** (all clean)
- G1-EN: ✅ **25/25 complete**
- G2-EN: **24/25** (haiku 2wiki 743/1500, running)

### Next: EN-G1, EN-G2 (English → Braille output)
- Tests model's ability to generate Braille
- Evaluation: compare model's braille output against braille gold answers (direct braille-to-braille comparison)
- Priority P1

---

## 2026-07-18 ~ 2026-07-19: EN→Braille Configs Complete

### All 5 ASCII configurations complete (Bedrock models)
- EN-EN, G1-EN, G2-EN, EN-G1, EN-G2 — each 25/25 (5 models × 5 datasets)

### Bug found & fixed: braille-output math metric
- `math_verify` cannot parse braille digit notation (`#ah` = 18)
- For braille-output configs (EN-G1, EN-G2), math datasets must use EM (braille-vs-braille), not math_verify
- Fixed in `evaluate_braille.py`: braille output always uses EM/sub_em/F1, never math_verify
- Initial mis-report: EN-G1 opus gsm8k showed 0.2% (math_verify) → corrected to 69.1% (EM)

### EN→Braille Key Finding
- **Only Opus can generate correct Braille numbers**: GSM8K EN-G1 = 69.1% (Opus) vs 0.0% (all others)
- Non-Opus models cannot convert digits to braille (18 → `#ah`)
- Braille generation (EN→B) is generally harder than braille reading (B→EN)

### Methodology Fix: Braille output validation via back-translation
- **Problem found**: For braille-output configs (EN-G1, EN-G2), initial approach compared model braille output against braille gold using `normalize_answer`. But normalize strips punctuation — including braille contraction symbols (`>`, `<`, `#`, `;`, `/`). This caused Grade 2 QA scores to be inflated by ~25% via false positives (different braille strings normalizing to the same thing).
- **Fix**: Back-translate model's braille output to English via liblouis, then compare to English gold. Correctly decodes braille before comparison. Math answers also work (back-translated `#ah` → `18` → math_verify).
- Added `back_translate()` and `compute_braille_output_metrics()` to `evaluate_braille.py`.
- Recomputed all EN-G1/EN-G2 metrics from saved responses (no API re-run needed).
- Example impact: EN-G2 opus CSQA 18.8% (normalize) → 44.2% (back-translate, correct).

### Methodology Fix 2: English-detection for braille output
- **Problem**: back-translate-only validation let models that ignored the "write Braille" instruction and answered in English pass — English text decodes to itself as Grade 1 Braille (letters map to themselves), so correct English answers accidentally matched.
- **Fix**: two-stage validation in `compute_braille_output_metrics()`:
  1. English detection: output with uppercase A-Z or literal digits 0-9 → wrote English → score 0 (records `wrote_english=True`)
  2. Valid braille → back-translate → compare to English gold
- Recomputed all EN-G1/EN-G2 (no API re-run).

### Core Finding: only Opus can write Braille
"Wrote English instead of Braille" rate (EN-G1):
| Model | GSM8K | CSQA | HotpotQA |
|-------|:-:|:-:|:-:|
| Opus 4.8 | 27% | 8% | 32% |
| Haiku 4.5 | 99% | 99% | 66% |
| Llama 3.3 70B | 14% | 73% | 33% |
| Llama 3.1 8B | 66% | 100% | 98% |
| Qwen3 32B | 92% | 90% | 68% |

- Most models ignore the Braille-output instruction entirely (write English) → capability gap, not just low accuracy
- Even when attempting braille (Llama 3.3: 14% English on GSM8K), they fail to encode digits correctly (score 0%)
- Only Opus has functional Braille generation

---

## 2026-07-19: G1-G1 & G2-G2 (full Braille configs)

### Pre-run validation review (all passed)
- Prompt: correctly tells model input is Braille and to answer in Braille ✓
- `extract_answer`: models mark answers with `\boxed{}` (math) or English "The answer is:" (QA), so extraction works even when answer body is braille ✓
- `\boxed{#ah}` → extracts `#ah` → back-translates to `18` → math_verify correct ✓
- English number `\boxed{18}` → wrote_english=True → score 0 ✓
- Braille answer `,bank` → wrote_english=False → back-translate → compare ✓

### Started G1-G1, G2-G2 (Braille input → Braille output)
- Will compare against EN-G1/EN-G2 (English input → Braille output)
- Analysis planned: does braille INPUT change braille-writing behavior? (wrote_english rate, score, output content)

### Pending Work
- [ ] G1-G1, G2-G2 (running)
- [ ] Comparison analysis: EN→Braille vs Braille→Braille output behavior
- [ ] Qwen3 1.5B — run on local GPU (all configs)
- [ ] Optional: unicode/dots format variants

---

## Code Structure

```
src/
├── model_client.py          # Bedrock API client (all models)
├── model_client_local.py    # Local GPU client (Qwen3-1.5B)
├── translator.py            # liblouis Braille translation
├── latex_preprocess.py      # LaTeX → plain math (rules + Opus fallback)
├── convert_dataset.py       # Dataset → Braille conversion
├── run_translate_all.py     # Batch translate all datasets
├── opus_clean_math.py       # Opus/Sonnet assisted math LaTeX cleaning
├── evaluate.py              # Baseline evaluation (EN→EN)
├── evaluate_braille.py      # Braille evaluation (all configs)
├── run_full_eval.py         # Orchestrator for baseline
├── run_braille_eval.py      # Orchestrator for braille eval
└── run_local_eval.py        # Local GPU evaluation script
```

```
data/
├── raw/                     # Original datasets (JSONL)
├── braille/                 # Translated Braille datasets
│   └── <dataset>/<grade>/<format>.jsonl
└── results/
    ├── baseline_full/       # EN→EN results
    │   └── <model>/<model>_<dataset>_details.jsonl
    └── braille_eval/        # Braille config results
        └── <model>_<dataset>_<config>_<format>_details.jsonl
```
