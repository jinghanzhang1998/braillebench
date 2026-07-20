# Migration Guide — Braille Reasoning Benchmark

This guide is for setting up the project on a **new device** (specifically to run
**Qwen3-1.5B on a GPU server**). Written for an AI coding assistant (Codex) to follow.

## What This Project Does

Evaluates how well LLMs can read and write **Braille** (Grade 1 & Grade 2, UEB, ASCII
representation) while doing reasoning tasks. 5 datasets × up to 6 configs.

**Core research question**: Can LLMs reason in Braille space and write Braille output?
Key finding so far: only Claude Opus 4.8 can genuinely read/write Braille; other models
mostly ignore Braille-output instructions and respond in English.

## Directory Structure

```
braille-benchmark/
├── EXPERIMENT_LOG.md          # Full chronological experiment log — READ THIS FIRST
├── MIGRATION_GUIDE.md         # This file
├── requirements.txt           # Python dependencies
├── docs/
│   └── available_models.md    # Model IDs, API formats, Bedrock config
├── src/
│   ├── model_client.py        # Bedrock API client (Claude/Llama/Qwen)
│   ├── model_client_local.py  # Local GPU client (Qwen3-1.5B) ← FOR THIS MIGRATION
│   ├── run_local_eval.py      # Local GPU evaluation runner ← FOR THIS MIGRATION
│   ├── translator.py          # liblouis Braille translation
│   ├── latex_preprocess.py    # LaTeX → plain math cleanup
│   ├── convert_dataset.py     # Dataset → Braille conversion
│   ├── run_translate_all.py   # Batch translate all datasets
│   ├── opus_clean_math.py     # LLM-assisted math LaTeX cleanup
│   ├── evaluate.py            # Baseline (EN→EN) evaluation
│   ├── evaluate_braille.py    # Braille config evaluation (all configs) ← KEY LOGIC
│   ├── run_full_eval.py       # Baseline orchestrator
│   ├── run_braille_eval.py    # Braille orchestrator
│   └── fix_failed_records.py  # Re-run only failed records in a result file
└── data/
    ├── raw/                   # Original datasets (JSONL) — 701MB
    ├── braille/               # Pre-translated Braille datasets — 107MB
    │   └── <dataset>/<grade1|grade2>/<ascii|unicode|dots>.jsonl
    └── results/
        ├── baseline_full/     # EN→EN results + baseline_results.md
        └── braille_eval/      # Braille config results + braille_results.md
```

## Setup on New Device

### 1. Python environment
```bash
pip install -r requirements.txt
```

### 2. liblouis (Braille library) — REQUIRED for translation/evaluation

The `louis` Python module needs the liblouis C library. Two options:

**Option A (recommended, Linux):**
```bash
sudo apt-get install liblouis-bin liblouis-dev
pip install louis
```

**Option B (build from source, matches original setup):**
```bash
curl -sL https://github.com/liblouis/liblouis/releases/download/v3.32.0/liblouis-3.32.0.tar.gz -o liblouis.tar.gz
tar xzf liblouis.tar.gz && cd liblouis-3.32.0
./configure --prefix=$HOME/local && make -j4 && make install
cd python && pip install .
export LOUIS_TABLEPATH=$HOME/local/share/liblouis/tables
export LD_LIBRARY_PATH=$HOME/local/lib   # Linux
# (macOS was DYLD_LIBRARY_PATH; the louis __init__.py was patched to load the dylib by absolute path)
```

**IMPORTANT — louis module library loading:**
The stock `louis/__init__.py` hardcodes `liblouis.dll`. On the original macOS setup it was
patched (around line 70) to load the platform library. On the new device, verify:
```python
import os
os.environ["LOUIS_TABLEPATH"] = "<path>/share/liblouis/tables"
import louis
print(louis.translateString(["en-ueb-g2.ctb"], "Hello world"))  # should print: ,hello _w
```
If it errors on library loading, patch `louis/__init__.py` to `cdll.LoadLibrary("<abs path to liblouis.so/.dylib>")`.

**NOTE:** If Braille data (`data/braille/`) is already translated and copied over, you do
NOT strictly need liblouis for *evaluation input* — but `evaluate_braille.py` uses
`louis.backTranslateString()` to validate Braille *output*, so liblouis IS needed for
scoring braille-output configs (EN-G1, EN-G2, G1-G1, G2-G2).

### 3. Verify data is present
```bash
ls data/braille/gsm8k/grade1/ascii.jsonl   # should exist
ls data/raw/gsm8k/test.jsonl               # should exist
```
If `data/` was not copied (it's ~900MB), regenerate:
- Raw data: re-download (see EXPERIMENT_LOG.md "Dataset Preparation" — GSM8K from OpenAI
  GitHub, AIME24 from HF AI-MO/aimo-validation-aime, QA from HF RUC-NLPIR/FlashRAG_datasets)
- Braille: `LOUIS_TABLEPATH=... python src/run_translate_all.py`

## Running Qwen3-1.5B on GPU (the migration task)

Qwen3-1.5B is NOT on Bedrock — must run locally on GPU.

### Files involved
- `src/model_client_local.py` — loads model via transformers, `invoke_local_model(name, prompt)`
- `src/run_local_eval.py` — evaluation runner using local model

### Model config
`model_client_local.py` `LOCAL_MODELS` dict maps `"qwen3-1.5b"` → `"Qwen/Qwen3-1.5B"`.
Verify the exact HuggingFace model ID exists; adjust if needed (e.g. `Qwen/Qwen3-1.5B-Instruct`).

### Commands
```bash
export LOUIS_TABLEPATH=<path>/share/liblouis/tables

# Baseline (EN→EN)
python src/run_local_eval.py --model qwen3-1.5b --mode baseline

# Braille reading (G1→EN, G2→EN)
python src/run_local_eval.py --model qwen3-1.5b --mode braille --configs G1-EN G2-EN --formats ascii

# Braille writing (EN→G1, EN→G2) and full braille (G1→G1, G2→G2)
python src/run_local_eval.py --model qwen3-1.5b --mode braille --configs EN-G1 EN-G2 G1-G1 G2-G2 --formats ascii
```
Results save to `data/results/local/qwen3-1.5b_<dataset>_<config>_details.jsonl`.

### Evaluation logic Codex must preserve (see evaluate_braille.py)
- **Math answers**: `math_verify` (SymPy) on English; for braille output, back-translate first
- **QA answers**: EM + F1 with FlashRAG `normalize_answer`
- **Braille output validation (two-stage)** — CRITICAL:
  1. If output has uppercase A-Z or literal digits 0-9 → model wrote English not Braille → score 0, flag `wrote_english=True`
  2. Else back-translate braille → English, compare to English gold
- Dataset limits: hotpotqa & 2wikimultihopqa capped at 1500; gsm8k(1319)/aime24(30)/commonsenseqa(1221) full
- Checkpoint/resume: results written incrementally; re-running resumes from last line

## Evaluation Configurations

| Config | Input | Output | Meaning |
|--------|-------|--------|---------|
| EN-EN | English | English | Baseline |
| G1-EN | Grade1 Braille | English | Read Grade 1 |
| G2-EN | Grade2 Braille | English | Read Grade 2 |
| EN-G1 | English | Grade1 Braille | Write Grade 1 |
| EN-G2 | English | Grade2 Braille | Write Grade 2 |
| G1-G1 | Grade1 Braille | Grade1 Braille | Full Grade 1 |
| G2-G2 | Grade2 Braille | Grade2 Braille | Full Grade 2 |

Bedrock models (already done): claude-opus-4.8, claude-haiku-4.5, llama3.3-70b, llama3.1-8b, qwen3-32b
Local GPU (this migration): qwen3-1.5b

## Key Docs to Read (in order)
1. `EXPERIMENT_LOG.md` — what was done, findings, bug fixes
2. `data/results/baseline_full/baseline_results.md` — baseline scores
3. `data/results/braille_eval/braille_results.md` — braille scores + findings
4. `docs/available_models.md` — model IDs and API formats
