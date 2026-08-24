# BrailleBench

BrailleBench evaluates whether a language model can read and write Unified English Braille
(UEB) while solving reasoning and question-answering tasks. Its primary evaluation surface is
six-dot UEB encoded as Braille ASCII. The release also provides synchronized Unicode Braille and
dot-number representations, parallel English fields, and a provider-neutral evaluation harness.

Version `0.1.0` is the first public release. It includes directly usable Braille records for all
five benchmark datasets. BrailleBench does not relicense the source questions and answers;
dataset-specific attribution and terms are documented in [DATA_LICENSES.md](DATA_LICENSES.md).

## What it tests

The seven core configurations separate Braille reading, Braille writing, and end-to-end
Braille reasoning:

| Config | Input | Output | Capability |
|---|---|---|---|
| `EN-EN` | English | English | Standard baseline |
| `G1-EN` | Grade 1 | English | Uncontracted Braille reading |
| `G2-EN` | Grade 2 | English | Contracted Braille reading |
| `EN-G1` | English | Grade 1 | Uncontracted Braille writing |
| `EN-G2` | English | Grade 2 | Contracted Braille writing |
| `G1-G1` | Grade 1 | Grade 1 | Full Grade 1 reasoning |
| `G2-G2` | Grade 2 | Grade 2 | Full Grade 2 reasoning |

The harness also exposes pure-Braille-prompt (`FULLBR-G1`, `FULLBR-G2`) and mapping
cheat-sheet (`G1-EN-CS`, `G2-EN-CS`) ablations. These are not part of the seven-config core
matrix unless reported explicitly.

## Data

| Dataset | Split | Records | Task | Primary metric |
|---|---|---:|---|---|
| GSM8K | test | 1,319 | Grade-school math | `math_verify` |
| AIME 2024 | I + II | 30 | Competition math | `math_verify` |
| CommonsenseQA | dev | 1,221 | Multiple-choice QA | Exact Match |
| HotpotQA | dev | 7,405 | Multi-hop QA | Exact Match |
| 2WikiMultiHopQA | dev | 12,576 | Multi-hop QA | Exact Match |

The complete release contains 22,551 logical records. Each record retains source attribution
through its dataset membership and is distributed with the dataset-specific notices in
`DATA_LICENSES.md`.

Each logical record appears in six files:

```text
data/braille/<dataset>/
  grade1/{ascii,unicode,dots}.jsonl
  grade2/{ascii,unicode,dots}.jsonl
```

The files retain the parallel English question, choices when applicable, and English gold
answer, so the packaged benchmark can run English-input configurations without a second copy
of the raw datasets.

### Representations

- `ascii`: printable Braille ASCII/BRF cells, for example `#ah` for the number 18.
- `unicode`: Unicode Braille Patterns (`U+2800` block).
- `dots`: space-separated dot numbers, such as `1-2-5`.

The benchmark uses the liblouis UEB tables `en-ueb-g1.ctb` and `en-ueb-g2.ctb`.

## Installation

Python 3.10-3.13 is supported. The reference environment uses Python 3.11 and conda-forge because
liblouis includes a native library and Python bindings:

```bash
conda env create -f environment.yml
conda activate braillebench
export LOUIS_TABLEPATH="$CONDA_PREFIX/share/liblouis/tables"
```

For English-output-only runs (`EN-EN`, `G1-EN`, `G2-EN`), liblouis is not needed at
evaluation time. A pip environment is sufficient:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Official liblouis installation and Python-binding documentation:
[liblouis installation](https://github.com/liblouis/liblouis) and
[Python bindings](https://liblouis.io/documentation/liblouis/Python-bindings.html).

## Validate the package

Run the dependency-free data audit first:

```bash
python src/validate_release.py --check-manifest DATA_MANIFEST.json
```

It parses every JSONL row, checks all 30 files and expected counts, verifies parallel fields
across grades/formats, validates required gold fields, and compares SHA-256 checksums.

Then run the scoring and checkpoint regression tests:

```bash
python -m unittest discover -s tests -v
python src/test_extract_answer.py
python src/test_english_gate.py
```

Braille-output scoring tests require a working `louis` import and `LOUIS_TABLEPATH`.

## Connect a model

Edit `src/model_client.py`. Register models in `MODELS` if you want to use `--models all`,
and implement one function:

```python
MODELS = {"my-model": {}}

def invoke_with_retry(model_name, prompt, max_tokens=1024, temperature=0.0):
    # Call an API, local inference server, or Transformers runtime.
    # Return the model's response as a string.
    ...
```

Keep provider credentials in environment variables or the provider's normal credential store.
Do not put them in this repository. Raise transient connection, timeout, throttling, and
credential errors; the harness pauses instead of permanently scoring those failures as zero.

## Run an evaluation

Small reading smoke run:

```bash
python src/evaluate_braille.py \
  --models my-model \
  --datasets gsm8k \
  --configs G1-EN \
  --formats ascii \
  --limit 20 \
  --output-dir runs/my-model-smoke
```

Seven-config ASCII core matrix:

```bash
python src/evaluate_braille.py \
  --models my-model \
  --datasets all \
  --configs EN-EN G1-EN G2-EN EN-G1 EN-G2 G1-G1 G2-G2 \
  --formats ascii \
  --output-dir runs/my-model-core
```

Use `--formats unicode` or `--formats dots` for another surface representation. Generation
defaults match the original protocol: temperature 0, 1,024 output tokens for most datasets,
and 4,096 for AIME24. Reasoning models that need a larger budget can use `--max-tokens` and
`--aime-max-tokens`; report any deviation with the results.

## Outputs and resume behavior

Per-item files are written under the selected output directory:

```text
<model>_<dataset>_<config>_<format>_details.jsonl
summary.json
```

Each detail row contains:

```json
{
  "id": "source or positional id",
  "gold_answer": ["English gold"],
  "predicted": "extracted model answer",
  "metrics": {"em": 0.0, "sub_em": 0.0, "f1": 0.0},
  "full_response": "verbatim model response"
}
```

Braille-output metrics additionally include `wrote_english`, `back_translated`,
`predicted_raw`, and `trailing_period_normalized`. Interrupted runs resume from the existing
JSONL prefix. Keep each output directory tied to one model, prompt protocol, and generation
configuration.

## Scoring

- Math: symbolic equivalence via `math_verify`, with a numeric fallback.
- QA: FlashRAG-style normalized Exact Match, substring EM, and token F1.
- Braille output: format normalization, conservative print-English detection, liblouis
  back-translation, then comparison against English gold.

Uppercase `A-Z` is a reliable print-English signal in Braille ASCII because capitals use a
comma prefix and lowercase cells. Literal digits are not rejected globally: they are valid
Braille ASCII punctuation cells. Math uses an additional numeral-specific check. A trailing
print sentence period is normalized only in the scoring layer and is recorded for auditing.

## Rebuilding translations

The release already contains translated data. To regenerate it from separately obtained raw
datasets, place source JSONL files under `data/raw/<dataset>/` and run:

```bash
python src/run_translate_all.py
```

The public preprocessor is deterministic and never calls an LLM. Complex LaTeX that remains
after rule-based cleanup must be reviewed or handled by a preprocessing step you provide.

## Known limitations

- Braille ASCII and lowercase print English share characters, so language detection must be
  conservative. The validator prioritizes avoiding rejection of valid Braille cells.
- Results depend on the liblouis version and UEB tables; record both in published evaluations.
- A generation cap can end a reasoning model before its final answer. Report token limits and
  inspect empty predictions or unclosed reasoning tags.
- The benchmark evaluates English UEB only; it does not measure other Braille languages or codes.
- No context passages are supplied for HotpotQA/2Wiki; this is a closed-book reasoning setting.

## Project layout

```text
data/braille/              parallel benchmark JSONL
src/evaluate_braille.py    prompts, inference loop, and scoring
src/model_client.py        provider adapter to implement
src/validate_release.py    data/checksum audit
src/translator.py          liblouis translation helpers
tests/                     regression tests
DATA_MANIFEST.json         counts, sizes, and SHA-256 checksums
DATA_LICENSES.md           upstream attribution and dataset terms
THIRD_PARTY_NOTICES.md     consolidated third-party notices
LICENSE                    MIT license for BrailleBench-authored code
CITATION.cff               machine-readable citation metadata
RELEASE_NOTES.md           versioned release summary
```

## Citation

Please cite the versioned repository release until the associated paper record is available:

```bibtex
@dataset{zhang2026braillebench,
  author    = {Jinghan Zhang},
  title     = {BrailleBench},
  year      = {2026},
  version   = {0.1.0},
  publisher = {GitHub},
  url       = {https://github.com/jinghanzhang1998/braillebench}
}
```

Also cite the upstream datasets used in the evaluation. GitHub-compatible citation metadata is
provided in [CITATION.cff](CITATION.cff).

## License

BrailleBench-authored code is released under the [MIT License](LICENSE). The MIT License does not
relicense source questions, answers, or other third-party dataset content contained in translated
records. Those materials remain subject to their upstream terms; see
[DATA_LICENSES.md](DATA_LICENSES.md).
