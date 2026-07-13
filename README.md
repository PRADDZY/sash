# Assistive VQA selective-reliability audit

This repository compares Qwen3-VL-8B with a public checkpoint described as VizWiz
domain-fine-tuned. It asks a stricter deployment question: can either model avoid
entirely wrong answers while retaining useful coverage? Sparse tuning provenance means
the comparison is not presented as an identified causal effect of fine-tuning.

The confirmatory protocol uses a seeded question-level 20/20/60 split:

- `fit`: learn one answer-confidence threshold and a Platt calibrator;
- `certification`: independently test the frozen threshold with a one-sided
  Clopper--Pearson bound; and
- `test`: report the operating point without further tuning.

The primary endpoint is the rate of entirely incorrect answers (`VQA score == 0`)
among nonempty, substantive answers; semantic `unanswerable` responses do not count as
coverage. Eighteen instrumentation-pilot sources are excluded before the seeded
4,301-question split (860 fit / 860 certification / 2,581 test). The primary confidence
score is mean generated-answer token log-likelihood. Grounding-gap and post-acquisition
scores are exploratory.

## Local checks

```powershell
uv sync --extra dev --extra modal
uv run pytest
uv run ruff check .
```

## Modal stages

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
uv run modal setup  # once per machine
uv run modal run modal_app.py --stage download
uv run modal run modal_app.py --stage prepare
uv run modal run modal_app.py --stage pilot
uv run modal run modal_app.py --stage infer --model base
uv run modal run modal_app.py --stage infer --model finetuned
uv run modal run modal_app.py --stage analyze
```

The pilot estimates full-run GPU cost on an A10 and stops if the projected primary spend
exceeds $55, preserving headroom within the reported roughly $70 credit balance. A10 is
used because this Modal workspace does not currently permit L40S functions without a
payment method.

Research artifacts are written to the Modal volume `sash-vlm-safety`. No result is
hard-coded into the paper; result macros are generated only from completed predictions.

## Rebuild the analysis and paper

The canonical run started from absent `predictions/base.jsonl` and
`predictions/finetuned.jsonl`; delete those two volume files before any intentionally
fresh rerun because inference otherwise resumes by case ID.

```powershell
New-Item -ItemType Directory -Force artifacts/predictions | Out-Null
uv run modal volume get sash-vlm-safety predictions/base.jsonl artifacts/predictions/base.jsonl --force
uv run modal volume get sash-vlm-safety predictions/finetuned.jsonl artifacts/predictions/finetuned.jsonl --force
uv run sash-audit artifacts/predictions/base.jsonl artifacts/predictions/finetuned.jsonl --output artifacts/analysis

# Tectonic 0.16.0 or newer
Push-Location paper
tectonic main.tex
Pop-Location
```

`read_predictions` refuses wrong checkpoint revisions, reserved-pilot leakage, stale
official VQA scores, semantic-abstention drift, duplicate IDs, and token-count drift.
The final provenance file records SHA-256 hashes for the two raw prediction files and
both frozen manifests.
