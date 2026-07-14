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

## Result

On the untouched natural test set, the public tuned checkpoint raises mean VQA score
from 69.5% to 72.6% and lowers the entirely-wrong rate from 20.5% to 16.6%. That
average-utility gain does not yield a deployable selective policy: the prespecified
base and tuned confidence gates both fail independent certification, with one-sided
Clopper--Pearson upper bounds of 15.71% and 41.28% against a 10% risk target. The
prescribed outcome is therefore zero deployed coverage for both models. Candidate
test outcomes from failed thresholds are retained as diagnostics, never as certified
results. See `paper/main.pdf` for the complete claim boundary and analysis.

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

Research artifacts are written to the Modal volume `sash-vlm-safety`. Quantitative
paper macros are generated only from completed predictions; manual review labels are
archived in `failure_review.csv`, with the four privacy-safe table rows in
`publication_examples.csv`. The generated raw failure-candidate table and images stay
untracked so privacy-screened-out cases are not redistributed.

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
$env:SOURCE_DATE_EPOCH = "1783900800"  # 2026-07-13 00:00:00 UTC
Push-Location paper
tectonic main.tex
Pop-Location
```

The Modal analysis stage also writes `analysis_bundle.zip`. This single-file download
avoids a recursive-folder issue in the Modal CLI on Windows:

```powershell
uv run modal volume get sash-vlm-safety analysis_bundle.zip artifacts/analysis_bundle.zip --force
Expand-Archive artifacts/analysis_bundle.zip artifacts/analysis -Force
```

`read_predictions` refuses wrong checkpoint revisions, reserved-pilot leakage, stale
official VQA scores, semantic-abstention drift, duplicate IDs, and token-count drift.
The final provenance file records SHA-256 hashes for the two raw prediction files and
both frozen manifests.

## arXiv upload

Upload `paper/arxiv-source.zip`. It contains only the manuscript, bibliography,
generated result macros, and the figure used by the paper. Inspect arXiv's generated
PDF preview before finalizing the submission; the local gate uses Tectonic 0.16.9,
not arXiv's AutoTeX environment.
