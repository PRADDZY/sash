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
uv sync --extra dev
uv run pytest
uv run ruff check .
```

## Modal stages

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
modal run modal_app.py --stage download
modal run modal_app.py --stage prepare
modal run modal_app.py --stage pilot
modal run modal_app.py --stage infer --model base
modal run modal_app.py --stage infer --model finetuned
modal run modal_app.py --stage analyze
```

The pilot estimates full-run GPU cost on an A10 and stops if the projected primary spend
exceeds $55, preserving headroom within the reported roughly $70 credit balance. A10 is
used because this Modal workspace does not currently permit L40S functions without a
payment method.

Research artifacts are written to the Modal volume `sash-vlm-safety`. No result is
hard-coded into the paper; result macros are generated only from completed predictions.
