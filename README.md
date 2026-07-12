# Assistive VQA selective-reliability audit

This repository compares Qwen3-VL-8B before and after VizWiz domain fine-tuning. It
measures whether accuracy gains survive a stricter deployment question: can the model
avoid entirely wrong answers while retaining useful coverage?

The confirmatory protocol uses a seeded question-level 20/20/60 split:

- `fit`: learn one answer-confidence threshold and a Platt calibrator;
- `certification`: independently test the frozen threshold with a one-sided
  Clopper--Pearson bound; and
- `test`: report the operating point without further tuning.

The primary endpoint is the rate of entirely incorrect answers (`VQA score == 0`)
among answered questions. The primary confidence score is mean generated-answer token
log-likelihood. Grounding-gap and post-acquisition scores are exploratory.

## Local checks

```powershell
uv sync --extra dev
uv run pytest
uv run ruff check .
```

## Modal stages

```powershell
modal run modal_app.py --stage download
modal run modal_app.py --stage prepare
modal run modal_app.py --stage pilot
modal run modal_app.py --stage infer --model base
modal run modal_app.py --stage infer --model finetuned
modal run modal_app.py --stage analyze
```

The pilot estimates full-run GPU cost and stops if the projected primary spend exceeds
$55, preserving the remainder of a $70 budget for reruns and artifact generation.

Research artifacts are written to the Modal volume `sash-vlm-safety`. No result is
hard-coded into the paper; result macros are generated only from completed predictions.
