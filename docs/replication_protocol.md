# Replication protocol

This protocol freezes the follow-up evaluation before inspecting full-run
metrics. It is an extension of the original Qwen base-versus-finetuned audit,
not a replacement for it.

## Data and splits

- Dataset: `lmms-lab/VizWiz-VQA`, validation split, revision
  `d428a2dae984f79cf1b9d99467dfa883e0c30686`.
- The existing frozen manifest supplies 4,301 natural rows and 1,200 paired
  shifted/acquired-view rows. The reserved pilot rows are excluded from every
  full prediction file.
- Natural rows use the existing 860/860/2,581 fit/certification/test split.
  Shift rows use 300/300/600. Every model must have identical case IDs within
  each regime and split.

## Checkpoints

The replication pair is pinned by immutable Hugging Face revisions in
`sash_audit.inference.MODEL_SPECS`:

- `smolvlm`: `HuggingFaceTB/SmolVLM2-2.2B-Instruct`
- `llava_onevision`: `llava-hf/llava-onevision-qwen2-7b-ov-hf`

Phi-4 was the first additional candidate. Its pinned runtime reached model
loading but failed at generation because the checkpoint's PEFT wrapper lacks
the generation hook required by the installed Transformers/PEFT combination.
It is retained in the registry for provenance and is not part of the reported
replication pair.

## Analysis and stopping rules

- Greedy decoding, `max_new_tokens=32`, with the existing blank-image and
  shifted/acquired-view counterfactual scores.
- Fit thresholds are learned on fit rows at empirical risk `0.05`.
- Certification uses target risk `0.10`, family-wise delta `0.05`, split evenly
  across the two replication models (`delta=0.025` each).
- One 18-row pilot per model is required before full inference. The combined
  projected A10 GPU cost must remain below `$20`; checkpointed JSONL writes
  permit safe resume after interruption.
- `analyze-replication` requires strict split counts and writes metrics,
  100-seed split sensitivity, and immutable protocol metadata. No result is
  treated as publication-ready until both full prediction files pass these
  checks.

W&B, when enabled, receives aggregate run metadata and progress only. Raw
questions, images, reference answers, and prediction rows are never logged.
