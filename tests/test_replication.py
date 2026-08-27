from __future__ import annotations

import pandas as pd

from sash_audit.inference import MODEL_SPECS, REPLICATION_MODEL_KEYS
from sash_audit.replication import (
    analyze_replication,
    validate_replication_frame,
    write_replication_analysis,
)
from sash_audit.tracking import create_tracker


def replication_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_key, offset in zip(REPLICATION_MODEL_KEYS, (0.0, 0.05), strict=True):
        model_id, revision = MODEL_SPECS[model_key]
        for regime in ("natural", "shift"):
            for split in ("fit", "certification", "test"):
                for index in range(60):
                    accuracy = 1.0 if index < 48 else 0.0
                    score = 1.0 - index / 100 + offset
                    case_id = f"{regime}-{split}-{index}"
                    rows.append(
                        {
                            "case_id": case_id,
                            "model_key": model_key,
                            "model_id": model_id,
                            "model_revision": revision,
                            "evaluation_split": split,
                            "regime": regime,
                            "source_case_id": None,
                            "image_path": f"/vol/{case_id}.jpg",
                            "clear_image_path": None,
                            "answerable": True,
                            "answer_type": "other",
                            "corruption": "blur" if regime == "shift" else "none",
                            "vqa_score": accuracy,
                            "entirely_wrong": not bool(accuracy),
                            "false_answer_on_unanswerable": False,
                            "generated_answer": "object" if accuracy else "wrong",
                            "reference_answers": ["object"] * 10,
                            "generated_token_ids": [1],
                            "normalized_answer": "object" if accuracy else "wrong",
                            "generated_tokens": 1,
                            "mean_logprob": score,
                            "grounding_score": score,
                            "acquisition_score": score,
                        }
                    )
    return pd.DataFrame(rows)


def test_replication_analysis_is_paired_and_seeded() -> None:
    summaries, sensitivity = analyze_replication(replication_rows())
    assert len(summaries) == 4
    assert len(sensitivity) == 200
    assert set(summaries.model_key) == set(REPLICATION_MODEL_KEYS)
    assert sensitivity.equals(
        analyze_replication(replication_rows())[1]
    )


def test_replication_rejects_missing_model() -> None:
    frame = replication_rows()
    frame = frame[frame.model_key == REPLICATION_MODEL_KEYS[0]]
    try:
        validate_replication_frame(frame)
    except ValueError as exc:
        assert "must contain" in str(exc)
    else:
        raise AssertionError("missing replication model was accepted")


def test_tracking_without_credentials_is_noop(monkeypatch) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    tracker = create_tracker(
        model_key="smolvlm",
        model_id=MODEL_SPECS["smolvlm"][0],
        model_revision=MODEL_SPECS["smolvlm"][1],
    )
    assert tracker.enabled is False
    tracker.log({"rows_written": 1})
    tracker.finish(summary={"rows_written": 1})


def test_replication_analysis_writes_tex_macros(tmp_path) -> None:
    outputs = write_replication_analysis(replication_rows(), tmp_path)
    macro_path = tmp_path / "replication_results.tex"
    assert outputs["latex"] == str(macro_path)
    macros = macro_path.read_text(encoding="utf-8")
    assert r"\renewcommand{\SmolNaturalCertErrorsAccepted}" in macros
    assert r"\renewcommand{\LlavaShiftTestCoverage}" in macros
