from __future__ import annotations

import pandas as pd
import pytest

from sash_audit.analysis import (
    analyze_predictions,
    read_predictions,
    validate_prediction_frame,
    write_analysis,
)
from sash_audit.inference import MODEL_SPECS


def prediction_rows() -> pd.DataFrame:
    rows = []
    for model_key, offset in (("base", 0.0), ("finetuned", 0.05)):
        model_id, revision = MODEL_SPECS[model_key]
        for regime in ("natural", "shift"):
            for split in ("fit", "certification", "test"):
                for index in range(60):
                    accuracy = 1.0 if index < 48 else 0.0
                    score = (1.0 - index / 100 + offset) if index < 48 else 0.0 + offset
                    case_id = f"{regime}-{split}-{index}"
                    rows.append(
                        {
                            "case_id": case_id,
                            "row_index": index,
                            "model_key": model_key,
                            "model_id": model_id,
                            "model_revision": revision,
                            "evaluation_split": split,
                            "regime": regime,
                            "source_case_id": None,
                            "answerable": index % 4 != 0,
                            "answer_type": "unanswerable" if index % 4 == 0 else "other",
                            "corruption": "blur" if regime == "shift" else "none",
                            "question": "What is shown?",
                            "reference_answers": ["object"] * 10,
                            "generated_answer": "object" if accuracy else "wrong",
                            "normalized_answer": "object" if accuracy else "wrong",
                            "generated_tokens": 1,
                            "generated_token_ids": [1],
                            "image_path": f"/vol/{case_id}.jpg",
                            "clear_image_path": None,
                            "vqa_score": accuracy,
                            "entirely_wrong": not bool(accuracy),
                            "false_answer_on_unanswerable": index % 4 == 0 and not bool(accuracy),
                            "mean_logprob": score - 2,
                            "grounding_score": score,
                            "acquisition_score": score + (0.1 if regime == "shift" else 0),
                        }
                    )
    return pd.DataFrame(rows)


def test_analysis_emits_all_model_regime_score_combinations() -> None:
    summary, paired = analyze_predictions(prediction_rows(), bootstrap_samples=20)
    assert len(summary) == 10
    assert set(summary.model_key) == {"base", "finetuned"}
    assert summary["confirmatory"].sum() == 2
    assert set(paired.regime) == {"natural", "shift"}
    assert {"vqa_accuracy", "aurc", "correctness_auroc"}.issubset(set(paired.metric))


def test_confirmatory_threshold_is_fit_then_independently_certified() -> None:
    summary, _ = analyze_predictions(prediction_rows(), bootstrap_samples=5)
    primary = summary[(summary.regime == "natural") & (summary.score == "likelihood")]
    assert primary["certified"].all()
    assert (primary["certification_upper_bound"] <= 0.10).all()
    assert (primary["test_coverage"] > 0).all()


def test_analysis_requires_complete_paired_splits() -> None:
    frame = prediction_rows().query("evaluation_split == 'test'")
    with pytest.raises(ValueError, match="missing"):
        analyze_predictions(frame, bootstrap_samples=5)

    mismatched = prediction_rows()
    mismatched = mismatched.drop(
        mismatched[
            (mismatched.model_key == "base")
            & (mismatched.regime == "natural")
            & (mismatched.evaluation_split == "test")
        ].index[0]
    )
    with pytest.raises(ValueError, match="IDs differ"):
        validate_prediction_frame(mismatched)


def test_prediction_reader_rejects_stale_vqa_scores(tmp_path) -> None:
    row = prediction_rows().iloc[[0]].copy()
    path = tmp_path / "prediction.jsonl"
    row.to_json(path, orient="records", lines=True)
    assert len(read_predictions([path])) == 1
    row.loc[:, "vqa_score"] = 0.5
    row.to_json(path, orient="records", lines=True)
    with pytest.raises(ValueError, match="official recomputation"):
        read_predictions([path])


def test_write_analysis_emits_reproducible_artifacts(tmp_path) -> None:
    write_analysis(
        prediction_rows(),
        tmp_path,
        strict_counts=False,
        bootstrap_samples=10,
    )
    expected = {
        "metrics.csv",
        "paired_differences.csv",
        "subgroup_metrics.csv",
        "failure_audit.csv",
        "paper_results.tex",
        "analysis_metadata.json",
    }
    assert expected.issubset({path.name for path in tmp_path.iterdir()})
    assert (tmp_path / "figures" / "risk_coverage_natural.pdf").exists()
    macros = (tmp_path / "paper_results.tex").read_text(encoding="utf-8")
    assert "\\renewcommand{\\BaseCertErrorsAccepted}" in macros
    assert "\\renewcommand{\\BaseCandidateErrorsAccepted}" in macros
    assert "\\renewcommand{\\BaseShiftCertErrorsAccepted}" in macros
    assert "\\renewcommand{\\BaseVisualAUROC}" in macros
    assert "\\renewcommand{\\PrimaryFinding}" in macros
