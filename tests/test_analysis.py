from __future__ import annotations

import pandas as pd
import pytest

from sash_audit.analysis import analyze_predictions, validate_prediction_frame, write_analysis


def prediction_rows() -> pd.DataFrame:
    rows = []
    for model_key, offset in (("base", 0.0), ("finetuned", 0.05)):
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
                            "evaluation_split": split,
                            "regime": regime,
                            "answerable": index % 4 != 0,
                            "answer_type": "unanswerable" if index % 4 == 0 else "other",
                            "corruption": "blur" if regime == "shift" else "none",
                            "question": "What is shown?",
                            "reference_answers": ["object"] * 10,
                            "generated_answer": "object" if accuracy else "wrong",
                            "image_path": f"/vol/{case_id}.jpg",
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
