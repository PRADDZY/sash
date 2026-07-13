"""Statistically separated fit, certification, and test analysis."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from .data import PILOT_CASES
from .inference import MODEL_SPECS
from .metrics import (
    aurc,
    best_effective_reliability_threshold,
    binary_auroc,
    bootstrap_mean_difference,
    brier_score,
    canonical_prediction,
    certify_selective_threshold,
    coverage_at_risk,
    effective_reliability,
    expected_calibration_error,
    fit_platt_calibrator,
    learn_selective_threshold,
    selective_curve,
    vqa_accuracy,
)

SCORE_COLUMNS = {
    "likelihood": "mean_logprob",
    "visual_delta": "grounding_score",
    "acquisition": "acquisition_score",
}
RISK_LIMITS = (0.01, 0.05, 0.10)
ERROR_COSTS = (1.0, 10.0, 100.0)
PRIMARY_ALPHA = 0.10
PRIMARY_FIT_RISK = PRIMARY_ALPHA / 2
PRIMARY_FAMILY_DELTA = 0.05
PRIMARY_PER_MODEL_DELTA = PRIMARY_FAMILY_DELTA / 2
EXPLORATORY_DELTA = 0.05
BOOTSTRAP_SAMPLES = 2_000
BOOTSTRAP_SEED = 20260712
EXPECTED_COUNTS = {
    ("natural", "fit"): 860,
    ("natural", "certification"): 860,
    ("natural", "test"): 2581,
    ("shift", "fit"): 300,
    ("shift", "certification"): 300,
    ("shift", "test"): 600,
}


def read_predictions(paths: Iterable[Path]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    frame = pd.DataFrame.from_records(records)
    required = {
        "case_id",
        "model_key",
        "model_id",
        "model_revision",
        "evaluation_split",
        "regime",
        "source_case_id",
        "image_path",
        "clear_image_path",
        "answerable",
        "answer_type",
        "corruption",
        "vqa_score",
        "entirely_wrong",
        "false_answer_on_unanswerable",
        "generated_answer",
        "reference_answers",
        "generated_token_ids",
        "normalized_answer",
        "generated_tokens",
        *SCORE_COLUMNS.values(),
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"prediction data missing columns: {sorted(missing)}")
    duplicates = frame.duplicated(["case_id", "model_key"])
    if duplicates.any():
        raise ValueError(f"duplicate predictions: {int(duplicates.sum())}")
    numeric = ["vqa_score", *SCORE_COLUMNS.values()]
    if not np.isfinite(frame[numeric].to_numpy(float)).all():
        raise ValueError("prediction scores must all be finite")
    if not frame["vqa_score"].between(0, 1).all():
        raise ValueError("VQA scores must be in [0, 1]")
    for model_key, (model_id, revision) in MODEL_SPECS.items():
        model = frame[frame["model_key"] == model_key]
        if not model.empty and (
            set(model["model_id"]) != {model_id}
            or set(model["model_revision"]) != {revision}
        ):
            raise ValueError(f"{model_key} predictions do not match the pinned checkpoint")
    source_ids = frame["source_case_id"].where(
        frame["source_case_id"].notna(), frame["case_id"]
    )
    leaked = set(source_ids.astype(str)) & set(PILOT_CASES)
    if leaked:
        raise ValueError(f"reserved pilot sources leaked into predictions: {sorted(leaked)}")
    recomputed = np.array(
        [
            vqa_accuracy(prediction, answers)
            for prediction, answers in zip(
                frame["generated_answer"], frame["reference_answers"], strict=True
            )
        ]
    )
    if not np.allclose(frame["vqa_score"].to_numpy(float), recomputed, atol=1e-12):
        raise ValueError("stored VQA scores do not match official recomputation")
    if not np.array_equal(frame["entirely_wrong"].to_numpy(bool), recomputed == 0):
        raise ValueError("stored entirely-wrong labels do not match VQA scores")
    normalized = frame["generated_answer"].map(canonical_prediction)
    if not normalized.equals(frame["normalized_answer"].astype(str)):
        raise ValueError("stored semantic abstention labels do not match recomputation")
    token_counts = frame["generated_token_ids"].map(len).to_numpy(int)
    if not np.array_equal(frame["generated_tokens"].to_numpy(int), token_counts):
        raise ValueError("stored generated-token counts do not match token IDs")
    return frame


def validate_prediction_frame(frame: pd.DataFrame, *, strict_counts: bool = False) -> None:
    """Verify split completeness and paired model IDs before analysis."""

    if set(frame["model_key"]) != {"base", "finetuned"}:
        raise ValueError("predictions must contain exactly base and finetuned models")
    if set(frame["regime"]) != {"natural", "shift"}:
        raise ValueError("predictions must contain exactly natural and shift regimes")
    if not set(frame["evaluation_split"]).issubset({"fit", "certification", "test"}):
        raise ValueError("unexpected evaluation split")
    for regime in ("natural", "shift"):
        for split in ("fit", "certification", "test"):
            subset = frame[
                (frame["regime"] == regime) & (frame["evaluation_split"] == split)
            ]
            if subset.empty:
                raise ValueError(f"missing {regime}/{split} predictions")
            ids_by_model = {
                model: set(model_frame["case_id"])
                for model, model_frame in subset.groupby("model_key")
            }
            if set(ids_by_model) != {"base", "finetuned"}:
                raise ValueError(f"missing a model for {regime}/{split}")
            if ids_by_model["base"] != ids_by_model["finetuned"]:
                raise ValueError(f"model case IDs differ for {regime}/{split}")
            if strict_counts:
                expected = EXPECTED_COUNTS[(regime, split)]
                for model, case_ids in ids_by_model.items():
                    if len(case_ids) != expected:
                        raise ValueError(
                            f"{model} {regime}/{split}: expected {expected}, found {len(case_ids)}"
                        )


def _split_pipeline(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fit = frame[frame["evaluation_split"] == "fit"]
    certification = frame[frame["evaluation_split"] == "certification"]
    test = frame[frame["evaluation_split"] == "test"]
    if fit.empty or certification.empty or test.empty:
        raise ValueError("every pipeline requires non-empty fit, certification, and test splits")
    return fit, certification, test


def _substantive(frame: pd.DataFrame) -> np.ndarray:
    return (
        (frame["normalized_answer"].astype(str) != "unanswerable")
        & (frame["generated_tokens"].to_numpy(int) > 0)
    ).to_numpy(bool)


def _accepted_metrics(
    test: pd.DataFrame, score_column: str, threshold: float
) -> dict[str, float | int]:
    mask = _substantive(test) & (test[score_column].to_numpy(float) >= threshold)
    accepted = int(mask.sum())
    if not accepted:
        return {
            "test_accepted": 0,
            "test_errors": 0,
            "test_coverage": 0.0,
            "test_entirely_wrong_risk": math.nan,
            "test_soft_risk": math.nan,
            "test_answered_vqa_accuracy": math.nan,
        }
    accuracy = test.loc[mask, "vqa_score"].to_numpy(float)
    return {
        "test_accepted": accepted,
        "test_errors": int((accuracy == 0).sum()),
        "test_coverage": accepted / len(test),
        "test_entirely_wrong_risk": float((accuracy == 0).mean()),
        "test_soft_risk": float((1 - accuracy).mean()),
        "test_answered_vqa_accuracy": float(accuracy.mean()),
    }


def _pipeline_summary(
    model_key: str,
    regime: str,
    score_name: str,
    frame: pd.DataFrame,
) -> dict[str, object]:
    score_column = SCORE_COLUMNS[score_name]
    fit, certification, test = _split_pipeline(frame)
    test_scores = test[score_column].to_numpy(float)
    test_accuracy = test["vqa_score"].to_numpy(float)
    test_correct = test_accuracy > 0
    fit_correct = fit["vqa_score"].to_numpy(float) > 0
    fit_substantive = _substantive(fit)
    certification_substantive = _substantive(certification)
    test_substantive = _substantive(test)
    calibrator = fit_platt_calibrator(fit[score_column], fit_correct)
    probabilities = calibrator.predict(test_scores)

    threshold, fit_risk, fit_accepted = learn_selective_threshold(
        fit[score_column],
        fit["vqa_score"].to_numpy(float) == 0,
        risk_limit=PRIMARY_FIT_RISK,
        eligible=fit_substantive,
    )
    is_primary = regime == "natural" and score_name == "likelihood"
    delta = PRIMARY_PER_MODEL_DELTA if is_primary else EXPLORATORY_DELTA
    certificate = certify_selective_threshold(
        certification[score_column],
        certification["vqa_score"].to_numpy(float) == 0,
        threshold=threshold,
        alpha=PRIMARY_ALPHA,
        delta=delta,
        eligible=certification_substantive,
    )
    deployed_threshold = threshold if certificate.certified else math.inf
    candidate_metrics = {
        f"candidate_{key}": value
        for key, value in _accepted_metrics(test, score_column, threshold).items()
    }

    row: dict[str, object] = {
        "model_key": model_key,
        "regime": regime,
        "score": score_name,
        "confirmatory": is_primary,
        "analysis_status": "confirmatory" if is_primary else "exploratory_marginal",
        "fit_n": len(fit),
        "certification_n": len(certification),
        "test_n": len(test),
        "vqa_accuracy": float(test_accuracy.mean()),
        "entirely_wrong_rate": float((test_accuracy == 0).mean()),
        "substantive_prediction_rate": float(test_substantive.mean()),
        "aurc": aurc(
            test_scores,
            test_correct.astype(float),
            eligible=test_substantive,
        ),
        "correctness_auroc": binary_auroc(test_scores, test_correct),
        "platt_status": calibrator.status,
        "correctness_ece_10bin": expected_calibration_error(
            probabilities, test_correct, bins=10
        ),
        "correctness_brier": brier_score(probabilities, test_correct),
        "fit_threshold": threshold,
        "fit_accepted": fit_accepted,
        "fit_coverage": fit_accepted / len(fit),
        "fit_entirely_wrong_risk": fit_risk,
        "certification_delta": delta,
        "certification_accepted": certificate.accepted,
        "certification_errors": certificate.errors,
        "certification_coverage": certificate.coverage,
        "certification_empirical_risk": certificate.empirical_risk,
        "certification_upper_bound": certificate.upper_bound,
        "certified": certificate.certified,
        "deployed_threshold": deployed_threshold,
        **candidate_metrics,
        **_accepted_metrics(test, score_column, deployed_threshold),
    }
    for risk_limit in RISK_LIMITS:
        point = coverage_at_risk(
            test_scores,
            test_correct.astype(float),
            risk_limit,
            eligible=test_substantive,
        )
        row[f"oracle_test_coverage_at_{risk_limit:.2f}_entirely_wrong_risk"] = (
            point.coverage if point else 0.0
        )
    for cost in ERROR_COSTS:
        phi_threshold, _ = best_effective_reliability_threshold(
            fit[score_column],
            fit["vqa_score"],
            cost,
            eligible=fit_substantive,
        )
        answered = test_substantive & (test_scores >= phi_threshold)
        row[f"phi_{int(cost)}"] = effective_reliability(test_accuracy, answered, cost)
        row[f"phi_{int(cost)}_threshold"] = phi_threshold
        row[f"phi_{int(cost)}_coverage"] = float(answered.mean())

    unanswerable = test[~test["answerable"].astype(bool)]
    row["false_answer_on_unanswerable_rate"] = (
        float(unanswerable["false_answer_on_unanswerable"].astype(bool).mean())
        if len(unanswerable)
        else math.nan
    )
    return row


def _bootstrap_statistic_difference(
    finetuned: pd.DataFrame,
    base: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    if list(finetuned["case_id"]) != list(base["case_id"]):
        raise ValueError("paired bootstrap frames are not aligned")
    point = statistic(finetuned) - statistic(base)
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(samples):
        indices = rng.integers(0, len(base), size=len(base))
        estimate = statistic(finetuned.iloc[indices]) - statistic(base.iloc[indices])
        if math.isfinite(estimate):
            estimates.append(estimate)
    if not estimates:
        return point, math.nan, math.nan
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return point, float(lower), float(upper)


def _aligned_models(test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = test[test["model_key"] == "base"].sort_values("case_id").reset_index(drop=True)
    finetuned = (
        test[test["model_key"] == "finetuned"].sort_values("case_id").reset_index(drop=True)
    )
    if list(base["case_id"]) != list(finetuned["case_id"]):
        raise ValueError("paired test IDs do not match")
    return finetuned, base


def _paired_rows(
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    bootstrap_samples: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seed_offset = 0
    for regime in ("natural", "shift"):
        test = frame[
            (frame["regime"] == regime) & (frame["evaluation_split"] == "test")
        ]
        finetuned, base = _aligned_models(test)
        for metric, first, second in (
            ("vqa_accuracy", finetuned["vqa_score"], base["vqa_score"]),
            (
                "entirely_wrong_rate",
                (finetuned["vqa_score"] == 0).astype(float),
                (base["vqa_score"] == 0).astype(float),
            ),
            (
                "substantive_prediction_rate",
                _substantive(finetuned).astype(float),
                _substantive(base).astype(float),
            ),
        ):
            difference, lower, upper = bootstrap_mean_difference(
                first,
                second,
                samples=bootstrap_samples,
                seed=BOOTSTRAP_SEED + seed_offset,
            )
            seed_offset += 1
            rows.append(
                {
                    "regime": regime,
                    "score": "all",
                    "metric": metric,
                    "n": len(base),
                    "finetuned_minus_base": difference,
                    "ci_2.5%": lower,
                    "ci_97.5%": upper,
                }
            )

        base_wrong = base["vqa_score"].to_numpy(float) == 0
        finetuned_wrong = finetuned["vqa_score"].to_numpy(float) == 0
        discordant_improved = int((base_wrong & ~finetuned_wrong).sum())
        discordant_worsened = int((~base_wrong & finetuned_wrong).sum())
        discordant = discordant_improved + discordant_worsened
        mcnemar_p = (
            float(
                binomtest(
                    min(discordant_improved, discordant_worsened),
                    discordant,
                    0.5,
                    alternative="two-sided",
                ).pvalue
            )
            if discordant
            else 1.0
        )
        rows.append(
            {
                "regime": regime,
                "score": "all",
                "metric": "mcnemar_entirely_wrong",
                "n": len(base),
                "finetuned_minus_base": (
                    float(finetuned_wrong.mean() - base_wrong.mean())
                ),
                "ci_2.5%": math.nan,
                "ci_97.5%": math.nan,
                "discordant_improved": discordant_improved,
                "discordant_worsened": discordant_worsened,
                "p_value": mcnemar_p,
            }
        )

        score_names = ("likelihood", "visual_delta")
        if regime == "shift":
            score_names = (*score_names, "acquisition")
        for score_name in score_names:
            column = SCORE_COLUMNS[score_name]
            statistics: Sequence[tuple[str, Callable[[pd.DataFrame], float]]] = (
                (
                    "aurc",
                    lambda sample, score_column=column: aurc(
                        sample[score_column],
                        (sample["vqa_score"].to_numpy(float) > 0).astype(float),
                        eligible=_substantive(sample),
                    ),
                ),
                (
                    "correctness_auroc",
                    lambda sample, score_column=column: binary_auroc(
                        sample[score_column], sample["vqa_score"].to_numpy(float) > 0
                    ),
                ),
            )
            for metric, statistic in statistics:
                difference, lower, upper = _bootstrap_statistic_difference(
                    finetuned,
                    base,
                    statistic,
                    samples=bootstrap_samples,
                    seed=BOOTSTRAP_SEED + seed_offset,
                )
                seed_offset += 1
                rows.append(
                    {
                        "regime": regime,
                        "score": score_name,
                        "metric": metric,
                        "n": len(base),
                        "finetuned_minus_base": difference,
                        "ci_2.5%": lower,
                        "ci_97.5%": upper,
                    }
                )
            if regime == "natural" and score_name == "likelihood":
                operating = summary[
                    (summary["regime"] == regime) & (summary["score"] == score_name)
                ].set_index("model_key")
                finetuned_answered = _substantive(finetuned) & (
                    finetuned[column].to_numpy(float)
                    >= float(operating.loc["finetuned", "deployed_threshold"])
                )
                base_answered = _substantive(base) & (
                    base[column].to_numpy(float)
                    >= float(operating.loc["base", "deployed_threshold"])
                )
                difference, lower, upper = bootstrap_mean_difference(
                    finetuned_answered.astype(float),
                    base_answered.astype(float),
                    samples=bootstrap_samples,
                    seed=BOOTSTRAP_SEED + seed_offset,
                )
                seed_offset += 1
                rows.append(
                    {
                        "regime": regime,
                        "score": score_name,
                        "metric": "deployed_test_coverage",
                        "n": len(base),
                        "finetuned_minus_base": difference,
                        "ci_2.5%": lower,
                        "ci_97.5%": upper,
                    }
                )
    return rows


def analyze_predictions(
    frame: pd.DataFrame, *, bootstrap_samples: int = BOOTSTRAP_SAMPLES
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-pipeline summaries and paired test differences."""

    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    validate_prediction_frame(frame)
    summary_rows: list[dict[str, object]] = []
    for model_key in ("base", "finetuned"):
        for regime in ("natural", "shift"):
            subset = frame[
                (frame["model_key"] == model_key) & (frame["regime"] == regime)
            ]
            score_names = ("likelihood", "visual_delta")
            if regime == "shift":
                score_names = (*score_names, "acquisition")
            for score_name in score_names:
                summary_rows.append(
                    _pipeline_summary(model_key, regime, score_name, subset)
                )
    summary = pd.DataFrame(summary_rows)
    paired = pd.DataFrame(
        _paired_rows(frame, summary, bootstrap_samples=bootstrap_samples)
    )
    paired["analysis_status"] = "secondary_paired_comparison"
    return summary, paired


def subgroup_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute predeclared test slices without threshold re-fitting."""

    test = frame[frame["evaluation_split"] == "test"]
    rows: list[dict[str, object]] = []
    for model_key in ("base", "finetuned"):
        model = test[test["model_key"] == model_key]
        slices: list[tuple[str, str, pd.DataFrame]] = []
        natural = model[model["regime"] == "natural"]
        answerability = natural["answerable"].map(
            {True: "answerable", False: "unanswerable"}
        )
        for value, subset in natural.groupby(answerability):
            slices.append(("answerability", str(value), subset))
        for value, subset in natural.groupby("answer_type"):
            slices.append(("answer_type", str(value), subset))
        shifted = model[model["regime"] == "shift"]
        for value, subset in shifted.groupby("corruption"):
            slices.append(("corruption", str(value), subset))
        for dimension, value, subset in slices:
            rows.append(
                {
                    "model_key": model_key,
                    "analysis_status": "exploratory_subgroup",
                    "dimension": dimension,
                    "value": value,
                    "n": len(subset),
                    "vqa_accuracy": float(subset["vqa_score"].mean()),
                    "entirely_wrong_rate": float((subset["vqa_score"] == 0).mean()),
                    "semantic_abstention_rate": float((~_substantive(subset)).mean()),
                    "false_answer_on_unanswerable_rate": (
                        float(subset["false_answer_on_unanswerable"].astype(bool).mean())
                        if not subset["answerable"].astype(bool).any()
                        else math.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def write_risk_coverage_figures(
    frame: pd.DataFrame, summary: pd.DataFrame, output_dir: Path
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for regime in ("natural", "shift"):
        test = frame[
            (frame["regime"] == regime) & (frame["evaluation_split"] == "test")
        ]
        fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
        for model_key, color in (("base", "#526D82"), ("finetuned", "#D35400")):
            model = test[test["model_key"] == model_key]
            points = selective_curve(
                model["mean_logprob"],
                (model["vqa_score"].to_numpy(float) > 0).astype(float),
                eligible=_substantive(model),
            )
            ax.plot(
                [point.coverage for point in points],
                [point.risk for point in points],
                color=color,
                label=model_key,
            )
            operating = summary[
                (summary["model_key"] == model_key)
                & (summary["regime"] == regime)
                & (summary["score"] == "likelihood")
            ].iloc[0]
            if bool(operating["certified"]) and math.isfinite(
                float(operating["test_entirely_wrong_risk"])
            ):
                ax.scatter(
                    [operating["test_coverage"]],
                    [operating["test_entirely_wrong_risk"]],
                    color=color,
                    edgecolor="black",
                    zorder=3,
                )
        ax.axhline(PRIMARY_ALPHA, color="black", linestyle="--", linewidth=1, alpha=0.7)
        ax.set(
            xlabel="Coverage",
            ylabel="Entirely-wrong risk among substantive answers",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        path = output_dir / f"risk_coverage_{regime}.pdf"
        fig.savefig(path)
        plt.close(fig)
        written.append(path)
    return written


def write_reliability_figures(frame: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    edges = np.linspace(0, 1, 11)
    for regime in ("natural", "shift"):
        fig, ax = plt.subplots(figsize=(5.0, 4.2), constrained_layout=True)
        for model_key, color in (("base", "#526D82"), ("finetuned", "#D35400")):
            pipeline = frame[
                (frame["regime"] == regime) & (frame["model_key"] == model_key)
            ]
            fit, _, test = _split_pipeline(pipeline)
            calibrator = fit_platt_calibrator(
                fit["mean_logprob"], fit["vqa_score"].to_numpy(float) > 0
            )
            probabilities = calibrator.predict(test["mean_logprob"])
            labels = test["vqa_score"].to_numpy(float) > 0
            bin_probabilities: list[float] = []
            bin_frequencies: list[float] = []
            for index in range(10):
                mask = (probabilities >= edges[index]) & (
                    (probabilities <= edges[index + 1])
                    if index == 9
                    else (probabilities < edges[index + 1])
                )
                if mask.any():
                    bin_probabilities.append(float(probabilities[mask].mean()))
                    bin_frequencies.append(float(labels[mask].mean()))
            ax.plot(
                bin_probabilities,
                bin_frequencies,
                marker="o",
                color=color,
                label=model_key,
            )
        ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
        ax.set(
            xlabel="Platt probability of any annotator match",
            ylabel="Observed frequency",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        path = output_dir / f"reliability_{regime}.pdf"
        fig.savefig(path)
        plt.close(fig)
        written.append(path)
    return written


def _write_result_macros(summary: pd.DataFrame, paired: pd.DataFrame, path: Path) -> None:
    primary = summary[(summary["regime"] == "natural") & (summary["score"] == "likelihood")]
    values = {row.model_key: row for row in primary.itertuples(index=False)}
    shift_values = {
        row.model_key: row
        for row in summary[
            (summary["regime"] == "shift") & (summary["score"] == "likelihood")
        ].itertuples(index=False)
    }
    visual_values = {
        row.model_key: row
        for row in summary[
            (summary["regime"] == "natural") & (summary["score"] == "visual_delta")
        ].itertuples(index=False)
    }
    accuracy_delta = paired[
        (paired["regime"] == "natural") & (paired["metric"] == "vqa_accuracy")
    ].iloc[0]

    def number(value: float, digits: int = 3) -> str:
        return "--" if not math.isfinite(float(value)) else f"{float(value):.{digits}f}"

    def percent(value: float, digits: int = 1) -> str:
        return (
            "--"
            if not math.isfinite(float(value))
            else f"{100 * float(value):.{digits}f}\\%"
        )

    def threshold(value: float) -> str:
        return "\\ensuremath{\\infty}" if math.isinf(float(value)) else number(value)

    base = values["base"]
    tuned = values["finetuned"]
    base_shift = shift_values["base"]
    tuned_shift = shift_values["finetuned"]
    base_visual = visual_values["base"]
    tuned_visual = visual_values["finetuned"]
    coverage_delta = paired[
        (paired["regime"] == "natural")
        & (paired["score"] == "likelihood")
        & (paired["metric"] == "deployed_test_coverage")
    ].iloc[0]
    wrong_delta = paired[
        (paired["regime"] == "natural") & (paired["metric"] == "entirely_wrong_rate")
    ].iloc[0]
    substantive_delta = paired[
        (paired["regime"] == "natural")
        & (paired["metric"] == "substantive_prediction_rate")
    ].iloc[0]
    shift_accuracy_delta = paired[
        (paired["regime"] == "shift") & (paired["metric"] == "vqa_accuracy")
    ].iloc[0]
    shift_wrong_delta = paired[
        (paired["regime"] == "shift") & (paired["metric"] == "entirely_wrong_rate")
    ].iloc[0]
    finding = (
        f"The base policy's conditional check "
        f"{'met' if base.certified else 'did not meet'} the numerical CP gate and would "
        f"answer at {percent(base.test_coverage)} test coverage; the tuned policy's "
        f"conditional check {'met' if tuned.certified else 'did not meet'} the gate and "
        f"would answer at {percent(tuned.test_coverage)} coverage."
    )

    lines = [
        "% Generated by sash_audit.analysis; do not edit.",
        f"\\renewcommand{{\\FitN}}{{{int(base.fit_n)}}}",
        f"\\renewcommand{{\\CertN}}{{{int(base.certification_n)}}}",
        f"\\renewcommand{{\\TestN}}{{{int(base.test_n)}}}",
        f"\\renewcommand{{\\BaseVQA}}{{{percent(base.vqa_accuracy)}}}",
        f"\\renewcommand{{\\TunedVQA}}{{{percent(tuned.vqa_accuracy)}}}",
        f"\\renewcommand{{\\BaseWrongRate}}{{{percent(base.entirely_wrong_rate)}}}",
        f"\\renewcommand{{\\TunedWrongRate}}{{{percent(tuned.entirely_wrong_rate)}}}",
        f"\\renewcommand{{\\BaseThreshold}}{{{threshold(base.fit_threshold)}}}",
        f"\\renewcommand{{\\TunedThreshold}}{{{threshold(tuned.fit_threshold)}}}",
        f"\\renewcommand{{\\BaseCertErrorsAccepted}}{{{int(base.certification_errors)}/{int(base.certification_accepted)}}}",
        f"\\renewcommand{{\\TunedCertErrorsAccepted}}{{{int(tuned.certification_errors)}/{int(tuned.certification_accepted)}}}",
        f"\\renewcommand{{\\BaseCertCoverage}}{{{percent(base.certification_coverage)}}}",
        f"\\renewcommand{{\\TunedCertCoverage}}{{{percent(tuned.certification_coverage)}}}",
        f"\\renewcommand{{\\BaseCertUpper}}{{{percent(base.certification_upper_bound, 2)}}}",
        f"\\renewcommand{{\\TunedCertUpper}}{{{percent(tuned.certification_upper_bound, 2)}}}",
        f"\\renewcommand{{\\BaseCertDecision}}{{{'Pass*' if base.certified else 'Fail'}}}",
        f"\\renewcommand{{\\TunedCertDecision}}{{{'Pass*' if tuned.certified else 'Fail'}}}",
        f"\\renewcommand{{\\BaseTestCoverage}}{{{percent(base.test_coverage)}}}",
        f"\\renewcommand{{\\TunedTestCoverage}}{{{percent(tuned.test_coverage)}}}",
        f"\\renewcommand{{\\BaseTestErrorsAccepted}}{{{int(base.test_errors)}/{int(base.test_accepted)}}}",
        f"\\renewcommand{{\\TunedTestErrorsAccepted}}{{{int(tuned.test_errors)}/{int(tuned.test_accepted)}}}",
        f"\\renewcommand{{\\BaseTestRisk}}{{{percent(base.test_entirely_wrong_risk)}}}",
        f"\\renewcommand{{\\TunedTestRisk}}{{{percent(tuned.test_entirely_wrong_risk)}}}",
        f"\\renewcommand{{\\BaseCandidateErrorsAccepted}}{{{int(base.candidate_test_errors)}/{int(base.candidate_test_accepted)}}}",
        f"\\renewcommand{{\\TunedCandidateErrorsAccepted}}{{{int(tuned.candidate_test_errors)}/{int(tuned.candidate_test_accepted)}}}",
        f"\\renewcommand{{\\BaseCandidateCoverage}}{{{percent(base.candidate_test_coverage)}}}",
        f"\\renewcommand{{\\TunedCandidateCoverage}}{{{percent(tuned.candidate_test_coverage)}}}",
        f"\\renewcommand{{\\BaseCandidateRisk}}{{{percent(base.candidate_test_entirely_wrong_risk)}}}",
        f"\\renewcommand{{\\TunedCandidateRisk}}{{{percent(tuned.candidate_test_entirely_wrong_risk)}}}",
        f"\\renewcommand{{\\BaseECE}}{{{number(base.correctness_ece_10bin)}}}",
        f"\\renewcommand{{\\TunedECE}}{{{number(tuned.correctness_ece_10bin)}}}",
        f"\\renewcommand{{\\BaseBrier}}{{{number(base.correctness_brier)}}}",
        f"\\renewcommand{{\\TunedBrier}}{{{number(tuned.correctness_brier)}}}",
        f"\\renewcommand{{\\VQADelta}}{{{percent(accuracy_delta['finetuned_minus_base'])}}}",
        f"\\renewcommand{{\\VQADeltaLow}}{{{percent(accuracy_delta['ci_2.5%'])}}}",
        f"\\renewcommand{{\\VQADeltaHigh}}{{{percent(accuracy_delta['ci_97.5%'])}}}",
        f"\\renewcommand{{\\WrongDelta}}{{{percent(wrong_delta['finetuned_minus_base'])}}}",
        f"\\renewcommand{{\\WrongDeltaLow}}{{{percent(wrong_delta['ci_2.5%'])}}}",
        f"\\renewcommand{{\\WrongDeltaHigh}}{{{percent(wrong_delta['ci_97.5%'])}}}",
        f"\\renewcommand{{\\SubstantiveDelta}}{{{percent(substantive_delta['finetuned_minus_base'])}}}",
        f"\\renewcommand{{\\SubstantiveDeltaLow}}{{{percent(substantive_delta['ci_2.5%'])}}}",
        f"\\renewcommand{{\\SubstantiveDeltaHigh}}{{{percent(substantive_delta['ci_97.5%'])}}}",
        f"\\renewcommand{{\\CoverageDelta}}{{{percent(coverage_delta['finetuned_minus_base'])}}}",
        f"\\renewcommand{{\\CoverageDeltaLow}}{{{percent(coverage_delta['ci_2.5%'])}}}",
        f"\\renewcommand{{\\CoverageDeltaHigh}}{{{percent(coverage_delta['ci_97.5%'])}}}",
        f"\\renewcommand{{\\BaseAURC}}{{{number(base.aurc)}}}",
        f"\\renewcommand{{\\TunedAURC}}{{{number(tuned.aurc)}}}",
        f"\\renewcommand{{\\BaseAUROC}}{{{number(base.correctness_auroc)}}}",
        f"\\renewcommand{{\\TunedAUROC}}{{{number(tuned.correctness_auroc)}}}",
        f"\\renewcommand{{\\BaseVisualAURC}}{{{number(base_visual.aurc)}}}",
        f"\\renewcommand{{\\TunedVisualAURC}}{{{number(tuned_visual.aurc)}}}",
        f"\\renewcommand{{\\BaseVisualAUROC}}{{{number(base_visual.correctness_auroc)}}}",
        f"\\renewcommand{{\\TunedVisualAUROC}}{{{number(tuned_visual.correctness_auroc)}}}",
        f"\\renewcommand{{\\BaseSubstantiveRate}}{{{percent(base.substantive_prediction_rate)}}}",
        f"\\renewcommand{{\\TunedSubstantiveRate}}{{{percent(tuned.substantive_prediction_rate)}}}",
        f"\\renewcommand{{\\ShiftN}}{{{int(base_shift.test_n)}}}",
        f"\\renewcommand{{\\BaseShiftVQA}}{{{percent(base_shift.vqa_accuracy)}}}",
        f"\\renewcommand{{\\TunedShiftVQA}}{{{percent(tuned_shift.vqa_accuracy)}}}",
        f"\\renewcommand{{\\BaseShiftWrongRate}}{{{percent(base_shift.entirely_wrong_rate)}}}",
        f"\\renewcommand{{\\TunedShiftWrongRate}}{{{percent(tuned_shift.entirely_wrong_rate)}}}",
        f"\\renewcommand{{\\BaseShiftSubstantiveRate}}{{{percent(base_shift.substantive_prediction_rate)}}}",
        f"\\renewcommand{{\\TunedShiftSubstantiveRate}}{{{percent(tuned_shift.substantive_prediction_rate)}}}",
        f"\\renewcommand{{\\ShiftVQADelta}}{{{percent(shift_accuracy_delta['finetuned_minus_base'])}}}",
        f"\\renewcommand{{\\ShiftVQADeltaLow}}{{{percent(shift_accuracy_delta['ci_2.5%'])}}}",
        f"\\renewcommand{{\\ShiftVQADeltaHigh}}{{{percent(shift_accuracy_delta['ci_97.5%'])}}}",
        f"\\renewcommand{{\\ShiftWrongDelta}}{{{percent(shift_wrong_delta['finetuned_minus_base'])}}}",
        f"\\renewcommand{{\\ShiftWrongDeltaLow}}{{{percent(shift_wrong_delta['ci_2.5%'])}}}",
        f"\\renewcommand{{\\ShiftWrongDeltaHigh}}{{{percent(shift_wrong_delta['ci_97.5%'])}}}",
        f"\\renewcommand{{\\BaseShiftCertErrorsAccepted}}{{{int(base_shift.certification_errors)}/{int(base_shift.certification_accepted)}}}",
        f"\\renewcommand{{\\TunedShiftCertErrorsAccepted}}{{{int(tuned_shift.certification_errors)}/{int(tuned_shift.certification_accepted)}}}",
        (
            "\\renewcommand{\\BaseShiftCertUpper}"
            f"{{{percent(base_shift.certification_upper_bound, 2)}}}"
        ),
        (
            "\\renewcommand{\\TunedShiftCertUpper}"
            f"{{{percent(tuned_shift.certification_upper_bound, 2)}}}"
        ),
        f"\\renewcommand{{\\BaseShiftDecision}}{{{'Pass*' if base_shift.certified else 'Fail'}}}",
        f"\\renewcommand{{\\TunedShiftDecision}}{{{'Pass*' if tuned_shift.certified else 'Fail'}}}",
        f"\\renewcommand{{\\BaseShiftTestErrorsAccepted}}{{{int(base_shift.test_errors)}/{int(base_shift.test_accepted)}}}",
        f"\\renewcommand{{\\TunedShiftTestErrorsAccepted}}{{{int(tuned_shift.test_errors)}/{int(tuned_shift.test_accepted)}}}",
        f"\\renewcommand{{\\BaseShiftTestCoverage}}{{{percent(base_shift.test_coverage)}}}",
        f"\\renewcommand{{\\TunedShiftTestCoverage}}{{{percent(tuned_shift.test_coverage)}}}",
        f"\\renewcommand{{\\BaseShiftTestRisk}}{{{percent(base_shift.test_entirely_wrong_risk)}}}",
        f"\\renewcommand{{\\TunedShiftTestRisk}}{{{percent(tuned_shift.test_entirely_wrong_risk)}}}",
        f"\\renewcommand{{\\TunedShiftCandidateErrorsAccepted}}{{{int(tuned_shift.candidate_test_errors)}/{int(tuned_shift.candidate_test_accepted)}}}",
        f"\\renewcommand{{\\PrimaryFinding}}{{{finding}}}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_analysis(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    strict_counts: bool = True,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validate_prediction_frame(frame, strict_counts=strict_counts)
    summary, paired = analyze_predictions(frame, bootstrap_samples=bootstrap_samples)
    subgroups = subgroup_metrics(frame)
    summary.to_csv(output_dir / "metrics.csv", index=False)
    paired.to_csv(output_dir / "paired_differences.csv", index=False)
    subgroups.to_csv(output_dir / "subgroup_metrics.csv", index=False)
    write_risk_coverage_figures(frame, summary, output_dir / "figures")
    write_reliability_figures(frame, output_dir / "figures")
    _write_result_macros(summary, paired, output_dir / "paper_results.tex")

    failures = frame[
        (frame["evaluation_split"] == "test")
        & (frame["vqa_score"] == 0)
        & _substantive(frame)
    ].copy()
    audit = failures.sort_values(
        ["model_key", "regime", "mean_logprob", "case_id"],
        ascending=[True, True, False, True],
    ).groupby(["model_key", "regime"], as_index=False).head(15)
    columns = [
        "case_id",
        "row_index",
        "model_key",
        "regime",
        "corruption",
        "image_path",
        "clear_image_path",
        "question",
        "reference_answers",
        "generated_answer",
        "vqa_score",
        "mean_logprob",
        "grounding_score",
    ]
    audit = audit[columns].copy()
    audit["failure_category"] = ""
    audit["recommended_action"] = ""
    audit["candidate_gate_behavior"] = ""
    audit["privacy_screening"] = ""
    audit["review_notes"] = ""
    audit.to_csv(output_dir / "failure_audit.csv", index=False)

    metadata = {
        "primary_error": "entirely incorrect under official VizWiz scoring (VQA score == 0)",
        "alpha": PRIMARY_ALPHA,
        "family_delta": PRIMARY_FAMILY_DELTA,
        "per_model_delta": PRIMARY_PER_MODEL_DELTA,
        "fit_risk_limit": PRIMARY_FIT_RISK,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "confirmatory_family": "natural-regime mean answer log-likelihood, base and finetuned",
        "secondary_status": "exploratory marginal per-pipeline",
    }
    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
