from __future__ import annotations

import math

import numpy as np
import pytest

from sash_audit.metrics import (
    aurc,
    binary_auroc,
    bootstrap_mean_difference,
    brier_score,
    canonical_prediction,
    certify_selective_threshold,
    clopper_pearson_upper,
    coverage_at_risk,
    effective_reliability,
    expected_calibration_error,
    fit_platt_calibrator,
    learn_selective_threshold,
    normalize_answer,
    selective_curve,
    vqa_accuracy,
)


def references(answer: str, count: int) -> list[str]:
    return [answer] * count + ["different"] * (10 - count)


@pytest.mark.parametrize(
    ("count", "expected"),
    [(0, 0.0), (1, 0.3), (2, 0.6), (3, 0.9), (4, 1.0), (10, 1.0)],
)
def test_vqa_consensus_matches_official_leave_one_out_values(count: int, expected: float) -> None:
    assert vqa_accuracy("cat", references("cat", count)) == pytest.approx(expected)


def test_official_normalization_and_abstention_mapping() -> None:
    assert normalize_answer("The two, cats!") == "2 cats"
    # The official evaluator removes every punctuation mark when a numeric comma is present.
    assert normalize_answer("1,000 well-known items") == "1000 wellknown items"
    assert normalize_answer("Im sure") == "i'm sure"
    assert canonical_prediction("I cannot determine that from the image.") == "unanswerable"
    assert canonical_prediction("I cannot see, maybe it says 7") != "unanswerable"
    assert vqa_accuracy("I cannot see", ["unanswerable"] * 10) == 0.0


def test_selective_curve_preserves_ties_and_uses_step_aurc() -> None:
    points = selective_curve([1.0, 1.0, 0.0], [1.0, 0.0, 1.0])
    assert [point.answered for point in points] == [2, 3]
    assert aurc([1.0, 1.0, 0.0], [1.0, 0.0, 1.0]) == pytest.approx(4 / 9)
    assert coverage_at_risk([0.9, 0.8, 0.7], [1.0, 1.0, 0.0], 0.1).coverage == pytest.approx(
        2 / 3
    )
    assert math.isnan(aurc([], []))


def test_selective_policy_excludes_semantic_abstentions() -> None:
    eligible = [True, False, True]
    points = selective_curve([0.9, 1.0, 0.8], [1.0, 0.0, 0.0], eligible=eligible)
    assert [point.answered for point in points] == [1, 2]
    assert [point.coverage for point in points] == pytest.approx([1 / 3, 2 / 3])
    threshold, risk, accepted = learn_selective_threshold(
        [0.9, 1.0, 0.8], [False, True, True], risk_limit=0.0, eligible=eligible
    )
    assert (threshold, risk, accepted) == (0.9, 0.0, 1)


def test_auroc_platt_calibration_ece_and_brier() -> None:
    scores = [0.0, 1.0, 2.0, 3.0]
    labels = [0, 0, 1, 1]
    assert binary_auroc(scores, labels) == 1.0
    calibrator = fit_platt_calibrator(scores, labels)
    probabilities = calibrator.predict(scores)
    assert calibrator.status == "fitted"
    assert np.all(np.diff(probabilities) > 0)
    assert 0 <= expected_calibration_error(probabilities, labels, bins=2) <= 1
    assert 0 <= brier_score(probabilities, labels) <= 1
    fallback = fit_platt_calibrator([1, 2, 3], [1, 1, 1])
    assert fallback.status == "constant_fallback"
    assert np.allclose(fallback.predict([1, 100]), fallback.predict([2, -5]))
    assert math.isnan(binary_auroc([1, 2], [1, 1]))


def test_effective_reliability_penalizes_only_entirely_wrong_answers() -> None:
    value = effective_reliability([1.0, 0.0, 0.5], [True, True, False], error_cost=10)
    assert value == pytest.approx(-3.0)
    partial = effective_reliability([0.3], [True], error_cost=100)
    assert partial == pytest.approx(0.3)


def test_fit_then_certify_uses_one_frozen_threshold() -> None:
    threshold, risk, accepted = learn_selective_threshold(
        [4, 3, 2, 1], [False, False, True, True], risk_limit=0.10
    )
    assert (threshold, risk, accepted) == (3.0, 0.0, 2)
    certificate = certify_selective_threshold(
        list(range(29)),
        [False] * 29,
        threshold=0,
        alpha=0.10,
        delta=0.05,
    )
    assert certificate.certified
    assert certificate.upper_bound <= 0.10


def test_clopper_pearson_edges_and_monte_carlo_false_certification() -> None:
    assert clopper_pearson_upper(0, 0) == 1.0
    assert clopper_pearson_upper(5, 5) == 1.0
    assert clopper_pearson_upper(0, 28, delta=0.05) > 0.10
    assert clopper_pearson_upper(0, 29, delta=0.05) <= 0.10

    rng = np.random.default_rng(7)
    false_certifications = 0
    trials = 2_000
    for errors in rng.binomial(200, 0.12, size=trials):
        false_certifications += clopper_pearson_upper(int(errors), 200, 0.05) <= 0.10
    assert false_certifications / trials <= 0.06


def test_paired_bootstrap_is_deterministic() -> None:
    first = np.array([1.0, 2.0, 3.0])
    second = np.array([0.0, 1.0, 2.0])
    assert bootstrap_mean_difference(first, second, samples=100) == (1.0, 1.0, 1.0)


def test_invalid_metric_inputs_fail_loudly() -> None:
    with pytest.raises(ValueError):
        vqa_accuracy("x", [])
    with pytest.raises(ValueError):
        selective_curve([1.0], [1.0, 0.0])
    with pytest.raises(ValueError):
        clopper_pearson_upper(2, 1)
    with pytest.raises(ValueError):
        fit_platt_calibrator([math.inf], [1])
