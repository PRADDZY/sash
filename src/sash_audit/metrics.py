"""Metrics and calibration for the selective-VQA audit.

Questions are the statistical unit. Higher confidence scores always mean that
an answer is more likely to have at least one annotator match.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import beta, rankdata

# Copied from the official VQA evaluator, then lower-cased because this module
# lower-cases text before applying the map.
_CONTRACTIONS = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hed've": "he'd've",
    "he'dve": "he'd've",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "id've": "i'd've",
    "i'dve": "i'd've",
    "im": "i'm",
    "ive": "i've",
    "isnt": "isn't",
    "itd": "it'd",
    "itd've": "it'd've",
    "it'dve": "it'd've",
    "itll": "it'll",
    "let's": "let's",
    "maam": "ma'am",
    "mightnt": "mightn't",
    "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "notve": "not've",
    "oclock": "o'clock",
    "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at",
    "shant": "shan't",
    "shed've": "she'd've",
    "she'dve": "she'd've",
    "she's": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll",
    "somebodys": "somebody's",
    "someoned": "someone'd",
    "someoned've": "someone'd've",
    "someone'dve": "someone'd've",
    "someonell": "someone'll",
    "someones": "someone's",
    "somethingd": "something'd",
    "somethingd've": "something'd've",
    "something'dve": "something'd've",
    "somethingll": "something'll",
    "thats": "that's",
    "thered": "there'd",
    "thered've": "there'd've",
    "there'dve": "there'd've",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyd've": "they'd've",
    "they'dve": "they'd've",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "twas": "'twas",
    "wasnt": "wasn't",
    "wed've": "we'd've",
    "we'dve": "we'd've",
    "weve": "we've",
    "werent": "weren't",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whod've": "who'd've",
    "who'dve": "who'd've",
    "wholl": "who'll",
    "whos": "who's",
    "whove": "who've",
    "whyll": "why'll",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've",
    "yall": "y'all",
    "yall'll": "y'all'll",
    "y'allll": "y'all'll",
    "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've",
    "youd": "you'd",
    "youd've": "you'd've",
    "you'dve": "you'd've",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}
_NUMBER_WORDS = {
    "none": "0",
    **{
        word: str(index)
        for index, word in enumerate(
            ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")
        )
    },
}
_ARTICLES = {"a", "an", "the"}
_PUNCTUATION = (
    ";", "/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-",
    ">", "<", "@", ",", "?", "!", chr(96),
)
_PERIOD = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA = re.compile(r"(\d)(,)(\d)")
_SPACE = re.compile(r"\s+")
_ABSTENTION_PATTERNS = (
    re.compile(r"^unanswerable$"),
    re.compile(
        r"^(?:i\s+)?(?:can(?:not|'t)|(?:am\s+)?unable to)\s+"
        r"(?:answer|tell|determine|see)(?:\s+(?:that|it|this|anything|answer))?"
        r"(?:\s+from\s+(?:this\s+)?image)?$"
    ),
    re.compile(r"^image\s+is\s+(?:too\s+)?(?:blurry|unclear|dark|illegible)$"),
    re.compile(r"^(?:not|insufficiently)\s+(?:visible|clear)$"),
)


def normalize_answer(value: object) -> str:
    """Apply the official VQA punctuation, digit, article, and contraction rules."""

    text = str(value or "").replace("\n", " ").replace("\t", " ").strip().lower()
    original = text
    for punctuation in _PUNCTUATION:
        has_adjacent_space = punctuation + " " in original or " " + punctuation in original
        if has_adjacent_space or _COMMA.search(original):
            text = text.replace(punctuation, "")
        else:
            text = text.replace(punctuation, " ")
    text = _PERIOD.sub("", text)
    words: list[str] = []
    for word in _SPACE.sub(" ", text).strip().split(" "):
        if not word or word in _ARTICLES:
            continue
        word = _NUMBER_WORDS.get(word, word)
        words.append(_CONTRACTIONS.get(word, word))
    return " ".join(words)


def canonical_prediction(value: object) -> str:
    """Map explicit natural-language abstentions to VizWiz's canonical label."""

    normalized = normalize_answer(value)
    if any(pattern.search(normalized) for pattern in _ABSTENTION_PATTERNS):
        return "unanswerable"
    return normalized


def _answer_text(answer: object) -> str:
    if isinstance(answer, dict):
        return str(answer.get("answer", ""))
    return str(answer)


def vqa_accuracy(prediction: object, answers: Sequence[object]) -> float:
    """Return official leave-one-annotator-out VQA consensus accuracy."""

    if not answers:
        raise ValueError("VQA scoring requires at least one reference answer")
    pred = normalize_answer(prediction)
    refs = [normalize_answer(_answer_text(answer)) for answer in answers]
    per_reference = [
        min(1.0, sum(pred == other for j, other in enumerate(refs) if j != index) / 3.0)
        for index in range(len(refs))
    ]
    return float(np.mean(per_reference))


def is_false_answer_on_unanswerable(prediction: object, answerable: bool | int) -> bool:
    return not bool(answerable) and canonical_prediction(prediction) != "unanswerable"


@dataclass(frozen=True)
class SelectivePoint:
    threshold: float
    coverage: float
    risk: float
    answered: int


def _metric_arrays(
    scores: Sequence[float], outcomes: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    score_array = np.asarray(scores, dtype=float)
    outcome_array = np.asarray(outcomes, dtype=float)
    if score_array.shape != outcome_array.shape or score_array.ndim != 1:
        raise ValueError("inputs must be equal-length 1D arrays")
    if not np.isfinite(score_array).all() or not np.isfinite(outcome_array).all():
        raise ValueError("metric inputs must be finite")
    return score_array, outcome_array


def _eligibility_array(eligible: Sequence[bool | int] | None, length: int) -> np.ndarray:
    if eligible is None:
        return np.ones(length, dtype=bool)
    values = np.asarray(eligible, dtype=bool)
    if values.shape != (length,):
        raise ValueError("eligible must be a matching 1D array")
    return values


def selective_curve(
    scores: Sequence[float],
    accuracies: Sequence[float],
    *,
    eligible: Sequence[bool | int] | None = None,
) -> list[SelectivePoint]:
    """Return all attainable tie-preserving risk/coverage points."""

    score_array, acc_array = _metric_arrays(scores, accuracies)
    eligible_array = _eligibility_array(eligible, len(score_array))
    if len(score_array) == 0 or not eligible_array.any():
        return []
    order = np.flatnonzero(eligible_array)[
        np.argsort(-score_array[eligible_array], kind="stable")
    ]
    sorted_scores = score_array[order]
    errors = 1.0 - np.clip(acc_array[order], 0.0, 1.0)
    cumulative_errors = np.cumsum(errors)
    points: list[SelectivePoint] = []
    for index, threshold in enumerate(sorted_scores):
        if index + 1 < len(sorted_scores) and sorted_scores[index + 1] == threshold:
            continue
        answered = index + 1
        points.append(
            SelectivePoint(
                threshold=float(threshold),
                coverage=answered / len(score_array),
                risk=float(cumulative_errors[index] / answered),
                answered=answered,
            )
        )
    return points


def coverage_at_risk(
    scores: Sequence[float],
    accuracies: Sequence[float],
    risk_limit: float,
    *,
    eligible: Sequence[bool | int] | None = None,
) -> SelectivePoint | None:
    """Descriptive oracle coverage at a test-set risk limit."""

    if not 0 <= risk_limit <= 1:
        raise ValueError("risk_limit must be in [0, 1]")
    valid = [
        point
        for point in selective_curve(scores, accuracies, eligible=eligible)
        if point.risk <= risk_limit
    ]
    return max(valid, key=lambda point: point.coverage, default=None)


def aurc(
    scores: Sequence[float],
    accuracies: Sequence[float],
    *,
    eligible: Sequence[bool | int] | None = None,
) -> float:
    """Coverage-normalized area over eligible substantive predictions."""

    points = selective_curve(scores, accuracies, eligible=eligible)
    if not points:
        return math.nan
    result = 0.0
    previous_coverage = 0.0
    for point in points:
        result += point.risk * (point.coverage - previous_coverage)
        previous_coverage = point.coverage
    return float(result / points[-1].coverage)


# Backward-compatible name used by the early scaffold.
risk_coverage_auc = aurc


def binary_auroc(scores: Sequence[float], labels: Sequence[bool | int]) -> float:
    score_array, label_values = _metric_arrays(scores, labels)
    label_array = label_values.astype(bool)
    positives = int(label_array.sum())
    negatives = len(label_array) - positives
    if positives == 0 or negatives == 0:
        return math.nan
    ranks = rankdata(score_array, method="average")
    positive_rank_sum = float(ranks[label_array].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


@dataclass(frozen=True)
class PlattCalibrator:
    slope: float
    intercept: float
    center: float
    scale: float
    status: str

    def predict(self, scores: Sequence[float]) -> np.ndarray:
        values = np.asarray(scores, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("calibration scores must be finite")
        return expit(self.slope * ((values - self.center) / self.scale) + self.intercept)


def fit_platt_calibrator(
    scores: Sequence[float], labels: Sequence[bool | int], *, l2: float = 1e-4
) -> PlattCalibrator:
    """Fit a deterministic sigmoid correctness map on fit-only data."""

    score_array, label_values = _metric_arrays(scores, labels)
    if len(score_array) == 0:
        raise ValueError("Platt fitting requires at least one observation")
    labels_array = label_values.astype(float)
    center = float(score_array.mean())
    scale = float(score_array.std())
    if not math.isfinite(scale) or scale < 1e-12:
        scale = 1.0
    prevalence = float((labels_array.sum() + 0.5) / (len(labels_array) + 1.0))
    initial = np.array([1.0, float(logit(prevalence))])
    if np.unique(labels_array).size < 2 or np.unique(score_array).size < 2:
        return PlattCalibrator(0.0, initial[1], center, scale, "constant_fallback")
    standardized = (score_array - center) / scale

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        slope, intercept = parameters
        logits = slope * standardized + intercept
        probabilities = expit(logits)
        loss = float(np.mean(np.logaddexp(0.0, logits) - labels_array * logits) + l2 * slope**2)
        residual = probabilities - labels_array
        gradient = np.array(
            [
                float(np.mean(residual * standardized) + 2 * l2 * slope),
                float(np.mean(residual)),
            ]
        )
        return loss, gradient

    result = minimize(
        lambda parameters: objective(parameters)[0],
        initial,
        jac=lambda parameters: objective(parameters)[1],
        method="L-BFGS-B",
    )
    if not result.success or not np.isfinite(result.x).all():
        return PlattCalibrator(0.0, initial[1], center, scale, "optimization_fallback")
    return PlattCalibrator(float(result.x[0]), float(result.x[1]), center, scale, "fitted")


def expected_calibration_error(
    probabilities: Sequence[float], labels: Sequence[bool | int], bins: int = 10
) -> float:
    probs, truth = _metric_arrays(probabilities, labels)
    if bins < 1 or np.any((probs < 0) | (probs > 1)):
        raise ValueError("probabilities must be in [0, 1] and bins must be positive")
    if len(probs) == 0:
        return math.nan
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        mask = (probs >= edges[index]) & (
            (probs <= edges[index + 1])
            if index == bins - 1
            else (probs < edges[index + 1])
        )
        if mask.any():
            result += mask.mean() * abs(float(probs[mask].mean() - truth[mask].mean()))
    return float(result)


def brier_score(probabilities: Sequence[float], labels: Sequence[bool | int]) -> float:
    probs, truth = _metric_arrays(probabilities, labels)
    if np.any((probs < 0) | (probs > 1)):
        raise ValueError("probabilities must be in [0, 1]")
    return float(np.mean((probs - truth) ** 2)) if len(probs) else math.nan


def effective_reliability(
    accuracies: Sequence[float], answered: Sequence[bool | int], error_cost: float
) -> float:
    acc, answered_values = _metric_arrays(accuracies, answered)
    cover = answered_values.astype(bool)
    if error_cost < 0:
        raise ValueError("error_cost must be non-negative")
    utility = np.zeros_like(acc)
    utility[cover & (acc > 0)] = acc[cover & (acc > 0)]
    utility[cover & (acc == 0)] = -error_cost
    return float(utility.mean()) if len(utility) else math.nan


def best_effective_reliability_threshold(
    scores: Sequence[float],
    accuracies: Sequence[float],
    error_cost: float,
    *,
    eligible: Sequence[bool | int] | None = None,
) -> tuple[float, float]:
    score_array, acc_array = _metric_arrays(scores, accuracies)
    if len(score_array) == 0:
        raise ValueError("threshold fitting requires at least one observation")
    eligible_array = _eligibility_array(eligible, len(score_array))
    candidates = [math.inf, *sorted(set(map(float, score_array[eligible_array])), reverse=True)]
    best_threshold, best_score = math.inf, 0.0
    for threshold in candidates:
        answered = eligible_array & (score_array >= threshold)
        value = effective_reliability(acc_array, answered, error_cost)
        if value > best_score:
            best_threshold, best_score = threshold, value
    return best_threshold, best_score


def learn_selective_threshold(
    scores: Sequence[float],
    entirely_wrong: Sequence[bool | int],
    *,
    risk_limit: float = 0.05,
    eligible: Sequence[bool | int] | None = None,
) -> tuple[float, float, int]:
    """Learn the maximum-coverage tie-preserving threshold on fit data."""

    if not 0 <= risk_limit <= 1:
        raise ValueError("risk_limit must be in [0, 1]")
    score_array, error_values = _metric_arrays(scores, entirely_wrong)
    errors = error_values.astype(bool)
    if len(score_array) == 0:
        raise ValueError("threshold fitting requires at least one observation")
    eligible_array = _eligibility_array(eligible, len(score_array))
    best: tuple[float, float, int] = (math.inf, math.nan, 0)
    for threshold in sorted(set(map(float, score_array[eligible_array])), reverse=True):
        mask = eligible_array & (score_array >= threshold)
        accepted = int(mask.sum())
        risk = float(errors[mask].mean())
        if risk <= risk_limit and accepted > best[2]:
            best = (threshold, risk, accepted)
    return best


def clopper_pearson_upper(errors: int, accepted: int, delta: float = 0.05) -> float:
    """One-sided exact upper confidence bound for a Bernoulli error rate."""

    if accepted < 0 or errors < 0 or errors > accepted:
        raise ValueError("require 0 <= errors <= accepted")
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    if accepted == 0 or errors == accepted:
        return 1.0
    return float(beta.ppf(1.0 - delta, errors + 1, accepted - errors))


@dataclass(frozen=True)
class CertificationResult:
    threshold: float
    accepted: int
    errors: int
    coverage: float
    empirical_risk: float
    upper_bound: float
    certified: bool


def certify_selective_threshold(
    scores: Sequence[float],
    entirely_wrong: Sequence[bool | int],
    *,
    threshold: float,
    alpha: float = 0.10,
    delta: float = 0.05,
    eligible: Sequence[bool | int] | None = None,
) -> CertificationResult:
    """Test one threshold frozen independently of the certification data."""

    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    score_array, error_values = _metric_arrays(scores, entirely_wrong)
    errors_array = error_values.astype(bool)
    eligible_array = _eligibility_array(eligible, len(score_array))
    mask = eligible_array & (score_array >= threshold)
    accepted = int(mask.sum())
    errors = int(errors_array[mask].sum())
    upper = clopper_pearson_upper(errors, accepted, delta)
    return CertificationResult(
        threshold=threshold,
        accepted=accepted,
        errors=errors,
        coverage=accepted / len(score_array) if len(score_array) else 0.0,
        empirical_risk=errors / accepted if accepted else math.nan,
        upper_bound=upper,
        certified=bool(accepted and upper <= alpha),
    )


def bootstrap_mean_difference(
    first: Sequence[float],
    second: Sequence[float],
    *,
    samples: int = 2_000,
    seed: int = 20260712,
) -> tuple[float, float, float]:
    """Paired bootstrap mean difference and percentile interval."""

    left, right = _metric_arrays(first, second)
    if len(left) == 0:
        raise ValueError("paired samples must be non-empty")
    if samples < 1:
        raise ValueError("samples must be positive")
    differences = left - right
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(samples, len(differences)))
    estimates = differences[indices].mean(axis=1)
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return float(differences.mean()), float(lower), float(upper)
