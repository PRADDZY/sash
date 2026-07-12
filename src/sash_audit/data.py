"""Deterministic VizWiz manifest construction and synthetic image shifts."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Mapping, Sequence, Sized
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

DEFAULT_SEED = 20260712
CORRUPTIONS = ("blur", "low_light", "low_resolution")
EVALUATION_SPLITS = ("fit", "certification", "test")
DEFAULT_SHIFT_COUNTS = {"fit": 300, "certification": 300, "test": 600}


@dataclass(frozen=True)
class ManifestRow:
    case_id: str
    row_index: int
    evaluation_split: str
    regime: str
    question: str
    answers: tuple[str, ...]
    answerable: bool
    answer_type: str
    image_path: str
    corruption: str = "none"
    clear_image_path: str | None = None
    source_case_id: str | None = None


def _answer_text(answer: object) -> str:
    if isinstance(answer, Mapping):
        return str(answer.get("answer", ""))
    return str(answer)


def _answer_type(row: Mapping[str, object]) -> str:
    return str(row.get("answer_type") or row.get("category") or "unknown")


def _is_answerable(row: Mapping[str, object]) -> bool:
    if "answerable" in row and row["answerable"] is not None:
        return bool(row["answerable"])
    return _answer_type(row) != "unanswerable"


def _filename(row: Mapping[str, object], index: int) -> str:
    return str(
        row.get("filename")
        or row.get("image_id")
        or row.get("question_id")
        or f"vizwiz_{index:08d}.jpg"
    )


def random_question_split(
    rows: Sequence[Mapping[str, object]],
    *,
    fit_fraction: float = 0.2,
    certification_fraction: float = 0.2,
    seed: int = DEFAULT_SEED,
) -> dict[int, str]:
    """Make a seeded, simple random question-level 20/20/60 split."""

    if not 0 < fit_fraction < 1 or not 0 < certification_fraction < 1:
        raise ValueError("split fractions must be in (0, 1)")
    if fit_fraction + certification_fraction >= 1:
        raise ValueError("fit and certification fractions must sum to less than 1")
    if len(rows) < 3:
        raise ValueError("at least three questions are required")

    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    fit_count = max(1, round(len(rows) * fit_fraction))
    certification_count = max(1, round(len(rows) * certification_fraction))
    if fit_count + certification_count >= len(rows):
        raise ValueError("split fractions leave no test questions")

    assignment: dict[int, str] = {}
    for index in indices[:fit_count]:
        assignment[index] = "fit"
    for index in indices[fit_count : fit_count + certification_count]:
        assignment[index] = "certification"
    for index in indices[fit_count + certification_count :]:
        assignment[index] = "test"
    return assignment


def select_shift_indices(
    rows: Sequence[Mapping[str, object]],
    assignment: Mapping[int, str],
    *,
    split_counts: Mapping[str, int] = DEFAULT_SHIFT_COUNTS,
    seed: int = DEFAULT_SEED,
) -> dict[str, list[tuple[int, str]]]:
    """Select answerable questions and balance the three corruption types."""

    if set(split_counts) != set(EVALUATION_SPLITS):
        raise ValueError(f"split_counts must have exactly {EVALUATION_SPLITS}")
    selected: dict[str, list[tuple[int, str]]] = {}
    for split_offset, split in enumerate(EVALUATION_SPLITS):
        count = int(split_counts[split])
        if count < len(CORRUPTIONS) or count % len(CORRUPTIONS):
            raise ValueError(f"{split} count must be a positive multiple of {len(CORRUPTIONS)}")
        candidates = [
            index
            for index, row in enumerate(rows)
            if assignment[index] == split and _is_answerable(row)
        ]
        if len(candidates) < count:
            raise ValueError(f"{split} has {len(candidates)} answerable questions; need {count}")
        random.Random(seed + 10_000 + split_offset).shuffle(candidates)
        selected[split] = [
            (index, CORRUPTIONS[position % len(CORRUPTIONS)])
            for position, index in enumerate(candidates[:count])
        ]
    return selected


def corrupt_image(image: Image.Image, corruption: str) -> Image.Image:
    """Apply one frozen deterministic acquisition-failure simulation."""

    rgb = image.convert("RGB")
    if corruption == "blur":
        return rgb.filter(ImageFilter.GaussianBlur(radius=4.0))
    if corruption == "low_light":
        return ImageEnhance.Brightness(rgb).enhance(0.25)
    if corruption == "low_resolution":
        width, height = rgb.size
        scale = 96.0 / max(width, height)
        small = rgb.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.BILINEAR,
        )
        return small.resize((width, height), Image.Resampling.BILINEAR)
    if corruption == "none":
        return rgb.copy()
    raise ValueError(f"unsupported corruption: {corruption}")


def build_manifest(
    dataset: Iterable[Mapping[str, object]],
    output_dir: Path,
    *,
    seed: int = DEFAULT_SEED,
    shift_counts: Mapping[str, int] = DEFAULT_SHIFT_COUNTS,
    expected_natural_rows: int | None = None,
) -> list[ManifestRow]:
    """Save original/corrupted images and return the frozen evaluation manifest."""

    output_dir = Path(output_dir)
    image_dir = output_dir / "images"
    corrupted_dir = output_dir / "corrupted"
    image_dir.mkdir(parents=True, exist_ok=True)
    corrupted_dir.mkdir(parents=True, exist_ok=True)

    if expected_natural_rows is not None:
        row_count = expected_natural_rows
    elif isinstance(dataset, Sized):
        row_count = len(dataset)
    else:
        raise ValueError("expected_natural_rows is required for streaming datasets")
    assignment = random_question_split([{}] * row_count, seed=seed)
    metadata: list[dict[str, object]] = []
    manifest: list[ManifestRow] = []
    seen_case_ids: set[str] = set()

    for index, row in enumerate(dataset):
        if index >= row_count:
            raise ValueError(f"dataset contains more than the expected {row_count} rows")
        filename = _filename(row, index)
        case_id = Path(filename).stem
        if case_id in seen_case_ids:
            raise ValueError(f"duplicate natural case_id: {case_id}")
        seen_case_ids.add(case_id)
        image_path = image_dir / f"{case_id}.jpg"
        image = row.get("image")
        if not isinstance(image, Image.Image):
            raise TypeError(f"row {index} image is not a PIL image")
        if not image_path.exists():
            rgb = image.convert("RGB")
            rgb.save(image_path, format="JPEG", quality=95)
            rgb.close()
        source_answers = row.get("answers_original")
        if source_answers is None:
            source_answers = row.get("answers")
        if not isinstance(source_answers, Sequence) or not source_answers:
            raise ValueError(f"row {index} has no reference answers")
        answers = tuple(_answer_text(answer) for answer in source_answers)
        row_metadata: dict[str, object] = {
            "filename": filename,
            "question": str(row["question"]),
            "answers": answers,
            "answerable": _is_answerable(row),
            "answer_type": _answer_type(row),
        }
        metadata.append(row_metadata)
        manifest.append(
            ManifestRow(
                case_id=case_id,
                row_index=index,
                evaluation_split=assignment[index],
                regime="natural",
                question=str(row_metadata["question"]),
                answers=answers,
                answerable=bool(row_metadata["answerable"]),
                answer_type=str(row_metadata["answer_type"]),
                image_path=str(image_path),
            )
        )

    if len(metadata) != row_count:
        raise ValueError(f"expected {row_count} natural rows, found {len(metadata)}")
    selected = select_shift_indices(
        metadata, assignment, split_counts=shift_counts, seed=seed
    )

    for split in EVALUATION_SPLITS:
        for index, corruption in selected[split]:
            source = metadata[index]
            filename = _filename(source, index)
            source_case_id = Path(filename).stem
            clear_path = image_dir / f"{source_case_id}.jpg"
            shifted_case_id = f"{source_case_id}__{corruption}"
            shifted_path = corrupted_dir / f"{shifted_case_id}.jpg"
            if not shifted_path.exists():
                with Image.open(clear_path) as clear:
                    shifted = corrupt_image(clear, corruption)
                    shifted.save(shifted_path, format="JPEG", quality=95)
                    shifted.close()
            source_answers = source.get("answers")
            if not isinstance(source_answers, Sequence) or not source_answers:
                raise ValueError(f"row {index} has no reference answers")
            answers = tuple(_answer_text(answer) for answer in source_answers)
            manifest.append(
                ManifestRow(
                    case_id=shifted_case_id,
                    row_index=index,
                    evaluation_split=split,
                    regime="shift",
                    question=str(source["question"]),
                    answers=answers,
                    answerable=True,
                    answer_type=_answer_type(source),
                    image_path=str(shifted_path),
                    corruption=corruption,
                    clear_image_path=str(clear_path),
                    source_case_id=source_case_id,
                )
            )
    return manifest


def validate_manifest(rows: Sequence[ManifestRow], *, expected_natural_rows: int = 4319) -> None:
    """Fail fast on split leakage, duplicates, or unexpected study counts."""

    natural = [row for row in rows if row.regime == "natural"]
    shifted = [row for row in rows if row.regime == "shift"]
    if len(natural) != expected_natural_rows:
        raise ValueError(f"expected {expected_natural_rows} natural rows, found {len(natural)}")
    expected_shift = sum(DEFAULT_SHIFT_COUNTS.values())
    if len(shifted) != expected_shift:
        raise ValueError(f"expected {expected_shift} shifted rows, found {len(shifted)}")
    case_ids = [row.case_id for row in rows]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("manifest case IDs are not unique")
    natural_split = {row.case_id: row.evaluation_split for row in natural}
    for row in shifted:
        crossed_split = (
            row.source_case_id is None
            or natural_split.get(row.source_case_id) != row.evaluation_split
        )
        if crossed_split:
            raise ValueError(f"shifted view crossed a split: {row.case_id}")
        if not row.clear_image_path:
            raise ValueError(f"shifted view lacks a clear acquisition image: {row.case_id}")


def write_manifest(rows: Iterable[ManifestRow], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def read_manifest(path: Path) -> list[ManifestRow]:
    with Path(path).open(encoding="utf-8") as handle:
        return [ManifestRow(**json.loads(line)) for line in handle if line.strip()]
