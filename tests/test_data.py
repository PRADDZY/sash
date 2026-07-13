from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from PIL import Image

from sash_audit.data import (
    CORRUPTIONS,
    build_manifest,
    corrupt_image,
    random_question_split,
    select_shift_indices,
)


def synthetic_rows(count: int = 300) -> list[dict[str, object]]:
    return [
        {
            "answerable": int(index % 3 != 0),
            "answer_type": (
                "unanswerable" if index % 3 == 0 else ("yes/no" if index % 2 else "other")
            ),
        }
        for index in range(count)
    ]


def test_random_question_split_is_deterministic_disjoint_and_20_20_60() -> None:
    rows = synthetic_rows(90)
    first = random_question_split(rows, seed=7)
    second = random_question_split(rows, seed=7)
    assert first == second
    assert first != random_question_split(rows, seed=8)
    assert Counter(first.values()) == {"fit": 18, "certification": 18, "test": 54}
    assert set(first) == set(range(len(rows)))


def test_shift_selection_balances_corruptions_without_crossing_splits() -> None:
    rows = synthetic_rows()
    assignment = random_question_split(rows, seed=9)
    counts = {"fit": 30, "certification": 30, "test": 30}
    selected = select_shift_indices(rows, assignment, split_counts=counts, seed=9)
    for split in ("fit", "certification", "test"):
        assert len({index for index, _ in selected[split]}) == 30
        assert Counter(corruption for _, corruption in selected[split]) == {
            corruption: 10 for corruption in CORRUPTIONS
        }
        assert all(
            assignment[index] == split and rows[index]["answerable"]
            for index, _ in selected[split]
        )


@pytest.mark.parametrize("corruption", CORRUPTIONS)
def test_corruptions_preserve_shape_and_are_deterministic(corruption: str) -> None:
    pixels = np.arange(64 * 48 * 3, dtype=np.uint8).reshape((48, 64, 3))
    image = Image.fromarray(pixels)
    first = corrupt_image(image, corruption)
    second = corrupt_image(image, corruption)
    assert first.size == image.size
    assert np.array_equal(np.asarray(first), np.asarray(second))
    assert not np.array_equal(np.asarray(first), np.asarray(image))


def test_manifest_keeps_clear_and_corrupted_views_in_same_split(tmp_path) -> None:
    rows = [
        {
            "filename": f"VizWiz_val_{index:08d}.jpg",
            "image": Image.new("RGB", (16, 12), color=(index, index, index)),
            "question": "What is shown?",
            "answers": ["object"] * 10,
            "answerable": 1,
            "answer_type": "other",
        }
        for index in range(30)
    ]
    counts = {"fit": 3, "certification": 3, "test": 3}
    manifest, pilot = build_manifest(
        rows,
        tmp_path,
        shift_counts=counts,
        seed=3,
        pilot_cases={},
    )
    assert pilot == []
    assert len(manifest) == 39
    natural_splits = {
        row.case_id: row.evaluation_split for row in manifest if row.regime == "natural"
    }
    for shifted in (row for row in manifest if row.regime == "shift"):
        assert natural_splits[shifted.source_case_id] == shifted.evaluation_split
        assert shifted.clear_image_path


def test_reserved_pilot_sources_are_disjoint_from_evaluation(tmp_path) -> None:
    rows = [
        {
            "filename": f"VizWiz_val_{index:08d}.jpg",
            "image": Image.new("RGB", (16, 12), color=(index, index, index)),
            "question": "What is shown?",
            "answers": ["object"] * 10,
            "answerable": 1,
            "answer_type": "other",
        }
        for index in range(32)
    ]
    reserved = {
        "VizWiz_val_00000000": "none",
        "VizWiz_val_00000001": "blur",
    }
    manifest, pilot = build_manifest(
        rows,
        tmp_path,
        shift_counts={"fit": 3, "certification": 3, "test": 3},
        seed=3,
        pilot_cases=reserved,
    )
    evaluation_sources = {
        row.source_case_id or row.case_id
        for row in manifest
    }
    assert not (set(reserved) & evaluation_sources)
    assert {row.case_id for row in pilot} == {
        "VizWiz_val_00000000",
        "VizWiz_val_00000001__blur",
    }


def test_unknown_corruption_is_rejected() -> None:
    with pytest.raises(ValueError):
        corrupt_image(Image.new("RGB", (10, 10)), "glare")
