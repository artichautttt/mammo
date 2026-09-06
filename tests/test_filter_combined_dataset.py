"""Tests des 5 critères de filtrage de l'étape 4."""

import pandas as pd
import pytest

from filter_combined_dataset import filter_dataset


def _row(**overrides):
    base = dict(
        dataset="CBIS-DDSM",
        patient_id="P1",
        exam_id="P1_LEFT_1",
        breast="LEFT",
        path_cc="/img/cc.jpg",
        path_mlo="/img/mlo.jpg",
        birads_subcategory="4A",
        birads_source="cbis_assessment",
    )
    base.update(overrides)
    return base


def test_incomplete_pairs_are_dropped():
    df = pd.DataFrame([_row(), _row(exam_id="P1_LEFT_2", path_mlo="")])
    out = filter_dataset(df, verbose=False)
    assert list(out["exam_id"]) == ["P1_LEFT_1"]


def test_missing_birads_dropped():
    df = pd.DataFrame([_row(), _row(exam_id="x", birads_subcategory="")])
    out = filter_dataset(df, verbose=False)
    assert len(out) == 1


def test_bare_four_dropped_but_subcategory_kept():
    df = pd.DataFrame([
        _row(exam_id="keep", birads_subcategory="4B"),
        _row(exam_id="drop", birads_subcategory="4"),
    ])
    out = filter_dataset(df, verbose=False)
    assert set(out["exam_id"]) == {"keep"}


def test_mvkl_unreliable_source_dropped():
    df = pd.DataFrame([
        _row(dataset="MVKL", exam_id="ok", birads_source="same_breast"),
        _row(dataset="MVKL", exam_id="ok2", birads_source="bilateral"),
        _row(dataset="MVKL", exam_id="bad", birads_source="unknown_context"),
    ])
    out = filter_dataset(df, verbose=False)
    assert set(out["exam_id"]) == {"ok", "ok2"}


def test_cbis_source_not_subject_to_mvkl_rule():
    df = pd.DataFrame([_row(birads_source="unknown_context")])
    out = filter_dataset(df, verbose=False)
    assert len(out) == 1


def test_exact_deduplication():
    df = pd.DataFrame([_row(), _row(), _row(exam_id="other")])
    out = filter_dataset(df, verbose=False)
    assert len(out) == 2
