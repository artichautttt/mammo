"""Tests des helpers de l'adaptateur ViKL (sans I/O Excel)."""

import pandas as pd

from adapt_vikl_annotations import (
    birads_from_text,
    label_to_pathology,
    parse_view_token,
)
from build_combined_dual_view_dataset import combine


def test_parse_view_token():
    assert parse_view_token("LMLO") == ("LEFT", "MLO")
    assert parse_view_token("RCC") == ("RIGHT", "CC")
    assert parse_view_token("LMO") == ("LEFT", "MLO")  # variante de MLO
    assert parse_view_token("rcc") == ("RIGHT", "CC")
    assert parse_view_token("garbage") is None


def test_label_to_pathology():
    assert label_to_pathology("恶性") == "MALIGNANT"
    assert label_to_pathology("良性") == "BENIGN"
    assert label_to_pathology("benign") == "BENIGN"
    assert label_to_pathology(None) is None


def test_birads_from_text():
    assert birads_from_text("... 结论 BI-RADS 4B ...") == "4B"
    assert birads_from_text("birads:5") == "5"
    assert birads_from_text("BI RADS 3") == "3"
    assert birads_from_text("aucune mention") is None
    assert birads_from_text(None) is None


def test_combine_accepts_mvkl_only():
    views = pd.DataFrame([
        {"patient_id": "P1", "laterality": "LEFT", "view": "CC",
         "image_path": "cc.png", "birads_subcategory": "3", "birads_source": "same_breast"},
        {"patient_id": "P1", "laterality": "LEFT", "view": "MLO",
         "image_path": "mlo.png", "birads_subcategory": "3", "birads_source": "same_breast"},
    ])
    out = combine(mvkl_views=views, cbis_cases=None)
    assert len(out) == 1
    assert set(out["dataset"]) == {"MVKL"}


def test_combine_requires_at_least_one_source():
    try:
        combine(None, None)
    except ValueError:
        return
    raise AssertionError("combine() aurait dû lever ValueError sans source")
