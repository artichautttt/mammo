"""Tests de la fusion MVKL + CBIS (étape 3)."""

import pandas as pd

from build_combined_dual_view_dataset import (
    MANIFESTATION_COLS,
    MORPHO_COLS,
    _align_cbis,
    _mvkl_to_cases,
    combine,
)


def test_align_cbis_adds_missing_manifestation_columns_as_na():
    cbis = pd.DataFrame([{
        "patient_id": "P1", "exam_id": "P1_LEFT_1", "breast": "LEFT",
        "path_cc": "cc.jpg", "path_mlo": "mlo.jpg", "mass_shape": "OVAL",
        "birads_subcategory": "3",
    }])
    out = _align_cbis(cbis)
    for col in MANIFESTATION_COLS:
        assert out[col].isna().all()


def test_mvkl_pairs_require_both_views():
    views = pd.DataFrame([
        {"patient_id": "P1", "laterality": "LEFT", "view": "CC",
         "image_path": "cc.png", "birads_subcategory": "2", "birads_source": "same_breast"},
        {"patient_id": "P1", "laterality": "LEFT", "view": "MLO",
         "image_path": "mlo.png", "birads_subcategory": "4A", "birads_source": "same_breast"},
        {"patient_id": "P2", "laterality": "RIGHT", "view": "CC",
         "image_path": "cc2.png", "birads_subcategory": "1", "birads_source": "same_breast"},
    ])
    cases = _mvkl_to_cases(views)
    assert list(cases["patient_id"]) == ["P1"]
    # BI-RADS le plus sévère du groupe conservé
    assert cases.iloc[0]["birads_subcategory"] == "4A"


def test_combine_produces_single_schema():
    views = pd.DataFrame([
        {"patient_id": "P1", "laterality": "LEFT", "view": "CC",
         "image_path": "cc.png", "birads_subcategory": "2", "birads_source": "same_breast"},
        {"patient_id": "P1", "laterality": "LEFT", "view": "MLO",
         "image_path": "mlo.png", "birads_subcategory": "2", "birads_source": "same_breast"},
    ])
    cbis = pd.DataFrame([{
        "patient_id": "C1", "exam_id": "C1_LEFT_1", "breast": "LEFT",
        "path_cc": "cc.jpg", "path_mlo": "mlo.jpg", "mass_margins": "SPICULATED",
        "birads_subcategory": "5",
    }])
    combined = combine(views, cbis)
    assert set(combined["dataset"]) == {"MVKL", "CBIS-DDSM"}
    assert len(combined) == 2
    for col in MORPHO_COLS + MANIFESTATION_COLS:
        assert col in combined.columns
