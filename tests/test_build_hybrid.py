"""Tests de l'augmentation virtuelle pondérée (étape 5)."""

import pandas as pd

from build_hybrid_augmented_weighted_dataset import (
    CLASS_TARGETS,
    DEFAULT_TARGET_PER_CLASS,
    _augmentation_params,
    build_hybrid_dataset,
)


def test_augmentation_params_are_deterministic():
    a = _augmentation_params(3, 2)
    b = _augmentation_params(3, 2)
    assert a == b
    assert _augmentation_params(3, 2) != _augmentation_params(4, 2)


def test_augmentation_params_within_bounds():
    p = _augmentation_params(10, 5)
    assert -8.0 <= p["rotation_deg"] <= 8.0
    assert 0.96 <= p["scale"] <= 1.04
    assert p["noise_sigma"] in (0.0, 1.5, 2.5, 3.5)
    assert p["hflip"] is False


def _dataset(counts):
    rows = []
    for cls, n in counts.items():
        for i in range(n):
            rows.append({
                "exam_id": f"{cls}_{i}", "birads_subcategory": cls,
                "path_cc": "cc.jpg", "path_mlo": "mlo.jpg",
            })
    return pd.DataFrame(rows)


def test_minority_classes_reach_target():
    df = _dataset({"4A": 429, "2": 10, "1": 4})
    out = build_hybrid_dataset(df)
    counts = out["birads_class"].value_counts().to_dict()
    assert counts["2"] == DEFAULT_TARGET_PER_CLASS
    assert counts["1"] == CLASS_TARGETS["1"]  # plafonnée à 40
    assert counts["4A"] == 429  # classe majoritaire inchangée


def test_augmented_rows_flagged_and_have_params():
    df = _dataset({"4A": 429, "3": 5})
    out = build_hybrid_dataset(df)
    aug = out[out["is_augmented"]]
    assert len(aug) == DEFAULT_TARGET_PER_CLASS - 5
    assert (aug["augmentation_params"] != "{}").all()


def test_sample_weight_present_for_all_rows():
    out = build_hybrid_dataset(_dataset({"4A": 20, "1": 4}))
    assert out["sample_weight"].notna().all()
    assert (out["sample_weight"] > 0).all()


def test_only_train_split_is_augmented():
    df = _dataset({"4A": 429, "3": 5})
    df["split"] = "train"
    val = _dataset({"3": 8})
    val["split"] = "val"
    out = build_hybrid_dataset(pd.concat([df, val], ignore_index=True))
    # les lignes val ne sont jamais dupliquées
    assert not out[out["split"] == "val"]["is_augmented"].any()
    assert (out[out["split"] == "val"]).shape[0] == 8
    # le train "3" est bien complété à la cible
    train3 = out[(out["split"] == "train") & (out["birads_class"] == "3")]
    assert len(train3) == DEFAULT_TARGET_PER_CLASS
