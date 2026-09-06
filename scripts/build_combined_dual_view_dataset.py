"""Étape 3 du pipeline — fusion MVKL + CBIS-DDSM en un seul schéma de cas bi-vues.

- **MVKL** apporte un vecteur de 35 manifestations binaires
  (``mf_<nom>``) mais aucun descripteur morphologique structuré.
- **CBIS-DDSM** apporte les colonnes catégorielles ``mass_shape``,
  ``mass_margins``, ``calc_type``, ``calc_distribution`` mais aucun vecteur de
  manifestations.

Les colonnes absentes d'une source sont laissées à ``pd.NA`` (elles seront
ignorées dans la fonction de perte via ``ignore_index=-100``).

En cas de sous-catégories BI-RADS conflictuelles pour un même cas, on conserve la
plus sévère (``pick_most_severe_subcategory`` — ordre défini dans
``src/labels.py``).

Sortie : ``combined_dual_view_cases.csv``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.labels import MVKL_MANIFESTATIONS, pick_most_severe_subcategory  # noqa: E402

MANIFESTATION_COLS = [f"mf_{name}" for name in MVKL_MANIFESTATIONS]
MORPHO_COLS = ["mass_shape", "mass_margins", "calc_type", "calc_distribution"]

CANONICAL_COLS = [
    "dataset", "patient_id", "exam_id", "breast",
    "path_cc", "path_mlo",
    *MORPHO_COLS,
    *MANIFESTATION_COLS,
    "acr_density", "birads_subcategory", "birads_source",
    "pathology", "description", "split",
]


def _mvkl_to_cases(mvkl_views: pd.DataFrame) -> pd.DataFrame:
    """Assemble les vues MVKL étiquetées (une ligne par image) en cas bi-vues
    (une ligne par ``(patient_id, laterality)`` avec CC + MLO)."""
    records = []
    for (patient_id, laterality), group in mvkl_views.groupby(
        ["patient_id", "laterality"], dropna=False
    ):
        cc = group[group["view"] == "CC"]
        mlo = group[group["view"] == "MLO"]
        if cc.empty or mlo.empty:
            continue
        cc_row, mlo_row = cc.iloc[0], mlo.iloc[0]

        birads = pick_most_severe_subcategory(
            list(group["birads_subcategory"].dropna().astype(str))
        )
        # source la plus fiable observée dans le groupe
        sources = set(group["birads_source"].dropna().astype(str))
        if "same_breast" in sources:
            birads_source = "same_breast"
        elif "bilateral" in sources:
            birads_source = "bilateral"
        elif sources:
            birads_source = sorted(sources)[0]
        else:
            birads_source = None

        record = {
            "dataset": "MVKL",
            "patient_id": patient_id,
            "exam_id": f"MVKL_{patient_id}_{laterality}",
            "breast": laterality,
            "path_cc": cc_row.get("image_path", ""),
            "path_mlo": mlo_row.get("image_path", ""),
            "acr_density": cc_row.get("acr_density"),
            "birads_subcategory": birads,
            "birads_source": birads_source,
            "pathology": cc_row.get("pathology"),
            "split": cc_row.get("split", "train"),
        }
        for col in MANIFESTATION_COLS:
            record[col] = cc_row[col] if col in mvkl_views.columns else pd.NA
        for col in MORPHO_COLS:
            record[col] = pd.NA
        record["description"] = _mvkl_description(record)
        records.append(record)
    return pd.DataFrame(records)


def _mvkl_description(record: dict) -> str:
    breast = "left breast" if str(record["breast"]).upper().startswith("L") else "right breast"
    active = [
        name.replace("mf_", "").replace("_", " ")
        for name in MANIFESTATION_COLS
        if str(record.get(name)) in {"1", "1.0", "True", "true"}
    ]
    parts = [f"Mammogram of the {breast}.", "Seen on CC and MLO views."]
    if active:
        parts.insert(1, "Findings: " + ", ".join(active) + ".")
    if record.get("birads_subcategory"):
        parts.append(f"BI-RADS {str(record['birads_subcategory']).upper()}.")
    return " ".join(parts)


def _align_cbis(cbis_cases: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les colonnes manquantes (manifestations, densité) à ``pd.NA``."""
    df = cbis_cases.copy()
    df["dataset"] = "CBIS-DDSM"
    for col in MANIFESTATION_COLS:
        df[col] = pd.NA
    if "acr_density" not in df.columns:
        df["acr_density"] = df.get("breast_density", pd.NA)
    if "birads_source" not in df.columns:
        df["birads_source"] = "cbis_assessment"

    # résolution BI-RADS la plus sévère si doublons de cas
    if "exam_id" in df.columns:
        severe = (
            df.groupby("exam_id")["birads_subcategory"]
            .apply(lambda s: pick_most_severe_subcategory(list(s.dropna().astype(str))))
        )
        df["birads_subcategory"] = df["exam_id"].map(severe).fillna(df["birads_subcategory"])
    return df


def combine(mvkl_views: pd.DataFrame, cbis_cases: pd.DataFrame) -> pd.DataFrame:
    """Fusionne les deux sources dans le schéma canonique."""
    mvkl_cases = _mvkl_to_cases(mvkl_views)
    cbis_aligned = _align_cbis(cbis_cases)

    for df in (mvkl_cases, cbis_aligned):
        for col in CANONICAL_COLS:
            if col not in df.columns:
                df[col] = pd.NA

    combined = pd.concat(
        [mvkl_cases[CANONICAL_COLS], cbis_aligned[CANONICAL_COLS]],
        ignore_index=True,
    )
    return combined


def main() -> None:
    from _bootstrap import bootstrap_console

    bootstrap_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mvkl", required=True, type=Path, help="mvkl_labeled_views.csv")
    parser.add_argument("--cbis", required=True, type=Path, help="dual_view_cases.csv")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    mvkl_views = pd.read_csv(args.mvkl)
    cbis_cases = pd.read_csv(args.cbis)
    combined = combine(mvkl_views, cbis_cases)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out, index=False)
    print(f"{len(combined)} cas combinés → {args.out}")
    print(combined["dataset"].value_counts().to_string())


if __name__ == "__main__":
    main()
