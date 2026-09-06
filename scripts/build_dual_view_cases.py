"""Étape 2b du pipeline — construction des cas bi-vues CBIS-DDSM.

À partir du CSV indexé (``cbis_indexed.csv``) :

- groupement par ``(patient_id, left_or_right_breast, abnormality_id)`` ;
- appariement d'une vue ``CC`` et d'une vue ``MLO`` ; les groupes qui n'ont pas
  les deux vues sont rejetés ;
- génération d'une description textuelle synthétique par cas (forme de masse,
  marges, type / distribution de calcification, sein, mention
  « seen on CC and MLO views », BI-RADS, pathologie) destinée à la supervision
  CLIP.

Sortie : ``dual_view_cases.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


def _clean(value: object) -> Optional[str]:
    """Normalise une valeur catégorielle CBIS (``str`` en minuscule, espaces)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "n/a", "none"}:
        return None
    return text.replace("_", " ").lower()


def build_case_description(row: pd.Series) -> str:
    """Phrase descriptive synthétique pour la supervision contrastive CLIP."""
    breast = "left breast" if str(row.get("left_or_right_breast", "")).upper().startswith("L") else "right breast"
    parts = [f"Mammogram of the {breast}."]

    shape = _clean(row.get("mass_shape"))
    margins = _clean(row.get("mass_margins"))
    if shape or margins:
        seg = "Mass with"
        if shape:
            seg += f" {shape} shape"
        if shape and margins:
            seg += " and"
        if margins:
            seg += f" {margins} margins"
        parts.append(seg + ".")

    calc_type = _clean(row.get("calc_type"))
    calc_dist = _clean(row.get("calc_distribution"))
    if calc_type or calc_dist:
        seg = "Calcifications"
        if calc_type:
            seg += f" of {calc_type} type"
        if calc_dist:
            seg += f" with {calc_dist} distribution"
        parts.append(seg + ".")

    parts.append("Seen on CC and MLO views.")

    birads = _clean(row.get("birads_subcategory")) or _clean(row.get("assessment"))
    if birads:
        parts.append(f"BI-RADS {birads.upper()}.")

    pathology = _clean(row.get("pathology"))
    if pathology:
        parts.append(f"Pathology: {pathology}.")

    return " ".join(parts)


def _pick_view(group: pd.DataFrame, view: str) -> Optional[pd.Series]:
    """Retourne la 1re ligne du groupe dont ``image_view`` vaut ``view`` et dont
    le JPEG est résolu, sinon la 1re ligne ``view`` quelconque, sinon ``None``."""
    subset = group[group["image_view"].astype(str).str.upper() == view]
    if subset.empty:
        return None
    with_jpeg = subset[subset.get("jpeg_path", "").astype(str) != ""]
    return (with_jpeg.iloc[0] if not with_jpeg.empty else subset.iloc[0])


def build_dual_view_cases(indexed: pd.DataFrame) -> pd.DataFrame:
    """Construit un DataFrame de cas bi-vues à partir du CSV indexé."""
    group_keys = ["patient_id", "left_or_right_breast", "abnormality_id"]
    records: list[Dict[str, object]] = []

    for (patient_id, breast, abn_id), group in indexed.groupby(group_keys, dropna=False):
        cc = _pick_view(group, "CC")
        mlo = _pick_view(group, "MLO")
        if cc is None or mlo is None:
            continue  # paire incomplète → rejet

        ref = cc
        record: Dict[str, object] = {
            "dataset": "CBIS-DDSM",
            "patient_id": patient_id,
            "exam_id": f"{patient_id}_{breast}_{abn_id}",
            "breast": "LEFT" if str(breast).upper().startswith("L") else "RIGHT",
            "abnormality_id": abn_id,
            "path_cc": cc.get("jpeg_path", ""),
            "path_mlo": mlo.get("jpeg_path", ""),
            "mass_shape": ref.get("mass_shape"),
            "mass_margins": ref.get("mass_margins"),
            "calc_type": ref.get("calc_type"),
            "calc_distribution": ref.get("calc_distribution"),
            "pathology": ref.get("pathology"),
            "assessment": ref.get("assessment"),
            "birads_subcategory": _clean(ref.get("assessment")),
            "split": ref.get("split"),
        }
        record["description"] = build_case_description(pd.Series(record))
        records.append(record)

    return pd.DataFrame(records)


def main() -> None:
    from _bootstrap import bootstrap_console

    bootstrap_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indexed", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    indexed = pd.read_csv(args.indexed)
    cases = build_dual_view_cases(indexed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cases.to_csv(args.out, index=False)
    print(f"{len(cases)} cas bi-vues CBIS-DDSM → {args.out}")


if __name__ == "__main__":
    main()
