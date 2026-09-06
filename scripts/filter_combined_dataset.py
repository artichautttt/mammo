"""Étape 4 du pipeline — filtrage du dataset combiné.

Cinq critères, appliqués dans cet ordre :

1. paires complètes obligatoires (``path_cc`` ET ``path_mlo`` renseignés) ;
2. ``birads_subcategory`` non nul ;
3. suppression des BI-RADS « 4 » sans sous-catégorie (A / B / C) — annotation
   trop ambiguë ;
4. pour MVKL uniquement : ne garder que les sources fiables (``same_breast``,
   ``bilateral``), exclure ``unknown_context`` ;
5. dédoublonnage exact sur
   ``(dataset, patient_id, exam_id, breast, path_cc, path_mlo)``.

Sortie attendue : **1930 cas** (852 CBIS-DDSM + 1078 MVKL).
Sortie : ``combined_dual_view_cases_filtered.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

RELIABLE_MVKL_SOURCES = {"same_breast", "bilateral"}
DEDUP_KEYS = ["dataset", "patient_id", "exam_id", "breast", "path_cc", "path_mlo"]


def _non_empty(series: pd.Series) -> pd.Series:
    """``True`` là où la valeur est une chaîne non vide et non ``NA``."""
    return series.notna() & (series.astype(str).str.strip() != "") & (
        series.astype(str).str.lower() != "nan"
    )


def filter_dataset(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Applique les 5 critères et retourne le DataFrame filtré."""
    steps = []

    # 1. paires complètes
    mask = _non_empty(df["path_cc"]) & _non_empty(df["path_mlo"])
    df = df[mask].copy()
    steps.append(("1. paires CC+MLO complètes", len(df)))

    # 2. birads_subcategory non nul
    df = df[_non_empty(df["birads_subcategory"])].copy()
    steps.append(("2. birads_subcategory renseigné", len(df)))

    # 3. supprimer les "4" nus (sans A/B/C)
    bare_four = df["birads_subcategory"].astype(str).str.strip().str.upper() == "4"
    df = df[~bare_four].copy()
    steps.append(("3. BI-RADS '4' ambigu retiré", len(df)))

    # 4. MVKL : sources fiables uniquement
    is_mvkl = df["dataset"].astype(str).str.upper() == "MVKL"
    reliable = df["birads_source"].astype(str).str.lower().isin(RELIABLE_MVKL_SOURCES)
    df = df[~is_mvkl | reliable].copy()
    steps.append(("4. MVKL sources fiables", len(df)))

    # 5. dédoublonnage exact
    df = df.drop_duplicates(subset=DEDUP_KEYS, keep="first").reset_index(drop=True)
    steps.append(("5. dédoublonnage exact", len(df)))

    if verbose:
        for label, count in steps:
            print(f"  {label:<35} → {count}")
        print(df["dataset"].value_counts().to_string())

    return df


def main() -> None:
    from _bootstrap import bootstrap_console

    bootstrap_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    df = pd.read_csv(args.in_path)
    filtered = filter_dataset(df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(args.out, index=False)
    print(f"\n{len(filtered)} cas retenus → {args.out}")
    if len(filtered) != 1930:
        print(f"  ⚠ attendu 1930 cas (852 CBIS-DDSM + 1078 MVKL), obtenu {len(filtered)}")


if __name__ == "__main__":
    main()
