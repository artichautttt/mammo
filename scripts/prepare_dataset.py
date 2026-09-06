"""Étape 2a du pipeline — normalisation et indexation des 4 CSV CBIS-DDSM
(masses train/test, calcifications train/test).

- normalisation des noms de colonnes (minuscules, underscores, nettoyage des
  retours à la ligne) ;
- extraction du ``SeriesUID`` depuis les chemins DICOM du type
  ``CaseName/StudyUID/SeriesUID/000000.dcm`` → avant-dernier segment ;
- indexation des JPEG présents sur disque par ``SeriesUID``.

Sortie : ``cbis_indexed.csv``.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd

CBIS_CSV_GLOBS = (
    "mass_case_description_train_set.csv",
    "mass_case_description_test_set.csv",
    "calc_case_description_train_set.csv",
    "calc_case_description_test_set.csv",
)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """minuscule, espaces/tirets → underscore, suppression des ``\\n``/``\\r``."""
    renamed = {}
    for col in df.columns:
        clean = str(col).strip().lower()
        clean = clean.replace("\n", " ").replace("\r", " ")
        clean = re.sub(r"[\s\-/]+", "_", clean)
        clean = re.sub(r"_+", "_", clean).strip("_")
        renamed[col] = clean
    return df.rename(columns=renamed)


def extract_series_uid(dicom_path: str) -> str:
    """``CaseName/StudyUID/SeriesUID/000000.dcm`` → ``SeriesUID`` (avant-dernier
    segment). Retourne ``""`` si le chemin est vide ou trop court."""
    if not isinstance(dicom_path, str) or not dicom_path.strip():
        return ""
    parts = [p for p in re.split(r"[\\/]+", dicom_path.strip()) if p]
    if len(parts) < 2:
        return ""
    return parts[-2]


def index_jpegs_by_series(jpeg_root: Path) -> Dict[str, str]:
    """Parcourt ``jpeg_root`` et associe chaque ``SeriesUID`` (nom du dossier
    parent) au premier JPEG trouvé."""
    index: Dict[str, str] = {}
    for jpg in jpeg_root.rglob("*.jpg"):
        series_uid = jpg.parent.name
        index.setdefault(series_uid, str(jpg))
    for jpg in jpeg_root.rglob("*.jpeg"):
        index.setdefault(jpg.parent.name, str(jpg))
    return index


def load_and_normalize(cbis_dir: Path) -> pd.DataFrame:
    """Charge et concatène les 4 CSV disponibles, avec une colonne ``abnormality``
    (``mass`` / ``calc``) et ``split`` (``train`` / ``test``)."""
    frames: List[pd.DataFrame] = []
    for name in CBIS_CSV_GLOBS:
        path = cbis_dir / name
        if not path.exists():
            continue
        df = normalize_columns(pd.read_csv(path))
        df["abnormality"] = "mass" if name.startswith("mass") else "calc"
        df["split"] = "train" if "train" in name else "test"
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"Aucun CSV CBIS-DDSM trouvé dans {cbis_dir}")
    return pd.concat(frames, ignore_index=True)


def prepare(cbis_dir: Path, jpeg_root: Path) -> pd.DataFrame:
    """Renvoie le DataFrame indexé prêt pour ``build_dual_view_cases``."""
    df = load_and_normalize(cbis_dir)

    # colonne du chemin DICOM de l'image "full" (le nom varie selon les versions)
    image_col = next(
        (c for c in ("image_file_path", "image_file_path_", "cropped_image_file_path")
         if c in df.columns),
        None,
    )
    if image_col is None:
        raise KeyError("Colonne de chemin image introuvable dans les CSV CBIS-DDSM")

    df["series_uid"] = df[image_col].map(extract_series_uid)

    jpeg_index = index_jpegs_by_series(jpeg_root) if jpeg_root.exists() else {}
    df["jpeg_path"] = df["series_uid"].map(lambda uid: jpeg_index.get(uid, ""))

    keep = [
        "patient_id", "abnormality", "abnormality_id", "abnormality_type",
        "left_or_right_breast", "image_view", "mass_shape", "mass_margins",
        "calc_type", "calc_distribution", "assessment", "pathology",
        "breast_density", "density", "split", "series_uid", "jpeg_path",
    ]
    present = [c for c in keep if c in df.columns]
    return df[present].copy()


def main() -> None:
    from _bootstrap import bootstrap_console

    bootstrap_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cbis", required=True, type=Path, help="Dossier contenant les 4 CSV")
    parser.add_argument("--jpeg-root", type=Path, default=None,
                        help="Racine des JPEG CBIS-DDSM (défaut: <cbis>/jpeg)")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    jpeg_root = args.jpeg_root or (args.cbis / "jpeg")
    df = prepare(args.cbis, jpeg_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"{len(df)} lignes indexées → {args.out}")
    print(f"  JPEG résolus : {(df['jpeg_path'] != '').sum()}/{len(df)}")


if __name__ == "__main__":
    main()
