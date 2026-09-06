"""Adaptateur pour le vrai jeu **MVKL / ViKL** (https://github.com/wxwxwwxxx/ViKL).

Le dataset publié fournit ses annotations dans un fichier **Excel** — et non des
PNG avec la vue incrustée en texte. Il est donc inutile de passer par
``detect_mvkl_views.py`` (template matching + OCR) : la vue, la latéralité et les
manifestations se lisent directement.

Format observé dans ``dataset/mkvl_dataset.py`` du dépôt ViKL :

- colonnes de tête : ``patient_id``, ``study_id``, ``label`` (``恶性`` = malin /
  ``良性`` = bénin), ``view_id`` (plusieurs vues séparées par ``|``),
  ``诊断结果`` (compte rendu, en chinois) ;
- à partir de la 9ᵉ colonne (index 8) : les **manifestations one-hot** ;
- codes de vue : ``RCC`` / ``LCC`` / ``LMLO`` / ``LMO`` / ``RMLO`` / ``RMO`` —
  la 1ʳᵉ lettre donne le côté, le reste la vue ;
- images : ``<images_root>/[<dcm_dir>/]<patient_id>/<study_id>/<view_id>.a.png``.

Ce script produit :

- ``mvkl_labeled_views.csv`` (même schéma que la sortie de l'étape 1) ;
- ``manifestations.json`` (liste ordonnée des noms de manifestations), repris
  automatiquement par :func:`src.labels.load_mvkl_manifestations`.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

_VIEW_TOKEN_RE = re.compile(r"^([LR])(CC|MLO|MLI|ML|MO)$", re.IGNORECASE)
_BIRADS_RE = re.compile(r"BI[-\s]?RADS?\s*[:：=]?\s*(0|1|2|3|4[ABC]?|5|6)", re.IGNORECASE)

META_COLUMNS_DEFAULT = ["patient_id", "study_id", "label", "view_id", "诊断结果"]
LABEL_PATHOLOGY = {"恶性": "MALIGNANT", "良性": "BENIGN", "malignant": "MALIGNANT", "benign": "BENIGN"}


def parse_view_token(token: str) -> Optional[tuple[str, str]]:
    """``"LMLO"`` → ``("LEFT", "MLO")``. Retourne ``None`` si non reconnu."""
    match = _VIEW_TOKEN_RE.match(token.strip())
    if not match:
        return None
    side = "LEFT" if match.group(1).upper() == "L" else "RIGHT"
    raw = match.group(2).upper()
    view = "CC" if raw == "CC" else "MLO"
    return side, view


def label_to_pathology(label: object) -> Optional[str]:
    """Mappe le label ViKL (``恶性`` / ``良性``) vers ``MALIGNANT`` / ``BENIGN``."""
    if label is None or (isinstance(label, float) and pd.isna(label)):
        return None
    return LABEL_PATHOLOGY.get(str(label).strip(), None)


def birads_from_text(text: object) -> Optional[str]:
    """Extrait une (sous-)catégorie BI-RADS d'un compte rendu libre."""
    if not isinstance(text, str):
        return None
    match = _BIRADS_RE.search(text)
    return match.group(1).upper() if match else None


def _sanitize(name: str) -> str:
    clean = re.sub(r"[^0-9A-Za-z一-鿿]+", "_", str(name).strip()).strip("_")
    return clean.lower() or "attr"


def _manifestation_columns(df: pd.DataFrame, meta_columns: List[str], first_index: int) -> List[str]:
    """Colonnes de manifestations = tout ce qui suit les colonnes de méta."""
    explicit = [c for c in df.columns if c not in meta_columns]
    # on respecte l'ordre du fichier et on retire les colonnes de méta connues
    ordered = [c for c in df.columns if c not in set(meta_columns)]
    if first_index is not None and first_index < len(df.columns):
        ordered = list(df.columns[first_index:])
    return [c for c in ordered if c not in set(meta_columns)] or explicit


def adapt(
    xlsx_path: Path,
    images_root: Path,
    dcm_dir: str = "",
    birads_col: Optional[str] = None,
    split: str = "train",
    meta_columns: Optional[List[str]] = None,
    manifestation_start: int = 8,
    sheet: object = 0,
) -> tuple[pd.DataFrame, List[str]]:
    """Convertit un xlsx ViKL en ``(DataFrame façon étape 1, noms_manifestations)``."""
    meta_columns = list(meta_columns or META_COLUMNS_DEFAULT)
    if birads_col:
        meta_columns.append(birads_col)

    raw = pd.read_excel(xlsx_path, sheet_name=sheet)
    raw.columns = [str(c).strip() for c in raw.columns]

    mf_source_cols = _manifestation_columns(raw, meta_columns, manifestation_start)
    mf_names = [f"mf_{_sanitize(c)}" for c in mf_source_cols]

    id_col = "patient_id" if "patient_id" in raw.columns else raw.columns[0]
    study_col = "study_id" if "study_id" in raw.columns else raw.columns[1]
    view_col = "view_id" if "view_id" in raw.columns else raw.columns[3]
    report_col = next((c for c in ("诊断结果", "report", "findings") if c in raw.columns), None)

    rows: List[Dict[str, object]] = []
    for _, r in raw.iterrows():
        patient_id = str(r[id_col])
        study_id = str(r[study_col])
        report = r[report_col] if report_col else ""
        pathology = label_to_pathology(r.get("label"))
        birads = (
            str(r[birads_col]).strip() if birads_col and pd.notna(r.get(birads_col))
            else birads_from_text(report)
        )
        tokens = [t for t in re.split(r"[|,;/\s]+", str(r[view_col])) if t]
        for token in tokens:
            parsed = parse_view_token(token)
            if parsed is None:
                continue
            side, view = parsed
            parts = [images_root, dcm_dir, patient_id, study_id, f"{token}.a.png"]
            image_path = str(Path(*[p for p in parts if p]))
            record: Dict[str, object] = {
                "image_path": image_path,
                "patient_id": patient_id,
                "laterality": side,
                "view": view,
                "method": "vikl_annotation",
                "template_score": 1.0,
                "birads_subcategory": birads,
                # les annotations ViKL portent sur le sein imagé → source fiable
                "birads_source": "same_breast",
                "acr_density": pd.NA,
                "pathology": pathology,
                "split": split,
            }
            for name, src in zip(mf_names, mf_source_cols):
                value = r.get(src)
                record[name] = 0 if value is None or pd.isna(value) else int(float(value) >= 0.5)
            rows.append(record)

    return pd.DataFrame(rows), mf_names


def main() -> None:
    from _bootstrap import bootstrap_console

    bootstrap_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xlsx", required=True, type=Path, help="Annotation ViKL (.xlsx)")
    parser.add_argument("--images-root", required=True, type=Path)
    parser.add_argument("--dcm-dir", default="", help="Sous-dossier des images sous images-root")
    parser.add_argument("--birads-col", default=None,
                        help="Nom de colonne BI-RADS si présente (sinon extraite du compte rendu)")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--manifestation-start", type=int, default=8,
                        help="Index de la 1ʳᵉ colonne de manifestations (défaut 8)")
    parser.add_argument("--sheet", default=0, help="Nom ou index de feuille")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--manifestations-out", type=Path,
                        default=Path("data/processed/manifestations.json"))
    args = parser.parse_args()

    df, mf_names = adapt(
        args.xlsx, args.images_root, args.dcm_dir, args.birads_col,
        args.split, manifestation_start=args.manifestation_start, sheet=args.sheet,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    args.manifestations_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifestations_out.write_text(
        json.dumps([n[3:] for n in mf_names], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"{len(df)} vues MVKL écrites → {args.out}")
    print(f"{len(mf_names)} manifestations → {args.manifestations_out}")
    if not df.empty:
        print(df.groupby(['laterality', 'view']).size().to_string())


if __name__ == "__main__":
    main()
