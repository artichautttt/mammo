"""Étape 1 du pipeline — détection de la vue (CC / MLO) et de la latéralité sur
les images MVKL, qui ne portent aucune métadonnée DICOM : l'incidence est
incrustée en texte en haut de l'image.

Deux méthodes combinées :

1. Template matching OpenCV (``cv2.matchTemplate`` + ``cv2.TM_CCOEFF_NORMED``)
   contre 4 templates de référence (LCC, RCC, LMLO, RMLO). Seuil : 0.38.
2. OCR Tesseract, en confirmation / repli, sur la région de label extraite.

La latéralité (LEFT / RIGHT) est ensuite ré-estimée par analyse des composantes
connexes sur la plus grande masse de tissu.

Sortie : ``mvkl_labeled_views.csv``.
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

try:  # pytesseract est optionnel au moment de l'import (binaire système requis)
    import pytesseract
except Exception:  # pragma: no cover - dépend de l'environnement
    pytesseract = None  # type: ignore

# --- Constantes de prétraitement -------------------------------------------

LABEL_REGION_RATIO = 0.30          # 30 % supérieurs de l'image
GAUSSIAN_KERNEL = (3, 3)
BINARY_THRESHOLD = 115
TEMPLATE_MATCH_THRESHOLD = 0.38    # score minimal cv2.TM_CCOEFF_NORMED
TEMPLATE_KEYS = ("LCC", "RCC", "LMLO", "RMLO")

VIEW_RE = re.compile(r"\b([LR])\s*-?\s*(CC|MLO|ML|MLI)\b", re.IGNORECASE)


@dataclass
class ViewDetection:
    """Résultat de détection pour une image."""

    image_path: str
    patient_id: str
    laterality: Optional[str]        # "LEFT" / "RIGHT"
    view: Optional[str]              # "CC" / "MLO"
    method: str                      # "template" / "ocr" / "template+ocr" / "none"
    template_score: float
    birads_subcategory: Optional[str]
    birads_source: Optional[str]


def crop_label_region(image: np.ndarray, ratio: float = LABEL_REGION_RATIO) -> np.ndarray:
    """Retourne les ``ratio`` premiers pourcents (en hauteur) de l'image."""
    height = image.shape[0]
    cut = max(1, int(round(height * ratio)))
    return image[:cut, :]


def preprocess_label_region(region: np.ndarray) -> np.ndarray:
    """Niveaux de gris → flou gaussien (3, 3) → seuillage binaire à 115."""
    if region.ndim == 3:
        region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(region, GAUSSIAN_KERNEL, 0)
    _, binary = cv2.threshold(blurred, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
    return binary


def load_templates(templates_dir: Path) -> Dict[str, np.ndarray]:
    """Charge les 4 templates de référence (``LCC.png`` … ``RMLO.png``)."""
    templates: Dict[str, np.ndarray] = {}
    for key in TEMPLATE_KEYS:
        path = templates_dir / f"{key}.png"
        if path.exists():
            tmpl = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if tmpl is not None:
                templates[key] = tmpl
    return templates


def match_templates(
    label_region: np.ndarray, templates: Dict[str, np.ndarray]
) -> Tuple[Optional[str], float]:
    """Template matching normalisé. Retourne ``(clé, score)`` du meilleur match,
    ou ``(None, best_score)`` si le meilleur score reste sous le seuil 0.38."""
    best_key: Optional[str] = None
    best_score = -1.0
    for key, tmpl in templates.items():
        if tmpl.shape[0] > label_region.shape[0] or tmpl.shape[1] > label_region.shape[1]:
            scale = min(
                label_region.shape[0] / tmpl.shape[0],
                label_region.shape[1] / tmpl.shape[1],
            ) * 0.95
            tmpl = cv2.resize(tmpl, (0, 0), fx=scale, fy=scale)
        result = cv2.matchTemplate(label_region, tmpl, cv2.TM_CCOEFF_NORMED)
        score = float(result.max())
        if score > best_score:
            best_score = score
            best_key = key
    if best_score >= TEMPLATE_MATCH_THRESHOLD:
        return best_key, best_score
    return None, best_score


def ocr_label(label_region: np.ndarray) -> Optional[str]:
    """OCR Tesseract sur la région de label. Retourne un token normalisé du type
    ``"LCC"`` / ``"RMLO"`` si détecté."""
    if pytesseract is None:
        return None
    tesseract_cmd = os.environ.get("TESSERACT_CMD")
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    try:
        text = pytesseract.image_to_string(label_region, config="--psm 6")
    except Exception:  # pragma: no cover - binaire absent
        return None
    match = VIEW_RE.search(text.upper().replace(" ", ""))
    if not match:
        # tentative moins stricte
        match = VIEW_RE.search(text.upper())
    if not match:
        return None
    side, view = match.group(1).upper(), match.group(2).upper()
    view = "MLO" if view in {"MLO", "ML", "MLI"} else view
    return f"{side}{view}"


def infer_laterality_from_tissue(image: np.ndarray) -> Optional[str]:
    """Estime LEFT / RIGHT par analyse des composantes connexes : on isole la
    plus grande masse de tissu et on compare la somme des pixels à gauche et à
    droite de l'axe vertical médian.

    Convention : si la masse est majoritairement à gauche de l'image, le sein est
    de type "image de sein droit" affiché à gauche → on retourne ``"RIGHT"`` ;
    sinon ``"LEFT"``.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    num, _labels, stats, _cent = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num <= 1:
        return None
    # composante 0 = fond ; on prend la plus grande des autres
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (_labels == largest).astype(np.uint8)
    width = mask.shape[1]
    mid = width // 2
    left_sum = int(mask[:, :mid].sum())
    right_sum = int(mask[:, mid:].sum())
    if left_sum == right_sum:
        return None
    return "RIGHT" if left_sum > right_sum else "LEFT"


# --- BI-RADS depuis les rapports textuels ---------------------------------

_BIRADS_TOKEN_RE = re.compile(r"BI[- ]?RADS?\s*[:=]?\s*(0|1|2|3|4[ABC]?|5|6)", re.IGNORECASE)

# priorité des sources, de la plus fiable à la moins fiable
_BIRADS_SOURCE_PRIORITY = ["same_breast", "bilateral", "unknown_context"]


def pick_birads_for_breast(
    annotations: List[Dict[str, str]], breast: str
) -> Tuple[Optional[str], Optional[str]]:
    """Choisit le BI-RADS applicable à ``breast`` (``"LEFT"`` / ``"RIGHT"``).

    ``annotations`` est une liste de dicts ``{"text": ..., "breast": ...,
    "context": ...}`` où ``context`` vaut ``"same_breast"``, ``"bilateral"``,
    ``"unknown_context"`` ou ``"no_birads_for_breast"``.

    Priorité : label du bon sein (``same_breast``) > bilatéral (``bilateral``) >
    contexte incertain (``unknown_context``). Les entrées
    ``no_birads_for_breast`` sont exclues.

    Retourne ``(sous_categorie, source)``.
    """
    buckets: Dict[str, List[str]] = {k: [] for k in _BIRADS_SOURCE_PRIORITY}
    for ann in annotations:
        context = (ann.get("context") or "").strip()
        if context == "no_birads_for_breast":
            continue
        if context == "same_breast" and (ann.get("breast") or "").upper() != breast.upper():
            continue
        if context not in buckets:
            continue
        match = _BIRADS_TOKEN_RE.search(ann.get("text", ""))
        if match:
            buckets[context].append(match.group(1).upper())

    for source in _BIRADS_SOURCE_PRIORITY:
        if buckets[source]:
            # en cas de valeurs multiples pour une même source, on garde la 1re
            return buckets[source][0], source
    return None, None


# --- Orchestration -------------------------------------------------------------

def detect_image(
    image_path: Path,
    templates: Dict[str, np.ndarray],
    annotations_by_patient: Dict[str, List[Dict[str, str]]],
) -> ViewDetection:
    """Applique le pipeline de détection complet à une image."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return ViewDetection(str(image_path), "", None, None, "none", -1.0, None, None)

    patient_id = image_path.stem.split("_")[0]
    region_raw = crop_label_region(image)
    label_region = preprocess_label_region(region_raw)

    tmpl_key, tmpl_score = match_templates(label_region, templates)
    ocr_key = ocr_label(label_region)

    method = "none"
    token: Optional[str] = None
    if tmpl_key and ocr_key and tmpl_key == ocr_key:
        token, method = tmpl_key, "template+ocr"
    elif tmpl_key:
        token, method = tmpl_key, "template"
    elif ocr_key:
        token, method = ocr_key, "ocr"

    view: Optional[str] = None
    laterality: Optional[str] = None
    if token:
        laterality = "LEFT" if token.startswith("L") else "RIGHT"
        view = "CC" if token.endswith("CC") else "MLO"

    # ré-estimation de la latéralité par le tissu (prioritaire si disponible)
    tissue_side = infer_laterality_from_tissue(image)
    if tissue_side is not None:
        laterality = tissue_side

    birads_sub, birads_src = (None, None)
    if laterality:
        birads_sub, birads_src = pick_birads_for_breast(
            annotations_by_patient.get(patient_id, []), laterality
        )

    return ViewDetection(
        image_path=str(image_path),
        patient_id=patient_id,
        laterality=laterality,
        view=view,
        method=method,
        template_score=round(tmpl_score, 4),
        birads_subcategory=birads_sub,
        birads_source=birads_src,
    )


def load_annotations(reports_csv: Optional[Path]) -> Dict[str, List[Dict[str, str]]]:
    """Charge les rapports textuels MVKL, si fournis.

    Format attendu : colonnes ``patient_id, text, breast, context``.
    """
    out: Dict[str, List[Dict[str, str]]] = {}
    if reports_csv is None or not reports_csv.exists():
        return out
    df = pd.read_csv(reports_csv, dtype=str).fillna("")
    for _, row in df.iterrows():
        out.setdefault(str(row["patient_id"]), []).append(
            {
                "text": row.get("text", ""),
                "breast": row.get("breast", ""),
                "context": row.get("context", ""),
            }
        )
    return out


def main() -> None:
    from _bootstrap import bootstrap_console

    bootstrap_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path, help="Dossier des PNG MVKL bruts")
    parser.add_argument("--templates", type=Path, default=Path("data/raw/MVKL/templates"))
    parser.add_argument("--reports", type=Path, default=None, help="CSV des rapports textuels")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    templates = load_templates(args.templates)
    annotations = load_annotations(args.reports)

    rows: List[Dict[str, object]] = []
    for image_path in sorted(args.raw.rglob("*.png")):
        detection = detect_image(image_path, templates, annotations)
        rows.append(asdict(detection))

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"{len(df)} images traitées → {args.out}")
    if not df.empty:
        print(df["method"].value_counts().to_string())


if __name__ == "__main__":
    main()
