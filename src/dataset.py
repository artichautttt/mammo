"""Dataset PyTorch bi-vues + utilitaires d'échantillonnage.

Le CSV consommé est ``combined_hybrid_augmented_weighted.csv`` (étape 5). Les
lignes marquées ``is_augmented`` portent des ``augmentation_params`` JSON qui
sont appliqués **à la volée** (aucune image augmentée n'existe sur le disque).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset, WeightedRandomSampler

from .labels import (
    ACR_DENSITY_CLASSES,
    BIRADS_CLASSES,
    CBIS_CALC_DISTRIBUTION,
    CBIS_CALC_TYPE,
    CBIS_MASS_MARGINS,
    CBIS_MASS_SHAPE,
    IGNORE_INDEX,
    MVKL_MANIFESTATIONS,
)

_CATEGORICAL_VOCAB = {
    "birads": BIRADS_CLASSES,
    "acr_density": ACR_DENSITY_CLASSES,
    "mass_shape": CBIS_MASS_SHAPE,
    "mass_margins": CBIS_MASS_MARGINS,
    "calc_type": CBIS_CALC_TYPE,
    "calc_distribution": CBIS_CALC_DISTRIBUTION,
}


def _to_index(value: object, vocab: List[str]) -> int:
    """Index de ``value`` dans ``vocab`` (comparaison insensible à la casse),
    ou ``IGNORE_INDEX`` si absent / manquant."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return IGNORE_INDEX
    token = str(value).strip().upper().replace(" ", "_")
    upper_vocab = [v.upper() for v in vocab]
    return upper_vocab.index(token) if token in upper_vocab else IGNORE_INDEX


def _maybe_apply_virtual_augmentation(img: Image.Image, params: dict) -> Image.Image:
    """Applique les transformations géométriques + photométriques déterministes
    décrites par ``params`` (voir ``build_hybrid_augmented_weighted_dataset``)."""
    if not params:
        return img

    w, h = img.size
    # échelle + translation via une transformation affine
    scale = float(params.get("scale", 1.0))
    tx = float(params.get("translate_x", 0.0)) * w
    ty = float(params.get("translate_y", 0.0)) * h
    if scale != 1.0 or tx or ty:
        a = 1.0 / scale
        img = img.transform(
            (w, h), Image.AFFINE, (a, 0, -tx, 0, a, -ty), resample=Image.BILINEAR
        )
    rot = float(params.get("rotation_deg", 0.0))
    if rot:
        img = img.rotate(rot, resample=Image.BILINEAR)

    for key, enhancer in (
        ("brightness", ImageEnhance.Brightness),
        ("contrast", ImageEnhance.Contrast),
        ("sharpness", ImageEnhance.Sharpness),
    ):
        factor = float(params.get(key, 1.0))
        if factor != 1.0:
            img = enhancer(img).enhance(factor)

    sigma = float(params.get("noise_sigma", 0.0))
    if sigma > 0:
        arr = np.asarray(img).astype(np.float32)
        arr += np.random.default_rng(0).normal(0.0, sigma, arr.shape)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    return img


class DualViewDataset(Dataset):
    """Retourne ``(img_cc, img_mlo, targets)`` où ``targets`` est un dict
    ``{nom_tache: LongTensor scalaire}`` avec ``IGNORE_INDEX`` pour les tâches
    non annotées sur la source de la ligne."""

    def __init__(
        self,
        csv_path: str | Path,
        preprocess: Callable[[Image.Image], torch.Tensor],
        split: Optional[str] = None,
        image_root: Optional[str | Path] = None,
    ) -> None:
        df = pd.read_csv(csv_path)
        if split is not None and "split" in df.columns:
            df = df[df["split"].astype(str) == split]
        self.df = df.reset_index(drop=True)
        self.preprocess = preprocess
        self.image_root = Path(image_root) if image_root else None

    def __len__(self) -> int:
        return len(self.df)

    # -- images -------------------------------------------------------------

    def _load_image(self, path: str, params: dict) -> torch.Tensor:
        full = Path(path)
        if self.image_root and not full.is_absolute():
            full = self.image_root / full
        img = Image.open(full).convert("RGB")
        img = _maybe_apply_virtual_augmentation(img, params)
        return self.preprocess(img)

    # -- targets ----------------------------------------------------------

    def _targets(self, row: pd.Series) -> Dict[str, torch.Tensor]:
        targets: Dict[str, int] = {
            "birads": _to_index(row.get("birads_subcategory"), BIRADS_CLASSES),
            "acr_density": _to_index(row.get("acr_density"), ACR_DENSITY_CLASSES),
        }
        for task in ("mass_shape", "mass_margins", "calc_type", "calc_distribution"):
            targets[task] = _to_index(row.get(task), _CATEGORICAL_VOCAB[task])
        for name in MVKL_MANIFESTATIONS:
            col = f"mf_{name}"
            value = row.get(col)
            if value is None or (isinstance(value, float) and np.isnan(value)):
                targets[f"mf_{name}"] = IGNORE_INDEX
            else:
                targets[f"mf_{name}"] = int(float(value) >= 0.5)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in targets.items()}

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        params: dict = {}
        raw = row.get("augmentation_params")
        if isinstance(raw, str) and raw not in ("", "{}"):
            try:
                params = json.loads(raw)
            except json.JSONDecodeError:
                params = {}
        img_cc = self._load_image(str(row["path_cc"]), params)
        img_mlo = self._load_image(str(row["path_mlo"]), params)
        return img_cc, img_mlo, self._targets(row)


# --- échantillonnage ---------------------------------------------------------

def build_weighted_sampler(
    csv_path: str | Path,
    main_task_col: str = "birads_subcategory",
    max_sample_weight: float = 8.0,
    seed: int = 42,
) -> WeightedRandomSampler:
    """``WeightedRandomSampler`` sur la tâche principale (BI-RADS) : poids par
    fréquence inverse normalisée, plafonnés à ``max_sample_weight``, générateur
    à seed fixe pour la reproductibilité."""
    df = pd.read_csv(csv_path)
    labels = df[main_task_col].astype(str).str.strip().str.upper()
    counts = labels.value_counts()
    inv = {c: 1.0 / n for c, n in counts.items()}
    mean_inv = sum(inv.values()) / len(inv)
    weights = labels.map(lambda c: min(inv[c] / mean_inv, max_sample_weight)).to_numpy()

    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


def compute_class_weights(csv_path: str | Path, col: str = "birads_subcategory") -> torch.Tensor:
    """Poids de classe (fréquence inverse normalisée) pour la tâche BI-RADS,
    aligné sur l'ordre de :data:`src.labels.BIRADS_CLASSES`."""
    df = pd.read_csv(csv_path)
    labels = df[col].astype(str).str.strip().str.upper()
    counts = labels.value_counts().to_dict()
    weights = []
    for cls in BIRADS_CLASSES:
        n = counts.get(cls.upper(), 0)
        weights.append(0.0 if n == 0 else 1.0 / n)
    w = torch.tensor(weights, dtype=torch.float)
    return w / w[w > 0].mean()
