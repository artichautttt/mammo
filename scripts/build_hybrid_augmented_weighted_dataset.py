"""Étape 5 du pipeline — dataset hybride augmenté + pondéré.

Problème : forte asymétrie de classes (ex. 4A ≈ 429 cas contre classe 1 ≈ 4 cas).

Stratégie :

- **Augmentation virtuelle** : chaque classe minoritaire est complétée jusqu'à
  ``DEFAULT_TARGET_PER_CLASS = 429`` cas, sauf la classe « 1 » plafonnée à 40
  (``CLASS_TARGETS = {"1": 40}``). Aucune image n'est copiée sur le disque : on
  stocke uniquement les **paramètres d'augmentation** dans le CSV, et le
  ``DataLoader`` les applique à la volée (``_maybe_apply_virtual_augmentation``).
- Les paramètres sont générés de façon **déterministe** à partir d'un seed
  reproductible : ``random.Random(RANDOM_STATE + source_index * 1009 +
  augmentation_index * 9173)``.
- **Pondération douce** (``sample_weight``) en complément pour les classes
  résiduellement sous-représentées.

Distribution finale cible : 0→429, 1→40, 2→429, 3→429, 4A→429, 4B→429, 4C→429,
5→429.

Sortie : ``combined_hybrid_augmented_weighted.csv``.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict

import pandas as pd

RANDOM_STATE = 42
DEFAULT_TARGET_PER_CLASS = 429
CLASS_TARGETS: Dict[str, int] = {"1": 40}

# bornes des transformations
ROTATION_DEG = 8.0
TRANSLATION_FRAC = 0.025
SCALE_MIN, SCALE_MAX = 0.96, 1.04
PHOTOMETRIC_FRAC = 0.15          # luminosité / contraste / netteté ±10-15 %
NOISE_SIGMAS = (0.0, 1.5, 2.5, 3.5)


def _augmentation_params(source_index: int, augmentation_index: int) -> dict:
    """Paramètres déterministes pour la ``augmentation_index``-ème copie virtuelle
    de la ligne source ``source_index``."""
    rng = random.Random(RANDOM_STATE + source_index * 1009 + augmentation_index * 9173)
    return {
        "rotation_deg": round(rng.uniform(-ROTATION_DEG, ROTATION_DEG), 3),
        "translate_x": round(rng.uniform(-TRANSLATION_FRAC, TRANSLATION_FRAC), 4),
        "translate_y": round(rng.uniform(-TRANSLATION_FRAC, TRANSLATION_FRAC), 4),
        "scale": round(rng.uniform(SCALE_MIN, SCALE_MAX), 4),
        "brightness": round(1.0 + rng.uniform(-PHOTOMETRIC_FRAC, PHOTOMETRIC_FRAC), 4),
        "contrast": round(1.0 + rng.uniform(-PHOTOMETRIC_FRAC, PHOTOMETRIC_FRAC), 4),
        "sharpness": round(1.0 + rng.uniform(-0.10, 0.10), 4),
        "noise_sigma": rng.choice(NOISE_SIGMAS),
        "hflip": False,  # jamais de miroir : changerait la latéralité
    }


def _target_for_class(birads_class: str) -> int:
    return CLASS_TARGETS.get(str(birads_class), DEFAULT_TARGET_PER_CLASS)


def build_hybrid_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Retourne le DataFrame augmenté virtuellement + colonne ``sample_weight``."""
    df = df.reset_index(drop=True).copy()
    df["birads_class"] = df["birads_subcategory"].astype(str).str.strip().str.upper()
    df["is_augmented"] = False
    df["augmentation_index"] = 0
    df["augmentation_params"] = "{}"
    df["source_row"] = df.index

    augmented_rows = []
    for birads_class, group in df.groupby("birads_class"):
        target = _target_for_class(birads_class)
        current = len(group)
        if current >= target:
            continue
        needed = target - current
        base = group.reset_index(drop=True)
        for k in range(needed):
            src = base.iloc[k % current]
            source_index = int(src["source_row"])
            aug_index = (k // current) + 1
            new_row = src.to_dict()
            new_row["is_augmented"] = True
            new_row["augmentation_index"] = aug_index
            new_row["augmentation_params"] = json.dumps(
                _augmentation_params(source_index, aug_index)
            )
            augmented_rows.append(new_row)

    out = pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)

    # pondération douce : fréquence inverse normalisée (moyenne = 1.0)
    counts = out["birads_class"].value_counts()
    inv = {c: 1.0 / n for c, n in counts.items()}
    mean_inv = sum(inv.values()) / len(inv)
    out["sample_weight"] = out["birads_class"].map(lambda c: round(inv[c] / mean_inv, 4))

    return out


def main() -> None:
    from _bootstrap import bootstrap_console

    bootstrap_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    df = pd.read_csv(args.in_path)
    out = build_hybrid_dataset(df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"{len(df)} → {len(out)} cas (dont {out['is_augmented'].sum()} virtuels)")
    print(out.groupby("birads_class").size().to_string())


if __name__ == "__main__":
    main()
