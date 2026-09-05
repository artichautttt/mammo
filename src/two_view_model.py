"""Inférence bi-vues + CLI ``predict``.

Usage :

    python src/two_view_model.py predict \
        --model best_two_view_model.pt \
        --left  chemin/vers/MLO.png \
        --right chemin/vers/CC.png
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from PIL import Image

from .clip_model import DualViewCLIP
from .labels import BIRADS_CLASSES, ACR_DENSITY_CLASSES, MVKL_MANIFESTATIONS

#: Seuil au-delà duquel une manifestation est considérée « active ».
MANIFESTATION_THRESHOLD = 0.5


@dataclass
class Prediction:
    """Sortie complète de l'inférence pour un cas."""

    birads_class: str
    birads_probs: Dict[str, float]
    acr_density_class: str
    acr_density_probs: Dict[str, float]
    manifestations: Dict[str, float]                    # nom → probabilité
    active_manifestations: List[str] = field(default_factory=list)
    morphology: Dict[str, str] = field(default_factory=dict)
    embedding: List[float] = field(default_factory=list)  # vecteur 1024-D CC++MLO

    def to_json(self) -> str:
        payload = {k: v for k, v in self.__dict__.items() if k != "embedding"}
        payload["embedding_dim"] = len(self.embedding)
        return json.dumps(payload, indent=2, ensure_ascii=False)


def load_model(model_path: str | Path, device: torch.device | None = None) -> DualViewCLIP:
    """Instancie ``DualViewCLIP`` et y charge les poids du checkpoint."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    tasks = checkpoint.get("tasks")
    model = DualViewCLIP(tasks=tasks).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()
    return model


def _argmax_probs(logits: torch.Tensor, vocab: List[str]) -> tuple[str, Dict[str, float]]:
    probs = F.softmax(logits, dim=-1).squeeze(0)
    mapping = {cls: round(float(p), 4) for cls, p in zip(vocab, probs.tolist())}
    top = vocab[int(torch.argmax(probs))]
    return top, mapping


@torch.no_grad()
def predict(
    model: DualViewCLIP,
    left_mlo: str | Path,
    right_cc: str | Path,
    device: torch.device | None = None,
) -> Prediction:
    """Prédit BI-RADS, densité, manifestations et l'embedding 1024-D pour une
    paire ``(MLO, CC)``."""
    device = device or next(model.parameters()).device
    img_mlo = model.preprocess(Image.open(left_mlo).convert("RGB")).unsqueeze(0).to(device)
    img_cc = model.preprocess(Image.open(right_cc).convert("RGB")).unsqueeze(0).to(device)

    logits = model(img_cc, img_mlo)
    embedding = model.embedding(img_cc, img_mlo).squeeze(0).cpu().tolist()

    birads_class, birads_probs = _argmax_probs(logits["birads"], BIRADS_CLASSES)
    acr_class, acr_probs = _argmax_probs(logits["acr_density"], ACR_DENSITY_CLASSES)

    manifestations: Dict[str, float] = {}
    for name in MVKL_MANIFESTATIONS:
        head = f"mf_{name}"
        if head in logits:
            p_active = float(F.softmax(logits[head], dim=-1).squeeze(0)[1])
            manifestations[name] = round(p_active, 4)
    active = [n for n, p in manifestations.items() if p >= MANIFESTATION_THRESHOLD]

    morphology: Dict[str, str] = {}
    for task in ("mass_shape", "mass_margins", "calc_type", "calc_distribution"):
        if task in logits:
            probs = F.softmax(logits[task], dim=-1).squeeze(0)
            from .dataset import _CATEGORICAL_VOCAB

            vocab = _CATEGORICAL_VOCAB[task]
            morphology[task] = vocab[int(torch.argmax(probs))]

    return Prediction(
        birads_class=birads_class,
        birads_probs=birads_probs,
        acr_density_class=acr_class,
        acr_density_probs=acr_probs,
        manifestations=manifestations,
        active_manifestations=active,
        morphology=morphology,
        embedding=embedding,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("predict", help="Prédire à partir d'une paire MLO/CC")
    p.add_argument("--model", required=True)
    p.add_argument("--left", required=True, help="Image MLO")
    p.add_argument("--right", required=True, help="Image CC")
    args = parser.parse_args()

    if args.command == "predict":
        model = load_model(args.model)
        result = predict(model, args.left, args.right)
        print(result.to_json())


if __name__ == "__main__":
    main()
