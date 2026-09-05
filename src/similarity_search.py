"""Recherche des cas d'entraînement les plus proches par similarité cosinus sur
l'embedding concaténé 1024-D (CC ++ MLO).

    sim(A, B) = (A · B) / (‖A‖ · ‖B‖)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from .clip_model import DualViewCLIP
from .two_view_model import load_model

DEFAULT_K = 5


@dataclass
class SimilarCase:
    """Un cas voisin renvoyé par la recherche."""

    exam_id: str
    dataset: str
    birads_subcategory: str
    pathology: str
    description: str
    similarity: float


def cosine_similarity(query: np.ndarray, bank: np.ndarray) -> np.ndarray:
    """Similarité cosinus entre un vecteur ``query`` (D,) et une matrice
    ``bank`` (N, D). Retourne un vecteur (N,)."""
    q = query / (np.linalg.norm(query) + 1e-8)
    b = bank / (np.linalg.norm(bank, axis=1, keepdims=True) + 1e-8)
    return b @ q


class SimilarityIndex:
    """Banque d'embeddings des cas d'entraînement + métadonnées."""

    def __init__(self, embeddings: np.ndarray, metadata: pd.DataFrame) -> None:
        assert len(embeddings) == len(metadata)
        self.embeddings = embeddings.astype(np.float32)
        self.metadata = metadata.reset_index(drop=True)

    # -- construction / persistance -----------------------------------------

    @classmethod
    def build(
        cls,
        bank_csv: str | Path,
        model: DualViewCLIP,
        image_root: Optional[str | Path] = None,
        limit: Optional[int] = None,
    ) -> "SimilarityIndex":
        """Encode toutes les paires du CSV ``bank_csv`` avec ``model``."""
        from PIL import Image

        df = pd.read_csv(bank_csv)
        if "is_augmented" in df.columns:
            df = df[~df["is_augmented"].astype(bool)]  # cas réels uniquement
        df = df.reset_index(drop=True)
        if limit:
            df = df.head(limit)

        device = next(model.parameters()).device
        root = Path(image_root) if image_root else None
        vectors: List[np.ndarray] = []
        kept_rows: List[int] = []
        for i, row in df.iterrows():
            try:
                pc, pm = Path(row["path_cc"]), Path(row["path_mlo"])
                if root:
                    pc, pm = root / pc, root / pm
                cc = model.preprocess(Image.open(pc).convert("RGB")).unsqueeze(0).to(device)
                mlo = model.preprocess(Image.open(pm).convert("RGB")).unsqueeze(0).to(device)
                with torch.no_grad():
                    vec = model.embedding(cc, mlo).squeeze(0).cpu().numpy()
                vectors.append(vec)
                kept_rows.append(i)
            except (FileNotFoundError, OSError):
                continue
        return cls(np.stack(vectors), df.loc[kept_rows])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        np.save(path.with_suffix(".npy"), self.embeddings)
        self.metadata.to_csv(path.with_suffix(".meta.csv"), index=False)

    @classmethod
    def load(cls, path: str | Path) -> "SimilarityIndex":
        path = Path(path)
        embeddings = np.load(path.with_suffix(".npy"))
        metadata = pd.read_csv(path.with_suffix(".meta.csv"))
        return cls(embeddings, metadata)

    # -- requête -----------------------------------------------------------

    def query(self, embedding: np.ndarray, k: int = DEFAULT_K) -> List[SimilarCase]:
        """Retourne les ``k`` cas les plus proches, triés par similarité
        décroissante."""
        sims = cosine_similarity(np.asarray(embedding, dtype=np.float32), self.embeddings)
        order = np.argsort(-sims)[:k]
        results: List[SimilarCase] = []
        for idx in order:
            row = self.metadata.iloc[int(idx)]
            results.append(
                SimilarCase(
                    exam_id=str(row.get("exam_id", "")),
                    dataset=str(row.get("dataset", "")),
                    birads_subcategory=str(row.get("birads_subcategory", "")),
                    pathology=str(row.get("pathology", "")),
                    description=str(row.get("description", "")),
                    similarity=round(float(sims[idx]), 4),
                )
            )
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--bank", required=True, help="CSV des cas d'entraînement")
    parser.add_argument("--out", required=True, help="Préfixe de sortie de l'index")
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    model = load_model(args.model)
    index = SimilarityIndex.build(args.bank, model, args.image_root, args.limit)
    index.save(args.out)
    print(f"Index construit : {len(index.metadata)} cas → {args.out}.npy")


if __name__ == "__main__":
    main()
