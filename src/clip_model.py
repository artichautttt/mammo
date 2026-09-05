"""Modèle ``DualViewCLIP`` — encodeur CLIP gelé partagé + fusion des deux vues +
têtes multi-tâches.
"""

from __future__ import annotations

from typing import Dict, Mapping

import torch
import torch.nn as nn

try:
    import open_clip
except Exception:  # pragma: no cover - dépend de l'environnement
    open_clip = None  # type: ignore

from .labels import TASK_NUM_CLASSES

CLIP_MODEL_NAME = "ViT-B-16"
CLIP_PRETRAINED = "laion2b_s34b_b88k"
CLIP_EMBED_DIM = 512
FUSED_DIM = 512


class DualViewCLIP(nn.Module):
    """Encode séparément les vues CC et MLO via un backbone CLIP ViT-B/16 **gelé**,
    concatène les deux embeddings (512 + 512 = 1024), les projette en un vecteur
    fusionné de 512, puis applique une tête linéaire par tâche.

    Args:
        tasks: dictionnaire ``{nom_tache: nb_classes}``. Par défaut
            :data:`src.labels.TASK_NUM_CLASSES`.
        clip_model_name / clip_pretrained: identifiants ``open_clip``.
        freeze_backbone: garder le backbone gelé (par défaut ``True``).
    """

    def __init__(
        self,
        tasks: Mapping[str, int] | None = None,
        clip_model_name: str = CLIP_MODEL_NAME,
        clip_pretrained: str = CLIP_PRETRAINED,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.tasks: Dict[str, int] = dict(tasks or TASK_NUM_CLASSES)

        if open_clip is None:
            raise ImportError("open_clip_torch est requis pour instancier DualViewCLIP")

        self.clip, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=clip_pretrained
        )
        self.clip_embed_dim = getattr(self.clip.visual, "output_dim", CLIP_EMBED_DIM)

        if freeze_backbone:
            for param in self.clip.parameters():
                param.requires_grad_(False)
            self.clip.eval()

        self.fusion = nn.Sequential(
            nn.Linear(self.clip_embed_dim * 2, FUSED_DIM),
            nn.LayerNorm(FUSED_DIM),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.heads = nn.ModuleDict(
            {name: nn.Linear(FUSED_DIM, n_classes) for name, n_classes in self.tasks.items()}
        )

    # -- encodage --------------------------------------------------------------

    def encode_view(self, images: torch.Tensor) -> torch.Tensor:
        """Encode un batch d'images (déjà prétraitées) → embeddings L2-normalisés."""
        with torch.no_grad():
            features = self.clip.encode_image(images)
        return features.float()

    def fuse(self, cc: torch.Tensor, mlo: torch.Tensor) -> torch.Tensor:
        """Concatène et projette les deux embeddings de vue."""
        return self.fusion(torch.cat([cc, mlo], dim=-1))

    # -- forward -------------------------------------------------------------

    def forward(
        self, img_cc: torch.Tensor, img_mlo: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Retourne un dict ``{nom_tache: logits}``."""
        emb_cc = self.encode_view(img_cc)
        emb_mlo = self.encode_view(img_mlo)
        fused = self.fuse(emb_cc, emb_mlo)
        return {name: head(fused) for name, head in self.heads.items()}

    def embedding(self, img_cc: torch.Tensor, img_mlo: torch.Tensor) -> torch.Tensor:
        """Vecteur concaténé brut de 1024-D (CC ++ MLO) utilisé par la recherche
        de similarité — indépendant des têtes."""
        emb_cc = self.encode_view(img_cc)
        emb_mlo = self.encode_view(img_mlo)
        return torch.cat([emb_cc, emb_mlo], dim=-1)
