"""Fonction de perte multi-tâches.

Cross-entropy par tâche, avec :

- ``ignore_index = -100`` pour les labels absents (colonnes non renseignées par
  la source) ;
- une pondération de classe optionnelle par tâche ;
- la perte totale = **somme** des pertes par tâche.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn

from .labels import IGNORE_INDEX


class MultiTaskLoss(nn.Module):
    """Somme de cross-entropies, une par tâche.

    Args:
        class_weights: dict optionnel ``{nom_tache: Tensor[nb_classes]}``.
        task_weights: pondération scalaire optionnelle par tâche (défaut 1.0).
    """

    def __init__(
        self,
        class_weights: Optional[Mapping[str, torch.Tensor]] = None,
        task_weights: Optional[Mapping[str, float]] = None,
    ) -> None:
        super().__init__()
        self.task_weights = dict(task_weights or {})
        self._criteria: Dict[str, nn.CrossEntropyLoss] = {}
        class_weights = class_weights or {}
        self._class_weights = {k: v for k, v in class_weights.items()}

    def _criterion(self, task: str, device: torch.device) -> nn.CrossEntropyLoss:
        if task not in self._criteria:
            weight = self._class_weights.get(task)
            self._criteria[task] = nn.CrossEntropyLoss(
                weight=weight.to(device) if weight is not None else None,
                ignore_index=IGNORE_INDEX,
            )
        return self._criteria[task]

    def forward(
        self,
        logits: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """Retourne ``(perte_totale, {nom_tache: valeur_detaillee})``.

        Une tâche dont tous les labels du batch valent ``ignore_index`` ne
        contribue pas (sa perte serait ``nan``).
        """
        total = torch.zeros((), device=next(iter(logits.values())).device)
        details: Dict[str, float] = {}
        for task, task_logits in logits.items():
            if task not in targets:
                continue
            target = targets[task].long()
            if (target != IGNORE_INDEX).sum() == 0:
                continue
            loss = self._criterion(task, task_logits.device)(task_logits, target)
            loss = loss * float(self.task_weights.get(task, 1.0))
            total = total + loss
            details[task] = float(loss.detach())
        return total, details
