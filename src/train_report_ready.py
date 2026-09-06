"""Boucle d'entraînement du modèle ``DualViewCLIP``.

- ``WeightedRandomSampler`` sur la tâche principale BI-RADS ;
- perte multi-tâches (somme des cross-entropies, ``ignore_index=-100``) ;
- 15 à 20 epochs max (au-delà : surapprentissage observé) ;
- le meilleur checkpoint est conservé **entre les epochs 12 et 15**.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .clip_model import DualViewCLIP
from .dataset import (
    DualViewDataset,
    build_weighted_sampler,
    compute_class_weights,
)
from .losses import MultiTaskLoss

BEST_CKPT_EPOCH_MIN = 12
BEST_CKPT_EPOCH_MAX = 15


def _move_targets(targets: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in targets.items()}


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> float:
    """Une passe complète. Retourne la perte moyenne."""
    model.train(train)
    model.clip.eval()  # le backbone reste gelé/en eval quoi qu'il arrive
    total, n_batches = 0.0, 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for img_cc, img_mlo, targets in tqdm(loader, leave=False):
            img_cc, img_mlo = img_cc.to(device), img_mlo.to(device)
            targets = _move_targets(targets, device)
            logits = model(img_cc, img_mlo)
            loss, _details = criterion(logits, targets)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += float(loss.detach())
            n_batches += 1
    return total / max(n_batches, 1)


def train(
    dataset_csv: str,
    out_path: str,
    epochs: int = 18,
    batch_size: int = 16,
    lr: float = 1e-4,
    val_fraction: float = 0.15,
    image_root: str | None = None,
    seed: int = 42,
) -> None:
    """Entraîne le modèle et écrit le meilleur checkpoint dans ``out_path``."""
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DualViewCLIP().to(device)

    full = DualViewDataset(dataset_csv, preprocess=model.preprocess, image_root=image_root)
    n_val = int(len(full) * val_fraction)
    n_train = len(full) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full, [n_train, n_val], generator=generator)

    sampler = build_weighted_sampler(dataset_csv, seed=seed)
    # le sampler porte sur le dataset complet ; on le restreint aux indices train
    train_indices = set(train_set.indices)
    sub_weights = torch.as_tensor(
        [sampler.weights[i] for i in train_set.indices], dtype=torch.double
    )
    train_sampler = torch.utils.data.WeightedRandomSampler(
        sub_weights, num_samples=len(sub_weights), replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    class_weights = {"birads": compute_class_weights(dataset_csv).to(device)}
    criterion = MultiTaskLoss(class_weights=class_weights)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4
    )

    ckpt_dir = Path(out_path).parent
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    best_state = None
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        print(f"epoch {epoch:02d} | train {train_loss:.4f} | val {val_loss:.4f}")

        torch.save(
            {"epoch": epoch, "model_state": model.state_dict(), "tasks": model.tasks},
            ckpt_dir / f"epoch_{epoch:02d}.pt",
        )

        # on ne retient le "meilleur" que dans la fenêtre [12, 15]
        if BEST_CKPT_EPOCH_MIN <= epoch <= BEST_CKPT_EPOCH_MAX and val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            print(f"  ↳ nouveau meilleur checkpoint (val={val_loss:.4f})")

    if best_state is None:  # repli si epochs < 12
        best_state = model.state_dict()
    torch.save(
        {"model_state": best_state, "tasks": model.tasks, "val_loss": best_val},
        out_path,
    )
    print(f"Meilleur modèle → {out_path}")


def main() -> None:
    from .console import enable_utf8_stdout

    enable_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", default="best_two_view_model.pt")
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-root", default=None)
    args = parser.parse_args()

    train(
        dataset_csv=args.dataset,
        out_path=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        image_root=args.image_root,
    )


if __name__ == "__main__":
    main()
