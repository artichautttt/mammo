"""Amorçage commun aux scripts CLI du pipeline.

- rend le paquet ``src`` importable ;
- force la sortie console en UTF-8 (la console Windows cp1252 fait planter
  ``print`` sur le moindre accent ou flèche).

Chaque script appelle :func:`bootstrap_console` au début de sa fonction ``main``.
"""

from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_console() -> None:
    """Ajoute la racine du dépôt au ``sys.path`` et passe stdout/stderr en UTF-8."""
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")
