"""Petit utilitaire : forcer la sortie console en UTF-8.

Sous Windows la console par défaut est en ``cp1252`` et ``print`` lève
``UnicodeEncodeError`` dès qu'un accent ou une flèche apparaît. On appelle
:func:`enable_utf8_stdout` en tête de chaque point d'entrée CLI.
"""

from __future__ import annotations

import sys


def enable_utf8_stdout() -> None:
    """Reconfigure ``stdout`` / ``stderr`` en UTF-8 si l'interpréteur le permet."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")
