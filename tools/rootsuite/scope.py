"""The committed scope: tests no Mac with Apple silicon runs, and why.

``scope.toml`` is a list of ``[[label]]`` tables, each a full-sentence
``reason`` and what it covers: ``ids``, globs over test ids (``*`` crosses
``/``); ``needs``, what discovery found a test needs (``dictionary``,
``executable``); or ``setup``, the kind of build step (``compiled-macro``).
A test under a label is OUT-OF-SCOPE and is not run; the first label that
covers it is the one it is counted under. Labels are only for what ROOT
itself cannot run on macOS ARM64 - CUDA, SYCL, X11, io_uring - never for
what xrdroot cannot do yet, dictionaries and compiled code included: that
is what the ratchet is for.
"""

from __future__ import annotations

import fnmatch
import importlib
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .model import Test

#: tomllib, or on Python 3.10 the tomli the dev dependencies bring.
_TOML = importlib.import_module("tomllib" if sys.version_info >= (3, 11) else "tomli")

__all__ = ["Label", "SCOPE", "load", "label_for"]

#: The labels kept in the repository beside this module.
SCOPE = Path(__file__).with_name("scope.toml")


@dataclass(frozen=True)
class Label:
    """One reason a group of tests is out of scope, and how to recognise them."""

    reason: str
    ids: tuple[str, ...] = ()
    needs: tuple[str, ...] = ()
    setup: tuple[str, ...] = ()

    def covers(self, test: Test) -> bool:
        return (
            any(fnmatch.fnmatchcase(test.id, pattern) for pattern in self.ids)
            or bool(set(self.needs) & set(test.needs))
            or test.setup_of in self.setup
        )


def load(path: Path = SCOPE) -> list[Label]:
    """The labels in a scope file, each checked to have a sentence for its reason."""
    with open(path, "rb") as stream:
        tables = _TOML.load(stream).get("label", [])
    labels = []
    for table in tables:
        reason = str(table.get("reason", "")).strip()
        if not (reason[:1].isupper() and reason.endswith(".")):
            raise ValueError(f"{path}: every scope reason must be a full sentence: {reason!r}")
        covers = (tuple(table.get(key, ())) for key in ("ids", "needs", "setup"))
        labels.append(Label(reason, *covers))
    return labels


def label_for(test: Test, labels: Iterable[Label]) -> Label | None:
    """The first label covering a test, or ``None`` when it is in scope."""
    return next((label for label in labels if label.covers(test)), None)
