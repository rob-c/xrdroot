"""Renaming a model's nodes, as a workspace does when names would clash.

``Import(k, RenameAllNodes="workspace")`` gives every node of ``k`` that is
not a variable the suffix ``_workspace``, saying so for each, so that it can
sit beside nodes of the same name.
"""

from __future__ import annotations

from typing import Any

from .messages import INFO, log

__all__ = ["rename_all"]


def rename_all(top: Any, suffix: str, existing: dict[str, Any]) -> None:
    for node in list(top._walk()):
        if node.isFundamental():
            continue
        new = f"{node.GetName()}_{suffix}"
        log(
            None,
            INFO,
            "ObjectHandling",
            "RooWorkspace::import(w) Resolving name conflict in "
            f"workspace by changing name of imported node  {node.GetName()} to {new}",
        )
        node.SetName(new)
