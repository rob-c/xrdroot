"""``%load_ext xrdroot``: ROOT's prompt commands and a few magics, in IPython or Jupyter.

    In [1]: %load_ext xrdroot
    In [2]: %root_open tests/data/graphs.root
    In [3]: .ls
    In [4]: %root_ls
    In [5]: %%root_macro
       ...: h = Histogram.book("h", (10, 0, 1))

Loading it puts :data:`~xrdroot.gROOT`, :data:`~xrdroot.gDirectory` and the
rest of the shell's names in the user's namespace, reads ``.ls``-style lines
as :mod:`.dot` translates them, and adds three magics: ``%root_ls [dir]``,
``%root_open FILE`` - which opens the file as the next ``_fileN`` - and the
cell magic ``%%root_macro``, which runs its cell the way ``.x`` runs a macro,
with the names preloaded and apart from the notebook's own.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..session import gROOT, preloaded
from .dot import transform

__all__ = ["load", "magics"]


def magics(ipython: Any) -> dict[str, tuple[str, Callable[..., Any]]]:
    """The magics, by name, each with its kind - ``line`` or ``cell`` - for ``ipython``."""

    def root_ls(line: str) -> None:
        """``%root_ls [dir]``: what the current directory, or ``dir``, holds."""
        print(gROOT.ls(line.strip() or None))

    def root_open(line: str) -> Any:
        """``%root_open FILE``: open a file, go into it, and name it ``_fileN``."""
        index = 0
        while f"_file{index}" in ipython.user_ns:
            index += 1
        name = f"_file{index}"
        opened = gROOT.open(line.strip())
        ipython.push({name: opened})
        print(f"{name} = {opened.name}")
        return opened

    def root_macro(line: str, cell: str) -> dict[str, Any]:
        """``%%root_macro``: run the cell as a macro, and hand back what it defined."""
        return gROOT.run(cell, filename="<root_macro>")

    return {
        "root_ls": ("line", root_ls),
        "root_open": ("line", root_open),
        "root_macro": ("cell", root_macro),
    }


def load(ipython: Any) -> None:
    """Load the extension into ``ipython``: the names, the dot-commands and the magics."""
    ipython.push(preloaded())
    if transform not in ipython.input_transformers_cleanup:
        ipython.input_transformers_cleanup.append(transform)
    for name, (kind, magic) in magics(ipython).items():
        ipython.register_magic_function(magic, magic_kind=kind, magic_name=name)
