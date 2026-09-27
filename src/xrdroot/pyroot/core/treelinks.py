"""Where the trees plug into the core: the directory a tree goes in, and what ``Draw`` fills.

:mod:`xrdroot.pyroot.trees` keeps four hooks in its ``_base.hooks`` - the
directory a new tree is made in, the mapping ``Draw("x>>h")`` puts ``h``
in, how an xrdroot object is handed back, and what drawing is - and this
sets them to the core's own: ``gDirectory`` when it is a file being
written, ``gDirectory`` again as a mapping of name to object, the wrapper
registry, and :func:`~.hooks.draw_hook`. :func:`connect` is called when the
core is imported and again when the trees are, whichever comes second.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterator, MutableMapping
from typing import Any

from . import hooks
from .directories import current_directory
from .wrapping import unwrap, wrap

__all__: list[str] = []

#: The module the trees keep their hooks in.
BASE = "xrdroot.pyroot.trees._base"


def directory() -> Any:
    """The directory a tree made now goes into: the current one, if a file is being written."""
    here = current_directory()
    writable = getattr(here, "IsWritable", None)
    return here if writable is not None and writable() and hasattr(here, "_writable") else None


def _wrapper(obj: Any) -> bool:
    """Is ``obj`` one of the core's wrappers of an xrdroot object - handed to ``Draw`` unwrapped?"""
    from .efficiencies import TEfficiency
    from .graphs import TGraph
    from .hists import TH1

    return isinstance(obj, (TH1, TGraph, TEfficiency))


class Registry(MutableMapping[str, Any]):
    """``gDirectory`` as ``Draw`` sees it: xrdroot histograms by name, anything else as it is."""

    def __getitem__(self, name: str) -> Any:
        found = current_directory().FindObject(str(name))
        if found is None:
            raise KeyError(name)
        return unwrap(found) if _wrapper(found) else found

    def __setitem__(self, name: str, value: Any) -> None:
        made = wrap(value)
        here = current_directory()
        if here.FindObject(str(name)) is made:
            return
        if made is not value and hasattr(made, "SetName"):
            made.SetName(name)
        here.Append(made, True)

    def __delitem__(self, name: str) -> None:
        found = current_directory().FindObject(str(name))
        if found is None:
            raise KeyError(name)
        current_directory().Remove(found)

    def __iter__(self) -> Iterator[str]:
        return iter([obj.GetName() for obj in current_directory().GetList()])

    def __len__(self) -> int:
        return len(current_directory().GetList())


def connect() -> bool:
    """Set the trees' hooks to the core's, if the trees are there and have finished loading."""
    base = sys.modules.get(BASE)
    if base is None and importlib.util.find_spec("xrdroot.pyroot.trees") is not None:
        base = importlib.import_module(BASE)
    held = getattr(base, "hooks", None)
    if held is None:
        return False
    registry = Registry()
    held.directory = directory
    held.registry = lambda: registry
    held.wrap = wrap
    held.draw = hooks.draw_hook
    return True
