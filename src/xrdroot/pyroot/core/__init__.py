"""ROOT's core classes by ROOT's names: objects, files, histograms, graphs, functions, maths.

Each submodule is one family - :mod:`.objects` for ``TObject`` and the
``TAtt`` mixins, :mod:`.files` for ``TFile``, :mod:`.hists` for ``TH1`` and
its kin, :mod:`.funcs` for ``TF1`` - and this gathers their names into one
namespace, as ROOT has them in one. ``TMath`` is the :mod:`.tmath` module and
``Math`` the :mod:`.rmath` module, since both are namespaces in ROOT too.

:func:`draw_hook` and :func:`set_draw_hook` are how the graphics part of the
kit takes over ``Draw``; see :mod:`.hooks`.
"""

from __future__ import annotations

import importlib
from typing import Any

from .hooks import DRAWN, draw_hook, set_draw_hook
from .hooks import _remember as _remember  # the default hook, which the graphics restore

#: The families, in the order their names are gathered: a later one's wins.
FAMILIES = (
    "colors",
    "cformat",
    "messages",
    "objects",
    "collections",
    "strings",
    "timing",
    "system",
    "directories",
    "troot",
    "randoms",
    "vectors",
    "axes",
    "hists",
    "profiles",
    "stacks",
    "funcs",
    "fits",
    "fitters",
    "graphs",
    "efficiencies",
    "files",
    "rootns",
    "typedefs",
)


def _gather(namespace: dict[str, Any]) -> list[str]:
    """Each family's ``__all__``, put in ``namespace``, and the names gathered."""
    names: list[str] = []
    for family in FAMILIES:
        found = importlib.import_module(f"{__name__}.{family}")
        for name in found.__all__:
            namespace[name] = getattr(found, name)
            names.append(name)
    return names


from . import rmath as Math  # noqa: E402
from . import tmath as TMath  # noqa: E402
from .mathtools import Fit  # noqa: E402

__all__ = [*_gather(globals()), "TMath", "Math", "Fit", "draw_hook", "set_draw_hook", "DRAWN"]

from . import mathformula as _mathformula  # noqa: E402, F401 - ROOT::Math in formulas
from . import treelinks  # noqa: E402

treelinks.connect()
