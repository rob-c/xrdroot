"""``ROOT``: where a translated macro finds ROOT's names - ``ROOT.TH1F``, ``ROOT.gRandom``.

Every name a macro uses that it did not declare itself is ROOT's, and the
translation spells it ``ROOT.<name>``. ``ROOT`` is the ROOT dialect's host, a
:class:`~xrdroot.pycxx.runtime.namespace.HostNamespace` that looks the name
up in :mod:`xrdroot.pyroot` the first time it is wanted - so a translation
can be imported, read and kept without pyroot loading - or in whatever
namespace :meth:`~xrdroot.pycxx.runtime.namespace.HostNamespace.bind` was
handed, which is how a test runs a macro against a small fake ROOT of its
own. ``ROOT.std`` is the standard library's containers, from
:mod:`xrdroot.pyroot.stl`, and ``ROOT.TestSupport`` is ROOT's unit tests'
check of the diagnostics a test prints.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from ...pycxx.runtime.namespace import DECLARED, HostNamespace

__all__ = ["RootProxy", "ROOT", "DECLARED", "DEFAULT", "USED"]

#: The module ROOT's names come from when nothing else was bound.
DEFAULT = "xrdroot.pyroot"
#: Namespaces whose names a macro uses unqualified - every RooFit tutorial says
#: ``using namespace RooFit``, TMVA's say ``using namespace TMVA``, dataframe ones
#: ``using namespace ROOT::RDF`` - looked in when
#: ROOT itself has no such name. A macro without the ``using`` would not have
#: compiled, so looking there is never wrong. ``RooStats::HistFactory``'s functions
#: are called unqualified by argument-dependent lookup - ``MakeModelAndMeasurementFast(meas)``
#: of a ``HistFactory::Measurement`` - which finds them there too. ``ROOT::Math``'s
#: vectors come last of all: ``using namespace ROOT::Math`` and then ``XYZTVector``.
USED = ("RooFit", "RooStats", "RooStats.HistFactory", "TMVA", "TMVA.Experimental", "RDF",
        "VecOps", "Math")  # VecOps late: RooFit's Range is not VecOps' Range


def _pyroot() -> Any:
    """:mod:`xrdroot.pyroot`, imported the first time a name is wanted - and only then."""
    loaded = sys.modules.get(DEFAULT)  # every name a macro uses comes this way: no import
    return loaded if loaded is not None else importlib.import_module(DEFAULT)


def _std() -> Any:
    """``ROOT.std``: the standard library, as pyroot registers ROOT's types with it."""
    return importlib.import_module(f"{DEFAULT}.stl").std


def _test_support() -> Any:
    """``ROOT::TestSupport``, which ROOT's unit tests are written against."""
    from .diags import TestSupport

    return TestSupport


class RootProxy(HostNamespace):
    """ROOT's namespace, looked up late: :mod:`xrdroot.pyroot`, or whatever is bound.

    It is plain C++'s host, configured: pyroot is the namespace, :data:`USED`
    are the namespaces a macro says ``using`` for, ``TestSupport`` is provided
    beside GoogleTest, and ``std`` is pyroot's.
    """

    def __init__(self) -> None:
        super().__init__(_pyroot, USED, {"TestSupport": _test_support}, _std)


ROOT = RootProxy()
