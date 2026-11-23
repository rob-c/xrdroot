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
from functools import partial
from typing import Any

from ...pycxx.runtime.namespace import DECLARED, HostNamespace
from ..typedefs import TYPEDEFS

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
USED = ("RooFit", "RooStats", "RooStats.HistFactory", "TMVA", "TMVA.DNN",
        "TMVA.Experimental", "RDF", "VecOps", "Math")  # VecOps late: RooFit's Range is not
# VecOps' Range; TMVA.DNN before TMVA.Experimental: a DNN test says ``using namespace TMVA::DNN``


def _pyroot() -> Any:
    """:mod:`xrdroot.pyroot`, imported the first time a name is wanted - and only then."""
    loaded = sys.modules.get(DEFAULT)  # every name a macro uses comes this way: no import
    return loaded if loaded is not None else importlib.import_module(DEFAULT)


def _std() -> Any:
    """``ROOT.std``: the standard library, as pyroot registers ROOT's types with it."""
    return importlib.import_module(f"{DEFAULT}.stl").std


def _application() -> Any:
    """``gApplication``, the ``root`` program a macro runs in."""
    from .application import gApplication

    return gApplication


def _application_class() -> Any:
    from .application import TApplication

    return TApplication


def _test_support() -> Any:
    """``ROOT::TestSupport``, which ROOT's unit tests are written against."""
    from .diags import TestSupport

    return TestSupport


#: The ``::testing::Test`` members a test asks unqualified about the test being run.
#: A test body says ``HasFailure()`` and reads the fixture's inherited static;
#: inside a lambda or a local class of one there is no ``self`` to read it from,
#: and the translation reaches it as ``ROOT.HasFailure`` - so it is provided here.
GTEST_STATE = ("HasFailure", "HasFatalFailure", "HasNonfatalFailure", "IsSkipped")


def _gtest_state(name: str) -> Any:
    """One of :data:`GTEST_STATE`, GoogleTest's own answer for the running test."""
    from ...pycxx.runtime.gtest.fixtures import Test

    return getattr(Test, name)


class RootProxy(HostNamespace):
    """ROOT's namespace, looked up late: :mod:`xrdroot.pyroot`, or whatever is bound.

    It is plain C++'s host, configured: pyroot is the namespace, :data:`USED`
    are the namespaces a macro says ``using`` for, ``TestSupport`` and the
    GoogleTest state its unit tests read are provided beside GoogleTest, and
    ``std`` is pyroot's.
    """

    def __init__(self) -> None:
        # ROOT's typedefs called like functions, ``Long64_t(x)``, are casts.
        casts = {name: _cast(target) for name, target in TYPEDEFS.items()}
        state = {name: partial(_gtest_state, name) for name in GTEST_STATE}
        #: ``gCling`` is ROOT's cling interpreter, which this kit does not have - its interpreter is
        #: ``gInterpreter`` - so it answers with nothing, which is what ROOT's own unit test asserts
        #: before an interpreter is started.
        provided = {**casts, **state, "TestSupport": _test_support, "gApplication": _application,
                    "TApplication": _application_class, "TRint": _application_class,
                    "gCling": _no_cling}  # fmt: skip
        super().__init__(_pyroot, USED, provided, _std)


#: The Python a value of each kind of C++ arithmetic type is, by the type's name.
CASTS: dict[str, Any] = {"float": float, "double": float, "long double": float, "bool": bool}


def _no_cling() -> None:
    """``gCling``: nothing at all, since this kit has no cling - a program asks ``gInterpreter``
    instead, which is the interpreter a translation runs on."""


def _cast(target: str) -> Any:
    """The conversion ``Int_t(x)`` makes: to ``int``, ``float`` or ``bool``."""
    return lambda: CASTS.get(target, int)


ROOT = RootProxy()
