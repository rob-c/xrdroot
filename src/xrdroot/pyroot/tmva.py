"""TMVA under ``import ROOT``: ``ROOT.TMVA.Factory``, ``ROOT.TMVA.Types.kBDT``, the ``Reader``.

``ROOT.TMVA`` is a namespace of :mod:`xrdroot.tmva`'s classes by TMVA's
names - the engine speaks TMVA's API already, so this only gathers it, as
``TMVA::Experimental`` gathers the tensor, reader and scaler classes. A
class of TMVA's this does not have is refused by that name.

``TMVA::DNN``'s own vocabulary is ``ROOT.TMVA.DNN``: its enumerations, the
``TCpu<T>`` CPU backend's matrix and tensor, and the ``evaluate`` functions
written ``evaluate<Arch>(...)`` - which ROOT's own DNN tests reach
unqualified after ``using namespace TMVA::DNN``.
"""

from __future__ import annotations

import enum
from typing import Any

from .. import tmva as _tmva
from ..tmva import batches as _batches
from ..tmva import dnn as _dnn

__all__ = ["DNN", "Experimental", "TMVA"]


class _Namespace:
    """A C++ namespace: its members as attributes, and a refusal naming any it has not."""

    def __init__(self, name: str, members: dict[str, Any]) -> None:
        self.__dict__.update(members)
        self._name = name

    def __getattr__(self, attribute: str) -> Any:
        if attribute.startswith("__"):
            raise AttributeError(attribute)
        raise AttributeError(f"ROOT has {self._name}::{attribute}; xrdroot.pyroot does not yet")

    def __repr__(self) -> str:
        return f"<namespace {self._name}>"


class EActivationFunction(enum.IntEnum):
    """``TMVA::DNN::EActivationFunction``: how a layer's output is made from its input."""

    kIdentity = 0
    kRelu = 1
    kSigmoid = 2
    kTanh = 3
    kSymmRelu = 4
    kSoftSign = 5
    kGauss = 6
    kFastTanh = 7


class EOutputFunction(enum.IntEnum):
    """``TMVA::DNN::EOutputFunction``: the output layer's, by the letter TMVA writes it with."""

    kIdentity = ord("I")
    kSigmoid = ord("S")
    kSoftmax = ord("M")


class ELossFunction(enum.IntEnum):
    """``TMVA::DNN::ELossFunction``: the objective a net is trained against, TMVA's letters."""

    kCrossEntropy = ord("C")
    kMeanSquaredError = ord("R")
    kSoftmaxCrossEntropy = ord("S")


class ATensor:
    """``TMVA::DNN::ATensor<T>``: the element type of a tensor, and the matrix of one slice.

    A DNN test writes ``typename ATensor::Scalar_t`` of its tensor template parameter,
    TMVA's default for which is a ``float``; here Python's ``float``, which is a double,
    so a test that names it computes without narrowing.
    """

    Scalar_t = float

    def __class_getitem__(cls, kind: Any) -> type[ATensor]:
        return cls


def _loss(member: Any) -> str:
    """``ELossFunction::kMeanSquaredError``, keyed as :data:`~xrdroot.tmva.dnn.LOSSES` is."""
    return str(member.name)[1:]


class _Losses:
    """``evaluate<Arch>(f, Y, output, weights)``: a loss function of the named architecture.

    The architecture in the angle brackets chooses whose arithmetic is written. Every
    one of TMVA's is the same formula over matrices, so the name is taken and not
    branched on, as the matrices themselves say which precision they hold.
    """

    #: Extra arguments the functions here take before the loss's own, and the last of them.
    leading = 0

    def __init__(self, table: dict[str, Any]) -> None:
        self.table = table

    def __getitem__(self, architecture: Any) -> Any:
        def evaluate(*args: Any) -> Any:
            head, member, rest = args[: self.leading], args[self.leading], args[self.leading + 1:]
            return self.table[_loss(member)](*head, *rest)

        return evaluate


class _LossGradients(_Losses):
    """``evaluateGradients<Arch>(dY, f, Y, output, weights)``: written into ``dY``."""

    leading = 1


def _gui(*_: Any) -> None:
    """``TMVA::TMVAGui``: a window, which batch mode never opens."""


DNN = _Namespace(
    "TMVA::DNN",
    {
        "ATensor": ATensor,
        "EActivationFunction": EActivationFunction,
        "ELossFunction": ELossFunction,
        "EOutputFunction": EOutputFunction,
        "TCpuMatrix": _dnn.matrix_type("double"),
        "evaluate": _Losses(_dnn.LOSSES),
        "evaluateGradients": _LossGradients(_dnn.LOSS_GRADIENTS),
    },
)

TMVA = _Namespace(
    "TMVA",
    {
        **{name: getattr(_tmva, name) for name in _tmva.__all__},
        "TMVAGui": _gui,
        "DNN": DNN,
        "Experimental": _Namespace("TMVA::Experimental", dict(_tmva.EXPERIMENTAL)),
    },
)

#: ``ROOT.Experimental``: of it, the machine-learning data loader.
Experimental = _Namespace(
    "ROOT::Experimental",
    {"ML": _Namespace("ROOT::Experimental::ML", {"RDataLoader": _batches.RDataLoader})},
)
