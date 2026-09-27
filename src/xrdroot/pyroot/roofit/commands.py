"""``ROOT.RooFit``: the namespace of command arguments, message levels and topics.

``RooFit.Save()``, ``RooFit.LineColor(kRed)``, ``RooFit.Range("signal")`` -
each makes a :class:`~xrdroot.roofit.cmdargs.RooCmdArg` of its name and
arguments, which every method of :mod:`xrdroot.roofit` reads. The
namespace also holds ``RooFit.INFO`` and the other levels, ``RooFit.Fitting``
and the other topics, and ``RooFit.RooConst``.
"""

from __future__ import annotations

from typing import Any

from ...roofit import messages
from ...roofit.cmdargs import RooCmdArg
from ...roofit.variables import RooConstVar

__all__ = ["RooFit", "COMMANDS"]

#: Every command function of ``RooGlobalFunc.h`` this namespace makes.
COMMANDS = (
    "AddTo",
    "AllBinned",
    "Asimov",
    "Asymmetry",
    "AsymptoticError",
    "AutoBinned",
    "AutoBinning",
    "AutoPrecision",
    "AutoRange",
    "AutoSymBinning",
    "AutoSymRange",
    "AxisLabel",
    "BaseClassName",
    "BatchMode",
    "Binned",
    "Binning",
    "Bins",
    "BootStrapData",
    "ClassName",
    "CloneData",
    "Color",
    "Columns",
    "Components",
    "Conditional",
    "ConditionalObservables",
    "Constrain",
    "Cut",
    "CutRange",
    "DataError",
    "DrawOption",
    "Efficiency",
    "Embedded",
    "EvalBackend",
    "EvalErrorValue",
    "EvalErrorWall",
    "EventRange",
    "ExpectedData",
    "Extended",
    "ExternalConstraints",
    "FillColor",
    "FillStyle",
    "FitGauss",
    "FitModel",
    "FitOptions",
    "FixedPrecision",
    "Format",
    "Frame",
    "FrameBins",
    "FrameRange",
    "GenBinned",
    "GlobalObservables",
    "GlobalObservablesSource",
    "GlobalObservablesTag",
    "Hesse",
    "Import",
    "ImportFromFile",
    "Index",
    "InitialHesse",
    "Integrate",
    "IntegrateBins",
    "IntegratedObservables",
    "IntrinsicBinning",
    "Invisible",
    "Label",
    "LatexStyle",
    "LatexTableStyle",
    "Layout",
    "LineColor",
    "LineStyle",
    "LineWidth",
    "Link",
    "MarkerColor",
    "MarkerSize",
    "MarkerStyle",
    "MaxCalls",
    "Minimizer",
    "Minos",
    "ModularL",
    "MoveToBack",
    "MultiArg",
    "Name",
    "NoRecursion",
    "NormSet",
    "Normalization",
    "NormRange",
    "NumCPU",
    "NumEvents",
    "NumIntConfig",
    "ObjectName",
    "Offset",
    "Optimize",
    "OutputFile",
    "OutputStream",
    "OwnLinked",
    "Parameters",
    "Precision",
    "Prefix",
    "PrintEvalErrors",
    "PrintLevel",
    "ProjWData",
    "Project",
    "ProjectionRange",
    "ProtoData",
    "Range",
    "RecoverFromUndefinedRegions",
    "RecycleConflictNodes",
    "RefreshNorm",
    "Rename",
    "RenameAllNodes",
    "RenameAllVariables",
    "RenameAllVariablesExcept",
    "RenameConflictNodes",
    "RenameVariable",
    "Rescale",
    "Restrict",
    "Save",
    "Scaling",
    "ScanAllCdf",
    "ScanNoCdf",
    "ScanNumCdf",
    "ScanParameters",
    "SelectVars",
    "ShiftToZero",
    "ShowAsymError",
    "ShowConstants",
    "ShowError",
    "ShowName",
    "ShowProgress",
    "ShowUnit",
    "ShowValue",
    "Sibling",
    "Silence",
    "Slice",
    "SplitParam",
    "SplitParamConstrained",
    "SplitRange",
    "StoreAsymError",
    "StoreError",
    "Strategy",
    "SumCoefRange",
    "SumW2Error",
    "SupNormSet",
    "SelectCompSet",
    "SliceCat",
    "TLatexStyle",
    "TagName",
    "Timer",
    "Title",
    "Topic",
    "VLines",
    "VerbatimName",
    "Verbose",
    "VisualizeError",
    "Warnings",
    "Weight",
    "WeightVar",
    "What",
    "XErrorSize",
    "YVar",
    "ZVar",
)


def _command(name: str) -> Any:
    def made(*args: Any, **kwargs: Any) -> RooCmdArg:
        for first in ("var", "what"):  # YVar(var=y, ...), Format(what="NE", ...): PyROOT's names
            if first in kwargs:
                args = (kwargs.pop(first), *args)
        if kwargs:  # FitOptions(Save=True), Format("NE", AutoPrecision=1)...
            from ...roofit.cmdargs import make

            args = args + tuple(make(key, value) for key, value in kwargs.items())
        return RooCmdArg(name, *args)

    made.__name__ = made.__qualname__ = name
    made.__doc__ = f"``RooFit::{name}(...)``: the ``{name}`` command argument."
    return made


class _Namespace:
    """``RooFit``: a namespace, as C++ has one."""

    def __init__(self) -> None:
        for name in COMMANDS:
            setattr(self, name, _command(name))
        for number, name in messages.LEVELS.items():
            setattr(self, name, number)
        for name, bit in messages.TOPICS.items():
            setattr(self, name, bit)
        self.NumIntegration = messages.TOPICS["NumericIntegration"]
        self.Relative, self.NumEvent, self.RelativeExpected, self.Raw = 0, 1, 2, 3

    @staticmethod
    def RooConst(value: float) -> RooConstVar:
        """``RooFit::RooConst``: a constant, named after its value."""
        from ...roofit.pdfs.basic import ref

        return ref(float(value))  # type: ignore[no-any-return]

    def __repr__(self) -> str:
        return "<namespace RooFit>"


#: ``ROOT.RooFit``.
RooFit = _Namespace()
