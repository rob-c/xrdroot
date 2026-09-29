"""What the tests of TMVA's fitters and density methods share beyond :mod:`tmvasupport`.

A regression loader and a three-class loader, made from :mod:`tmvasupport`'s
trees, and a Reader that books a method from the weight file a Factory run
wrote - so that each method can be trained for every analysis it has and
read back as a user of TMVA would read it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import xrdroot.pyroot as ROOT
from tmvasupport import VARIABLES, gaussian, regression_tree, weights

#: How every loader here splits its events.
SPLIT = "SplitMode=Random:NormMode=NumEvents:!V"
#: A Factory's options for a regression.
REGRESSION = "!V:!Silent:AnalysisType=Regression"
#: A Factory's options for several classes.
MULTICLASS = "!V:!Silent:AnalysisType=Multiclass"


def regression_loader(name: str = "dataset") -> Any:
    """A loader of two variables and the target ``fvalue`` over :func:`regression_tree`."""
    made = ROOT.TMVA.DataLoader(name)
    made.AddVariable("var1", "F")
    made.AddVariable("var2", "F")
    made.AddTarget("fvalue")
    made.AddRegressionTree(regression_tree())
    made.PrepareTrainingAndTestTree("", SPLIT)
    return made


def multiclass_loader(name: str = "dataset", count: int = 120) -> Any:
    """A loader of the four variables over three classes, apart in their means."""
    made = ROOT.TMVA.DataLoader(name)
    for variable in VARIABLES:
        made.AddVariable(variable, "F")
    for index, (cls, shift) in enumerate((("Signal", 1.5), ("Background", -1.5), ("Third", 0.0))):
        made.AddTree(gaussian(f"Tree{index}", shift, index + 1, count), cls)
    made.PrepareTrainingAndTestTree("", SPLIT)
    return made


def reader(title: str, variables: tuple[str, ...] = VARIABLES, silent: bool = False) -> Any:
    """A Reader of ``variables``, the method ``title`` booked from the weight file it was given."""
    made = ROOT.TMVA.Reader("!Color:Silent" if silent else "!Color:!Silent")
    for variable in variables:
        made.AddVariable(variable, np.zeros(1, dtype=np.float32))
    made.BookMVA(title, weights(title))
    return made
