"""``TMVA::DataLoader``: the variables, the trees, the classes and how to split them.

A loader is named - ``"dataset"`` - and everything TMVA writes for it goes
under that name: the directory in the output file, and ``<name>/weights/``
for the weight files. Its declarations are a :class:`~.dataset.DataSetInfo`;
the events are read from the trees only when a method first needs them, as
TMVA's ``Rebuilding Dataset`` does, and are kept until something is declared
that would change them.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .building import create_dataset
from .dataset import MAX_TREE_TYPE, TESTING, TRAINING, DataSet, DataSetInfo
from .log import Logger
from .reading import TreeInput
from .variables import VariableInfo

__all__ = ["DataLoader", "EventTree"]


def _char(value: Any) -> str:
    """A ``char`` argument - ``'F'``, or the ``ord('F')`` a translated macro passes - as text."""
    return chr(value) if isinstance(value, int) else str(value)


def _is_type(args: tuple[Any, ...]) -> bool:
    """Is ``AddVariable``'s second argument the variable's type - ``'F'`` - rather than a title?

    C++ tells the two apart by the argument's type; here a one-letter string
    followed by nothing but numbers is the type, as it would be the ``char``.
    """
    first = args[0]
    if isinstance(first, int):
        return True
    text = first if isinstance(first, str) else None
    rest_numbers = all(isinstance(arg, (int, float)) for arg in args[1:])
    return text is not None and len(text) == 1 and rest_numbers


def _tree_type(value: Any) -> int:
    """``Types::ETreeType``, or ``"Training"``/``"Test"`` as ``AddTree``'s string reads it."""
    if isinstance(value, int):
        return value
    text = str(value).lower()
    if "train" in text and "test" in text:
        return MAX_TREE_TYPE
    if "train" in text:
        return TRAINING
    if "test" in text:
        return TESTING
    raise Logger("DataLoader").fatal(
        f'<AddTree> cannot interpret tree type: "{value}" should be "Training" or "Test" '
        'or "Training and Testing"'
    )


class EventTree:
    """A tree of events added one at a time - ``AddSignalTrainingEvent`` - read like any other."""

    def __init__(self, name: str, nvar: int) -> None:
        self.name = name
        self.rows: list[list[float]] = []
        self.weights: list[float] = []
        self.nvar = nvar

    def GetName(self) -> str:
        return self.name

    def GetEntries(self) -> int:
        return len(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def arrays(self, names: list[str]) -> dict[str, Any]:
        table = np.array(self.rows, dtype=np.float64).reshape(len(self.rows), self.nvar)
        columns = {f"__var{index}": table[:, index] for index in range(self.nvar)}
        columns["__weight"] = np.array(self.weights, dtype=np.float64)
        return {name: columns[name] for name in names}


class DataLoader:
    """``TMVA::DataLoader(name)``."""

    def __init__(self, name: Any = "default") -> None:
        self.name = str(name)
        self.info = DataSetInfo(self.name)
        self.inputs: list[TreeInput] = []
        self.log = Logger("DataLoader")
        self._dataset: DataSet | None = None
        self._event_trees: dict[tuple[str, int], EventTree] = {}
        self.transformations: list[str] = []

    def GetName(self) -> str:
        return self.name

    def GetDataSetInfo(self) -> DataSetInfo:
        return self.info

    def DefaultDataSetInfo(self) -> DataSetInfo:
        return self.info

    def _changed(self) -> None:
        self._dataset = None

    # -- variables ------------------------------------------------------------------

    def AddVariable(self, expression: Any, *args: Any) -> None:
        """``AddVariable(expr, type)`` or ``(expr, title, unit, type, min, max)``."""
        if args and _is_type(args):
            vartype, rest = _char(args[0]), args[1:]
            low, high = [*rest, 0.0, 0.0][:2]
            self.info.variables.append(VariableInfo(expression, "", "", vartype, low, high))
        else:
            title, unit, vartype, low, high = list(args) + ["", "", "F", 0.0, 0.0][len(args) :]
            made = VariableInfo(expression, str(title), str(unit), _char(vartype), low, high)
            self.info.variables.append(made)
        self._changed()

    def AddVariablesArray(self, expression: Any, size: int, vartype: Any = "F", *_: Any) -> None:
        """``AddVariablesArray("vars", 256)``: one variable per element of an array expression."""
        for index in range(int(size)):
            made = VariableInfo(expression, f"[{index}]", "", _char(vartype), index=index)
            made.title = f"{made.label}[{index}]"
            self.info.variables.append(made)
        self._changed()

    def AddTarget(self, expression: Any, title: Any = "", unit: Any = "", *_: Any) -> None:
        self.info.targets.append(VariableInfo(expression, str(title), str(unit)))
        self._changed()

    AddRegressionTarget = AddTarget

    def AddSpectator(self, expression: Any, title: Any = "", unit: Any = "", *_: Any) -> None:
        self.info.spectators.append(VariableInfo(expression, str(title), str(unit)))
        self._changed()

    # -- trees ----------------------------------------------------------------------

    def AddTree(
        self,
        tree: Any,
        class_name: Any,
        weight: float = 1.0,
        cut: Any = "",
        tree_type: Any = MAX_TREE_TYPE,
    ) -> None:
        """``AddTree(tree, className, weight, cut, treetype)``: every event of the tree is of it."""
        if tree is None:
            raise self.log.fatal("Tree does not exist (empty pointer).")
        name = str(class_name)
        self.info.AddClass(name)
        self.log.info(f"Add Tree {tree.GetName()} of type {name} with {tree.GetEntries()} events")
        kind = _tree_type(tree_type)
        if str(cut):
            tree = tree.CopyTree(str(cut))
        self.inputs.append(TreeInput(tree, name, float(weight), kind))
        self._changed()

    def AddSignalTree(self, tree: Any, weight: float = 1.0, tree_type: Any = MAX_TREE_TYPE) -> None:
        self.AddTree(tree, "Signal", weight, "", tree_type)

    def AddBackgroundTree(
        self, tree: Any, weight: float = 1.0, tree_type: Any = MAX_TREE_TYPE
    ) -> None:
        self.AddTree(tree, "Background", weight, "", tree_type)

    def AddRegressionTree(
        self, tree: Any, weight: float = 1.0, tree_type: Any = MAX_TREE_TYPE
    ) -> None:
        self.AddTree(tree, "Regression", weight, "", tree_type)

    SetSignalTree = AddSignalTree
    SetBackgroundTree = AddBackgroundTree

    def SetInputTrees(self, signal: Any, background: Any, *weights: float) -> None:
        signal_weight, background_weight = [*weights, 1.0, 1.0][:2]
        self.AddSignalTree(signal, signal_weight)
        self.AddBackgroundTree(background, background_weight)

    # -- events one by one -------------------------------------------------------------

    def AddEvent(self, class_name: Any, tree_type: int, event: Any, weight: float = 1.0) -> None:
        """``AddEvent``: one event of a class, for training or testing, gathered into a tree."""
        key = (str(class_name), int(tree_type))
        tree = self._event_trees.get(key)
        if tree is None:
            if not self._event_trees:
                self._variables_for_events()
            kind = "Train" if tree_type == TRAINING else "Test"
            tree = EventTree(f"{kind}AssignTree_{class_name}", self.info.GetNVariables())
            self._event_trees[key] = tree
            self.info.AddClass(str(class_name))
        tree.rows.append([float(value) for value in event])
        tree.weights.append(float(weight))
        self._changed()

    def _variables_for_events(self) -> None:
        """Events added one by one are read through columns named after their position."""
        for index, variable in enumerate(self.info.variables):
            variable.expression = f"__var{index}"

    def AddTrainingEvent(self, class_name: Any, event: Any, weight: float = 1.0) -> None:
        self.AddEvent(class_name, TRAINING, event, weight)

    def AddTestEvent(self, class_name: Any, event: Any, weight: float = 1.0) -> None:
        self.AddEvent(class_name, TESTING, event, weight)

    def AddSignalTrainingEvent(self, event: Any, weight: float = 1.0) -> None:
        self.AddEvent("Signal", TRAINING, event, weight)

    def AddBackgroundTrainingEvent(self, event: Any, weight: float = 1.0) -> None:
        self.AddEvent("Background", TRAINING, event, weight)

    def AddSignalTestEvent(self, event: Any, weight: float = 1.0) -> None:
        self.AddEvent("Signal", TESTING, event, weight)

    def AddBackgroundTestEvent(self, event: Any, weight: float = 1.0) -> None:
        self.AddEvent("Background", TESTING, event, weight)

    # -- weights, cuts and the split ----------------------------------------------------

    def SetWeightExpression(self, expression: Any, class_name: Any = "") -> None:
        if not str(class_name):
            self.SetSignalWeightExpression(expression)
            self.SetBackgroundWeightExpression(expression)
        else:
            self.info.SetWeightExpression(expression, str(class_name))
        self._changed()

    def SetSignalWeightExpression(self, expression: Any) -> None:
        self.info.SetWeightExpression(expression, "Signal")
        self._changed()

    def SetBackgroundWeightExpression(self, expression: Any) -> None:
        self.info.SetWeightExpression(expression, "Background")
        self._changed()

    def SetCut(self, cut: Any, class_name: Any = "") -> None:
        self.info.SetCut(cut, str(class_name))
        self._changed()

    def AddCut(self, cut: Any, class_name: Any = "") -> None:
        self.info.AddCut(cut, str(class_name))
        self._changed()

    def _from_event_trees(self) -> None:
        """``SetInputTreesFromEventAssignTrees``: the events added one by one, as trees."""
        for (name, kind), tree in self._event_trees.items():
            self.inputs.append(TreeInput(tree, name, 1.0, kind))
            self.info.SetWeightExpression("__weight", name)
        self._event_trees = {}

    def PrepareTrainingAndTestTree(self, *args: Any) -> None:
        """``(cut, options)``, ``(sigcut, bkgcut, options)``, or the numbers-of-events forms."""
        self._from_event_trees()
        options = str(args[-1]) if args else ""
        numbers = [arg for arg in args[1:] if isinstance(arg, int) and not isinstance(arg, bool)]
        if numbers:
            rest = options if not isinstance(args[-1], int) else ""
            options = _numbered(numbers, rest)
            self.AddCut(args[0])
        elif len(args) >= 3:
            self.AddCut(args[0], "Signal")
            self.AddCut(args[1], "Background")
        elif len(args) == 2:
            for number, cls in enumerate(self.info.classes):
                self.info.log.info(
                    f"Dataset[{self.info.name}] : Class index : {number}  name : {cls.name}"
                )
            self.AddCut(args[0])
        self.info.split_options = options
        self._changed()

    # -- the events ---------------------------------------------------------------------

    def dataset(self) -> DataSet:
        """The training and test samples, read and split the first time they are asked for."""
        if self._dataset is None:
            self._from_event_trees()
            self._dataset = create_dataset(self.info, self.inputs)
        return self._dataset

    def has_dataset(self) -> bool:
        return self._dataset is not None


def _numbered(numbers: list[int], rest: str) -> str:
    """The options ``PrepareTrainingAndTestTree(cut, Ntrain, Ntest)`` and its kin stand for."""
    if len(numbers) >= 4:
        sig_train, bkg_train, sig_test, bkg_test = numbers[:4]
        return (
            f"nTrain_Signal={sig_train}:nTrain_Background={bkg_train}:"
            f"nTest_Signal={sig_test}:nTest_Background={bkg_test}:{rest}"
        )
    train, test = [*numbers, -1][:2]
    return (
        f"nTrain_Signal={train}:nTrain_Background={train}:nTest_Signal={test}:"
        f"nTest_Background={test}:SplitMode=Random:EqualTrainSample:!V"
    )
