"""What the tests of :mod:`xrdroot.tmva` start from: small samples, and a Factory run over them.

The samples are made here, in memory, from a fixed seed - a signal and a
background of four Gaussian variables apart in their means, three classes
of the same, and a regression target that is a smooth function of two
variables - so that every method can be trained in a fraction of a second
and asked what TMVA would ask it.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import fresh
from xrdroot.tmva.log import CONFIG

#: How many events each class has.
EVENTS = 200
#: The classification's variables.
VARIABLES = ("var1", "var2", "var3", "var4")


def _tree(name: str, columns: dict[str, Any]) -> Any:
    """A pyroot tree of ``columns``, filled entry by entry through its branches' addresses."""
    tree = ROOT.TTree(name, name)
    cells = {key: np.zeros(1, dtype=np.float32) for key in columns}
    for key, cell in cells.items():
        tree.Branch(key, cell, f"{key}/F")
    for row in range(len(next(iter(columns.values())))):
        for key, cell in cells.items():
            cell[0] = columns[key][row]
        tree.Fill()
    return tree


def gaussian(name: str, shift: float, seed: int, count: int = EVENTS) -> Any:
    """A tree of the four variables, each Gaussian with mean ``shift``; a weight beside them."""
    rng = np.random.default_rng(seed)
    columns = {v: rng.normal(shift * (i + 1) / 4, 1.0, count) for i, v in enumerate(VARIABLES)}
    columns["weight"] = rng.uniform(0.5, 1.5, count)
    return _tree(name, columns)


def regression_tree(name: str = "TreeR", count: int = 2 * EVENTS, seed: int = 7) -> Any:
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(0, 5, count), rng.uniform(0, 5, count)
    return _tree(
        name, {"var1": x, "var2": y, "fvalue": 10 * x + 5 * y * y + rng.normal(0, 1, count)}
    )


@pytest.fixture
def session(tmp_path: Any) -> Iterator[None]:
    """A fresh ROOT session in ``tmp_path``, TMVA's shared configuration put back after."""
    saved = (CONFIG.use_color, CONFIG.silent, CONFIG.draw_progress_bar)
    yield from fresh(tmp_path)
    CONFIG.use_color, CONFIG.silent, CONFIG.draw_progress_bar = saved


def loader(
    name: str = "dataset",
    split: str = "SplitMode=Random:NormMode=NumEvents:!V",
    spectator: bool = False,
) -> Any:
    """A loader of the four variables over a signal and a background tree, split as asked."""
    made = ROOT.TMVA.DataLoader(name)
    for variable in VARIABLES:
        made.AddVariable(variable, "F")
    if spectator:
        made.AddSpectator("spec := var1*2", "Spectator", "units", "F")
    made.AddSignalTree(gaussian("TreeS", 1.0, 1))
    made.AddBackgroundTree(gaussian("TreeB", -1.0, 2))
    made.SetBackgroundWeightExpression("weight")
    made.PrepareTrainingAndTestTree("", "", split)
    return made


def classify(
    methods: list[tuple[Any, str, str]],
    options: str = "!V:!Silent:AnalysisType=Classification",
    output: str = "out.root",
    data: Any = None,
) -> Any:
    """A Factory that has booked, trained, tested and evaluated ``methods`` over :func:`loader`."""
    target = ROOT.TFile.Open(output, "RECREATE") if output else None
    factory = (
        ROOT.TMVA.Factory("job", target, options) if target else ROOT.TMVA.Factory("job", options)
    )
    data = data if data is not None else loader()
    for kind, title, text in methods:
        factory.BookMethod(data, kind, title, text)
    factory.TrainAllMethods()
    factory.TestAllMethods()
    factory.EvaluateAllMethods()
    if target:
        target.Close()
    return factory


def weights(title: str, job: str = "job", dataset: str = "dataset") -> str:
    """Where the Factory put a method's weight file."""
    return os.path.join(dataset, "weights", f"{job}_{title}.weights.xml")
