"""PDERS: its volume modes, its kernels, regression, and the search tree it writes and reads."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from tmvafitsupport import REGRESSION, reader, regression_loader
from tmvasupport import classify, loader, session
from xrdroot.tmva import TMVAError
from xrdroot.tmva.methods.pderskernels import KERNELS, kernel_values
from xrdroot.tmva.searchtree import TreeEvents, build, read_tree_xml, tree_xml
from xrdroot.tmva.xmlfile import Node

__all__ = ["session"]

PDERS = ROOT.TMVA.Types.kPDERS
#: An adaptive box small enough to be quick.
SMALL = "!H:!V:NEventsMin=20:NEventsMax=40:MaxVIterations=30"


def _parsed(node: Node) -> ET.Element:
    """A written element read back, as a weight file is."""
    return ET.fromstring("\n".join(node.lines()))


@pytest.mark.parametrize(
    "options",
    [
        f"{SMALL}:VolumeRangeMode=Adaptive:KernelEstimator=Gauss:NormTree",
        "!H:!V:VolumeRangeMode=RMS:DeltaFrac=1.5:KernelEstimator=Teepee",
        "!H:!V:VolumeRangeMode=MinMax:DeltaFrac=0.3:KernelEstimator=Sinc3",
        "!H:!V:VolumeRangeMode=Unscaled:DeltaFrac=0.5:KernelEstimator=Lanczos2",
    ],
)
def test_every_volume_mode_is_trained_and_its_tree_read_back_by_a_reader(session, options):
    classify([(PDERS, "PDERS", options)])
    made = reader("PDERS", silent=True)
    high = made.EvaluateMVA([2.0, 2.0, 2.0, 2.0], "PDERS")
    low = made.EvaluateMVA([-2.0, -2.0, -2.0, -2.0], "PDERS")
    assert 0.0 <= low <= high <= 1.0


def test_an_event_far_from_all_training_events_is_given_a_half(session):
    factory = classify([(PDERS, "PDERS", "!H:!V:VolumeRangeMode=Unscaled:DeltaFrac=0.2")])
    trained = factory.GetMethod("dataset", "PDERS")
    # The method the Factory hands back was read from its weight file, which - as in TMVA -
    # leaves it adaptive; a fixed box is what leaves an event far away with nothing in it.
    trained.mode = "Unscaled"
    far = np.array([[50.0, 50.0, 50.0, 50.0], [1.0, 1.0, 1.0, 1.0]])
    assert trained.evaluate(far)[0] == 0.5
    assert trained.evaluate(np.zeros((0, 4))).shape == (0,)


def test_boxes_holding_too_few_events_are_grown_until_they_hold_enough(session):
    options = "!H:!V:NEventsMin=150:NEventsMax=250:DeltaFrac=0.05:MaxVIterations=20"
    factory = classify([(PDERS, "PDERS", options)])
    assert factory.GetMethod("dataset", "PDERS") is not None


def test_boxes_that_never_hold_what_is_asked_are_left_as_the_best_found(session):
    options = "!H:!V:NEventsMin=150:NEventsMax=150.5:MaxVIterations=3"
    factory = classify([(PDERS, "PDERS", options)])
    assert factory.GetMethod("dataset", "PDERS") is not None


def test_a_regression_is_the_kernel_weighted_mean_target(session):
    options = f"{SMALL}:KernelEstimator=Box"
    classify([(PDERS, "PDERS", options)], options=REGRESSION, data=regression_loader())
    found = reader("PDERS", ("var1", "var2"), silent=True).EvaluateRegression(
        0, [2.0, 2.0], "PDERS"
    )
    assert found == pytest.approx(10 * 2 + 5 * 4, abs=15)


@pytest.mark.parametrize(
    ("options", "said"),
    [
        ("VolumeRangeMode=kNN", "VolumeRangeMode=kNN"),
        ("KernelEstimator=Trim", "KernelEstimator=Trim"),
        ("IgnoreNegWeightsInTraining", "ignore events with negative weights"),
    ],
)
def test_what_xrdroot_s_pders_does_not_have_is_refused(session, options, said):
    factory = ROOT.TMVA.Factory("job", "!V:Silent:AnalysisType=Classification")
    with pytest.raises(TMVAError, match=said):
        factory.BookMethod(loader(), PDERS, "PDERS", options)


@pytest.mark.parametrize("name", KERNELS)
def test_every_kernel_is_one_at_the_event(name):
    distance = np.array([0.0, 0.5, 1.0])
    for nvar in (3, 4):
        found = kernel_values(name, distance, 0.1, nvar)
        assert found[0] == pytest.approx(1.0) and np.all(np.isfinite(found))


def test_a_class_without_weight_has_no_spread(session):
    factory = classify([(PDERS, "PDERS", "!H:!V:VolumeRangeMode=RMS")])
    trained = factory.GetMethod("dataset", "PDERS")
    assert trained._rms(np.zeros(len(trained.tree.weights), dtype=bool)).tolist() == [0.0] * 4


def test_a_balanced_tree_puts_the_first_of_equal_medians_at_its_root():
    values = np.array([[1.0], [2.0], [2.0], [2.0], [3.0]])
    root = build(values, True)
    assert values[root.index, 0] == 2.0 and root.left is not None
    events = TreeEvents(values, np.zeros(5, dtype=int), np.ones(5), None)
    node = Node("Weights")
    tree_xml(node, events, False)
    back = read_tree_xml(_parsed(node.children[0]))
    assert sorted(back.values[:, 0]) == [1.0, 2.0, 2.0, 2.0, 3.0] and back.targets is None


def test_a_tree_of_no_events_is_empty():
    node = Node("Weights")
    tree_xml(node, TreeEvents(np.zeros((0, 2)), np.zeros(0), np.zeros(0), None), True)
    assert len(read_tree_xml(_parsed(node.children[0])).values) == 0
