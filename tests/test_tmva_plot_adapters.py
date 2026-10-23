"""TMVA computes ROC arrays independently and delegates only its public ROOT pictures."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

import xrdroot
from xrdroot import plugins
from xrdroot.tmva.dataset import DataSet, Events
from xrdroot.tmva.envelope import ClassificationResult
from xrdroot.tmva.evaluation import roc_curve
from xrdroot.tmva.factory import Booked, Factory
from xrdroot.tmva.loader import DataLoader
from xrdroot.tmva.log import CONFIG
from xrdroot.tmva.method import MULTICLASS, Method


@pytest.fixture(autouse=True)
def _configuration():
    saved = CONFIG.use_color, CONFIG.silent, CONFIG.draw_progress_bar
    yield
    CONFIG.use_color, CONFIG.silent, CONFIG.draw_progress_bar = saved


@pytest.fixture
def root_session(tmp_path):
    from pyrootsupport import fresh

    yield from fresh(tmp_path)


def _provider(monkeypatch, point, callback):
    name = "_test_" + point.replace(".", "_")
    module = ModuleType(name)
    module.provide = callback
    monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(plugins.PROVIDERS, (point, "default"), ("core", name, "provide"))


def _factory():
    factory = Factory("job", "Silent:AnalysisType=Classification")
    loader = DataLoader("dataset")
    loader.info.AddClass("Signal")
    loader.info.AddClass("Background")
    empty = np.zeros((4, 0))
    events = Events(
        np.zeros((4, 1)), empty, empty, np.array([0, 1, 0, 1]), np.array([1., 2., 3., 4.])
    )
    loader._dataset = DataSet(events, events.take([3, 1, 2, 0]))
    method = Method("job", "classifier", loader.info)
    item = Booked(method, loader, np.array([.1, .1, .8, .4]), np.array([.2, .2, .7, .3]))
    factory.booked[loader.GetName()] = [item]
    return factory, loader, item


def test_the_classification_result_passes_unchanged_roc_arrays_to_its_provider(monkeypatch):
    values = np.array([0.1, 0.1, 0.8, 0.4])
    signal = np.array([True, False, True, False])
    weights = np.array([1 / 3, 1.25, 2 / 3, 2.5])
    result = ClassificationResult("dataset", "LD", "classifier", values, signal, weights)
    received = []
    expected = object()

    def graph(*args):
        received.append(args)
        return expected

    _provider(monkeypatch, "tmva.result_roc_graph", graph)
    assert result.GetROCGraph(False) is expected
    assert received[0][0] == "classifier"
    for actual, wanted in zip(received[0][1:], roc_curve(values, signal, weights), strict=True):
        assert np.array_equal(actual.view(np.uint64), wanted.view(np.uint64))


def test_factory_graph_queries_keep_training_selection_and_title_suppression(monkeypatch):
    factory, loader, item = _factory()
    received = []
    expected = object()
    _provider(
        monkeypatch, "tmva.factory_roc_graph", lambda *args: received.append(args) or expected
    )
    assert factory.GetROCCurve(loader, "classifier", False, 0, 0) is expected
    assert received[0][0] == "classifier" and received[0][3] is False
    train = loader.dataset().train
    wanted = roc_curve(item.train_values, train.classes == 0, train.weights)
    assert all(np.array_equal(a, b) for a, b in zip(received[0][1:3], wanted, strict=True))


def test_factory_graph_queries_keep_multiclass_column_selection(monkeypatch):
    factory, loader, item = _factory()
    factory.analysis = MULTICLASS
    item.test_values = np.column_stack((item.test_values, 1 - item.test_values))
    received = []
    _provider(monkeypatch, "tmva.factory_roc_graph", lambda *args: received.append(args))
    assert factory.GetROCCurve("dataset", "classifier", True, 1, 1) is None
    test = loader.dataset().test
    wanted = roc_curve(item.test_values[:, 1], test.classes == 1, test.weights)
    assert all(np.array_equal(a, b) for a, b in zip(received[0][1:3], wanted, strict=True))
    assert received[0][3] is True


def test_canvas_overloads_pass_the_existing_factory_and_dataset_to_the_provider(monkeypatch):
    factory, loader, _ = _factory()
    received = []
    expected = object()
    _provider(monkeypatch, "tmva.roc_canvas", lambda *args: received.append(args) or expected)
    assert factory.GetROCCurve(loader) is expected
    assert factory.GetROCCurve("dataset", True) is expected
    assert received == [(factory, loader), (factory, "dataset")]


def test_public_result_and_factory_graphs_keep_their_root_types_names_and_axes(root_session):
    import xrdroot.pyroot as ROOT

    factory, loader, item = _factory()
    test = loader.dataset().test
    result = ClassificationResult("dataset", "LD", "classifier", item.test_values,
                                  test.classes == 0, test.weights)
    graph = result.GetROCGraph()
    assert isinstance(graph, ROOT.TGraph)
    assert (graph.GetName(), graph.GetTitle()) == ("classifier", "classifier")
    titled = factory.GetROCCurve("dataset", "classifier")
    assert isinstance(titled, ROOT.TGraph)
    assert titled.GetTitle() == "Signal efficiency vs. Background rejection"
    assert (titled.GetXaxis().GetTitle(), titled.GetYaxis().GetTitle()) == (
        "Signal efficiency", "Background rejection",
    )
    bare = factory.GetROCCurve("dataset", "classifier", False)
    assert bare.GetTitle() == "Graph"
    wanted = roc_curve(item.test_values, test.classes == 0, test.weights)
    assert list(graph.GetX()) == list(titled.GetX()) == list(wanted[0])
    assert list(graph.GetY()) == list(titled.GetY()) == list(wanted[1])
    assert ROOT.TMVA.Factory is Factory


def test_public_roc_canvases_keep_the_original_color_sequence_and_legend(root_session):
    import xrdroot.pyroot as ROOT
    from xrdroot.pyroot.graphics import hook

    # The fresh session fixture disables drawing until the test requests graphics.
    hook.install()
    factory, loader, item = _factory()
    factory.booked[loader.GetName()] = [
        Booked(Method("job", f"method{i}", loader.info), loader,
               item.train_values, item.test_values)
        for i in range(10)
    ]
    canvas = factory.GetROCCurve(loader)
    graphs = [obj for obj in canvas.GetListOfPrimitives() if isinstance(obj, ROOT.TGraph)]
    assert isinstance(canvas, ROOT.TCanvas)
    assert canvas.GetName() == "dataset"
    assert [graph.GetLineColor() for graph in graphs] == [1, 2, 3, 4, 6, 7, 8, 9, 11, 12]
    assert all(graph.GetLineWidth() == 2 for graph in graphs)
    assert canvas._tmva_legend.GetNRows() == 11


def test_raw_roc_and_public_queries_work_with_downward_providers_and_root_imports_blocked(tmp_path):
    program = """
        import sys
        from types import ModuleType

        class BlockAdapters:
            def find_spec(self, fullname, path=None, target=None):
                if fullname.startswith(("xrdroot.pyroot", "xrdroot.cint", "xrdroot.canvas",
                                        "xrdroot.plot")):
                    raise AssertionError("TMVA imported " + fullname)

        sys.meta_path.insert(0, BlockAdapters())
        import numpy as np
        from xrdroot.plugins import PROVIDERS
        from xrdroot.tmva.dataset import DataSet, Events
        from xrdroot.tmva.envelope import ClassificationResult
        from xrdroot.tmva.evaluation import roc_curve
        from xrdroot.tmva.factory import Booked, Factory
        from xrdroot.tmva.loader import DataLoader
        from xrdroot.tmva.method import Method

        module = ModuleType("_tmva_downward")
        module.result = lambda title, sensitivity, specificity: (title, sensitivity, specificity)
        module.graph = lambda title, sensitivity, specificity, titled: (
            title, sensitivity, specificity
        )
        module.canvas = lambda factory, dataset: factory
        sys.modules[module.__name__] = module
        for point, attribute in (("tmva.result_roc_graph", "result"),
                                 ("tmva.factory_roc_graph", "graph"),
                                 ("tmva.roc_canvas", "canvas")):
            PROVIDERS[point, "default"] = ("core", module.__name__, attribute)
        values, signal, weights = np.array([.1, .1, .8, .4]), np.array([1, 0, 1, 0]), np.ones(4)
        result = ClassificationResult("dataset", "LD", "classifier", values, signal, weights)
        graph = result.GetROCGraph()
        assert graph[0] == "classifier"
        assert np.array_equal(graph[1], roc_curve(values, signal, weights)[0])
        factory = Factory("job", "Silent:AnalysisType=Classification")
        loader = DataLoader("dataset")
        loader.info.AddClass("Signal")
        loader.info.AddClass("Background")
        empty = np.zeros((4, 0))
        events = Events(np.zeros((4, 1)), empty, empty, 1 - signal, weights)
        loader._dataset = DataSet(events, events)
        item = Booked(Method("job", "classifier", loader.info), loader, values, values)
        factory.booked[loader.GetName()] = [item]
        assert factory.GetROCCurve(loader) is factory
        assert factory.GetROCCurve(loader, "classifier")[0] == "classifier"
        assert not any(name.startswith(("xrdroot.pyroot", "xrdroot.cint", "xrdroot.canvas",
                                        "xrdroot.plot")) for name in sys.modules)
        """
    source = str(Path(xrdroot.__file__).parent.parent)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((source, environment.get("PYTHONPATH", "")))
    subprocess.run(
        [sys.executable, "-c", textwrap.dedent(program)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
