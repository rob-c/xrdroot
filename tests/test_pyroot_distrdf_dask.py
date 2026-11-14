"""Distributed RDataFrame on Dask, ``ROOT.RDF.Distributed`` and what every worker is sent.

A frame given a Dask client as its ``executor`` runs each range as a task on
the cluster - a cluster of one worker thread in this process, here - and
the code, headers and libraries registered are declared, sent and loaded
on every worker before its tasks start.
"""

from __future__ import annotations

import logging
import sys
import types
import warnings

import pytest

import xrdroot.pyroot as ROOT
from xrdroot.errors import UnsupportedFeatureError
from xrdroot.pyroot import _distrdf
from xrdroot.pyroot._distrdf.Backends import Base, Utils
from xrdroot.pyroot._distrdf.Backends.Dask import Backend
from xrdroot.pyroot._distrdf.Backends.Dask import RDataFrame as dask_frame

distributed = pytest.importorskip("distributed")
dask_distributed = pytest.importorskip("dask.distributed")


@pytest.fixture(scope="module")
def client():
    cluster = distributed.LocalCluster(n_workers=1, threads_per_worker=1, processes=False,
                                       dashboard_address=None)  # fmt: skip
    made = distributed.Client(cluster)
    yield made
    made.close()
    cluster.close()


@pytest.fixture(autouse=True)
def _registrations():
    """What a test registers on every backend is gone after it."""
    kept = {name: getattr(Base.BaseBackend, name) for name in
            ("headers", "files", "pcms", "shared_libraries", "strings_to_declare")}  # fmt: skip
    copies = {name: type(value)(value) for name, value in kept.items()}
    initialization = Base.BaseBackend.__dict__["initialization"]
    for name, value in copies.items():
        setattr(Base.BaseBackend, name, value)
    yield
    for name, value in kept.items():
        setattr(Base.BaseBackend, name, value)
    Base.BaseBackend.initialization = initialization


def _header(path, name):
    path.write_text(f"bool {name}(int x) {{ return x > 0; }}\n")
    return str(path)


def test_a_frame_given_a_dask_client_runs_on_its_cluster(client):
    df = ROOT.RDataFrame(100, executor=client, npartitions=4)
    defined = df.Define("x", "rdfentry_ * 1.0")
    histo = defined.Histo1D(("h", "h", 10, 0, 100), "x")
    assert histo.GetEntries() == 100 and defined.Mean("x").GetValue() == 49.5
    assert ROOT.RDataFrame(10, executor=client).Count().GetValue() == 10
    assert _distrdf._graph_cache._RDF_REGISTER == {}


def test_rungraphs_runs_each_graph_once_and_says_how_many(client, caplog):
    sums = [ROOT.RDataFrame(10, executor=client).Define("x", "1").Sum("x") for _ in range(3)]
    sums.append(sums[0])
    assert ROOT.RDF.RunGraphs(sums) == 3
    assert [each.proxied_node.value for each in sums] == [10] * 4
    with caplog.at_level(logging.WARNING):
        assert _distrdf.RunGraphs([]) == 0
    assert "empty list of handles" in caplog.text
    local = ROOT.RDataFrame(3).Count()
    assert ROOT.RDF.RunGraphs([local]) == 1


def test_the_old_dask_constructor_warns_and_refuses_two_clients(client):
    with pytest.warns(FutureWarning, match="'daskclient' is not necessary"):
        assert dask_frame(5, daskclient=client).Count().GetValue() == 5
    with pytest.raises(ValueError, match="Both the 'daskclient' and 'executor'"):
        with pytest.warns(FutureWarning):
            dask_frame(5, daskclient=client, executor=client)


def test_an_executor_that_is_no_client_is_refused(monkeypatch):
    with pytest.raises(ValueError, match="Missing keyword argument 'executor'"):
        _distrdf.RDataFrame(5)
    with pytest.raises(TypeError, match="not a supported connection type"):
        _distrdf.RDataFrame(5, executor=object())
    SparkContext = type("SparkContext", (), {})
    with pytest.raises(UnsupportedFeatureError, match="Spark is not supported"):
        ROOT.RDataFrame(5, executor=SparkContext())
    monkeypatch.setattr(_distrdf.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(TypeError, match="not a supported connection type"):
        _distrdf.RDataFrame(5, executor=object())


def test_what_rdf_distributed_has_and_what_it_refuses(client):
    module = _distrdf.distributed_module()
    assert module is _distrdf.distributed_module()
    assert module.Dask.__package__ == "ROOT.RDF.Distributed" and module.__package__ == "ROOT.RDF"
    assert module.Spark.RDataFrame is not None and module.initialize is _distrdf.initialize
    with pytest.raises(UnsupportedFeatureError, match="from JSON"):
        module.FromSpec("spec.json", executor=None)
    with pytest.raises(UnsupportedFeatureError, match="LiveVisualize draws partial results"):
        module.LiveVisualize({})
    with pytest.raises(UnsupportedFeatureError, match="no Vary"):
        module.VariationsFor(ROOT.RDataFrame(5, executor=client).Count())


class _FakeClient:
    """Enough of a client to make a frame on, though nothing will run."""

    def ncores(self):
        return {"worker": 1}


def test_a_frame_needs_no_cluster_until_a_result_is_asked_for(monkeypatch):
    monkeypatch.setattr(distributed, "Client", _FakeClient)
    df = ROOT.RDataFrame(10, executor=_FakeClient(), npartitions=3)
    assert df._headnode.npartitions == 3
    assert Backend.DaskBackend(_FakeClient()).optimize_npartitions() == 1


def test_a_backend_given_no_client_makes_a_cluster_of_this_machine(monkeypatch):
    made = []
    monkeypatch.setattr(dask_distributed, "LocalCluster", lambda **options: made.append(options))
    monkeypatch.setattr(dask_distributed, "Client", lambda cluster: "client")
    assert Backend.DaskBackend().client == "client"
    assert made[0]["processes"] is True and made[0]["threads_per_worker"] == 1


def test_without_dask_a_distributed_frame_says_what_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "dask", None)
    with pytest.raises(UnsupportedFeatureError, match=r'pip install "xrdroot\[distrdf\]"'):
        Backend.dask_modules()


def test_a_workers_scripts_import_root_as_this_namespace(monkeypatch):
    monkeypatch.delitem(sys.modules, "ROOT", raising=False)
    Backend._root_for_scripts()
    assert sys.modules["ROOT"] is ROOT
    shim = types.ModuleType("ROOT")
    monkeypatch.setitem(sys.modules, "ROOT", shim)
    Backend._root_for_scripts()
    assert sys.modules["ROOT"] is shim


def _note(path):
    with open(path, "a") as file:
        file.write("ran\n")


def test_initialization_runs_here_and_first_in_every_task(client, tmp_path):
    path = tmp_path / "ran.txt"
    _distrdf.initialize(_note, str(path))
    assert path.read_text() == "ran\n"
    assert ROOT.RDataFrame(4, executor=client, npartitions=2).Count().GetValue() == 4
    assert path.read_text() == "ran\n" * 3


def test_code_headers_and_files_reach_every_worker(client, tmp_path):
    _distrdf.DistributeCppCode("int distrdf_twice(int x) { return 2 * x; }")
    _distrdf.DistributeCppCode("int distrdf_twice(int x) { return 2 * x; }")
    assert len(Base.BaseBackend.strings_to_declare) == 1
    _distrdf.DistributeHeaders(_header(tmp_path / "one.hxx", "distrdf_one"))
    (tmp_path / "more").mkdir()
    _distrdf.DistributeHeaders([_header(tmp_path / "more" / "two.hxx", "distrdf_two")])
    (tmp_path / "data.txt").write_text("7")
    _distrdf.DistributeFiles(str(tmp_path / "data.txt"))
    _distrdf.DistributeSharedLibs(str(tmp_path / "more"))
    assert ROOT.distrdf_one(1) and ROOT.distrdf_two(1)
    df = ROOT.RDataFrame(4, executor=client, npartitions=2)
    assert df.Define("y", "distrdf_twice(3)").Sum("y").GetValue() == 24
    assert len(Base.BaseBackend.headers) == 2 and len(Base.BaseBackend.files) == 1


def test_shared_libraries_load_or_say_why_not(tmp_path):
    Utils.distribute_shared_libraries(["libHist"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(OSError, match="does not exist"):
            Utils.distribute_shared_libraries([str(tmp_path / "libnone.so")])
        (tmp_path / "libbad.so").write_text("not a library")
        (tmp_path / "libbad.pcm").write_text("")
        (tmp_path / "libother.so").write_text("")
        with pytest.raises(RuntimeError, match=r"loads the \.xrdlib libraries"):
            _distrdf.DistributeSharedLibs([str(tmp_path / "libbad.so")])
    pcms, libraries = Utils.check_pcm_in_library_path(str(tmp_path / "libbad.so"))
    assert (pcms, libraries) == ({str(tmp_path / "libbad.pcm")}, {str(tmp_path / "libbad.so")})
    assert Utils.register_files(str(tmp_path / "missing")) == set()


def test_the_workers_forget_a_run_once_it_is_over(client):
    cache = _distrdf._graph_cache
    key = cache.ExecutionIdentifier("rdf", "graph")
    cache._RDF_REGISTER[key] = cache._ACTIONS_REGISTER[key] = "kept"
    Backend.DaskBackend(client).cleanup_cache(key)
    assert key not in cache._RDF_REGISTER and key not in cache._ACTIONS_REGISTER


def test_a_backend_without_its_methods_cannot_be_made():
    with pytest.raises(TypeError):
        Base.BaseBackend()
    Base.BaseBackend.cleanup_cache(Backend.DaskBackend(_FakeClient()), None)
