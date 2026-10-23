"""Statistical engines can use plain providers while ROOT and translator imports are blocked."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import xrdroot


def test_engine_operations_use_downward_providers_without_loading_root_adapters(tmp_path):
    program = """
        import math
        import sys
        from types import ModuleType, SimpleNamespace

        class BlockAdapters:
            def find_spec(self, fullname, path=None, target=None):
                if fullname.startswith(("xrdroot.pyroot", "xrdroot.cint", "xrdroot.canvas",
                                        "xrdroot.plot")):
                    raise AssertionError("An engine imported " + fullname)

        sys.meta_path.insert(0, BlockAdapters())
        from xrdroot.plugins import PROVIDERS
        from xrdroot.roofit.bindings import bind_function
        from xrdroot.roofit.fitting import contour
        from xrdroot.roofit.fitting.minimizer import RooMinimizer
        from xrdroot.roofit.functions import RooAddition, RooFormulaVar
        from xrdroot.roofit.variables import RooRealVar
        from xrdroot.roostats import inverterlimits, inverterscan
        from xrdroot.roostats.inverterresult import HypoTestInverterResult
        from xrdroot.roostats.mcmc import MCMCInterval
        from xrdroot.histfactory import make, model
        from xrdroot.histfactory.channel import Channel
        from xrdroot.histfactory.model import Sample

        module = ModuleType("_statistical_downward")
        sys.modules[module.__name__] = module

        def install(point, callback, key="default"):
            attribute = point.replace(".", "_") + "_" + key
            setattr(module, attribute, callback)
            PROVIDERS[point, key] = ("core", module.__name__, attribute)

        install("roofit.binding_name", lambda function: ("plain::erf", ("input",)))
        x, y = RooRealVar("x", "x", 0.5, -2, 2), RooRealVar("y", "y", 1, -2, 2)
        scope = {"__name__": "__cint__", "bind": bind_function, "function": math.erf, "x": x}
        exec("bound = bind('f', function, x)", scope)
        assert scope["bound"].getVal() == math.erf(0.5)
        assert "function=plain::erf input=x" in scope["bound"].printArgs()
        install("root.class", lambda *values: values, "TMarker")
        objective = RooAddition("nll", "nll", [RooFormulaVar("formula", "x*x+y*y", [x, y])])
        minimizer = RooMinimizer(objective)
        frame = minimizer.contour(x, y)
        assert frame.items[0][0] == (0.5, 1.0, 8)
        install("roofit.contour_graph", lambda *values: values[1])
        points = [(0.5, 1.0), (1.0, 0.5)]
        assert contour._graph(minimizer, points, 1.0, 0) is points
        install("roostats.posterior_hist", lambda interval: interval)
        interval = MCMCInterval("interval")
        assert interval.GetPosteriorHist() is interval
        sentinel = object()
        install("roostats.inverter_error_graph", lambda *values: (sentinel, 3))
        install("roostats.inverter_error_fit", lambda *values: 0.125)
        assert inverterlimits._error_graph(sentinel, 1.0, 3.0) == (sentinel, 3)
        assert inverterlimits._fitted_error(sentinel, sentinel, .05, False, 2., (1., 3.)) == .125
        expo = SimpleNamespace(GetParameter=lambda index: 2.0, GetParError=lambda index: .01)
        plot = SimpleNamespace(GetX=lambda: [1., 2., 3.], GetN=lambda: 3,
                               Sort=lambda: None, Fit=lambda *values: None)
        install("roostats.inverter_plot", lambda result: plot)
        install("roostats.inverter_exponential", lambda *values: expo)
        drawn = []
        install("roostats.inverter_draw", lambda *values: drawn.append(values))
        result = HypoTestInverterResult("r", x)
        it = SimpleNamespace(_var=x, _verbose=0, _results=result, _size=.05)
        search = inverterscan._Search(it, .05, (.1, .1))
        search.r_min, search.r_max, search.limit, search.fit_range = 1., 3., 2., (1., 3.)
        assert search.fit() is expo
        assert inverterscan._finish(it, search, expo) == (True, 2.0, .01)
        assert drawn == [(it, .05, 2.0, .01, expo)]
        counted = lambda name, value: SimpleNamespace(GetName=lambda: name)
        install("histfactory.count_data", counted)
        install("histfactory.count_sample", counted)
        channel, sample = Channel("observed"), Sample("background")
        channel.SetData(7)
        sample.SetValue(2.5)
        assert channel.GetData().GetHistoName() == "observed_data"
        assert sample.GetHistoName() == "background_hist"
        install("histfactory.read_histogram", lambda *values: sentinel)
        assert model._histogram({}, "input.root", "region", "data") is sentinel
        written = []
        install("histfactory.write_model", lambda *values: written.append(values))
        make._written(sentinel, "output.root", sentinel)
        assert written == [(sentinel, "output.root", sentinel, None)]
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
