"""Declared built-in extension targets, resolved only when a caller needs them.

The table records deliberate dispatch across component boundaries. Resolving a
target does not discover packages or retain objects whose module may be replaced.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from typing import Any, cast

from .errors import UnsupportedFeatureError

__all__ = ["PROVIDERS", "keys", "lookup", "lookup_loaded", "require"]

#: Extension point and key against the owning component, module and attribute.
PROVIDERS: dict[tuple[str, str], tuple[str, str, str]] = {
    ("session.preloaded", "default"): ("app", "xrdroot.apinamespace", "preloaded"),
    ("display.plot", "default"): ("graphics", "xrdroot.plot", "plot"),
    ("display.html", "default"): ("graphics", "xrdroot.plot.notebook", "figure"),
    ('io.workspace', 'RooWorkspace'): (
        'roofit', 'xrdroot.roofit.io.workspace', 'read_workspace',
    ),
    ('canvas.streamer', 'TCanvas'): (
        'graphics', 'xrdroot.canvas.streamer', 'streamer_for',
    ),
    ('canvas.streamer', 'TQObject'): (
        'graphics', 'xrdroot.canvas.streamer', 'streamer_for',
    ),
    ('canvas.dress', 'TCanvas'): (
        'graphics', 'xrdroot.canvas', 'Canvas',
    ),
    ('canvas.dress', 'TPad'): (
        'graphics', 'xrdroot.canvas', 'Pad',
    ),
    ('canvas.dress', 'TFrame'): (
        'graphics', 'xrdroot.canvas', 'Primitive',
    ),
    ('canvas.dress', 'TPave'): (
        'graphics', 'xrdroot.canvas', 'Primitive',
    ),
    ('canvas.dress', 'TPaveText'): (
        'graphics', 'xrdroot.canvas', 'Primitive',
    ),
    ('canvas.dress', 'TPavesText'): (
        'graphics', 'xrdroot.canvas', 'Primitive',
    ),
    ('canvas.dress', 'TPaveLabel'): (
        'graphics', 'xrdroot.canvas', 'Primitive',
    ),
    ('canvas.dress', 'TPaveStats'): (
        'graphics', 'xrdroot.canvas', 'Primitive',
    ),
    ('canvas.dress', 'TDiamond'): (
        'graphics', 'xrdroot.canvas', 'Primitive',
    ),
    ('canvas.dress', 'TPaletteAxis'): (
        'graphics', 'xrdroot.canvas', 'Primitive',
    ),
    ('canvas.dress', 'TLegend'): (
        'graphics', 'xrdroot.canvas', 'Primitive',
    ),
    ('canvas.dress', 'TLegendEntry'): (
        'graphics', 'xrdroot.canvas', 'Primitive',
    ),
    ('canvas.dress', 'TText'): (
        'graphics', 'xrdroot.canvas', 'Primitive',
    ),
    ('canvas.dress', 'TLatex'): (
        'graphics', 'xrdroot.canvas', 'Primitive',
    ),
    ('canvas.dress', 'TLine'): (
        'graphics', 'xrdroot.canvas', 'Primitive',
    ),
    ('canvas.dress', 'TArrow'): (
        'graphics', 'xrdroot.canvas', 'Primitive',
    ),
    ('canvas.dress', 'TBox'): (
        'graphics', 'xrdroot.canvas', 'Primitive',
    ),
    ('canvas.dress', 'TWbox'): (
        'graphics', 'xrdroot.canvas', 'Primitive',
    ),
    ('canvas.dress', 'TEllipse'): (
        'graphics', 'xrdroot.canvas', 'Primitive',
    ),
    ('canvas.dress', 'TArc'): (
        'graphics', 'xrdroot.canvas', 'Primitive',
    ),
    ('canvas.dress', 'TCrown'): (
        'graphics', 'xrdroot.canvas', 'Primitive',
    ),
    ('canvas.dress', 'TMarker'): (
        'graphics', 'xrdroot.canvas', 'Primitive',
    ),
    ('canvas.dress', 'TPolyLine'): (
        'graphics', 'xrdroot.canvas', 'Primitive',
    ),
    ('canvas.dress', 'TCurlyLine'): (
        'graphics', 'xrdroot.canvas', 'Primitive',
    ),
    ('canvas.dress', 'TCurlyArc'): (
        'graphics', 'xrdroot.canvas', 'Primitive',
    ),
    ('canvas.dress', 'TPolyMarker'): (
        'graphics', 'xrdroot.canvas', 'Primitive',
    ),
    ('canvas.dress', 'TGaxis'): (
        'graphics', 'xrdroot.canvas', 'Primitive',
    ),
    ('canvas.dress', 'TColor'): (
        'graphics', 'xrdroot.canvas', 'Primitive',
    ),
    ('io.layouts', 'RooStats::HistFactory::Measurement'): (
        'histfactory', 'xrdroot.histfactory.layouts', 'described',
    ),
    ('io.layouts', 'RooStats::HistFactory::Channel'): (
        'histfactory', 'xrdroot.histfactory.layouts', 'described',
    ),
    ('io.layouts', 'RooStats::HistFactory::Data'): (
        'histfactory', 'xrdroot.histfactory.layouts', 'described',
    ),
    ('io.layouts', 'RooStats::HistFactory::HistRef'): (
        'histfactory', 'xrdroot.histfactory.layouts', 'described',
    ),
    ('io.layouts', 'RooStats::HistFactory::StatErrorConfig'): (
        'histfactory', 'xrdroot.histfactory.layouts', 'described',
    ),
    ('io.layouts', 'RooStats::HistFactory::Sample'): (
        'histfactory', 'xrdroot.histfactory.layouts', 'described',
    ),
    ('io.layouts', 'RooStats::HistFactory::OverallSys'): (
        'histfactory', 'xrdroot.histfactory.layouts', 'described',
    ),
    ('io.layouts', 'RooStats::HistFactory::NormFactor'): (
        'histfactory', 'xrdroot.histfactory.layouts', 'described',
    ),
    ('io.layouts', 'RooStats::HistFactory::HistogramUncertaintyBase'): (
        'histfactory', 'xrdroot.histfactory.layouts', 'described',
    ),
    ('io.layouts', 'RooStats::HistFactory::HistoSys'): (
        'histfactory', 'xrdroot.histfactory.layouts', 'described',
    ),
    ('io.layouts', 'RooStats::HistFactory::HistoFactor'): (
        'histfactory', 'xrdroot.histfactory.layouts', 'described',
    ),
    ('io.layouts', 'RooStats::HistFactory::ShapeSys'): (
        'histfactory', 'xrdroot.histfactory.layouts', 'described',
    ),
    ('io.layouts', 'RooStats::HistFactory::ShapeFactor'): (
        'histfactory', 'xrdroot.histfactory.layouts', 'described',
    ),
    ('io.layouts', 'RooStats::HistFactory::StatError'): (
        'histfactory', 'xrdroot.histfactory.layouts', 'described',
    ),
    ('io.layouts', 'RooStats::HistFactory::PreprocessFunction'): (
        'histfactory', 'xrdroot.histfactory.layouts', 'described',
    ),
    ('io.layouts', 'RooStats::HistFactory::Asimov'): (
        'histfactory', 'xrdroot.histfactory.layouts', 'described',
    ),
    ('workspace.maker', 'RooStats::ModelConfig'): (
        'roostats', 'xrdroot.roostats.workspaceio', 'make_model_config',
    ),
    ("canvas.streamer.custom", "default"): (
        "graphics", "xrdroot.canvas.streamer", "streamer_for",
    ),
    ("canvas.classes", "default"): ("graphics", "xrdroot.canvas", "class_registry"),
    ('roofit.binding_name', 'default'): (
        'pyroot', 'xrdroot.pyroot.roofit.bindings', 'registered_name',
    ),
    ('root.class', 'TMarker'): ('pyroot', 'xrdroot.pyroot.graphics.shapes', 'TMarker'),
    ('roofit.contour_graph', 'default'): ('pyroot', 'xrdroot.pyroot.roofit.contour', 'make_graph'),
    ('roostats.posterior_hist', 'default'): (
        'pyroot', 'xrdroot.pyroot.roostats.mcmcplot', 'posterior_hist',
    ),
    ('roostats.inverter_draw', 'default'): (
        'pyroot', 'xrdroot.pyroot.roostats.inverterplot', 'draw_limit_plot',
    ),
    ('roostats.inverter_plot', 'default'): (
        'pyroot', 'xrdroot.pyroot.roostats.inverterplot', 'make_plot',
    ),
    ('roostats.inverter_exponential', 'default'): (
        'pyroot', 'xrdroot.pyroot.roostats.inverterfits', 'exponential',
    ),
    ('roostats.inverter_error_graph', 'default'): (
        'pyroot', 'xrdroot.pyroot.roostats.inverterfits', 'error_graph',
    ),
    ('roostats.inverter_error_fit', 'default'): (
        'pyroot', 'xrdroot.pyroot.roostats.inverterfits', 'fitted_error',
    ),
    ('histfactory.count_data', 'default'): (
        'pyroot', 'xrdroot.pyroot.roostats.histfactoryio', 'counting_data',
    ),
    ('histfactory.read_histogram', 'default'): (
        'pyroot', 'xrdroot.pyroot.roostats.histfactoryio', 'read_histogram',
    ),
    ('histfactory.count_sample', 'default'): (
        'pyroot', 'xrdroot.pyroot.roostats.histfactoryio', 'counting_sample',
    ),
    ('histfactory.write_model', 'default'): (
        'pyroot', 'xrdroot.pyroot.roostats.histfactoryio', 'write_model',
    ),
    ('tmva.result_roc_graph', 'default'): ('pyroot', 'xrdroot.pyroot.tmvaplot', 'result_graph'),
    ('tmva.factory_roc_graph', 'default'): ('pyroot', 'xrdroot.pyroot.tmvaplot', 'factory_graph'),
    ('tmva.roc_canvas', 'default'): ('pyroot', 'xrdroot.pyroot.tmvaplot', 'roc_canvas'),
}


def keys(point: str) -> tuple[str, ...]:
    """The declared keys of an extension point, without loading its providers."""
    return tuple(key for name, key in PROVIDERS if name == point)


def lookup(point: str, key: str = "default") -> Callable[..., Any] | None:
    """Resolve a declared target, leaving absent entries to the caller's fallback."""
    target = PROVIDERS.get((point, key))
    if target is None:
        return None
    _, module, attribute = target
    return cast(Callable[..., Any], getattr(importlib.import_module(module), attribute))


def require(point: str, key: str = "default") -> Callable[..., Any]:
    """Resolve a required target and name the missing registration if it is absent."""
    found = lookup(point, key)
    if found is None:
        raise UnsupportedFeatureError(f"No provider is registered for {point!r} with key {key!r}.")
    return found


def lookup_loaded(point: str, key: str = "default") -> Callable[..., Any] | None:
    """Resolve an already-loaded provider without importing a component for fallback."""
    target = PROVIDERS.get((point, key))
    if target is None:
        return None
    _, module, attribute = target
    loaded = sys.modules.get(module)
    return None if loaded is None else cast(Callable[..., Any], getattr(loaded, attribute))
