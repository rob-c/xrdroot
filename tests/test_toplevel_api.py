"""The package facade keeps engine imports out of a cold bootstrap without changing its API."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import ModuleType

import pytest

import xrdroot

#: The public import contract predates lazy loading, including its star-import order.
_PUBLIC = """
open_root chain ROOTFile Directory Key
create update WritableFile WritableDirectory WritableTree WritableRNTuple
merge copy Merged MergeWarning
TTree Branch Group Jagged RNTuple RField Chain ChainedBranch EntryList
Histogram Axis Profile Efficiency SparseHistogram Stack Graph MultiGraph Function Canvas Image
fit FitResult loc rebin underflow overflow stats plot
RDataFrame RunGraphs EnableImplicitMT TRandom3 gRandom gROOT gDirectory
compile_formula Formula FormulaError ROOTError FormatError UnsupportedFeatureError
""".split()

#: These are the defining modules of the existing public API, rather than replacement wrappers.
_ORIGINS = {
    "canvas": "Canvas",
    "chain": "Chain ChainedBranch chain",
    "efficiency": "Efficiency",
    "entries": "EntryList",
    "errors": "FormatError ROOTError UnsupportedFeatureError",
    "file": "Directory Key ROOTFile open_root",
    "fit": "FitResult",
    "formula": "Formula FormulaError compile_formula",
    "function": "Function",
    "graph": "Graph",
    "hist": "Axis Histogram",
    "image": "Image",
    "merging": "Merged MergeWarning copy merge",
    "profile": "Profile",
    "random": "TRandom3 gRandom",
    "rdf": "EnableImplicitMT RDataFrame RunGraphs",
    "rntuple": "RField RNTuple WritableRNTuple",
    "session": "gDirectory gROOT",
    "slicing": "loc overflow rebin underflow",
    "sparse": "SparseHistogram",
    "stacks": "MultiGraph Stack",
    "tree": "Branch Group Jagged TTree",
    "writer": "WritableDirectory WritableFile create",
    "wtree": "WritableTree",
    "wupdate": "update",
}


def _fresh_python(program: str, tmp_path: Path) -> None:
    """Use the same interpreter and selected source, outside the suite's imported namespace."""
    source = str(Path(xrdroot.__file__).parent.parent)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((source, environment.get("PYTHONPATH", "")))
    subprocess.run(
        [sys.executable, "-c", textwrap.dedent(program)],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_a_cold_import_and_directory_listing_do_not_load_engines_or_transport(tmp_path):
    _fresh_python(
        """
        import sys

        class BlockEngines:
            def find_spec(self, fullname, path=None, target=None):
                if fullname.startswith(("xrdroot.", "xrdclient", "numpy")):
                    raise AssertionError("The bootstrap imported " + fullname)

        sys.meta_path.insert(0, BlockEngines())
        import xrdroot

        names = dir(xrdroot)
        assert set(xrdroot.__all__) <= set(names)
        assert {"file", "tree", "buffer", "compression", "objects",
                "random", "load_ipython_extension"} <= set(names)
        assert "Histogram" not in vars(xrdroot)
        assert names == sorted(set(names))
        assert [name for name in sys.modules if name.startswith("xrdroot")] == ["xrdroot"]
        assert not hasattr(xrdroot, "no_such_export")
        """,
        tmp_path,
    )


def test_the_public_export_list_and_star_imports_keep_the_existing_contract():
    assert xrdroot.__all__ == _PUBLIC
    namespace: dict = {}
    exec("from xrdroot import *", namespace)
    assert {name for name in namespace if name != "__builtins__"} == set(_PUBLIC)
    assert all(namespace[name] is getattr(xrdroot, name) for name in _PUBLIC)


@pytest.mark.parametrize(
    ("module", "name"),
    [(module, name) for module, names in _ORIGINS.items() for name in names.split()],
)
def test_public_exports_are_the_defining_objects_and_are_cached(module, name, monkeypatch):
    expected = getattr(importlib.import_module(f"xrdroot.{module}"), name)
    monkeypatch.delattr(xrdroot, name, raising=False)
    assert getattr(xrdroot, name) is expected
    assert vars(xrdroot)[name] is expected
    assert getattr(xrdroot, name) is expected


@pytest.mark.parametrize(
    "name",
    ["fit", "plot", "stats", "tree", "random", "file", "buffer", "compression", "objects"],
)
def test_explicit_submodules_remain_accessible_by_their_names(name, monkeypatch):
    expected = importlib.import_module(f"xrdroot.{name}")
    monkeypatch.delattr(xrdroot, name, raising=False)
    assert getattr(xrdroot, name) is expected
    assert vars(xrdroot)[name] is expected
    assert name in dir(xrdroot)


def test_importing_the_chain_submodule_first_keeps_the_public_function(tmp_path):
    _fresh_python(
        """
        import importlib
        import xrdroot

        module = importlib.import_module("xrdroot.chain")
        assert xrdroot.chain is module.chain
        from xrdroot import chain
        assert chain is module.chain
        assert xrdroot.Chain is module.Chain
        """,
        tmp_path,
    )


def test_a_chain_submodule_binding_resolves_back_to_the_original_function(monkeypatch):
    module = importlib.import_module("xrdroot.chain")
    monkeypatch.setattr(xrdroot, "chain", module)
    assert xrdroot.chain is module.chain
    assert vars(xrdroot)["chain"] is module.chain


def test_explicit_overrides_of_the_chain_function_remain_untouched(monkeypatch):
    replacement = object()
    monkeypatch.setattr(xrdroot, "chain", replacement)
    assert xrdroot.chain is replacement
    foreign_module = ModuleType("another_chain")
    monkeypatch.setattr(xrdroot, "chain", foreign_module)
    assert xrdroot.chain is foreign_module


def test_missing_attributes_keep_the_standard_module_error():
    with pytest.raises(AttributeError, match="module 'xrdroot' has no attribute 'not_an_export'"):
        assert xrdroot.not_an_export
    assert not hasattr(xrdroot, "not_an_export")


def test_loading_the_ipython_extension_still_delegates_to_the_existing_loader(monkeypatch):
    magics = importlib.import_module("xrdroot.cli.magics")
    seen = []
    monkeypatch.setattr(magics, "load", seen.append)
    shell = object()
    xrdroot.load_ipython_extension(shell)
    assert seen == [shell]


@pytest.mark.parametrize("name", ["create", "writer"])
def test_writer_exports_can_be_the_first_package_attribute_loaded(name, tmp_path):
    _fresh_python(
        """
        import importlib
        import sys
        import xrdroot

        assert "xrdroot.writer" not in sys.modules
        assert "xrdroot.efficiency" not in sys.modules
        exported = getattr(xrdroot, FIRST_EXPORT)
        writer = importlib.import_module("xrdroot.writer")
        assert exported is (writer.create if FIRST_EXPORT == "create" else writer)
        assert xrdroot.create is writer.create
        efficiency = importlib.import_module("xrdroot.efficiency")
        distributions = importlib.import_module("xrdroot.function.distributions")
        assert distributions.regularized_beta is efficiency.regularized_beta
        assert distributions.beta_quantile is efficiency.beta_quantile
        # Symmetry is stable; log/exp rounding can differ across platform libm versions.
        assert abs(efficiency.regularized_beta(0.5, 2.0, 2.0) - 0.5) < 1e-14
        assert abs(efficiency.beta_quantile(0.5, 2.0, 2.0) - 0.5) < 1e-14
        """.replace("FIRST_EXPORT", repr(name)),
        tmp_path,
    )


@pytest.mark.parametrize("module", ["xrdroot.efficiency", "xrdroot.function"])
def test_scalar_beta_import_order_keeps_shared_functions_and_mutable_terms(module, tmp_path):
    _fresh_python(
        """
        import importlib

        first = importlib.import_module(FIRST_MODULE)
        assert first.__name__ == FIRST_MODULE
        efficiency = importlib.import_module("xrdroot.efficiency")
        function = importlib.import_module("xrdroot.function")
        distributions = importlib.import_module("xrdroot.function.distributions")
        assert function.Function.__module__ == "xrdroot.function.function"
        assert distributions.regularized_beta is efficiency.regularized_beta
        assert distributions.beta_quantile is efficiency.beta_quantile
        assert efficiency.regularized_beta.__globals__ is vars(efficiency)
        assert efficiency.beta_quantile.__globals__ is vars(efficiency)
        # Symmetry is stable; log/exp rounding can differ across platform libm versions.
        assert abs(distributions.regularized_beta(0.5, 2.0, 2.0) - 0.5) < 1e-14
        original = efficiency._TERMS
        try:
            efficiency._TERMS = 2
            try:
                distributions.regularized_beta(0.3, 200.0, 300.0)
            except ArithmeticError as error:
                assert "did not converge" in str(error)
            else:
                raise AssertionError("The shared efficiency._TERMS setting was ignored.")
        finally:
            efficiency._TERMS = original
        assert efficiency._TERMS == original
        """.replace("FIRST_MODULE", repr(module)),
        tmp_path,
    )
