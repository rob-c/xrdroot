"""Shared column values and primitive array layouts keep their compatibility identities."""

from __future__ import annotations

import importlib
import os
import pickle
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

import xrdroot
from xrdroot import jagged, rootarrays
from xrdroot.cxx import Prim


@pytest.mark.parametrize("name", ["Jagged", "concatenate", "scattered", "take", "_bounds"])
def test_tree_compatibility_imports_keep_the_shared_primitive_objects(name):
    tree = importlib.import_module("xrdroot.tree")
    expected = getattr(jagged, name)
    assert getattr(tree, name) is expected
    legacy_global = f"cxrdroot.tree\n{name}\n.".encode()
    assert pickle.loads(legacy_global) is expected


def test_every_consumer_shares_the_same_array_layout_table_and_descriptors():
    for name in ("interp", "hist", "slicing", "writer"):
        module = importlib.import_module(f"xrdroot.{name}")
        assert module.ARRAYS is rootarrays.ARRAYS
    assert all(isinstance(layout, Prim) for layout in rootarrays.ARRAYS.values())
    assert set(rootarrays.ARRAYS) == {
        "TArrayC", "TArrayS", "TArrayI", "TArrayL", "TArrayL64", "TArrayF", "TArrayD",
    }


def test_formula_and_library_consumers_recognize_legacy_and_shared_rows():
    tree = importlib.import_module("xrdroot.tree")
    library = importlib.import_module("xrdroot.library")
    columns = importlib.import_module("xrdroot.formula.columns")
    formula = importlib.import_module("xrdroot.formula.formula")
    select = importlib.import_module("xrdroot.formula.select")
    for module in (library, columns, formula, select):
        assert module.Jagged is jagged.Jagged
    rows = tree.Jagged(np.array([2.0, 5.0, 7.0]), [0, 2, 2, 3])
    result = formula.compile_formula("Sum$(x)", {"x": rows}).evaluate({"x": rows})
    assert result.tolist() == [7.0, 0.0, 7.0]
    assert library.convert({"x": rows}, "np")["x"] is rows


def test_shared_scattered_selection_keeps_duplicates_and_original_row_order():
    values = jagged.Jagged(np.array([10, 11, 30, 40, 41]), [0, 2, 2, 3, 5])
    visits = []

    def read(block, rows):
        visits.append((block, rows.tolist()))
        return values.take(rows)

    result = jagged.scattered([3, 0, 3, 1], [0, 2, 4], read, lambda: values.take([]))
    assert visits == [(0, [0, 1]), (1, [3, 3])]
    assert result.tolist() == [[40, 41], [10, 11], [40, 41], []]
    assert result.offsets.tolist() == [0, 2, 4, 6, 6]
    assert result.content.dtype == values.content.dtype


def test_graph_iteration_and_layer_pairing_still_reject_mismatched_lengths():
    graph = importlib.import_module("xrdroot.graph").Graph.new("g", [0, 1], [2, 3])
    graph.y = np.array([2.0])
    with pytest.raises(ValueError, match="shorter than argument 1"):
        list(graph)
    graph.members["fEyL"], graph.members["fEyH"] = [[0.5, 0.5]], []
    with pytest.raises(ValueError, match="shorter than argument 1"):
        _ = graph.layers


def test_shared_primitives_histograms_and_formulas_work_with_io_imports_blocked(tmp_path):
    program = """
        import importlib
        import sys

        blocked = {
            "chain", "file", "interp", "objects", "tree", "writer", "wtree", "wupdate",
            "rntuple", "streamers", "rdf", "pyroot", "cint", "session", "roofit",
            "roostats", "histfactory", "tmva", "canvas", "plot",
        }

        class BlockHigherComponents:
            def find_spec(self, fullname, path=None, target=None):
                parts = fullname.split(".")
                if parts[0] == "xrdroot" and len(parts) > 1 and parts[1] in blocked:
                    raise AssertionError("A primitive consumer imported " + fullname)

        sys.meta_path.insert(0, BlockHigherComponents())
        import numpy as np
        from xrdroot import jagged, rootarrays, library, hist, formula
        import xrdroot.formula.select
        import xrdroot.graph

        rows = jagged.Jagged(np.array([2.0, 5.0, 7.0]), [0, 2, 2, 3])
        assert rows.take([2, 0]).tolist() == [[7.0], [2.0, 5.0]]
        assert library.convert({"x": rows}, "np")["x"] is rows
        value = formula.compile_formula("Sum$(x)", {"x": rows}).evaluate({"x": rows})
        assert value.tolist() == [7.0, 0.0, 7.0]
        histogram = hist.Histogram.book("h", (2, 0, 2))
        histogram.fill([0.25, 1.25, 1.75])
        assert histogram.values().tolist() == [1.0, 2.0]
        assert rootarrays.ARRAYS["TArrayD"].typename == "float64"
        assert all("xrdroot." + name not in sys.modules for name in blocked)
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
