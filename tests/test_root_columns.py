"""Trees written a column at a time, from arrays and from tables.

Filling a tree entry by entry is how ROOT's own API reads, and it is slow in
any language that is not compiled; a whole column of NumPy values packed at
once is how the rest of scientific Python hands data over. These check that
the fast way writes the same tree the careful way does, and refuses the same
things it refuses.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from xrdroot import create, open_root
from xrdroot.buffer import gather
from xrdroot.wtree import _typecode, spec_of


def read_back(data: bytes):
    return open_root(io.BytesIO(data))


def tree_of(columns, filler, basket_size=64) -> bytes:
    buf = io.BytesIO()
    with create(buf) as out:
        filler(out.tree("events", columns, basket_size=basket_size))
    return buf.getvalue()


def test_numpy_spells_every_column_type_as_well_as_python_does():
    assert _typecode("x", np.float32) == ("f", 1)
    assert _typecode("x", "float64") == ("d", 1)
    assert _typecode("x", np.dtype("<i8")) == (_typecode("x", "q")[0], 1)
    assert _typecode("x", (np.uint8, 16)) == ("B", 16)
    assert _typecode("x", np.bool_) == ("?", 1)
    with pytest.raises(ValueError, match="is of type 'float16'"):
        _typecode("x", np.dtype("float16"))


def test_a_column_at_a_time_makes_the_baskets_an_entry_at_a_time_makes():
    energies = np.arange(50, dtype=np.float64)
    hits = np.arange(100, dtype=np.int32).reshape(50, 2)

    def by_rows(tree):
        for energy, hit in zip(energies, hits):
            tree.fill(energy=energy, hits=hit)

    def by_columns(tree):
        tree.extend({"energy": energies[:7], "hits": hits[:7]})
        tree.fill(energy=energies[7], hits=hits[7])
        tree.extend({"energy": energies[8:], "hits": hits[8:]})

    columns = {"energy": "d", "hits": ("i", 2)}
    with (
        read_back(tree_of(columns, by_rows)) as one,
        read_back(tree_of(columns, by_columns)) as two,
    ):
        for name in columns:
            assert (
                one["events"][name].record.basket_entry == two["events"][name].record.basket_entry
            )
        assert two["events"]["energy"].array().tolist() == energies.tolist()
        assert two["events"]["hits"].array().tolist() == hits.tolist()
        assert len(two["events"]) == 50


def test_a_table_of_arrays_written_under_a_name_becomes_a_tree():
    table = {
        "pt": np.array([10.5, 20.25], dtype=np.float32),
        "n": np.array([1, 2], dtype=np.int16),
        "p4": np.ones((2, 4)),
        "ok": np.array([True, False]),
    }
    buf = io.BytesIO()
    with create(buf) as out:
        out["events"] = table
    with read_back(buf.getvalue()) as back:
        tree = back["events"]
        assert tree.typenames() == {"pt": "float32", "n": "int16", "p4": "float64", "ok": "bool"}
        assert tree["p4"].array().shape == (2, 4)
        assert tree["ok"].array().tolist() == [True, False]


def test_frames_from_pandas_polars_and_arrow_are_trees_too():
    import pandas as pd
    import polars as pl
    import pyarrow as pa

    values = {"a": np.arange(3), "b": np.linspace(0.0, 1.0, 3)}
    buf = io.BytesIO()
    with create(buf) as out:
        out["pandas"] = pd.DataFrame(values)
        out["polars"] = pl.DataFrame(values)
        out["arrow"] = pa.table(values)
    with read_back(buf.getvalue()) as back:
        for name in ("pandas", "polars", "arrow"):
            assert back[name].arrays(library="pd")["b"].tolist() == [0.0, 0.5, 1.0]


def test_a_batch_of_columns_that_does_not_fit_is_refused_whole():
    buf = io.BytesIO()
    with create(buf) as out:
        tree = out.tree("events", {"x": "d", "n": "b"})
        with pytest.raises(ValueError, match="different numbers of entries"):
            tree.extend({"x": np.zeros(3), "n": np.zeros(2, np.int8)})
        with pytest.raises(ValueError, match="without losing what they are"):
            tree.extend({"x": np.zeros(2), "n": np.array([0.5, 1.5])})
        with pytest.raises(ValueError, match="from -128 to 127, and these run from 0 to 300"):
            tree.extend({"x": np.zeros(2), "n": np.array([0, 300])})
        with pytest.raises(ValueError, match="takes 1 values per entry, and these are shaped"):
            tree.extend({"x": np.zeros((2, 2)), "n": np.zeros(2, np.int8)})
        with pytest.raises(ValueError, match="nothing for n"):
            tree.extend({"x": np.zeros(2)})
        assert len(tree) == 0
        tree.extend({"x": np.zeros(0), "n": np.zeros(0, np.int8)})
        assert len(tree) == 0


def test_a_column_given_as_one_value_is_not_a_column():
    assert spec_of("x", [1.0, 2.0]) == np.dtype("float64")
    assert spec_of("x", np.zeros((3, 2, 2))) == (np.dtype("float64"), 4)
    with pytest.raises(ValueError, match="a single float64 rather than a value per entry"):
        spec_of("x", 1.0)


def test_runs_are_gathered_in_order_or_out_of_it():
    data = bytes(range(20))
    assert gather(data, [2, 2, 5, 9], [3, 0, 2, 1]) == data[2:5] + data[5:7] + data[9:10]
    assert gather(data, [0, 3], [3, 2]) == data[0:5]
    assert gather(data, [5, 0], [2, 3]) == data[5:7] + data[0:3]
    assert gather(data, [1], [0]) == b""
