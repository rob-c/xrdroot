"""What is read here goes straight into the rest of scientific Python.

Every column is a NumPy array, every histogram speaks the plotting protocol
``hist`` and ``mplhep`` share, and a tree is one keyword from being a pandas,
Awkward, Arrow or Polars table. These check each of those hand-overs against
the library on the other side, rather than against what it is believed to do.
"""

from __future__ import annotations

import importlib
import io
import pathlib

import numpy as np
import pytest

from xrdroot import Histogram, Jagged, create, open_root
from xrdroot.errors import UnsupportedFeatureError
from xrdroot.library import convert

DATA = pathlib.Path(__file__).parent / "data"
COLUMNS = ["Int32", "Float64", "ArrayInt32", "SliceFloat32", "Str"]


@pytest.fixture
def flat():
    with open_root(str(DATA / "small-flat-tree.root")) as handle:
        yield handle["tree"]


@pytest.fixture
def h1d():
    with open_root(str(DATA / "gauss-h1.root")) as handle:
        yield handle["h1d"]


def written(**objects) -> bytes:
    buf = io.BytesIO()
    with create(buf) as out:
        for name, obj in objects.items():
            out[name] = obj
    return buf.getvalue()


def read_back(data: bytes):
    return open_root(io.BytesIO(data))


# -- columns ----------------------------------------------------------------


def test_every_column_shape_comes_back_as_numpy(flat):
    batch = flat.arrays(COLUMNS, 0, 3)
    assert batch["Int32"].dtype == np.int32 and batch["Int32"].tolist() == [0, 1, 2]
    assert batch["ArrayInt32"].shape == (3, 10)
    assert batch["SliceFloat32"].content.dtype == np.float32
    assert flat["ArrayInt32"].column.dtype == np.int32
    assert batch["Str"] == ["evt-000", "evt-001", "evt-002"]


def test_jagged_rows_slice_into_jagged_rows_and_compare_by_value():
    rows = Jagged(np.array([1.0, 2.0, 3.0, 4.0]), [0, 2, 2, 4])
    middle = rows[1:3]
    assert isinstance(middle, Jagged) and middle.tolist() == [[], [3.0, 4.0]]
    assert middle.flat.tolist() == [3.0, 4.0]
    assert [row.tolist() for row in rows[::2]] == [[1.0, 2.0], [3.0, 4.0]]
    assert rows[5:1].tolist() == []
    assert middle == Jagged(np.array([3.0, 4.0]), [0, 0, 2])
    assert middle != Jagged(np.array([3.0, 5.0]), [0, 0, 2])
    assert (rows == "rows") is False


def test_jagged_rows_go_to_awkward_and_arrow_as_lists(flat):
    import awkward as ak
    import pyarrow as pa

    rows = flat["SliceFloat32"].array(0, 4)
    assert ak.to_list(rows.to_awkward()) == rows.tolist()
    arrow = rows.to_arrow()
    assert pa.types.is_large_list(arrow.type) and arrow.to_pylist() == rows.tolist()


def test_a_tree_is_a_pandas_frame_with_rows_as_objects(flat):
    frame = flat.arrays(COLUMNS, 0, 3, library="pd")
    assert list(frame.columns) == COLUMNS
    assert frame["Float64"].tolist() == [0.0, 1.0, 2.0]
    assert frame["ArrayInt32"][1].tolist() == [1] * 10
    assert frame["SliceFloat32"][2].tolist() == [2.0, 2.0]
    assert frame["Str"][0] == "evt-000"


def test_a_tree_is_an_awkward_record_array(flat):
    import awkward as ak

    events = flat.arrays(COLUMNS, 0, 3, library="ak")
    assert ak.fields(events) == COLUMNS
    assert ak.to_list(events["SliceFloat32"]) == [[], [1.0], [2.0, 2.0]]
    assert ak.to_list(events["ArrayInt32"][2]) == [2] * 10
    assert ak.to_list(events["Str"]) == ["evt-000", "evt-001", "evt-002"]


def test_a_tree_is_an_arrow_table_and_a_polars_frame(flat):
    import pyarrow as pa

    table = flat.arrays(COLUMNS, 0, 3, library="pa")
    assert table.column_names == COLUMNS
    assert pa.types.is_fixed_size_list(table["ArrayInt32"].type)
    assert table["SliceFloat32"].to_pylist() == [[], [1.0], [2.0, 2.0]]
    frame = flat.arrays(COLUMNS, 0, 3, library="polars")
    assert frame["Int32"].to_list() == [0, 1, 2]
    assert frame["SliceFloat32"].to_list() == [[], [1.0], [2.0, 2.0]]


def test_iterating_hands_each_batch_to_the_library_asked_for(flat):
    frames = list(flat.iterate(["Int32"], step=40, library="pd"))
    assert [len(frame) for frame in frames] == [40, 40, 20]


def test_a_library_nobody_has_heard_of_is_refused_with_the_ones_there_are(flat):
    with pytest.raises(ValueError, match="not one of np, pd, ak, pa and pl"):
        flat.arrays(["Int32"], library="excel")


def test_a_library_that_is_not_installed_says_how_to_install_it(monkeypatch):
    real = importlib.import_module

    def missing(name, *args):
        if name == "polars":
            raise ImportError(name)
        return real(name, *args)

    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(UnsupportedFeatureError, match="pip install polars"):
        convert({"x": np.zeros(2)}, "pl")


# -- histograms -------------------------------------------------------------


def test_an_axis_is_a_sequence_of_bins_as_the_plotting_protocol_asks(h1d):
    axis = h1d.axes[0]
    assert (axis.traits.circular, axis.traits.discrete) == (False, False)
    assert axis[0] == (axis.low, axis.edges()[1])
    assert axis[-1][1] == axis.high
    assert list(axis) == [axis[index] for index in range(len(axis))]
    assert axis.widths().tolist() == pytest.approx([0.8] * 10)
    assert axis == h1d.axes[0] and axis != Histogram.new("x", [0, 1], [1]).axes[0]
    assert (axis == "xaxis") is False
    assert axis.label == "xaxis"
    with pytest.raises(IndexError, match="bin 10 of an axis of 10"):
        axis[10]


def test_a_weighted_histogram_gives_variances_and_effective_counts(h1d):
    assert h1d.kind == "COUNT" and h1d.weighted
    variances = h1d.variances()
    assert variances.tolist() == h1d.members["TH1"]["fSumw2"][1:-1].tolist()
    expected = np.where(variances > 0, h1d.values() ** 2 / np.where(variances > 0, variances, 1), 0)
    assert h1d.counts().tolist() == pytest.approx(expected.tolist())
    assert h1d.errors().tolist() == pytest.approx(np.sqrt(variances).tolist())
    unweighted = Histogram.new("n", [0, 1, 2], [4, 9])
    assert not unweighted.weighted and unweighted.counts().tolist() == [4.0, 9.0]
    assert unweighted.variances().tolist() == [4.0, 9.0]


def test_a_density_integrates_to_one_and_the_flow_has_none(h1d):
    density = h1d.density()
    assert float((density * h1d.axes[0].widths()).sum()) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="flow bins"):
        h1d.density(flow=True)
    empty = Histogram.new("e", ([0, 1, 2], [0, 1]), np.zeros((2, 1)))
    assert empty.density().tolist() == [[0.0], [0.0]]


def test_to_numpy_gives_what_numpy_histogram_would(h1d):
    values, edges = h1d.to_numpy()
    assert values.tolist() == h1d.values().tolist() and edges.tolist() == h1d.edges().tolist()
    flowed, wide = h1d.to_numpy(flow=True)
    assert len(flowed) == 12 and wide[0] == -np.inf and wide[-1] == np.inf


def test_to_hist_keeps_values_variances_and_the_flow(h1d):
    made = h1d.to_hist()
    assert made.values(flow=True).tolist() == h1d.values(flow=True).tolist()
    assert made.variances(flow=True).tolist() == h1d.variances(flow=True).tolist()
    assert made.axes[0].edges.tolist() == h1d.edges().tolist()
    plain = Histogram.new("n", [0, 1, 2], [4, 9]).to_hist()
    assert plain.values().tolist() == [4.0, 9.0]


def test_to_hist_without_hist_installed_points_at_to_numpy(monkeypatch, h1d):
    import builtins

    real = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == "hist":
            raise ImportError(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(UnsupportedFeatureError, match=r"pip install hist"):
        h1d.to_hist()


def test_a_new_histogram_of_two_or_three_dimensions_knows_its_moments():
    two = Histogram.new(
        "map",
        ([0, 1, 2], [0, 5, 10]),
        [[1, 2], [3, 4]],
        labels=["x", "y"],
        errors=[[1, 1], [1, 2]],
    )
    assert (two.classname, two.shape) == ("TH2D", (2, 2))
    moments = two.members["TH2"]
    assert moments["fTsumwy"] == pytest.approx(1 * 2.5 + 2 * 7.5 + 3 * 2.5 + 4 * 7.5)
    expected = 0.5 * 2.5 + 0.5 * 7.5 * 2 + 1.5 * 2.5 * 3 + 1.5 * 7.5 * 4
    assert moments["fTsumwxy"] == pytest.approx(expected)
    assert [axis.title for axis in two.axes] == ["x", "y"]
    assert two.errors().tolist() == [[1.0, 1.0], [1.0, 2.0]]
    three = Histogram.new("cube", ([0, 1], [0, 1], [0, 1, 2]), np.ones((1, 1, 2)))
    assert three.classname == "TH3D" and three.sum() == 2.0
    assert "fTsumwyz" in three.members["TH3"]


def test_a_new_histogram_refuses_shapes_and_edges_that_do_not_agree():
    with pytest.raises(ValueError, match="need 2 sets of edges"):
        Histogram.new("m", [0, 1, 2], [[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="one, two and three"):
        Histogram.new("m", [0, 1], 5.0)
    with pytest.raises(ValueError, match="errors or variances, not both"):
        Histogram.new("m", [0, 1], [1], errors=[1], variances=[1])
    with pytest.raises(ValueError, match="3 x 1 values for 2 x 2 bins"):
        Histogram.new("m", ([0, 1, 2], [0, 1, 2]), [[1], [2], [3]])


def test_a_2d_histogram_writes_and_reads_back_as_a_th2d():
    made = Histogram.new(
        "map", ([0, 1, 2], [0, 5, 10]), [[1, 2], [3, 4]], variances=[[1, 4], [9, 16]]
    )
    with read_back(written(map=made)) as back:
        again = back["map"]
        assert again.classname == "TH2D"
        assert again.values().tolist() == [[1.0, 2.0], [3.0, 4.0]]
        assert again.variances().tolist() == [[1.0, 4.0], [9.0, 16.0]]
        assert again.edges(1).tolist() == [0.0, 5.0, 10.0]


def test_numpy_histograms_of_every_dimension_are_taken_as_they_come():
    rng = np.random.default_rng(7)
    x, y = rng.normal(size=500), rng.normal(size=500)
    one = np.histogram(x, bins=5)
    two = np.histogram2d(x, y, bins=(3, 4))
    many = np.histogramdd(np.stack([x, y], axis=1), bins=(2, 2))
    with read_back(written(one=one, two=two, many=many)) as back:
        assert back["one"].values().tolist() == one[0].tolist()
        assert back["two"].values().tolist() == two[0].tolist()
        assert back["two"].edges(1).tolist() == two[2].tolist()
        assert back["many"].shape == (2, 2)
    assert Histogram.of(one, name="named").name == "named"
    assert not Histogram.recognises((np.zeros(3), np.zeros((2, 2))))
    assert not Histogram.recognises((np.zeros((2, 2)), np.zeros(3)))
    assert not Histogram.recognises(("not", "arrays"))


def test_a_hist_object_is_written_with_its_flow_variances_and_labels():
    import hist

    made = hist.Hist(
        hist.axis.Regular(4, 0, 4, name="x", label="energy [GeV]"),
        storage=hist.storage.Weight(),
        name="spectrum",
        label="a spectrum",
    )
    made.fill([0.5, 1.5, 1.5, 9.0], weight=[1.0, 2.0, 2.0, 3.0])
    with read_back(written(spectrum=made)) as back:
        again = back["spectrum"]
        assert again.title == "a spectrum" and again.axes[0].title == "energy [GeV]"
        assert again.values(flow=True).tolist() == [0.0, 1.0, 4.0, 0.0, 0.0, 3.0]
        assert again.variances().tolist() == [1.0, 8.0, 0.0, 0.0]
    assert Histogram.of(again) is again


def test_a_boost_histogram_without_flow_is_padded_with_empty_flow_bins():
    import boost_histogram as bh

    made = bh.Histogram(bh.axis.Variable([0, 1, 3], underflow=False, overflow=False))
    made.fill([0.5, 2.0, 2.0])
    again = Histogram.of(made, name="b")
    assert again.values(flow=True).tolist() == [0.0, 1.0, 2.0, 0.0]
    assert again.name == "b"


class Protocol:
    """Just enough of the plotting protocol, for the shapes no library makes."""

    def __init__(self, kind="COUNT", discrete=False, flow_keyword=True, flow_only_below=False):
        self.kind = kind
        self.axes = [Bins(discrete)]
        self._flow_keyword = flow_keyword
        self._flow_only_below = flow_only_below

    def values(self, **flow):
        if flow and not self._flow_keyword:
            raise TypeError("no flow keyword here")
        if flow and self._flow_only_below:
            return np.array([9.0, 1.0, 2.0])  # an underflow bin, and no overflow
        return np.array([1.0, 2.0])

    def variances(self, **flow):
        return None


class Bins(list):
    def __init__(self, discrete):
        super().__init__([(0.0, 1.0), (1.0, 2.0)])
        self.traits = type("Traits", (), {"discrete": discrete, "circular": False})()


def test_a_protocol_histogram_without_a_flow_keyword_is_read_without_one():
    again = Histogram.of(Protocol(flow_keyword=False))
    assert (again.name, again.values().tolist()) == ("", [1.0, 2.0])


def test_a_protocol_histogram_with_flow_at_one_end_only_is_read_without_it():
    again = Histogram.of(Protocol(flow_only_below=True))
    assert again.values(flow=True).tolist() == [0.0, 1.0, 2.0, 0.0]


def test_protocol_histograms_that_are_not_counts_or_have_categories_are_refused():
    with pytest.raises(UnsupportedFeatureError, match="TProfile"):
        Histogram.of(Protocol(kind="MEAN"))
    with pytest.raises(UnsupportedFeatureError, match="axis of categories"):
        Histogram.of(Protocol(discrete=True))
    with pytest.raises(TypeError, match="is not a histogram"):
        Histogram.of([1, 2, 3])


def test_numpy_arrays_write_as_the_tarray_their_dtype_names():
    arrays = {"ints": np.arange(3, dtype=np.int64), "floats": np.ones(2, np.float32)}
    with read_back(written(**arrays)) as back:
        assert back.classnames() == {"ints": "TArrayL64", "floats": "TArrayF"}
        assert back["ints"].tolist() == [0, 1, 2]
    with pytest.raises(UnsupportedFeatureError, match="2 dimensions has no ROOT class"):
        written(grid=np.zeros((2, 2)))
