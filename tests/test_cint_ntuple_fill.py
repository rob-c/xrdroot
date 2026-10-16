"""Batching preserves the scalar loop's bins, statistics, and last-read entry."""
from __future__ import annotations

import numpy as np
import pytest

from xrdroot import pyroot as ROOT
from xrdroot.cint.execute import run_source
from xrdroot.cint.runtime.ntuplefill import fill_ntuple_histogram
from xrdroot.hist import Axis


@pytest.fixture
def ntuple(tmp_path):
    path = str(tmp_path / 'entries.root')
    with ROOT.TFile(path, 'RECREATE'):
        tree = ROOT.TNtuple('events', '', 'x:y')
        values = np.array([-5., -4., -1.25, -.0, .1, 1.75, 4., 5.], dtype=np.float32)
        tree.Fill(values, values * 2)
        tree.Write()
    with ROOT.TFile.Open(path) as handle:
        yield handle.Get('events')


def histogram():
    return ROOT.TH1F('output', '', 8, -4., 4.)



def _assert_same_histogram(actual, expected):
    assert actual._xrd._cells().tobytes() == expected._xrd._cells().tobytes()
    assert ([float(x).hex() for x in actual.GetStats()]
            == [float(x).hex() for x in expected.GetStats()])
    assert actual.GetEntries() == expected.GetEntries()


@pytest.mark.parametrize('column', [0, 1])
@pytest.mark.parametrize('bounds', [(0, 8), (1, 7), (3, 4)])
@pytest.mark.parametrize('squares', [False, True])
def test_a_batched_loop_keeps_every_scalar_histogram_bit_and_last_entry(
    ntuple, column, bounds, squares
):
    expected, actual = histogram(), histogram()
    if squares:
        expected.Sumw2()
        actual.Sumw2()
    for hist in (expected, actual):
        hist.Fill(-.75, .3)
    begin, end = bounds
    for index in range(begin, end):
        ntuple.GetEntry(index)
        expected.Fill(ntuple.GetArgs()[column])
    last = ntuple.GetArgs().copy()
    assert fill_ntuple_histogram(ntuple, actual, ROOT.TSeqL(begin, end), column)
    _assert_same_histogram(actual, expected)
    assert ntuple.GetReadEntry() == end - 1
    assert ntuple.GetArgs().tobytes() == last.tobytes()
    assert actual._xrd._sumw2().tobytes() == expected._xrd._sumw2().tobytes()


def test_an_empty_loop_does_not_read_its_tree_or_histogram():
    assert fill_ntuple_histogram(None, None, ROOT.TSeqL(0), 0)


@pytest.mark.parametrize('sequence', [range(8), None, ROOT.TSeqL(-1, 3),
                                     ROOT.TSeqL(0, 9), ROOT.TSeqL(0, 8, 2),
                                     ROOT.TSeqL(7, 0, -1)])
def test_non_native_or_non_contiguous_sequences_leave_the_scalar_loop_available(ntuple, sequence):
    hist = histogram()
    assert not fill_ntuple_histogram(ntuple, hist, sequence, 0)
    assert hist.GetEntries() == 0 and ntuple.GetReadEntry() == -1


@pytest.mark.parametrize('column', [-1, 2, 1.0, True])
def test_unknown_or_non_integer_columns_are_left_to_the_scalar_loop(ntuple, column):
    hist = histogram()
    assert not fill_ntuple_histogram(ntuple, hist, ROOT.TSeqL(8), column)
    assert hist.GetEntries() == 0 and ntuple.GetReadEntry() == -1


@pytest.mark.parametrize('field,value', [('_store', object()), ('_entry_list', object()),
    ('_addresses', {'x': object()}), ('_disabled', {'x'}), ('_friends', [object()]),
    ('_aliases', {'alias': 'x'}), ('_addresses', []), ('_code', object()), ('_code', 'i')])
def test_tree_state_that_requires_per_entry_work_keeps_the_original_loop(ntuple, field, value):
    hist = histogram()
    setattr(ntuple, field, value)
    assert not fill_ntuple_histogram(ntuple, hist, ROOT.TSeqL(8), 0)
    assert hist.GetEntries() == 0 and ntuple.GetReadEntry() == -1


@pytest.mark.parametrize('target,method', [('tree', 'GetArgs'), ('tree', 'GetEntry'),
                                          ('hist', 'Fill'), ('hist', '_fill_arrays')])
def test_user_callbacks_are_not_entered_during_batch_eligibility(
    ntuple, target, method, monkeypatch
):
    hist = histogram()
    def callback(*args):
        raise AssertionError('The speculative batch called a user callback.')
    monkeypatch.setattr(ntuple if target == 'tree' else hist, method, callback)
    assert not fill_ntuple_histogram(ntuple, hist, ROOT.TSeqL(8), 0)
    assert hist.GetEntries() == 0


@pytest.mark.parametrize('target,method,alias', [('tree', 'GetArgs', 'GetEntriesFast'),
    ('tree', 'GetEntry', 'LoadTree'), ('hist', 'Fill', 'GetBinContent'),
    ('hist', '_fill_arrays', '_fill_any'), ('engine', 'fill', 'fill_one')])
def test_native_method_aliases_also_keep_the_original_loop(
    ntuple, target, method, alias, monkeypatch
):
    hist = histogram()
    obj = {'tree': ntuple, 'hist': hist, 'engine': hist._xrd}[target]
    monkeypatch.setattr(type(obj), method, getattr(type(obj), alias))
    assert not fill_ntuple_histogram(ntuple, hist, ROOT.TSeqL(8), 0)
    assert hist.GetEntries() == 0


@pytest.mark.parametrize('entries', [.5, -1., float('inf'), float('nan'), float(2**53)])
def test_entries_that_cannot_be_added_in_one_exact_step_keep_the_original_loop(ntuple, entries):
    hist = histogram()
    hist._xrd._core['fEntries'] = entries
    assert not fill_ntuple_histogram(ntuple, hist, ROOT.TSeqL(8), 0)
    assert np.count_nonzero(hist._xrd._cells()) == 0


@pytest.mark.parametrize('method', ['find_bin', 'edges', 'even'])
def test_axis_callbacks_keep_the_original_loop_without_being_called(ntuple, method, monkeypatch):
    hist = histogram()
    def callback(*args):
        raise AssertionError('The speculative batch called an axis callback.')
    monkeypatch.setattr(Axis, method, callback)
    assert not fill_ntuple_histogram(ntuple, hist, ROOT.TSeqL(8), 0)


def test_a_translated_read_and_fill_loop_uses_the_same_entry_range_and_values(ntuple):
    source = '''
    int fill(TNtuple * t, TH1F * h, int begin, int end) {
        for (auto i : ROOT::TSeqL(begin, end)) {
            t->GetEntry(i);
            h->Fill(t->GetArgs()[1]);
        }
        return t->GetReadEntry();
    }
    '''
    namespace = run_source(source, call=False)
    hist = histogram()
    assert namespace['fill'](ntuple, hist, 2, 7) == 6
    assert hist.GetEntries() == 5
    assert hist.GetBinContent(0) == 0
    assert hist.GetBinContent(9) == 1


def test_a_translated_loop_keeps_user_getentry_calls_in_their_original_order(ntuple):
    namespace = run_source('''
    void fill(TNtuple * t, TH1F * h) {
        for (auto i : ROOT::TSeqL(8)) { t->GetEntry(i); h->Fill(t->GetArgs()[0]); }
    }''', call=False)
    calls = []
    original = ntuple.GetEntry
    def read(index):
        calls.append(index)
        return original(index)
    ntuple.GetEntry = read
    hist = histogram()
    namespace['fill'](ntuple, hist)
    assert calls == list(range(8)) and hist.GetEntries() == 8


@pytest.mark.parametrize('field,value', [('_source', None), ('_batch', None)])
def test_foreign_backing_readers_keep_the_original_loop(ntuple, field, value):
    hist = histogram()
    setattr(ntuple, field, value)
    assert not fill_ntuple_histogram(ntuple, hist, ROOT.TSeqL(8), 0)
    assert hist.GetEntries() == 0


def test_an_injected_batch_source_is_not_called_during_eligibility(ntuple):
    class Source:
        def __getattr__(self, name):
            raise AssertionError('Batch eligibility inspected an arbitrary callback.')
    ntuple._batch._source = Source()
    assert not fill_ntuple_histogram(ntuple, histogram(), ROOT.TSeqL(8), 0)


@pytest.mark.parametrize('field,value', [('_auto_axes', {0}), ('_auto_axes', object()),
                                       ('_xrd', None)])
def test_automatic_histogram_axes_and_foreign_engines_keep_the_original_loop(ntuple, field, value):
    hist = histogram()
    setattr(hist, field, value)
    assert not fill_ntuple_histogram(ntuple, hist, ROOT.TSeqL(8), 0)


@pytest.mark.parametrize('axes', [[], (), (None,), (None, None)])
def test_foreign_or_non_one_dimensional_axis_storage_keeps_the_original_loop(ntuple, axes):
    hist = histogram()
    hist._xrd.axes = axes
    assert not fill_ntuple_histogram(ntuple, hist, ROOT.TSeqL(8), 0)


def test_a_foreign_histogram_core_keeps_the_original_loop(ntuple):
    hist = histogram()
    hist._xrd._core = []
    assert not fill_ntuple_histogram(ntuple, hist, ROOT.TSeqL(8), 0)


@pytest.mark.parametrize('field,value', [('typename', 'Int_t'), ('size', 2),
                                       ('counter', 'count'), ('column', object())])
def test_ntuple_records_that_are_not_scalar_float_columns_keep_the_original_loop(
    ntuple, field, value
):
    branches = ntuple._layout()
    branch = branches[0]
    leaf = branch.leaves[0]._replace(**{field: value})
    ntuple._layout_cache = [branch._replace(leaves=[leaf]), *branches[1:]]
    hist = histogram()
    assert not fill_ntuple_histogram(ntuple, hist, ROOT.TSeqL(8), 0)
    assert hist.GetEntries() == 0


def test_an_ntuple_without_columns_keeps_the_original_loop(ntuple):
    ntuple._layout_cache = []
    assert not fill_ntuple_histogram(ntuple, histogram(), ROOT.TSeqL(8), 0)


def test_callbacks_on_the_backing_column_keep_the_original_loop(ntuple, monkeypatch):
    from xrdroot.tree import Branch
    def callback(*args):
        raise AssertionError('Batch eligibility invoked a backing-column callback.')
    monkeypatch.setattr(Branch, 'array', callback)
    assert not fill_ntuple_histogram(ntuple, histogram(), ROOT.TSeqL(8), 0)


@pytest.mark.parametrize('field,value', [('nbins', 0), ('low', object()),
                                       ('high', object()), ('_edges', object())])
def test_axis_metadata_that_can_run_numeric_callbacks_keeps_the_original_loop(ntuple, field, value):
    hist = histogram()
    setattr(hist._xrd.axes[0], field, value)
    assert not fill_ntuple_histogram(ntuple, hist, ROOT.TSeqL(8), 0)


@pytest.mark.parametrize('low,high', [(float('-inf'), 4.), (-4., float('inf')),
                                     (-4., float('nan')), (4., -4.)])
def test_non_finite_or_inverted_axis_ranges_keep_the_original_loop(ntuple, low, high):
    hist = histogram()
    axis = hist._xrd.axes[0]
    axis.low, axis.high = low, high
    assert not fill_ntuple_histogram(ntuple, hist, ROOT.TSeqL(8), 0)


def test_variable_histogram_bins_retain_the_original_loop(ntuple):
    hist = ROOT.TH1F('variable', '', 3, np.array([-4., -1., 1., 4.]))
    assert not fill_ntuple_histogram(ntuple, hist, ROOT.TSeqL(8), 0)


def test_double_precision_values_keep_their_bits_across_multiple_read_batches(tmp_path):
    path = str(tmp_path / 'double.root')
    values = np.tile(np.array([.1, np.nextafter(.1, 1.), -3.25, 4.5]), 3001)
    with ROOT.TFile(path, 'RECREATE'):
        written = ROOT.TNtupleD('double', '', 'x')
        written.Fill(values)
        written.Write()
    with ROOT.TFile.Open(path) as handle:
        tree = handle.Get('double')
        expected = ROOT.TH1D('expected', '', 8, -4., 4.)
        actual = ROOT.TH1D('actual', '', 8, -4., 4.)
        for entry in range(len(values)):
            tree.GetEntry(entry)
            expected.Fill(tree.GetArgs()[0])
        assert fill_ntuple_histogram(tree, actual, ROOT.TSeqL(len(values)), 0)
        assert actual._xrd._cells().tobytes() == expected._xrd._cells().tobytes()
        assert ([float(x).hex() for x in actual.GetStats()]
                == [float(x).hex() for x in expected.GetStats()])
        assert actual.GetEntries() == expected.GetEntries() == len(values)
        assert tree.GetReadEntry() == len(values) - 1
        assert tree.GetArgs().tobytes() == values[-1:].tobytes()
