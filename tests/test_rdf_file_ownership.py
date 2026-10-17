"""A file opened to identify a named tree belongs to its chain until the frame closes."""
import pickle

import pytest

import xrdroot
from frames import write_xyn
from xrdroot import RDataFrame
from xrdroot.rdf import sources


def _opened(monkeypatch):
    from xrdroot import file
    original = file.open_root
    opened = []
    def track(*args, **kwargs):
        result = original(*args, **kwargs)
        opened.append(result)
        return result
    monkeypatch.setattr(sources, 'open_root', track)
    monkeypatch.setattr(file, 'open_root', track)
    return opened


def test_a_named_tree_is_opened_once_until_its_frame_is_closed(tmp_path, monkeypatch):
    path = write_xyn(tmp_path / 'tree.root')
    opened = _opened(monkeypatch)
    with RDataFrame('tree', path) as frame:
        assert len(opened) == 1 and not opened[0]._closed
        assert frame.Sum('x').GetValue() == 4950.
        expected = f'Dataframe from TChain tree in file {path}'
        assert frame._graph.source.root_description() == expected
        assert len(opened) == 1
    assert opened[0]._closed


def test_each_file_of_a_named_chain_is_opened_once(tmp_path, monkeypatch):
    paths = [write_xyn(tmp_path / 'a.root'), write_xyn(tmp_path / 'b.root', entries=7)]
    opened = _opened(monkeypatch)
    with RDataFrame('tree', paths) as frame:
        assert len(opened) == 1
        assert frame.Sum('x').GetValue() == 4971.
        assert frame.Count().GetValue() == 107
        assert frame._graph.source.table.files == paths
        assert len(opened) == 2
    assert all(file._closed for file in opened)


def test_pickled_sources_reopen_by_target_without_serializing_live_handles(tmp_path, monkeypatch):
    path = write_xyn(tmp_path / 'tree.root', entries=7)
    opened = _opened(monkeypatch)
    source = sources.open_named('tree', path)
    assert source.file is None
    replica = pickle.loads(pickle.dumps(source))
    assert len(opened) == 1
    assert replica.read('x', 0, 7).tolist() == list(range(7))
    assert len(opened) == 2
    replica.close()
    assert opened[1]._closed and not opened[0]._closed
    source.close()
    assert opened[0]._closed


def test_a_closed_source_can_reopen_its_original_target(tmp_path, monkeypatch):
    path = write_xyn(tmp_path / 'tree.root', entries=7)
    opened = _opened(monkeypatch)
    source = sources.open_named('tree', path)
    source.close()
    assert opened[0]._closed
    assert source.read('x', 0, 7).tolist() == list(range(7))
    assert len(opened) == 2 and not opened[1]._closed
    source.close()
    assert opened[1]._closed


@pytest.mark.parametrize('options', [{'workers': 0}, {'step': 0}])
def test_failed_frame_initialization_closes_its_owned_file(tmp_path, monkeypatch, options):
    path = write_xyn(tmp_path / 'tree.root')
    opened = _opened(monkeypatch)
    with pytest.raises(ValueError, match='at least'):
        RDataFrame('tree', path, **options)
    assert len(opened) == 1 and opened[0]._closed


def test_missing_tree_errors_close_every_opened_handle(tmp_path, monkeypatch):
    path = write_xyn(tmp_path / 'tree.root')
    opened = _opened(monkeypatch)
    with pytest.raises(KeyError):
        RDataFrame('missing', path)
    assert opened and all(file._closed for file in opened)


def test_failed_initialization_leaves_a_borrowed_tree_open(tmp_path):
    path = write_xyn(tmp_path / 'tree.root')
    with xrdroot.open_root(path) as file:
        with pytest.raises(ValueError, match='at least'):
            RDataFrame(file['tree'], workers=0)
        assert not file._closed
        assert len(file['tree']) == 100


class _Probe:
    def __init__(self, kind='TTree', error=None):
        self.kind, self.error, self.closed = kind, error, 0

    def classnames(self):
        if self.error is not None:
            raise self.error
        return {'tree': self.kind}

    def close(self):
        self.closed += 1


@pytest.mark.parametrize('error', [RuntimeError('probe failed'), KeyboardInterrupt()])
def test_classification_errors_close_the_probe_even_when_interrupted(monkeypatch, error):
    probe = _Probe(error=error)
    monkeypatch.setattr(sources, 'open_root', lambda path: probe)
    with pytest.raises(type(error)):
        sources.open_named('tree', 'memory.root')
    assert probe.closed == 1


@pytest.mark.parametrize('constructor', ['Chain', 'Table'])
def test_source_construction_errors_close_the_untransferred_file(monkeypatch, constructor):
    probe = _Probe()
    monkeypatch.setattr(sources, 'open_root', lambda path: probe)
    def fail(*args, **kwargs):
        raise RuntimeError('source failed')
    monkeypatch.setattr(sources, constructor, fail)
    with pytest.raises(RuntimeError, match='source failed'):
        sources.open_named('tree', 'memory.root')
    assert probe.closed == 1


def test_rntuple_probes_are_closed_before_constructing_the_joined_source(monkeypatch):
    probe = _Probe('ROOT::RNTuple')
    monkeypatch.setattr(sources, 'open_root', lambda path: probe)
    original = sources.Joined
    def joined(*args):
        assert probe.closed == 1
        return original(*args)
    monkeypatch.setattr(sources, 'Joined', joined)
    result = sources.open_named('tree;1', 'memory.root')
    assert result.name == 'tree;1' and result.targets == ['memory.root']
    assert probe.closed == 1
