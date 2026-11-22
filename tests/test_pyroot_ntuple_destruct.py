"""``ROOT::RNTupleWriter``: what C++'s ``~unique_ptr`` does to it where its scope ends.

``RNTupleWriter::Recreate`` gives a ``std::unique_ptr`` back in ROOT, so a macro that fills a
writer in a block writes its file as the block ends - which is what lets it open the same
file straight afterwards. ``tests/test_cint_scope_destructors.py`` checks the translator
runs it; here it is the letting go itself.
"""

from __future__ import annotations

import pytest

import xrdroot
import xrdroot.pyroot as ROOT
from pyrootsupport import expect, fresh


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


def _writer(path):
    model = ROOT.RNTupleModel.Create()
    model.AddField(ROOT.RFieldBase.Create("value", "std::int32_t").Unwrap())
    entry = model.GetDefaultEntry()
    writer = ROOT.RNTupleWriter.Recreate(ROOT.std.move(model), "rs", str(path))
    entry["value"] = 7
    writer.Fill()
    return writer


def test_letting_a_writer_go_writes_the_file_it_was_filling(tmp_path):
    path = tmp_path / "held.root"
    writer = _writer(path)
    writer._destruct()
    with xrdroot.open_root(path) as found:
        expect((found["rs"].arrays()["value"].tolist(), [7]))


def test_letting_a_writer_go_twice_leaves_the_file_it_wrote_alone(tmp_path):
    path = tmp_path / "twice.root"
    writer = _writer(path)
    writer._destruct()
    written = path.read_bytes()
    writer._destruct()
    expect((path.read_bytes(), written), (writer._sink.closed, True))
