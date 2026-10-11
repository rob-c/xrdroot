"""``RooDataSet::write`` and ``RooDataSet::read``: rf102's dataset as text, and back.

ROOT 6.40 writes each event as its values with a space after each -
``-4.21601 -4.71476 `` - and reads a file into ``dataset``, titled by the
file, with a ``blindState`` category added, saying how many events it read
and how many it left out for lying outside a variable's range.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from xrdroot.cint.runtime.streams import ofstream
from xrdroot.roofit.categories import RooCategory
from xrdroot.roofit.data.dataset import RooDataSet
from xrdroot.roofit.variables import RooRealVar


def _data() -> tuple[RooRealVar, RooRealVar, RooDataSet]:
    x = RooRealVar("x", "x", -10, 10)
    y = RooRealVar("y", "y", -10, 10)
    data = RooDataSet("ds", "ds", [x, y])
    for a, b in ((-4.216011, -4.714762), (0.1019841, 7.067449)):
        x.setVal(a)
        y.setVal(b)
        data.add([x, y])
    return x, y, data


def test_a_dataset_is_written_a_line_an_event_as_cpps_stream_writes_numbers(
    tmp_path: Path,
) -> None:
    _, _, data = _data()
    path = tmp_path / "d.txt"
    stream = ofstream(str(path))
    assert data.write(stream) is False
    stream.close()
    assert path.read_text() == "-4.21601 -4.71476 \n0.101984 7.06745 \n"


def test_a_category_is_written_by_its_label(tmp_path: Path) -> None:
    c = RooCategory("c", "c")
    c.defineType("Plus", 1)
    c.setLabel("Plus")
    data = RooDataSet("dc", "dc", [c])
    data.add([c])
    path = tmp_path / "c.txt"
    stream = ofstream(str(path))
    data.write(stream)
    stream.close()
    assert path.read_text() == "Plus \n"


def test_a_file_is_read_back_as_roots_dataset_of_it(tmp_path: Path, capsys: Any) -> None:
    x, y, _ = _data()
    path = tmp_path / "d.txt"
    path.write_text("# a comment\n-4.21601 -4.71476 \n\n0.101984 7.06745 \n12 1\n3\n")
    capsys.readouterr()
    read = RooDataSet.read(str(path), [x, y], "Q")
    assert capsys.readouterr().out == (
        f"[#1] INFO:DataHandling -- RooDataSet::read: reading file {path}\n"
        "[#1] INFO:DataHandling -- RooDataSet::read: read 2 events (ignored 2 out of range "
        "events)\n"
    )
    assert (read.GetName(), read.GetTitle(), read.numEntries()) == ("dataset", str(path), 2)
    assert [one.GetName() for one in read.get()] == ["x", "y", "blindState"]
    assert read.get(0)["x"].getVal() == pytest.approx(-4.21601)
    assert read.get(1).find("blindState").getLabel() == "Normal"


def test_files_are_read_one_after_another_from_a_common_path(tmp_path: Path) -> None:
    x, _, _ = _data()
    (tmp_path / "a.txt").write_text("1\n")
    (tmp_path / "b.txt").write_text("2\n3\n")
    blind = RooCategory("blindState", "mine")
    read = RooDataSet.read("a.txt, b.txt", [x, blind], "", str(tmp_path))
    assert [read.get(i)["x"].getVal() for i in range(3)] == [1.0, 2.0, 3.0]


def test_an_empty_file_reads_as_an_empty_dataset_with_a_blind_state_of_its_own(
    tmp_path: Path,
) -> None:
    x, _, _ = _data()
    (tmp_path / "e.txt").write_text("# nothing\n")
    blind = RooCategory("blindState", "Blinding State")
    blind.defineType("Normal", 0)
    read = RooDataSet.read(str(tmp_path / "e.txt"), [x, blind])
    assert read.numEntries() == 0 and blind.hasLabel("Blind")
