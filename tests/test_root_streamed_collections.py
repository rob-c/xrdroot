"""ROOT's collections that stream themselves, written back as the file that held them did.

A ``TObjArray`` and a ``TList`` are written by a ``Streamer`` of their own rather
than by the layout a file describes them with, so one read out of a file comes
back as the entries it held and the name it was given - a
:class:`~xrdroot.buffer.Written` or a :class:`~xrdroot.buffer.Listed` - and not
as the members its description names. Writing those bytes back is what makes
them a collection again: the version of the ``Streamer`` that wrote them, the
``TObject``, the name, how many entries there are, and then each entry naming
the class it was read as.

The geometry ROOT keeps in a ROOT file is why this is here: a ``TGeoManager``
holds its matrices, shapes and volumes in arrays of exactly this kind, so
``TGeoManager::Export`` cannot write a manager back without it. Each test below
reads what it wrote back with the reader the other direction of the same file
uses, so the two agree about the bytes rather than about a hex string.
"""

from __future__ import annotations

import numpy as np
import pytest

from xrdroot.buffer import Buffer, Listed, Named, Written
from xrdroot.errors import UnsupportedFeatureError
from xrdroot.wdeclared import Catalogue, Declaration, Field
from xrdroot.wdobjects import Instance, write_instance, write_value
from xrdroot.winfo import INFOS
from xrdroot.writer import WBuffer


def _tnamed(buf: Buffer) -> dict[str, str]:
    """One ``TNamed``, the class the entries below are of: ROOT's own, so no layout is
    declared for it and the writer's carried description of it is used."""
    _version, end = buf.header()
    buf.tobject()
    found = {"fName": buf.string(), "fTitle": buf.string()}
    buf.resume(end)
    return found


def _named(name: str) -> Named:
    return Named({"fUniqueID": 0, "fBits": 0, "fName": name, "fTitle": ""}, "TNamed")


def _version_at_zero(data: bytes) -> int:
    """The version the record starting the buffer carries, past its own byte count."""
    return int.from_bytes(data[4:6], "big")


def test_a_tobjarray_read_from_a_file_is_written_as_root_streams_one() -> None:
    buf = WBuffer()
    write_value(buf, "TObjArray", Written([_named("a"), None], name="mine"), Catalogue({}))
    read = Buffer(bytes(buf.data)).objarray({"TNamed": _tnamed})
    assert read.name == "mine"
    assert [type(item).__name__ for item in read] == ["Named", "NoneType"]
    assert read[0].classname == "TNamed"
    assert read[0]["fName"] == "a"
    assert read[1] is None


def test_a_tobjarrays_record_is_the_version_of_the_streamer_that_writes_it() -> None:
    buf = WBuffer()
    write_value(buf, "TObjArray", Written([], name=""), Catalogue({}))
    assert _version_at_zero(bytes(buf.data)) == INFOS["TObjArray"][1]


def test_a_tlist_read_from_a_file_keeps_the_option_each_entry_was_added_under() -> None:
    buf = WBuffer()
    entries = Listed([_named("a"), None], options=["hist same", ""])
    write_value(buf, "TList", entries, Catalogue({}))
    read = Buffer(bytes(buf.data)).tlist({"TNamed": _tnamed})
    assert read.options == ["hist same", ""]
    assert read[0]["fName"] == "a"
    assert read[1] is None


def test_a_hashlist_is_written_by_the_version_of_the_list_streamer_it_inherits() -> None:
    buf = WBuffer()
    write_value(buf, "THashList", Listed([_named("a")], options=[""]), Catalogue({}))
    assert _version_at_zero(bytes(buf.data)) == INFOS["TList"][1]
    assert len(Buffer(bytes(buf.data)).tlist({"TNamed": _tnamed})) == 1


def test_a_collection_that_does_not_stream_itself_is_refused_by_name() -> None:
    with pytest.raises(UnsupportedFeatureError, match="a TGeoManager read out of a file"):
        write_value(WBuffer(), "TGeoManager", Written([], name="cdf"), Catalogue({}))


def test_an_array_read_as_a_list_is_refused_rather_than_written_as_one() -> None:
    with pytest.raises(UnsupportedFeatureError, match="a TList read out of a file"):
        write_value(WBuffer(), "TList", Written([_named("a")], name=""), Catalogue({}))


def test_an_entry_that_names_no_class_is_refused_rather_than_guessed_at() -> None:
    with pytest.raises(UnsupportedFeatureError, match="a dict does not"):
        write_value(WBuffer(), "TObjArray", Written([{"fName": "a"}]), Catalogue({}))


def test_a_counted_pointer_holding_nothing_is_written_as_a_null_pointer() -> None:
    """ROOT writes the marker byte alone for a pointer that was null, whatever the counter
    says - and reading one back gives nothing, so a file that round-trips keeps its meaning."""
    counter, counted = Field("n", "int", "", ()), Field("v", "float*", "[n]", ())
    catalogue = Catalogue({"C": Declaration("C", 2, False, (), (counter, counted))})
    empty, full = WBuffer(), WBuffer()
    write_instance(empty, Instance("C", {"n": 3, "v": None}), catalogue)
    write_instance(full, Instance("C", {"n": 3, "v": [1.0, 2.0, 3.0]}), catalogue)
    head = len(bytes(empty.data)) - 1  # up to the marker byte, past the two byte counts
    assert bytes(full.data)[4:head] == bytes(empty.data)[4:head]
    assert bytes(empty.data)[head] == 0
    assert bytes(full.data)[head] == 1
    assert bytes(full.data)[head + 1:] == np.array([1.0, 2.0, 3.0], dtype=">f4").tobytes()
