"""Input stream implementations are shared while ROOT supplies its own typedefs."""

from __future__ import annotations

import pytest

from cppstdsupport import isolated


def test_input_stream_compatibility_paths_share_the_classes_and_type_registry() -> None:
    from xrdroot.cint.runtime import istreams as legacy
    from xrdroot.cppstd import istreams

    assert legacy.__all__ is istreams.__all__
    assert legacy.KINDS is istreams.KINDS
    assert legacy.PATTERNS is istreams.PATTERNS
    for name in legacy.__all__:
        assert getattr(legacy, name) is getattr(istreams, name), name


@pytest.mark.parametrize(("kind", "expected"), [
    ("Int_t", -7), ("UInt_t", -7), ("Long64_t", -7),
    ("Double_t", -7.25), ("Float_t", -7.25),
])
def test_runtime_streams_keep_root_numeric_typedef_extraction(kind, expected) -> None:
    from xrdroot.cint.runtime.istreams import istringstream

    stream = istringstream(" -7.25 word")
    actual = stream.extract(kind)
    assert actual == expected
    assert type(actual) is type(expected)
    assert stream.good()


def test_standard_input_streams_read_files_and_words_without_root() -> None:
    isolated("""
import tempfile
from pathlib import Path
from xrdroot.cppstd.istreams import KINDS, find_if, ifstream, istringstream, stream_after
assert not {'Int_t', 'Long64_t', 'UInt_t', 'Float_t', 'Double_t'} & KINDS.keys()
stream = istringstream(' -7 2.5 q rest')
assert stream.extract('int') == -7
assert stream.extract('double') == 2.5
assert stream.extract('char') == ord('q')
assert stream.extract('std::string') == 'rest'
assert stream_after(stream, 1) is stream
assert stream.extract('int') == 0
assert stream.fail() and stream.eof() and not stream
assert stream.str('head/tail') == ''
assert stream.getline(ord('/')) == 'head'
assert stream.getline() == 'tail'
assert stream.str() == 'head/tail'
items = [object(), object(), object()]
assert find_if(items, items[2], lambda item: item is items[1]) is items[1]
assert find_if(items, items[1], lambda item: item is items[2]) is None
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / 'input.txt'
    path.write_text('42')
    file = ifstream(path)
    assert file.is_open() and file.extract('int') == 42
    file.close()
    assert not file.is_open()
    absent = ifstream(path.with_name('absent'))
    assert absent.fail() and not absent.is_open()
""")
