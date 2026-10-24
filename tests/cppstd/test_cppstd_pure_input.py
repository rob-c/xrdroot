"""Standard extraction retains stream failure, delimiters and file lifecycle semantics."""

from __future__ import annotations

from pathlib import Path

from xrdroot.cppstd import istreams


def _read_standard_targets(stream: istreams.istream) -> None:
    assert stream.extract("int") == -7
    assert stream.extract("double") == 2.5
    assert stream.extract("char") == ord("q")
    assert stream.extract("std::string") == "rest"
    assert istreams.stream_after(stream, 1) is stream


def _check_exhausted_stream(stream: istreams.istream) -> None:
    assert stream.extract("int") == 0
    assert stream.fail() and stream.eof() and not stream
    assert stream.extract("std::string") == ""


def test_standard_stream_extraction_preserves_each_target_type_and_failure_state() -> None:
    stream = istreams.istringstream(" -7 2.5 q rest")
    _read_standard_targets(stream)
    _check_exhausted_stream(stream)
    assert stream.str("word 9") == "" and stream.good()
    assert stream.extract("int") == 0 and stream.fail() and not stream.eof()
    assert stream.extract("std::string") == ""


def test_delimited_reads_and_reset_keep_the_underlying_text_and_cursor() -> None:
    stream = istreams.istringstream("head/tail\nlast")
    assert stream.getline(ord("/")) == "head"
    assert stream.getline() == "tail"
    assert stream.getline() == "last"
    assert stream.good()
    assert stream.getline() == "" and stream.fail() and stream.eof()
    assert stream.str("new") == ""
    assert stream.str() == "new" and stream.extract("std::string") == "new"


def test_file_streams_reopen_after_a_failed_open_and_close_explicitly(tmp_path: Path) -> None:
    path = tmp_path / "input.txt"
    stream = istreams.ifstream(path)
    assert not stream.is_open() and stream.fail()
    path.write_text("42 3.5")
    stream.open(path)
    assert stream.is_open() and stream.good()
    assert stream.extract("unsigned int") == 42 and stream.extract("float") == 3.5
    stream.close()
    assert not stream.is_open()


def test_find_if_stops_before_the_end_sentinel_without_testing_it() -> None:
    values = [object(), object(), object()]
    visited = []

    def predicate(value):
        visited.append(value)
        return value is values[2]

    assert istreams.find_if(values, values[2], predicate) is None
    assert visited == values[:2]
    assert istreams.find_if(values, None, lambda value: value is values[1]) is values[1]
