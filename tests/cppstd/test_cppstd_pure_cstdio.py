"""C's files: opened or null, written and read in bytes, and printed to by ``fprintf``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from xrdroot.cppstd import cstdio, printf
from xrdroot.cppstd.buffers import CharBuffer


def test_a_file_that_cannot_be_opened_is_a_null_pointer_and_closing_one_is_eof(
        tmp_path: Path) -> None:
    assert cstdio.fopen(tmp_path / "missing" / "f.txt", "r") is None
    assert cstdio.fclose(None) == cstdio.EOF


def test_text_bytes_and_arrays_are_written_as_bytes(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    out = cstdio.fopen(path, "wb")
    assert out is not None and "f.bin" in repr(out)
    written = [cstdio.fwrite("hello", 1, 3, out), cstdio.fwrite(b"!!", 2, 1, out),
               cstdio.fwrite(np.array([65, 66], dtype=np.uint8), 1, 2, out),
               cstdio.fwrite([67, 68], 1, 2, out), cstdio.fwrite(b"", 0, 1, out),
               cstdio.fputs("xy", out), printf.fprintf(out, "%d", 7)]
    assert written == [3, 1, 2, 2, 0, 1, 1]
    assert (cstdio.fflush(out), cstdio.fflush(), cstdio.ftell(out)) == (0, 0, 12)
    assert cstdio.fclose(out) == 0 and path.read_bytes() == b"hel!!ABCDxy7"


def test_a_file_is_read_in_bytes_to_its_end_and_from_where_it_is_sought(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"hel!!ABC")
    back = cstdio.fopen(path, "r")
    assert back is not None
    buffer, numbers = bytearray(4), [0] * 16
    assert cstdio.fread(buffer, 1, 4, back) == 4 and bytes(buffer) == b"hel!"
    assert cstdio.fgetc(back) == ord("!") and not cstdio.feof(back)
    assert cstdio.fread(numbers, 2, 8, back) == 1 and cstdio.feof(back)
    assert cstdio.fgetc(back) == cstdio.EOF
    cstdio.rewind(back)
    assert cstdio.fread(buffer, 0, 1, back) == 0
    assert cstdio.fseek(back, -1, cstdio.SEEK_END) == 0 and cstdio.fgetc(back) == ord("C")
    cstdio.fclose(back)


def test_a_line_is_read_into_a_buffer_with_its_newline_and_nul_terminated(
        tmp_path: Path) -> None:
    path = tmp_path / "lines.txt"
    path.write_bytes(b"first\nsecond line\n")
    read = cstdio.fopen(path, "r")
    assert read is not None
    buffer = CharBuffer(80)
    assert cstdio.fgets(buffer, 80, read) is buffer
    assert str(buffer) == "first\n"
    assert cstdio.fgets(buffer, 8, read) is buffer and str(buffer) == "second "
    assert cstdio.fgets(buffer, 80, read) is buffer and str(buffer) == "line\n"
    assert cstdio.fgets(buffer, 80, read) is None and cstdio.feof(read)
    cstdio.fclose(read)


def test_a_size_of_one_or_less_reads_nothing_and_leaves_the_buffer_as_it_was(
        tmp_path: Path) -> None:
    path = tmp_path / "lines.txt"
    path.write_bytes(b"first\n")
    read = cstdio.fopen(path, "r")
    assert read is not None
    buffer = CharBuffer(8)
    assert cstdio.fgets(buffer, 1, read) is None and cstdio.fgets(buffer, 0, read) is None
    assert cstdio.fgets(buffer, 8, read) is buffer and str(buffer) == "first\n"
    cstdio.fclose(read)


def test_a_line_read_into_text_python_cannot_change_in_place_is_refused(
        tmp_path: Path) -> None:
    """A translated ``char line[80]`` is a ``str`` here, and C's read into it is refused
    rather than dropped, since what it read would never reach the caller."""
    path = tmp_path / "lines.txt"
    path.write_bytes(b"first\n")
    read = cstdio.fopen(path, "r")
    assert read is not None
    with pytest.raises(TypeError) as refused:
        cstdio.fgets("", 80, read)
    assert "new char[n]" in str(refused.value)
    cstdio.fclose(read)


def test_a_line_read_into_the_cell_a_char_array_stands_for_gives_the_cell_back(
        tmp_path: Path) -> None:
    """A translated ``char line[80]`` has no memory for ``fgets`` to fill, so the emitter
    passes a cell in its place and the variable takes the cell's text afterwards: the read
    hands the cell back, as C hands its buffer back, and the text is in the cell.
    """
    from xrdroot.cppstd.cells import Cell

    path = tmp_path / "lines.txt"
    path.write_bytes(b"first\nsecond\n")
    read = cstdio.fopen(path, "r")
    assert read is not None
    line = Cell("")
    assert cstdio.fgets(line, 80, read) is line and line.value == "first\n"
    assert cstdio.written_after(cstdio.fgets(line, 80, read), line) is line
    assert line.value == "second\n"
    assert cstdio.fgets(line, 80, read) is None and line.value == "second\n"
    cstdio.fclose(read)


def test_rand_draws_what_the_macos_c_library_draws_from_each_seed() -> None:
    """The numbers of macOS's ``rand`` after ``srand(1)``, ``srand(0)`` and ``srand(42)``."""
    from xrdroot.cppstd import stdlib
    from xrdroot.cppstd.containers import std

    stdlib.srand(1)
    assert [std.rand() for _ in range(3)] == [16807, 282475249, 1622650073]
    std.srand(0)
    assert [stdlib.rand(), stdlib.rand()] == [520932930, 28925691]
    stdlib.srand(42)
    assert [stdlib.rand(), stdlib.rand()] == [705894, 1126542223]
    stdlib.srand(2147483000)  # Schrage's product comes out negative, and is wrapped
    assert [stdlib.rand(), stdlib.rand()] == [2136609518, 1922107539]
    assert stdlib.RAND_MAX == 2**31 - 1
    stdlib.srand(1)


def test_the_standard_streams_are_named_by_the_translation_and_flushed_by_name(capsys) -> None:
    """A translation has no ``FILE`` object for ``stderr`` - it writes ``'stderr'`` in its
    place - so both ``fprintf`` and ``fflush`` read that name as the stream it stands for,
    and ``fflush(NULL)`` flushes every stream that is open."""
    assert cstdio.standard("stderr") is not None
    assert cstdio.standard("stdout") is not None
    assert cstdio.standard("a name that is no stream") == "a name that is no stream"
    assert printf.fprintf("stderr", "%d", 3) == 1
    assert (cstdio.fflush("stderr"), cstdio.fflush("stdout"), cstdio.fflush()) == (0, 0, 0)
    assert capsys.readouterr().err == "3"
