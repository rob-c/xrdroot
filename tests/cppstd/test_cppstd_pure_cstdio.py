"""C's files: opened or null, written and read in bytes, and printed to by ``fprintf``."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from xrdroot.cppstd import cstdio, printf


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
