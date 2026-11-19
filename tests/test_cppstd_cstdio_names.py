"""``fgets`` and ``sscanf`` are the runtime's own names, so a translation calls them bare."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np


def test_c_files_and_the_scan_are_what_a_translation_finds_by_name() -> None:
    runtime = importlib.import_module("xrdroot.pycxx.runtime")
    from xrdroot.cppstd import cformat, cstdio

    assert runtime.fgets is cstdio.fgets and runtime.sscanf is cstdio.sscanf
    assert {"fgets", "sscanf"} <= set(runtime.LIBRARY)
    assert runtime.sscanf is cformat.sscanf


def test_reading_a_file_and_scanning_what_was_read_works_through_the_runtime(
        tmp_path: Path) -> None:
    runtime = importlib.import_module("xrdroot.pycxx.runtime")
    from xrdroot.cppstd.buffers import CharBuffer
    from xrdroot.cppstd.cells import Cell

    path = tmp_path / "data.txt"
    path.write_bytes(b"a 12 3.5\n")
    handle = runtime.fopen(path, "r")
    line, count, real = CharBuffer(80), Cell(0), Cell(0.0)
    assert runtime.fgets(line, 80, handle) is line
    name = CharBuffer(8)
    assert runtime.sscanf(line, "%s %d %lf", name, count, real) == 3
    assert (str(name), count.value, real.value) == ("a", 12, 3.5)
    assert runtime.fgets(line, 80, handle) is None
    runtime.fclose(handle)


def test_a_pointer_to_memory_is_true_whatever_bytes_it_holds() -> None:
    """``if (p)`` of a ``char *`` that holds text - and of one that holds none at all."""
    from xrdroot.cppstd.buffers import CharBuffer

    empty = CharBuffer(4)
    assert empty is not None and bool(empty)
    text = CharBuffer(5)
    text[:4] = np.frombuffer(b"text", np.int8)
    assert str(text) == "text" and bool(text)
