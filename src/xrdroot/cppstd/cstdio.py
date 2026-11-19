"""C's ``<cstdio>`` files: ``fopen`` and ``fclose``, ``fwrite``, ``fread``, ``fgets``,
``fputs``, ``fgetc``, and the ``sscanf`` that reads text back (:mod:`.cformat`).

A ``FILE *`` is a :class:`FILE` here, wrapping a Python binary file, and a
null one is ``None``, which is what ``fopen`` returns for a file it cannot
open - as C's does, rather than raising. What is written may be text, bytes
or an array of numbers; ``fwrite(data, size, count, f)`` writes ``size *
count`` bytes of it and returns ``count`` when they were all written. A
:class:`FILE` has ``write`` too, so ``fprintf`` prints to one.

A ``char name[100]`` is a ``str`` here, which Python cannot write into, so a
callee that fills one - ``fgets``, ``sscanf`` - is handed a cell in the
variable's place and the variable is given the text afterwards: that is
:func:`written_after`, which the emitter wraps such a call in.
"""

from __future__ import annotations

import os
from typing import IO, Any

from .buffers import text_into
from .cformat import EOF as EOF
from .cformat import sscanf as sscanf

__all__ = ["FILE", "fopen", "fclose", "fwrite", "fread", "fgets", "fputs", "fgetc", "fflush",
           "feof", "ftell", "fseek", "rewind", "sscanf", "written_after", "EOF", "SEEK_SET",
           "SEEK_CUR", "SEEK_END"]  # fmt: skip

#: ``fseek``'s origins: the start, the current position, the end.
SEEK_SET, SEEK_CUR, SEEK_END = 0, 1, 2


class FILE:
    """``FILE``: an open file, read and written in bytes."""

    def __init__(self, stream: IO[bytes], name: str = "") -> None:
        self.stream, self.name, self.eof = stream, name, False

    def write(self, text: Any) -> int:
        """Text or bytes written: what ``fprintf`` and ``fputs`` hand it."""
        data = _bytes(text)
        return self.stream.write(data)

    def __repr__(self) -> str:
        return f"<FILE {self.name!r}>"


def _mode(mode: str) -> str:
    """C's mode as Python's, always binary: ``"r"``, ``"w+"``, ``"ab"``..."""
    return mode.replace("b", "").replace("t", "") + "b"


def fopen(name: Any, mode: Any = "r") -> FILE | None:
    """``fopen``: the file opened, or ``None`` - a null ``FILE *`` - if it cannot be."""
    try:
        stream = open(os.fspath(str(name)), _mode(str(mode)))
    except OSError:
        return None
    return FILE(stream, str(name))


def fclose(file: FILE | None) -> int:
    """``fclose``: 0 once closed, ``EOF`` for a null file."""
    if file is None:
        return EOF
    file.stream.close()
    return 0


def _bytes(data: Any) -> bytes:
    """What a ``const void *`` points at, as bytes: text, bytes, or an array of numbers."""
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    if isinstance(data, str):
        return data.encode("latin-1", "replace")
    tobytes = getattr(data, "tobytes", None)
    if tobytes is not None:
        return bytes(tobytes())
    return bytes(int(item) & 0xFF for item in data)


def fwrite(data: Any, size: Any, count: Any, file: FILE) -> int:
    """``fwrite``: ``size * count`` bytes of ``data`` written; how many items were."""
    wanted = int(size) * int(count)
    written = file.stream.write(_bytes(data)[:wanted])
    return written // int(size) if int(size) else 0


def fread(buffer: Any, size: Any, count: Any, file: FILE) -> int:
    """``fread``: up to ``size * count`` bytes into a ``bytearray`` or array; items read."""
    data = file.stream.read(int(size) * int(count))
    file.eof = len(data) < int(size) * int(count)
    buffer[: len(data)] = data if isinstance(buffer, bytearray) else list(data)
    return len(data) // int(size) if int(size) else 0


def fgets(buffer: Any, size: Any, file: FILE) -> Any:
    """``fgets``: up to ``size - 1`` bytes of the next line, its newline included, written into
    ``buffer`` and NUL terminated; the buffer, or ``None`` - a null ``char *`` - when nothing
    was read, as at the end of the file. A ``size`` of one or less leaves no room for any
    character, so nothing is read and the null pointer is returned rather than a buffer that
    was not advanced, which would leave a program that reads a line at a time spinning.

    The buffer is memory to write into, a :class:`~.buffers.CharBuffer`, or the cell the
    emitter passes for a ``char name[100]`` the translation holds as a ``str`` - the text is
    given the cell and the cell is handed back, as C hands back its buffer, for the
    variable to take the text from (:func:`written_after`). Text Python cannot change in
    place is otherwise refused by :func:`~.buffers.text_into` rather than read and dropped.
    """
    data = file.stream.readline(int(size) - 1) if int(size) > 1 else b""
    if not data:
        file.eof = True
        return None
    text = data.decode("latin-1")
    if hasattr(buffer, "value"):
        buffer.value = text
        return buffer
    return text_into(buffer, text, True)


def written_after(result: Any, *assigned: Any) -> Any:
    """``fgets``'s buffer, or the fields ``sscanf`` stored, once the text it wrote into a
    ``char[]`` has been given to the variable that array stands for.

    C writes into the memory of ``char name[100]``; the translation holds that as a ``str``,
    which Python cannot write into, so the callee is handed a cell in the variable's place
    and the variable takes the cell's text afterwards. ``assigned`` are those assignments -
    ``name := text_1.value`` - made in order as this call's arguments, before the value of
    the call itself (the count, or the buffer) is used.
    """
    return result


def fputs(text: Any, file: Any) -> int:
    """``fputs``: the text written, with no newline added."""
    file.write(str(text))
    return 1


def fgetc(file: FILE) -> int:
    """``fgetc``: the next byte, or ``EOF`` at the end."""
    byte = file.stream.read(1)
    if not byte:
        file.eof = True
        return EOF
    return byte[0]


def fflush(file: Any = None) -> int:
    """``fflush``: what was written handed to the system."""
    if file is not None:
        (file.stream if isinstance(file, FILE) else file).flush()
    return 0


def feof(file: FILE) -> bool:
    """``feof``: whether a read has run past the end."""
    return file.eof


def ftell(file: FILE) -> int:
    return file.stream.tell()


def fseek(file: FILE, offset: Any, origin: Any = SEEK_SET) -> int:
    file.stream.seek(int(offset), int(origin))
    file.eof = False
    return 0


def rewind(file: FILE) -> None:
    fseek(file, 0)
