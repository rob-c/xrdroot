"""Character buffers: the memory ``new char[n]`` hands out, which C writes into in place.

A ``const char*`` naming a literal, a ``char name[20]`` and a ``std::string``
are Python ``str`` here, and a function that "writes" one gives its variable
a new string. Memory from ``new char[n]`` is different: a program keeps
pointers into it (``fName``, ``p = buf + 3``) and writes through any of
them, so it must be one object every pointer shares. A :class:`CharBuffer`
is that object - a NumPy array of bytes, so ``buf + 3`` is the view
``buf[3:]`` into the same memory, as for an array of numbers - that reads as
the C string it holds wherever text is wanted (``printf("%s")``, a
``TString`` made of it, ``strlen``).

:func:`put_char` and :func:`text_into` are the writes: into a buffer in
place, and - where what a ``char*`` holds is a string Python cannot change -
the new string for its variable, or a refusal where the variable is not the
one the caller sees.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["CharBuffer", "put_char", "text_into", "pointer_diff"]


class CharBuffer(np.ndarray):  # type: ignore[misc]
    """``new char[n]``: bytes in place, read as the C string they hold."""

    def __new__(cls, size: Any, dtype: Any = np.int8) -> CharBuffer:
        made: CharBuffer = np.zeros(int(size), dtype=dtype).view(cls)
        return made

    def __str__(self) -> str:
        """The C string: the bytes up to the first NUL, as Latin-1 (one char, one byte)."""
        raw: bytes = np.asarray(self).astype(np.uint8).tobytes()
        return raw.split(b"\0", 1)[0].decode("latin-1")

    def __format__(self, spec: str) -> str:
        return format(str(self), spec)

    def __repr__(self) -> str:
        return f"CharBuffer({str(self)!r})"


def put_char(buffer: Any, index: Any, char: Any) -> Any:
    """``p[i] = c`` through a ``char*``: into the buffer it points into, in place."""
    if not isinstance(buffer, np.ndarray):
        raise TypeError(
            "writing a character through a char* that holds a string Python cannot change in "
            "place; only memory from new char[n] is written through a pointer here"
        )
    byte = int(char) & 0xFF  # wrapped to the char's width, as C stores it
    buffer[int(index)] = byte - 256 if byte > 127 and buffer.dtype == np.int8 else byte
    return char


def text_into(buffer: Any, text: Any, passed: bool = False) -> Any:
    """``strcpy(buffer, text)`` and kin: ``text`` and its NUL written into the buffer in place,
    which is handed back; or, for a string - ``char name[20]`` - the text itself, for the
    variable to hold. A string the caller passed (``passed``) has no variable here to give it
    to, and is refused."""
    if not isinstance(buffer, np.ndarray):
        if passed:
            raise TypeError(
                "writing a string into a buffer the caller passed, which here is text Python "
                "cannot change in place; pass memory from new char[n] instead"
            )
        return str(text)
    data = str(text).encode("latin-1", "replace")
    if len(data) >= len(buffer):
        raise IndexError(
            f"writing {len(data) + 1} bytes into a buffer of {len(buffer)}, past its end, which "
            "C leaves undefined"
        )
    buffer[: len(data)] = np.frombuffer(data, dtype=np.uint8).astype(buffer.dtype)
    buffer[len(data)] = 0
    return buffer


def pointer_diff(later: Any, earlier: Any) -> int:
    """``q - p`` of two pointers into one array - views of it here: how many elements apart
    they are. Anything else (text, a cell, an object) is no array to measure in, and is
    refused."""
    if not (isinstance(later, np.ndarray) and isinstance(earlier, np.ndarray)):
        raise TypeError(
            "the distance between two pointers is measured here only within one array of "
            "numbers or characters"
        )
    if np.shares_memory(later, earlier) is False and later.size and earlier.size:
        raise ValueError("the distance between two pointers into different arrays, which C "
                         "leaves undefined")  # fmt: skip
    start: int = later.__array_interface__["data"][0] - earlier.__array_interface__["data"][0]
    return int(start // later.itemsize)
