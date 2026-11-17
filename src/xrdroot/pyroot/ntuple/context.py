"""The ``with`` protocol ROOT's Python layer gives ``RNTupleWriter`` and ``RNTupleReader``.

ROOT wraps each of them in a ``RNTupleContextWrapper``: ``with`` hands the
object itself back, the end of the block gives the file back - so what was
written is closed - and every public attribute afterwards is refused, since
the object the wrapper held is gone. Asking for two blocks on one object at
once, or a second block after the first, is refused too. The messages are
ROOT's own word for word, because a test matches on them.
"""

from __future__ import annotations

from typing import Any, Literal

__all__: list[str] = []


class Guarded:
    """One ``with`` at a time, then nothing: what :class:`RNTupleReader` and a writer are.

    Subclasses say what ROOT's wrapper calls them - ``CONTEXT_NAME`` - and
    what giving the object up means, :meth:`_close`.
    """

    #: The word ROOT's wrapper puts in its refusals.
    CONTEXT_NAME = "RNTuple object"

    _in_context = False
    _closed = False

    def __getattribute__(self, name: str) -> Any:
        if not name.startswith("_") and object.__getattribute__(self, "_closed"):
            known = object.__getattribute__(self, "CONTEXT_NAME")
            raise RuntimeError(f"cannot access {known} after the `with` statement is exited")
        return object.__getattribute__(self, name)

    def __enter__(self) -> Any:
        known = type(self).CONTEXT_NAME
        if self._closed:
            raise RuntimeError(f"cannot reuse {known} in multiple `with` statements")
        if self._in_context:
            raise RuntimeError(f"cannot nest `with` statements using the same {known}")
        self._in_context = True
        return self

    def __exit__(self, *args: Any) -> Literal[False]:
        self._close()
        self._in_context = False
        self._closed = True
        return False

    def _close(self) -> None:
        """What the end of the ``with`` block does to the object: ROOT's lets its pointer go."""
