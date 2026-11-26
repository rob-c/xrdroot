"""``TPython``: the C++ side's way into Python, which here runs the code a macro hands it.

A macro that calls ``TPython::Exec("x = 1")`` runs Python in the very interpreter the macro
itself is running in, so the names one ``Exec`` makes are the ones the next sees. ROOT hands
back the value of ``_anyresult`` through a ``std::any*`` the caller passes; that is what
:meth:`TPython.Exec` writes into, converting nothing - the any keeps the Python value and the
C++ ``std::any_cast<T>`` is what converts it.
"""

from __future__ import annotations

import os
from typing import Any, ClassVar

__all__ = ["TPython"]


class TPython:
    """``TPython``: ``Exec``, ``Eval`` and ``ExecScript`` over one namespace kept per session.

    ROOT keeps a single dictionary for the session's ``Exec`` calls, so a macro's second call
    sees the first one's names; this does too, and puts ``ROOT`` in it, as ROOT's Python has it.
    """

    #: The names the macro's Python writes into: one dictionary for the session, as ROOT's is.
    namespace: ClassVar[dict[str, Any]] = {"__name__": "__main__"}

    @staticmethod
    def _globals() -> dict[str, Any]:
        """The session's namespace, with ``ROOT`` in it the first time it is asked for."""
        import sys

        scope = TPython.namespace
        scope.setdefault("ROOT", sys.modules["xrdroot.pyroot"])
        return scope

    @staticmethod
    def Exec(code: Any, result: Any = None) -> None:
        """Run ``code`` as Python; ``result``, when given, takes ``_anyresult``'s value.

        That second argument is ROOT's ``std::any*``: the macro's Python leaves the value it
        wants back in ``_anyresult``, which is how ``TPython::Exec(cmd, &out)`` carries a Python
        object into C++. An ``Exec`` that sets nothing leaves the any as it was.
        """
        scope = TPython._globals()
        exec(compile(str(code), "<TPython::Exec>", "exec"), scope)
        if result is None or "_anyresult" not in scope:
            return
        held = scope["_anyresult"]
        result.set(held.held if hasattr(held, "held") else held)

    @staticmethod
    def Eval(code: Any) -> Any:
        """``TPython::Eval``: the value of one Python expression, run in the session's namespace."""
        return eval(compile(str(code), "<TPython::Eval>", "eval"), TPython._globals())

    @staticmethod
    def LoadMacro(name: Any) -> None:
        """``TPython::LoadMacro``: the named file run as Python, in the session's namespace.

        ROOT's own ``LoadMacro`` runs a Python macro through the interpreter it already has; the
        name is a path and nothing more, so a quote or a newline in it is the file's name and not
        something to escape - which is what the four ``LoadMacro*InName`` cases check.
        """
        with open(os.fspath(name), encoding="utf-8") as reading:
            TPython.Exec(reading.read())

    @staticmethod
    def ExecScript(path: Any) -> None:
        """``TPython::ExecScript``: the named file run as Python in the session's namespace."""
        with open(str(path), encoding="utf-8") as reading:
            TPython.Exec(reading.read())
