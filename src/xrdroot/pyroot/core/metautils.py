"""``ROOT::TMetaUtils`` and ``ROOT::FoundationUtils``: the utilities ROOT's own unit tests are
written against.

``ROOT::TMetaUtils::GetCppName`` turns any text into a C++ variable name - every character C++
would not take is spelled out as two letters, ``+`` being ``pL`` and ``<`` ``lE`` - which is how
ROOT names the variables of a macro it writes (``lEoBsPnOaTdAp...``). ``GetRealPath`` is llvm's
canonical path: the symlinks followed and a leading ``~`` expanded. ``ROOT::FoundationUtils::
GetCurrentDir`` is the working directory with the separator on the end, which is what ROOT's own
``cwd + "file"`` concatenations need. The numbers and the names are README of ROOT's own
``core/clingutils/src/TClingUtils.cxx`` and ``core/foundation/src/FoundationUtils.cxx``.
"""

from __future__ import annotations

import os
from typing import Any

from .refs import store

__all__ = ["FoundationUtils", "GetCppName", "GetRealPath", "TMetaUtils"]

#: The two letters each character C++ would not take is spelled as, from
#: ``ROOT::TMetaUtils::GetCppName``'s own switch - which is why ROOT's macros have variable names
#: like ``lEoBsPnOaTdAp...``: a ``<[`` at the front of an object's name is ``lEoB``.
REPLACEMENTS = {
    "+": "pL", "-": "mI", "*": "mU", "/": "dI", "&": "aN", "%": "pE", "|": "oR",
    "^": "hA", ">": "gR", "<": "lE", "=": "eQ", "~": "wA", ".": "dO", "(": "oP",
    ")": "cP", "[": "oB", "]": "cB", "{": "lB", "}": "rB", ";": "sC", "#": "hS",
    "?": "qM", "`": "bT", "!": "nO", ",": "cO", "$": "dA", " ": "sP", ":": "cL",
    '"': "dQ", "@": "aT", "'": "sQ", "\\": "fI",
}  # fmt: skip


def _cpp_name(text: str) -> str:
    """``GetCppName``'s own answer: every special character spelled out, and an underscore in
    front of an empty name or one that starts with a digit, since neither is a C++ name."""
    spelled = "".join(REPLACEMENTS.get(character, character) for character in text)
    return f"_{spelled}" if not spelled or spelled[0].isdigit() else spelled


def GetCppName(out: Any, text: Any) -> None:
    """``GetCppName(name, text)``: into ``name``, the text as a C++ variable name ROOT would make.

    The name is an out-parameter - ``std::string &`` - so the local a macro hands over is a cell,
    as ``TGraph::GetPoint``'s ``x`` and ``y`` are.
    """
    store(out, _cpp_name(str(text)))


def GetRealPath(path: Any) -> str:
    """``GetRealPath``: the path made canonical - every symlink followed, ``~`` expanded, and the
    working directory put in front of a relative one."""
    return os.path.realpath(os.path.expanduser(str(path)))


class _FoundationUtils:
    """``ROOT::FoundationUtils``: the filesystem helpers ROOT's own code is written against."""

    def __repr__(self) -> str:
        return "<namespace ROOT::FoundationUtils>"

    @staticmethod
    def GetCurrentDir() -> str:
        """``GetCurrentDir``: the working directory, with the separator on the end - ROOT's own
        callers concatenate a file name onto it."""
        return os.path.join(os.getcwd(), "")


class _MetaUtils:
    """``ROOT::TMetaUtils``: ROOT's name and path utilities."""

    def __repr__(self) -> str:
        return "<namespace ROOT::TMetaUtils>"

    #: ``GetCppName`` and ``GetRealPath``, as ROOT's namespace holds them.
    GetCppName = staticmethod(GetCppName)
    GetRealPath = staticmethod(GetRealPath)


#: ``ROOT::TMetaUtils``, and ``ROOT::FoundationUtils`` beside it.
TMetaUtils = _MetaUtils()
FoundationUtils = _FoundationUtils()
