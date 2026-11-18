"""``TClassEdit``: ROOT's opinions about how a C++ type name is spelled.

A dictionary, a rootmap and a branch title all name types, and ROOT's own code
arguments with itself about how: the standard library's long names shortened
(``basic_string<char,char_traits<char>,allocator<char> >`` is ``string``), the
default template arguments left out, two closing brackets written apart, and the
type a ``unique_ptr`` member is read and written by. The parsing is
:mod:`xrdroot.cppstd.typenames`; what is here is ROOT's class and the flags a
macro passes it.
"""

from __future__ import annotations

from typing import Any

from ...cppstd import typenames

__all__ = ["TClassEdit"]


class TClassEdit:
    """``TClassEdit``: the class whose functions answer how a name is spelled.

    ``ShortType(name, mode)`` takes a mode of the ``EModType`` bits below, which
    a macro writes as ``TClassEdit::kDropAllDefault`` and reaches here as an
    attribute of the class.
    """

    #: ``EModType``: what ``ShortType`` leaves out. Only the ones a macro passes are here
    #: by name; the others' bits are honoured where they matter.
    kDropStlDefault = 0x1
    kDropDefaultTemplate = 0x4
    kKeepOuterConst = 0x8
    kDropStd = typenames.DROP_STD
    kDropTrailSpace = 0x20
    kDropRootTypedef = 0x40
    kInternalNamespace = 0x80
    kDropAllDefault = 0x100

    #: ``TClassEdit::EModType``'s names for the kinds of STL container, which a macro
    #: compares ``IsSTLcontainers``' answer with.
    kNotSTL, kVector, kList, kDeque, kMap, kMultiMap, kSet, kMultiSet = range(1, 9)
    kEnd = kMultiSet + 1

    @staticmethod
    def ShortType(name: Any, mode: Any = 0) -> str:
        """``ShortType(name, mode)``: the name with the standard library's long spellings
        shortened and its default arguments dropped."""
        return typenames.short_type(str(name), int(mode or 0))

    @staticmethod
    def GetNameForIO(name: Any) -> str:
        """``GetNameForIO(name)``: the name a member of this type is read and written by -
        a ``unique_ptr<T>`` inside standard containers reads as a ``T*``."""
        return typenames.name_for_io(str(name))

    @staticmethod
    def CleanType(name: Any, *rest: Any) -> str:
        """``CleanType(name)``: the name without ROOT's own typedefs or a trailing space -
        which, for the names a macro writes, is the name as written."""
        return str(name).strip()
