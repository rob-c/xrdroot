"""Which Python exceptions a C++ ``catch`` of a named type would have caught.

``EXPECT_THROW(f(), std::invalid_argument)`` names a C++ type; the
translation of ``f`` raises a Python exception. A C++ ``throw`` becomes
``RuntimeError``, ``ValueError``, ``IndexError`` as the translator writes
it, and the library raises the like where it throws. What no C++ program
throws - a ``NameError``, an ``AttributeError``, the translator's own refusal
- is :func:`internal`: it is never caught as the exception a test expects, so
a missing piece of the runtime can never pass for the error a test is about.

A host whose library has exceptions of its own says which are its gaps and
which its errors, and what its exception classes catch, with :func:`adopt`.
"""

from __future__ import annotations

from collections.abc import Mapping

from ...errors import CxxError, Unsupported

__all__ = ["internal", "caught", "adopt", "refusal", "INTERNAL", "LIBRARY", "CAUGHT"]

#: What Python raises that no C++ throw becomes: the runtime's gaps and the translation's slips.
INTERNAL: tuple[type[BaseException], ...] = (
    CxxError,
    NameError,
    AttributeError,
    TypeError,
    NotImplementedError,
    ImportError,
    AssertionError,
    RecursionError,
    ZeroDivisionError,
)

#: What a C++ library error is in Python: every error a throw of the STL's becomes.
LIBRARY: tuple[type[BaseException], ...] = (
    RuntimeError, ValueError, LookupError, OSError, ArithmeticError,
)  # fmt: skip

#: The C++ exception types, and the Python exceptions a ``catch`` of each takes - by name, or
#: as ``"library"`` for every library error, which a host may add to.
CAUGHT: dict[str, tuple[type[BaseException], ...] | str] = {
    "std::exception": (Exception,),
    "std::runtime_error": "library",
    "std::logic_error": (RuntimeError, ValueError, LookupError),
    "std::invalid_argument": (ValueError,),
    "std::domain_error": (ValueError,),
    "std::length_error": (ValueError,),
    "std::out_of_range": (LookupError,),
    "std::range_error": (RuntimeError, ArithmeticError),
    "std::overflow_error": (RuntimeError, ArithmeticError),
    "std::underflow_error": (RuntimeError, ArithmeticError),
    "std::bad_alloc": (MemoryError,),
}

#: What a part of GoogleTest the runtime does not provide is refused with.
REFUSAL: list[type[Exception]] = []


def refusal(message: str) -> Exception:
    """The refusal of a missing part of GoogleTest: the host's kind, else the runtime's own."""
    kind = REFUSAL[-1] if REFUSAL else Unsupported
    return kind(message)


def adopt(
    *,
    gaps: tuple[type[BaseException], ...] = (),
    errors: tuple[type[BaseException], ...] = (),
    caught: Mapping[str, str] | None = None,
    refused: type[Exception] | None = None,
) -> None:
    """Take a host's exceptions in: its ``gaps`` (never caught as what a test expects), its
    library ``errors`` (caught as ``std::runtime_error``), the C++ classes it throws
    that catch every library error, and what it refuses a missing feature with."""
    global INTERNAL, LIBRARY
    INTERNAL = (*INTERNAL, *(gap for gap in gaps if gap not in INTERNAL))
    LIBRARY = (*LIBRARY, *(error for error in errors if error not in LIBRARY))
    CAUGHT.update(dict.fromkeys(caught or (), "library"))
    if refused is not None and refused not in REFUSAL:
        REFUSAL.append(refused)


def internal(why: BaseException) -> bool:
    """Is ``why`` something no C++ program throws: a gap in the runtime, or not an error?"""
    return isinstance(why, INTERNAL) or not isinstance(why, Exception)


def caught(why: BaseException, spelled: str) -> bool:
    """Would ``catch (spelled &)`` catch ``why``? ``spelled`` as written, ``const`` and all."""
    name = spelled.replace("const", "").replace("&", "").replace(" ", "").lstrip(":")
    kinds = CAUGHT.get(name)
    if isinstance(kinds, str):
        return isinstance(why, LIBRARY)
    if kinds is not None:
        return isinstance(why, kinds)
    short = name.rsplit("::", 1)[-1]  # one of the test's own classes: by its name
    return any(kind.__name__ == short for kind in type(why).__mro__)
