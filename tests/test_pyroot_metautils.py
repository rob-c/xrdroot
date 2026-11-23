"""``ROOT::TMetaUtils``, ``ROOT::FoundationUtils`` and POSIX's ``symlink``.

ROOT's own unit test (``core/clingutils/test/TClingUtilsTests.cxx``) is what these numbers are
taken from: the 31 two-letter spellings ``GetCppName`` uses for the characters C++ would not put
in a name, the path made canonical by ``GetRealPath``, and the trailing separator
``GetCurrentDir`` ends with.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import expect, fresh
from xrdroot.cppstd import unistd
from xrdroot.cppstd.cells import Cell


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


def test_a_name_is_made_of_the_letters_root_spells_each_character_as():
    """ROOT's own test's input and answer, character for character: ``+`` is ``pL``, ``<`` ``lE``,
    a space ``sP`` - which is why ROOT's macros hold names like ``lEoBsPnOaTdAp...``."""
    out = Cell("")
    ROOT.TMetaUtils.GetCppName(out, "some+input-string*with/special&characters")
    expect((out.value, "somepLinputmIstringmUwithdIspecialaNcharacters"))


def test_a_name_that_is_empty_or_starts_with_a_digit_gets_an_underscore():
    empty, digit = Cell(""), Cell("")
    ROOT.TMetaUtils.GetCppName(empty, "")
    ROOT.TMetaUtils.GetCppName(digit, "3rd")
    expect((empty.value, "_"), (digit.value, "_3rd"))


def test_a_character_cpp_takes_is_left_alone():
    out = Cell("")
    ROOT.GetCppName(out, "already_aName9")
    expect((out.value, "already_aName9"))


def test_the_path_root_answers_with_is_the_one_the_symlinks_resolve_to(tmp_path):
    """``GetRealPath`` is llvm's canonical path: a symlink and a path through ``./`` both come
    back as the file they name, which is what ROOT's own test compares."""
    real = tmp_path / "realfile1"
    real.write_text("")
    link = tmp_path / "symlink_realfile1"
    os.symlink("./realfile1", link)
    expect(
        (ROOT.TMetaUtils.GetRealPath(str(link)), str(real)),
        (ROOT.GetRealPath("./realfile1"), str(real)),
    )


def test_the_working_directory_root_reports_ends_with_a_separator():
    """``GetCurrentDir`` is what ROOT's own callers paste a file name onto, so it carries the
    separator already."""
    expect((ROOT.FoundationUtils.GetCurrentDir(), os.path.join(os.getcwd(), "")))


def test_symlink_makes_one_and_answers_as_c_does(tmp_path):
    target = tmp_path / "target"
    target.write_text("")
    link = tmp_path / "link"
    expect((unistd.symlink("target", str(link)), 0), (Path(link).is_symlink(), True))
    expect((unistd.symlink("target", str(link)), -1))


def test_the_cling_pointer_root_keeps_is_nothing_at_all():
    """ROOT's own test asserts ``gCling`` is null before an interpreter starts; this kit has no
    cling at all, its interpreter being ``gInterpreter``."""
    expect((ROOT.gCling is None, True))


def test_each_namespace_prints_as_the_class_root_names_it():
    """``ROOT::TMetaUtils`` and ``ROOT::FoundationUtils`` are namespaces, not objects of ROOT's -
    printing one says which namespace it is."""
    expect(
        (repr(ROOT.TMetaUtils), "<namespace ROOT::TMetaUtils>"),
        (repr(ROOT.FoundationUtils), "<namespace ROOT::FoundationUtils>"),
    )
