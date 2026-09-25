"""``TList``, ``TObjArray``, ``TIter``, ``TObjString`` and ``TString``."""

from __future__ import annotations

import os

import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import fresh


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


def named(*names):
    return [ROOT.TNamed(name, name.upper()) for name in names]


def test_a_list_adds_finds_and_removes_as_roots_does():
    a, b, c = named("a", "b", "c")
    held = ROOT.TList()
    held.Add(b)
    held.AddFirst(a)
    held.AddLast(c)
    assert [item.GetName() for item in held] == ["a", "b", "c"]
    assert held.GetSize() == held.GetEntries() == len(held) == 3
    assert held.FindObject("b") is b and held.FindObject(b) is b and held.FindObject("z") is None
    assert b in held and "c" in held and held.IndexOf(c) == 2 and held.IndexOf(ROOT.TObject()) == -1
    assert held.Remove(b) is b and held.Remove(b) is None
    assert held[0] is a and bool(ROOT.TList()) and ROOT.TList().IsEmpty()
    held.AddAll([b])
    held.Clear()
    held.Delete()
    assert held.IsEmpty()


def test_a_sequence_is_indexed_and_walked_both_ways():
    a, b, c = named("b", "a", "c")
    held = ROOT.TList()
    for item in (a, b, c):
        held.Add(item)
    assert held.At(1) is b and held.At(9) is None and held.UncheckedAt(0) is a
    assert held.First() is a and held.Last() is c and held.GetLast() == 2
    assert held.After(a) is b and held.Before(b) is a and held.Before(a) is None
    assert held.After(ROOT.TObject()) is None
    d, e, f = named("d", "e", "f")
    held.AddAt(d, 0)
    held.AddBefore(a, e)
    held.AddAfter(c, f)
    held.AddAfter(ROOT.TObject(), ROOT.TNamed("g", ""))
    assert [item.GetName() for item in held] == ["d", "e", "b", "a", "c", "f", "g"]
    assert held.RemoveAt(0) is d and held.RemoveAt(99) is None
    assert held.RemoveFirst() is e and held.RemoveLast().GetName() == "g"
    held.Sort()
    assert held.IsSorted() and [item.GetName() for item in held] == ["a", "b", "c", "f"]
    held.Sort(False)
    assert not held.IsSorted()


def test_links_walk_a_list():
    held = ROOT.TList()
    assert held.FirstLink() is None and held.LastLink() is None
    for item in named("a", "b"):
        held.Add(item)
    link = held.FirstLink()
    assert link.GetObject().GetName() == "a" and link.GetOption() == ""
    assert link.Next().GetObject().GetName() == "b" and link.Next().Next() is None
    assert held.LastLink().Prev().GetObject().GetName() == "a" and link.Prev() is None


def test_an_object_array_grows_to_the_slot_asked_for():
    array = ROOT.TObjArray()
    a, b = named("a", "b")
    array.AddAt(a, 3)
    array.AddAtAndExpand(b, 1)
    assert array.GetEntriesFast() == 4 and array.GetEntries() == 2 and array.GetLast() == 3
    assert array.At(0) is None and array.LowerBound() == 0
    assert array.RemoveAt(3) is a and array.RemoveAt(10) is None and array.GetLast() == 1
    assert array.Remove(b) is b and array.Remove(b) is None and array.GetLast() == -1
    array.Expand(2)
    assert array.GetEntriesFast() == 2
    array.Add(a)
    array.Compress()
    assert list(array) == [a]


def test_print_and_ls_list_every_entry_one_level_in(capsys):
    held = ROOT.TList()
    held.SetName("things")
    held.SetOwner()
    assert held.IsOwner() and held.GetName() == "things"
    held.Add(ROOT.TNamed("a", "A"))
    held.Print()
    held.ls("noaddr")
    assert capsys.readouterr().out.splitlines() == [
        "Collection name='things', class='TList', size=1",
        " OBJ: TNamed\ta\tA",
        "OBJ: TList\tthings\tBasic ROOT object : 0",
        " OBJ: TNamed\ta\tA : 0",
    ]


def test_an_iterator_hands_out_none_at_the_end():
    held = ROOT.THashList()
    for item in named("a", "b"):
        held.Add(item)
    walk = ROOT.TIter(held)
    assert walk.Next().GetName() == "a" and walk().GetName() == "b" and walk.Next() is None
    walk.Reset()
    assert [item.GetName() for item in walk] == ["a", "b"]
    backwards = held.MakeIterator(False)
    assert backwards.Next().GetName() == "b" and backwards.GetCollection() is held
    assert ROOT.TIter(None).Next() is None
    assert ROOT.TOrdCollection().GetSize() == 0


def test_an_obj_string_is_a_string_in_a_list():
    text = ROOT.TObjString("hello")
    assert text.GetString() == text.String() == str(text) == "hello"
    text.SetString("bye")
    assert text == "bye" and hash(text) == hash("bye")


def test_a_tstring_changes_in_place_as_roots_does():
    s = ROOT.TString("Hello World")
    s.ReplaceAll("o", "0")
    assert s == "Hell0 W0rld" and s.Data() == "Hell0 W0rld" and s.Length() == len(s) == 11
    s.ToUpper()
    assert str(s) == "HELL0 W0RLD"
    s.ToLower().Append("!").Prepend(">")
    assert s == ">hell0 w0rld!"
    s.Insert(1, "[").Remove(1, 1)
    s.Replace(0, 1, "<")
    s.Chop()
    assert s == "<hell0 w0rld"
    s.Remove(5)
    assert s == "<hell" and s.Capacity() == 5 and s.Sizeof() == 6
    s.Append(ord("x"), 2).Prepend(ord("y"))
    assert s == "y<hellxx"
    s.Replace(0, 1, "abc", 1)
    s.Resize(3)
    assert s == "a<h"
    s.Clear()
    assert s.IsNull() and TStringEquality()


def TStringEquality():
    a = ROOT.TString("b")
    return a > "a" and a >= "b" and a < "c" and a <= "b" and a != "c" and not (a == "c")


def test_a_tstring_is_searched_as_roots_is():
    s = ROOT.TString("Some Text")
    assert s.Contains("text", ROOT.TString.kIgnoreCase) and not s.Contains("text")
    assert s.BeginsWith("some", 1) and s.EndsWith("Text") and s.Index("e") == 3
    assert s.Index("x", 0) == 7 and s.Index("zz") == ROOT.kNPOS == -1
    assert s.First("e") == 3 and s.Last(ord("e")) == 6 and s.First(ord("S")) == 0
    assert s.Last("x") == 7 and s.CountChar("e") == 2 and s.CountChar(ord("T")) == 1
    assert s.CompareTo("some text", 1) == 0 and s.CompareTo("A") == 1 and s.EqualTo("Some Text")
    assert s(0) == "S" and s(5, 4) == "Text" and s[0] == "S" and list(s)[:2] == ["S", "o"]
    assert "Text" in s and s.upper() == "SOME TEXT" and s.Hash() == s.Hash(0)


def test_a_tstring_reads_as_numbers():
    assert ROOT.TString("42").Atoi() == 42 and ROOT.TString("x").Atoi() == 0
    assert ROOT.TString("2.5").Atof() == 2.5 and ROOT.TString("x").Atof() == 0.0
    assert ROOT.TString("7").Atoll() == 7 and ROOT.TString("12").IsDigit()
    assert ROOT.TString("12").IsDec() and not ROOT.TString("").IsDigit()
    assert ROOT.TString("ab").IsAlpha() and ROOT.TString("a1").IsAlnum()
    assert ROOT.TString("ab").IsAscii() and ROOT.TString("  ").IsWhitespace()
    assert ROOT.TString("1e3").IsFloat() and not ROOT.TString("e").IsFloat()


def test_a_tstring_strips_splits_formats_and_copies():
    s = ROOT.TString("  pad  ")
    assert s.Strip() == "  pad" and s.Strip(ROOT.TString.kBoth) == "pad"
    assert (
        s.Strip(ROOT.TString.kLeading) == "pad  " and ROOT.TString("xxa").Strip(1, ord("x")) == "a"
    )
    tokens = ROOT.TString("a,b;;c").Tokenize(",;")
    assert [str(item) for item in tokens] == ["a", "b", "c"]
    made = ROOT.TString.Format("%d-%s", 3, "x")
    assert isinstance(made, ROOT.TString) and made == "3-x"
    made.Form("%.1f", 2.25)
    assert made == "2.2" and made.Copy() == "2.2"
    assert ROOT.TString.Itoa(255, 16) == "ff" and ROOT.TString.Itoa(-5, 2) == "-101"
    assert ROOT.TString.UItoa(0, 10) == "0"


def test_a_tstring_stands_in_for_a_python_string():
    s = ROOT.TString("abcdef", 3)
    assert s == "abc" and repr(s) == "'abc'" and f"{s:>5}" == "  abc" and os.fspath(s) == "abc"
    assert s + "d" == "abcd" and "z" + s == "zabc" and bool(ROOT.TString(""))
    s += "d"
    assert s == "abcd" and len({ROOT.TString("q"), ROOT.TString("q")}) == 1
    assert ROOT.TString(None) == ""
    with pytest.raises(AttributeError):
        s._missing  # noqa: B018
