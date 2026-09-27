"""``TList``, ``TObjArray``, ``TIter``, ``TObjString`` and ``TString``."""

from __future__ import annotations

import os

import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import expect, fresh


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
    expect(
        ([item.GetName() for item in held], ["a", "b", "c"]),
        (bool(held.GetSize() == held.GetEntries() == len(held) == 3), True),
        (bool(held.FindObject("b") is b), True),
        (bool(held.FindObject(b) is b), True),
        (bool(held.FindObject("z") is None), True),
        (bool(b in held), True),
        (bool("c" in held), True),
        (held.IndexOf(c), 2),
        (held.IndexOf(ROOT.TObject()), -1),
        (bool(held.Remove(b) is b), True),
        (bool(held.Remove(b) is None), True),
        (bool(held[0] is a), True),
        (bool(bool(ROOT.TList())), True),
        (bool(ROOT.TList().IsEmpty()), True),
    )
    held.AddAll([b])
    held.Clear()
    held.Delete()
    assert held.IsEmpty()


def test_a_sequence_is_indexed_and_walked_both_ways():
    a, b, c = named("b", "a", "c")
    held = ROOT.TList()
    for item in (a, b, c):
        held.Add(item)
    expect(
        (bool(held.At(1) is b), True),
        (bool(held.At(9) is None), True),
        (bool(held.UncheckedAt(0) is a), True),
        (bool(held.First() is a), True),
        (bool(held.Last() is c), True),
        (held.GetLast(), 2),
        (bool(held.After(a) is b), True),
        (bool(held.Before(b) is a), True),
        (bool(held.Before(a) is None), True),
        (bool(held.After(ROOT.TObject()) is None), True),
    )
    d, e, f = named("d", "e", "f")
    held.AddAt(d, 0)
    held.AddBefore(a, e)
    held.AddAfter(c, f)
    held.AddAfter(ROOT.TObject(), ROOT.TNamed("g", ""))
    expect(
        ([item.GetName() for item in held], ["d", "e", "b", "a", "c", "f", "g"]),
        (bool(held.RemoveAt(0) is d), True),
        (bool(held.RemoveAt(99) is None), True),
        (bool(held.RemoveFirst() is e), True),
        (held.RemoveLast().GetName(), "g"),
    )
    held.Sort()
    expect(
        (bool(held.IsSorted()), True),
        ([item.GetName() for item in held], ["a", "b", "c", "f"]),
    )
    held.Sort(False)
    assert not held.IsSorted()


def test_links_walk_a_list():
    held = ROOT.TList()
    expect(
        (bool(held.FirstLink() is None), True),
        (bool(held.LastLink() is None), True),
    )
    for item in named("a", "b"):
        held.Add(item)
    link = held.FirstLink()
    expect(
        (link.GetObject().GetName(), "a"),
        (link.GetOption(), ""),
        (link.Next().GetObject().GetName(), "b"),
        (bool(link.Next().Next() is None), True),
        (held.LastLink().Prev().GetObject().GetName(), "a"),
        (bool(link.Prev() is None), True),
    )


def test_an_object_array_grows_to_the_slot_asked_for():
    array = ROOT.TObjArray()
    a, b = named("a", "b")
    array.AddAt(a, 3)
    array.AddAtAndExpand(b, 1)
    expect(
        (array.GetEntriesFast(), 4),
        (array.GetEntries(), 2),
        (array.GetLast(), 3),
        (bool(array.At(0) is None), True),
        (array.LowerBound(), 0),
        (bool(array.RemoveAt(3) is a), True),
        (bool(array.RemoveAt(10) is None), True),
        (array.GetLast(), 1),
        (bool(array.Remove(b) is b), True),
        (bool(array.Remove(b) is None), True),
        (array.GetLast(), -1),
    )
    array.Expand(2)
    assert array.GetEntriesFast() == 2
    array.Add(a)
    array.Compress()
    assert list(array) == [a]


def test_print_and_ls_list_every_entry_one_level_in(capsys):
    held = ROOT.TList()
    held.SetName("things")
    held.SetOwner()
    expect(
        (bool(held.IsOwner()), True),
        (held.GetName(), "things"),
    )
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
    expect(
        (walk.Next().GetName(), "a"),
        (walk().GetName(), "b"),
        (bool(walk.Next() is None), True),
    )
    walk.Reset()
    assert [item.GetName() for item in walk] == ["a", "b"]
    backwards = held.MakeIterator(False)
    expect(
        (backwards.Next().GetName(), "b"),
        (bool(backwards.GetCollection() is held), True),
        (bool(ROOT.TIter(None).Next() is None), True),
        (ROOT.TOrdCollection().GetSize(), 0),
    )


def test_an_obj_string_is_a_string_in_a_list():
    text = ROOT.TObjString("hello")
    assert text.GetString() == text.String() == str(text) == "hello"
    text.SetString("bye")
    expect(
        (text, "bye"),
        (hash(text), hash("bye")),
    )


def test_a_tstring_changes_in_place_as_roots_does():
    s = ROOT.TString("Hello World")
    s.ReplaceAll("o", "0")
    expect(
        (s, "Hell0 W0rld"),
        (s.Data(), "Hell0 W0rld"),
        (bool(s.Length() == len(s) == 11), True),
    )
    s.ToUpper()
    assert str(s) == "HELL0 W0RLD"
    s.ToLower().Append("!").Prepend(">")
    assert s == ">hell0 w0rld!"
    s.Insert(1, "[").Remove(1, 1)
    s.Replace(0, 1, "<")
    s.Chop()
    assert s == "<hell0 w0rld"
    s.Remove(5)
    expect(
        (s, "<hell"),
        (s.Capacity(), 5),
        (s.Sizeof(), 6),
    )
    s.Append(ord("x"), 2).Prepend(ord("y"))
    assert s == "y<hellxx"
    s.Replace(0, 1, "abc", 1)
    s.Resize(3)
    assert s == "a<h"
    s.Clear()
    expect(
        (bool(s.IsNull()), True),
        (bool(TStringEquality()), True),
    )


def TStringEquality():
    a = ROOT.TString("b")
    return a > "a" and a >= "b" and a < "c" and a <= "b" and a != "c" and not (a == "c")


def test_a_tstring_is_searched_as_roots_is():
    s = ROOT.TString("Some Text")
    expect(
        (bool(s.Contains("text", ROOT.TString.kIgnoreCase)), True),
        (bool(not s.Contains("text")), True),
        (bool(s.BeginsWith("some", 1)), True),
        (bool(s.EndsWith("Text")), True),
        (s.Index("e"), 3),
        (s.Index("x", 0), 7),
        (bool(s.Index("zz") == ROOT.kNPOS == -1), True),
        (s.First("e"), 3),
        (s.Last(ord("e")), 6),
        (s.First(ord("S")), 0),
        (s.Last("x"), 7),
        (s.CountChar("e"), 2),
        (s.CountChar(ord("T")), 1),
        (s.CompareTo("some text", 1), 0),
        (s.CompareTo("A"), 1),
        (bool(s.EqualTo("Some Text")), True),
        (s(0), "S"),
        (s(5, 4), "Text"),
        (s[0], "S"),
        (list(s)[:2], ["S", "o"]),
        (bool("Text" in s), True),
        (s.upper(), "SOME TEXT"),
        (s.Hash(), s.Hash(0)),
    )


def test_a_tstring_reads_as_numbers():
    expect(
        (ROOT.TString("42").Atoi(), 42),
        (ROOT.TString("x").Atoi(), 0),
        (ROOT.TString("2.5").Atof(), 2.5),
        (ROOT.TString("x").Atof(), 0.0),
        (ROOT.TString("7").Atoll(), 7),
        (bool(ROOT.TString("12").IsDigit()), True),
        (bool(ROOT.TString("12").IsDec()), True),
        (bool(not ROOT.TString("").IsDigit()), True),
        (bool(ROOT.TString("ab").IsAlpha()), True),
        (bool(ROOT.TString("a1").IsAlnum()), True),
        (bool(ROOT.TString("ab").IsAscii()), True),
        (bool(ROOT.TString("  ").IsWhitespace()), True),
        (bool(ROOT.TString("1e3").IsFloat()), True),
        (bool(not ROOT.TString("e").IsFloat()), True),
    )


def test_a_tstring_strips_splits_formats_and_copies():
    s = ROOT.TString("  pad  ")
    expect(
        (s.Strip(), "  pad"),
        (s.Strip(ROOT.TString.kBoth), "pad"),
        (s.Strip(ROOT.TString.kLeading), "pad  "),
        (ROOT.TString("xxa").Strip(1, ord("x")), "a"),
    )
    tokens = ROOT.TString("a,b;;c").Tokenize(",;")
    assert [str(item) for item in tokens] == ["a", "b", "c"]
    made = ROOT.TString.Format("%d-%s", 3, "x")
    expect(
        (bool(isinstance(made, ROOT.TString)), True),
        (made, "3-x"),
    )
    made.Form("%.1f", 2.25)
    expect(
        (made, "2.2"),
        (made.Copy(), "2.2"),
        (ROOT.TString.Itoa(255, 16), "ff"),
        (ROOT.TString.Itoa(-5, 2), "-101"),
        (ROOT.TString.UItoa(0, 10), "0"),
    )


def test_a_tstring_stands_in_for_a_python_string():
    s = ROOT.TString("abcdef", 3)
    expect(
        (s, "abc"),
        (repr(s), "'abc'"),
        (f"{s:>5}", "  abc"),
        (os.fspath(s), "abc"),
        (s + "d", "abcd"),
        ("z" + s, "zabc"),
        (bool(bool(ROOT.TString(""))), True),
    )
    s += "d"
    expect(
        (s, "abcd"),
        (len({ROOT.TString("q"), ROOT.TString("q")}), 1),
        (ROOT.TString(None), ""),
    )
    with pytest.raises(AttributeError):
        s._missing  # noqa: B018
