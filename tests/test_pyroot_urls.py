"""``TUrl`` taken apart as ROOT takes it, and ``TObject``'s comparisons through ``Compare``."""

from __future__ import annotations

import warnings

import pytest

import xrdroot.pyroot as ROOT
from xrdroot.pyroot.core import messages


@pytest.mark.parametrize(
    ("given", "url", "parts"),
    [
        ("a", "http://a/", ("http", "a", 80, "", "", "")),
        ("root://host.cern.ch:1094//data/f.root?x=1#tree",
         "root://host.cern.ch//data/f.root?x=1#tree",
         ("root", "host.cern.ch", 1094, "/data/f.root", "x=1", "tree")),
        ("http://a.b:8080/c/d.root", "http://a.b:8080/c/d.root",
         ("http", "a.b", 8080, "c/d.root", "", "")),
        ("/tmp/f.root", "file:///tmp/f.root", ("file", "", 0, "/tmp/f.root", "", "")),
        ("file:///tmp/g.root", "file:///tmp/g.root", ("file", "", 0, "/tmp/g.root", "", "")),
    ],
)  # fmt: skip
def test_a_url_comes_apart_into_roots_pieces(given, url, parts):
    found = ROOT.TUrl(given)
    assert found.GetUrl() == url and found.GetName() == url and found.IsValid()
    assert (found.GetProtocol(), found.GetHost(), found.GetPort(), found.GetFile(),
            found.GetOptions(), found.GetAnchor()) == parts  # fmt: skip
    assert found.GetHostFQDN() == parts[1]


def test_a_word_said_to_be_a_file_is_a_file_with_its_options_and_anchor_after_it():
    found = ROOT.TUrl("a?x=1#t", True)
    assert (found.GetProtocol(), found.GetFile(), found.GetFileAndOptions()) == (
        "file", "a", "a?x=1#t")  # fmt: skip


def test_urls_compare_by_their_text_so_a_list_of_them_sorts():
    a1, a2, b = ROOT.TUrl("a"), ROOT.TUrl("a"), ROOT.TUrl("b")
    assert a1 < b and not b < a1 and a1 <= a2 and b > a1 and b >= a1 and a1 >= a2
    assert a1 != a2 and a1 == a1  # equal is the same object, as TObject::IsEqual has it
    assert sorted([b, a1]) == [a1, b]
    assert a1.__lt__(1) is NotImplemented and a1.__le__(1) is NotImplemented
    assert a1.__gt__(1) is NotImplemented and a1.__ge__(1) is NotImplemented
    assert a1.__eq__(1) is NotImplemented and a1.__ne__(1) is NotImplemented
    assert a1.__eq__(a2) is NotImplemented and a1.__ne__(a1) is False
    assert {a1: 1}[a1] == 1


def test_a_plain_object_compares_equal_with_itself_only_and_warns_that_compare_is_abstract(
    capsys, monkeypatch
):
    o = ROOT.TObject()
    assert o == o and o != ROOT.TObject() and o != None  # noqa: E711
    assert not o < ROOT.TObject()
    warned = "Warning in <TObject::Compare>: this method must be overridden!"
    assert warned in capsys.readouterr().err
    monkeypatch.setattr(messages, "PYROOT_SCRIPT", True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        o.Compare(o)
    assert [(str(w.message), w.filename, w.lineno) for w in caught] == [
        ("this method must be overridden!", "TObject::Compare", 0)]  # fmt: skip


def test_dynamic_cast_sees_an_object_as_its_base_and_back_again():
    tobject, tobjstring = ROOT.TClass.GetClass("TObject"), ROOT.TClass.GetClass("TObjString")
    o = ROOT.TObjString("a")
    up = tobjstring.DynamicCast(tobject, o)
    assert type(up) is ROOT.TObject
    assert tobjstring.DynamicCast(tobject, up, False) is o
    assert tobjstring.DynamicCast(tobject, up) is up
    assert ROOT.TClass("Nothing").DynamicCast(tobject, up, False) is up
    again = ROOT.TClass.GetClass("TNamed").DynamicCast(tobject, up, False)
    assert type(again) is ROOT.TNamed
