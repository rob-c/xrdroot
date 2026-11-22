"""``TPRegexp`` and ``TPMERegexp``: ROOT's regular expressions over Python's ``re``."""

from __future__ import annotations

import pytest

import xrdroot.pyroot as ROOT


def test_a_pattern_matches_and_hands_back_the_whole_and_its_groups():
    """``MatchB`` says whether, ``MatchS`` gives a ``TObjArray`` of ``TObjString`` with the
    whole match first - which is what regexp.C reads a protocol and a port out of."""
    email = ROOT.TPRegexp(r"^([\w\-\.]+)@((\[[0-9]{1,3}\.)|(([\w\-]+\.)+))([a-zA-Z]{2,4})$")
    assert email.MatchB(ROOT.TString("fons.rademakers@cern.ch")) is True
    assert email.MatchB(ROOT.TString("not an address")) is False
    found = email.MatchS(ROOT.TString("fons.rademakers@cern.ch"))
    assert found.GetLast() == 6 and found.At(1).GetString() == "fons.rademakers"
    assert ROOT.TPRegexp(r"^\d+$").MatchS(ROOT.TString("12a")).GetLast() == -1


def test_the_call_is_the_text_the_pattern_matches_and_the_position_is_where_it_starts():
    """``s(re)`` is the match itself - Perl's ``$&`` - and ``Match`` answers with its offset."""
    text = ROOT.TString("lekkere pepernotenkoek")
    pattern = ROOT.TPRegexp(r"\bpeper(\w+)koek\b")
    assert str(text(pattern)) == "pepernotenkoek"
    assert pattern.Match("lekkere pepernotenkoek") == 8
    assert pattern.Match("lekkere pepernotenkoek", 10) == -1
    assert str(ROOT.TString("nothing here")(pattern)) == ""


def test_a_substitution_writes_the_groups_and_the_g_option_takes_every_match():
    """``$1`` in the replacement is the first group, and ``g`` - given to the call or to the
    pattern - substitutes every match rather than the first, as both regexp tutorials do."""
    once = ROOT.TString("lekkere pepernotenkoek")
    assert ROOT.TPRegexp(r"\bpeper(\w+)koek\b").Substitute(once, "wal$1boom") == 1
    assert str(once) == "lekkere walnotenboom"
    everywhere = ROOT.TString("neutron proton electron neutron")
    assert ROOT.TPRegexp(r"(n\w+n)").Substitute(everywhere, "neutrino", "g") == 2
    assert str(everywhere) == "neutrino proton electron neutrino"


def test_the_g_option_is_the_object_s_as_well_as_the_call_s():
    """A pattern made with ``g`` substitutes every match even when the call gives no options."""
    by_pattern = ROOT.TString("one two one")
    assert ROOT.TPRegexp("one", "g").Substitute(by_pattern, "1") == 2
    assert str(by_pattern) == "1 two 1"
    spare = ROOT.TString("a")
    assert ROOT.TPRegexp("(a)(b)?").Substitute(spare, r"\2x$2y") == 1
    assert str(spare) == "xy"
    seventh = ROOT.TString("a")
    assert ROOT.TPRegexp("(a)").Substitute(seventh, "$7!") == 1  # no seventh group to write
    assert str(seventh) == "!"
    assert ROOT.TPRegexp("a").Substitute("banana", "x") == 1  # a plain str is not written over


def test_a_pattern_says_what_it_is():
    pattern = ROOT.TPRegexp("a+", "i")
    said = (pattern.GetPattern(), pattern.GetOptions(), repr(pattern))
    assert said == ("a+", "i", "<TPRegexp 'a+'>")
    assert ROOT.TPRegexp("A+", "imx").MatchB("aaa") is True
    assert ROOT.TPRegexp("a.b", "s").MatchB("a\nb") is True
    assert ROOT.TPRegexp("^b", "m").MatchB("a\nb") is True


def test_the_perl_walk_continues_where_the_last_match_ended():
    """``TPMERegexp::Match`` with ``g`` walks the string one match per call, remembering the
    string, the count and every group for ``Print`` - the shape of regexp_pme.C's first block."""
    pattern = ROOT.TPMERegexp("ba[rz]", "g")
    assert pattern.Match("foobarbaz") is True
    assert (pattern.NMatches(), pattern.GetMatch(0)) == (1, "bar")
    assert pattern.Match("foobarbaz") is True
    assert pattern.GetMatch(0) == "baz"
    assert pattern.Match("foobarbaz") is False and pattern.NMatches() == 0
    assert pattern.GetMatch(3) == ""
    with_groups = ROOT.TPMERegexp(r"([\w\.-]+)@((\d+)\.(\d+)\.(\d+)\.(\d+))")
    assert with_groups.Match("matevz.tadel@137.138.170.210") is True
    assert [with_groups.GetMatch(at) for at in (1, 2, 6)] == [
        "matevz.tadel",
        "137.138.170.210",
        "210",
    ]
    assert with_groups.Match("no address here") is False


def test_printing_says_the_pattern_and_all_that_was_found(capsys: pytest.CaptureFixture[str]):
    """``Print()`` is ROOT's one line about the pattern; ``Print("all")`` adds the string and
    every group of the last match."""
    pattern = ROOT.TPMERegexp("([,-])")
    pattern.Split("1-10,20")
    pattern.Print()
    assert capsys.readouterr().out == "Regexp='([,-])', Opts=''\n"
    pattern.Print("all")
    assert capsys.readouterr().out == (
        "Regexp='([,-])', Opts=''\n"
        "  last string='1-10,20'\n"
        "  number of matches = 5\n"
        "  0 - 1\n  1 - -\n  2 - 10\n  3 - ,\n  4 - 20\n"
    )


@pytest.mark.parametrize(
    ("pattern", "text", "maxfields", "expected"),
    [
        (":", "root:x:0:0:root:/root:/bin/bash", 0,
         ["root", "x", "0", "0", "root", "/root", "/bin/bash"]),
        (":", "root:x:0:0:root:/root:/bin/bash", 5,
         ["root", "x", "0", "0", "root:/root:/bin/bash"]),
        (":", "root::0:0:root:/root::", 0, ["root", "", "0", "0", "root", "/root"]),
        (":", ":x:0:0:root::", -1, ["", "x", "0", "0", "root", "", ""]),
        (":", "A dummy line of text.", 0, ["A dummy line of text."]),
        (" *", "hi there", 0, ["h", "i", "t", "h", "e", "r", "e"]),
        ("([,-])", "1-10,20", 0, ["1", "-", "10", ",", "20"]),
    ],
)
def test_a_split_is_the_fields_pcre_makes_of_a_string(
    pattern: str, text: str, maxfields: int, expected: list[str]
) -> None:
    """ROOT's seven split cases from regexp_pme.C, field for field: the maximum, the empty
    fields a trailing match leaves, and a pattern that can match nothing at all."""
    made = ROOT.TPMERegexp(pattern)
    assert made.Split(text, maxfields) == len(expected)
    assert [made.GetMatch(at) for at in range(len(expected))] == expected


def test_a_perl_substitution_takes_the_case_controls_a_replacement_carries():
    """``\\u`` lifts one letter and ``\\U...\\E`` every letter until ``\\E``; ``\\l`` and
    ``\\L`` lower them, as the last block of regexp_pme.C writes a name."""
    text = ROOT.TString("rene.brun@cern.ch, philippe.canal@fnal.gov")
    pattern = ROOT.TPMERegexp(r"(\w+)\.(\w+)@[\w\.-]+", "g")
    assert pattern.Substitute(text, r"\u$1 \U$2\E") == 2
    assert str(text) == "Rene BRUN, Philippe CANAL"
    lowered = ROOT.TString("ABC DEF")
    assert ROOT.TPMERegexp(r"\w+").Substitute(lowered, r"\l\L$0\E", "g") == 2
    assert str(lowered) == "abc def"
    assert ROOT.TPMERegexp("a").Substitute("banana", "x") == 1  # a plain str stays as it is
    single = ROOT.TString("x")
    assert ROOT.TPMERegexp("(x)").Substitute(single, "unterminated \\U") == 1
    assert str(single) == "unterminated "  # ``\U`` with nothing after it changes nothing
