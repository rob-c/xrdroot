"""``RooCmdArg``: RooFit's named options, and PyROOT's keywords that stand for them.

A command made with fewer arguments than ROOT's function takes gets ROOT's
defaults from ``RooGlobalFunc.h``, and a string where ROOT takes a colour,
a style or an error type is read as ROOT's ``TColorNumber`` and friends read
it, so ``LineColor("r")`` is ``kRed``.
"""

from __future__ import annotations

import pytest

from xrdroot.roofit.cmdargs import Commands, RooCmdArg, RooLinkedList, commands, make


def test_a_command_made_bare_takes_roots_defaults() -> None:
    """``Save()`` is ``Save(true)`` and ``AutoBinning()`` is ``AutoBinning(100, 0.1)``."""
    assert RooCmdArg("Save").args == (True,)
    assert RooCmdArg("AutoBinning").args == (100, 0.1)
    assert RooCmdArg("AutoBinning", 50).args == (50,)
    assert RooCmdArg("Unheard").args == ()


def test_a_command_reads_its_values_by_position_with_a_fallback() -> None:
    """A method asks for the n-th value and says what to use when there are fewer."""
    command = RooCmdArg("Range", -1.0, 1.0)
    assert command.value() == -1.0
    assert command.value(1) == 1.0
    assert command.value(2, "none") == "none"
    assert command.GetName() == "Range"
    assert repr(command) == "RooFit::Range(-1.0, 1.0)"


def test_the_none_command_says_nothing() -> None:
    """``RooCmdArg::none()`` has no name, so a call ignores it."""
    empty = RooCmdArg.none()
    assert empty.name == ""
    assert Commands([empty]).names() == []


@pytest.mark.parametrize(
    ("name", "text", "number"),
    [
        ("LineColor", "r", 632),
        ("FillColor", "kWhite", 0),
        ("MarkerColor", "kAzure+2", 862),
        ("Color", "kRed-3", 629),
        ("LineStyle", "--", 2),
        ("LineStyle", "kDotted", 3),
        ("FillStyle", "3004", 3004),
        ("MarkerStyle", "kFullCircle", 20),
        ("DataError", "SumW2", 1),
    ],
)
def test_a_string_where_root_takes_a_number_is_read_as_root_reads_it(
    name: str, text: str, number: int
) -> None:
    """Matplotlib's letters, ROOT's enumerations and plain numbers are all accepted."""
    assert RooCmdArg(name, text, 7).args == (number, 7)


def test_a_bad_colour_name_is_refused() -> None:
    """A name ROOT's interpreter would not know is an error, not a silent black."""
    with pytest.raises(ValueError, match="kPurple"):
        RooCmdArg("LineColor", "kPurple")


def test_data_error_none_from_python_is_roots_none_error_type() -> None:
    """PyROOT's ``DataError=None`` means ``RooAbsData::None``, which is two."""
    assert RooCmdArg("DataError", None).args == (2,)
    assert make("DataError", None).args == (2,)


def test_a_linked_list_collects_commands() -> None:
    """``fitTo(data, list)`` takes a ``RooLinkedList`` of commands made up front."""
    given = RooLinkedList()
    given.Add(RooCmdArg("Save"))
    given.Add(RooCmdArg("PrintLevel", -1))
    assert given.GetSize() == 2
    assert commands([given]).get("PrintLevel") == -1


def test_a_keyword_flag_is_the_command_or_nothing() -> None:
    """``MoveToBack=True`` makes the command; ``MoveToBack=False`` leaves it out."""
    assert make("MoveToBack", True).name == "MoveToBack"
    assert make("MoveToBack", False).name == ""


def test_a_keyword_tuple_is_the_commands_arguments() -> None:
    """``Range=(-1, 1)`` is ``Range(-1, 1)``, and a single value is its one argument."""
    assert make("Range", (-1, 1)).args == (-1, 1)
    assert make("Range", [-2, 2]).args == (-2, 2)
    assert make("PrintLevel", -1).args == (-1,)


def test_a_keyword_dict_is_a_variable_and_the_commands_that_go_with_it() -> None:
    """``YVar=dict(var=y, Binning=50)`` is ``YVar(y, Binning(50))``."""
    made = make("YVar", {"var": "y", "Binning": 50})
    assert made.args[0] == "y"
    assert made.args[1].name == "Binning"
    assert made.args[1].args == (50,)
    what = make("Project", {"what": "p"})
    assert what.args == ("p",)


def test_a_keyword_dict_for_a_map_command_is_the_map_itself() -> None:
    """``Import={"a": data}`` hands the whole map to the command, as ROOT's ``std::map``."""
    table = {"a": 1, "b": 2}
    assert make("Import", table).args == (table,)


def test_the_last_of_a_repeated_option_wins_but_each_is_kept() -> None:
    """``RooCmdConfig`` takes the last value, while ``Slice`` and ``Cut`` may repeat."""
    found = commands([RooCmdArg("Slice", "a"), RooCmdArg("Slice", "b")], {"Save": True})
    assert found.get("Slice") == "b"
    assert [one.args for one in found.every("Slice")] == [("a",), ("b",)]
    assert found.args("Slice") == ("b",)
    assert found.args("Missing") == ()
    assert found.get("Missing", default=3) == 3
    assert "Save" in found
    assert "Missing" not in found
    assert found.names() == ["Slice", "Save"]


def test_commands_may_come_as_a_tuple_of_them() -> None:
    """A macro can build its options as a tuple and pass it as one argument."""
    found = commands([(RooCmdArg("Save"), RooCmdArg("Hesse", False))])
    assert found.get("Hesse") is False
    assert commands([]).names() == []


def test_something_that_is_not_a_command_is_refused() -> None:
    """A misplaced argument - a dataset where a command should be - says so."""
    with pytest.raises(TypeError, match="not a RooFit command argument"):
        commands([42])
    with pytest.raises(TypeError, match="not a RooFit command argument"):
        commands([(RooCmdArg("Save"), 42)])


def test_a_duplicated_option_is_warned_about_once_for_each_repeat(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """ROOT's ``RooCmdConfig::process`` warns for every option given twice."""
    found = commands([RooCmdArg("Save"), RooCmdArg("Save"), RooCmdArg("Hesse")])
    found.warn_duplicates("fitTo(model)")
    out = capsys.readouterr().out
    assert out == (
        "[#0] WARNING:InputArguments -- fitTo(model) WARNING: argument Save is duplicated\n"
    )
