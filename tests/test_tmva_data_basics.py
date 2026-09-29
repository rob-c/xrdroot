"""TMVA's small pieces: option strings, the logger, the tools, the type numbers and variables."""

from __future__ import annotations

import pytest

import xrdroot.pyroot as ROOT
from tmvasupport import session
from xrdroot.tmva.log import CONFIG, DEBUG, ERROR, FATAL, VERBOSE, Config, Logger, TMVAError
from xrdroot.tmva.options import Options, truth
from xrdroot.tmva.tools import Tools, gConfig, gTools
from xrdroot.tmva.types import Types
from xrdroot.tmva.variables import ClassInfo, VariableInfo

__all__ = ["session"]


def test_a_flag_given_as_a_word_is_read_as_tmva_reads_it():
    assert truth(True) and truth(" Yes ") and truth("kTRUE") and not truth("no")


def test_an_option_given_twice_is_remembered_with_the_value_it_had_before():
    options = Options("NTrees=10::NTrees=20")
    assert options.integer("NTrees") == 20 and options.repeated == [("ntrees", "10")]


def test_an_array_option_takes_its_whole_value_then_each_index_within_its_size():
    options = Options("Smooth=3:Smooth[1]=7:Smooth[9]=1")
    assert options.array("Smooth", 3, 0) == [3, 7, 3]
    assert options.given("smooth") and not options.given("other")


def test_an_array_option_is_read_as_the_type_of_its_default():
    options = Options("A[0]=True:B[0]=2.5:C[0]=x")
    assert options.array("A", 1, False) == [True]
    assert options.array("B", 1, 0.0) == [2.5]
    assert options.array("C", 1, "") == ["x"]


def test_options_read_numbers_text_and_flags_with_the_askers_defaults():
    options = Options("~Eff=12%:Name=abc:!H")
    assert options.number("Eff") == 12.0 and options.number("None", 1.5) == 1.5
    assert options.text_of("Name") == "abc" and options.text_of("None", "d") == "d"
    assert options.flag("H", True) is False and repr(options) == "Options('~Eff=12%:Name=abc:!H')"


def test_a_line_without_colour_is_marked_with_its_kind(session, capsys):
    CONFIG.use_color = False
    logger = Logger("Source", minimum=DEBUG)
    logger.send(DEBUG, "d")
    logger.info("i")
    logger.verbose("v")
    logger.error("e")
    logger.warning("w")
    printed = capsys.readouterr().out.splitlines()
    assert printed[0].startswith("<DEBUG> ") and printed[1].endswith(": i")
    assert printed[2].startswith("<VERBOSE>") and printed[3].startswith("<ERROR>")
    assert printed[4].startswith("<WARNING> <WARNING>")
    CONFIG.use_color = True


def test_a_coloured_error_is_escaped_and_a_verbose_line_is_left_plain(session, capsys):
    CONFIG.use_color = True
    logger = Logger("Source", minimum=VERBOSE)
    logger.verbose("v")
    logger.send(ERROR, "e")
    printed = capsys.readouterr().out.splitlines()
    assert printed[0].endswith(": v") and printed[1].startswith("\033[31m<ERROR>")


def test_a_source_name_too_long_is_cut_short_with_an_ellipsis(capsys):
    Logger("A" * 40).header("text")
    assert capsys.readouterr().out.startswith("A" * 22 + "...: text")


def test_a_silent_logger_still_prints_what_is_fatal(session, capsys):
    CONFIG.silent = True
    error = Logger("Source").fatal("stop")
    Logger("Source").info("hidden")
    CONFIG.silent = False
    printed = capsys.readouterr().out
    assert "stop" in printed and "hidden" not in printed and "abort program" in printed
    assert isinstance(error, TMVAError)


def test_a_refusal_is_both_a_tmva_error_and_an_unsupported_feature(capsys):
    refused = Logger("Source").refuse("not here")
    assert isinstance(refused, TMVAError) and "not here" in capsys.readouterr().out
    assert FATAL > ERROR


def test_the_configuration_is_one_and_is_set_through_its_setters(session):
    config = Config.Instance()
    assert config is CONFIG is gConfig()
    config.SetUseColor(False)
    config.SetSilent(True)
    config.SetDrawProgressBar(True)
    config.SetNumWorkers(3)
    assert (config.use_color, config.silent, config.draw_progress_bar) == (False, True, True)
    assert config.GetNumWorkers() == 3 and config.GetNCpu() >= 1
    config.SetNumWorkers(1)


def test_a_split_string_is_a_vector_a_macro_walks_with_iterators():
    pieces = ROOT.TMVA.gTools().SplitString("a,,bb,c", ord(","))
    assert list(pieces) == ["a", "bb", "c"] and pieces.size() == 3 and pieces.at(1) == "bb"
    pieces.push_back("d")
    assert not pieces.empty() and Tools.Instance() is gTools()
    walked, it = [], pieces.begin()
    while it != pieces.end():
        walked.append(it.upper())
        it += 1
    assert walked == ["A", "BB", "C", "D"] and pieces.begin() < pieces.end()
    assert (pieces.begin() == "a") is False and pieces.begin() == pieces.begin()


def test_an_iterator_reads_as_the_number_it_points_at_and_hides_its_dunders():
    numbers = Tools.SplitString("1.5 2", " ")
    it = numbers.begin()
    assert float(it) == 1.5 and it.__deref__() == "1.5"
    with pytest.raises(AttributeError):
        _ = it.__missing_thing__
    with pytest.raises(TypeError):
        hash(it)


def test_a_colour_is_an_escape_only_while_colour_is_on(session):
    CONFIG.use_color = True
    assert Tools.Color("bold") == "\033[1m" and Tools.Color("nothing") == ""
    CONFIG.use_color = False
    assert Tools.Color("bold") == ""


def test_a_method_is_named_by_its_number_and_numbered_by_its_name():
    types = Types.Instance()
    assert types is Types.Instance() and types.GetMethodName(Types.kBDT) == "BDT"
    assert Types.GetMethodName(999) == "" and Types.GetMethodName("MLP") == "MLP"
    assert (
        Types.GetMethodType("LD") == Types.kLD and Types.GetMethodType("None") == Types.kMaxMethod
    )


def test_a_variable_knows_its_expression_label_title_unit_type_and_range():
    variable = VariableInfo("sum := var1 + var2", "Sum", "GeV", ord("I"), -1, 2, index=3)
    assert (variable.GetExpression(), variable.GetLabel(), variable.GetTitle()) == (
        "var1+var2",
        "sum",
        "Sum",
    )
    assert (variable.GetUnit(), variable.GetVarType(), variable.GetInternalName()) == (
        "GeV",
        "I",
        "sum[3]",
    )
    assert (variable.GetMin(), variable.GetMax()) == (-1.0, 2.0)
    assert repr(variable) == "VariableInfo('sum' := 'var1+var2')"
    assert VariableInfo("a*b", vartype=None).GetInternalName() == "a_T_b"


def test_a_class_knows_its_name_number_weight_and_cut():
    found = ClassInfo("Signal", 0, "w", "x>0")
    assert (found.GetName(), found.GetNumber(), found.GetWeight(), found.GetCut()) == (
        "Signal",
        0,
        "w",
        "x>0",
    )
