"""RooFit's message service: which stream takes a message, and how the streams are listed.

The listings and the error lines below are what ROOT 6.40.04 printed for
``RooMsgService::instance().Print()`` and ``Print("v")`` after the same
calls. The service here is a fresh one in most tests, so that nothing a
test does to the streams leaks into another's output.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest

from xrdroot.roofit import messages
from xrdroot.roofit.cmdargs import RooCmdArg
from xrdroot.roofit.messages import (
    DEBUG,
    ERROR,
    INFO,
    PROGRESS,
    SERVICE,
    TOPICS,
    WARNING,
    RooMsgService,
    StreamConfig,
)
from xrdroot.roofit.variables import RooRealVar

#: ``Print()`` of the streams RooFit starts with.
DEFAULT_STREAMS = (
    "[0] MinLevel = PROGRESS Topic = Generation Minimization Plotting Fitting Integration "
    "LinkStateMgmt Eval Caching Optimization ObjectHandling InputArguments Tracing Contents "
    "DataHandling NumericIntegration FastEvaluations \n"
    "[1] MinLevel = INFO Topic = Minimization Plotting Fitting Eval Caching ObjectHandling "
    "InputArguments DataHandling NumericIntegration \n"
    "[2] MinLevel = INFO Topic = HistFactory \n"
)


class Tagged:
    """Something a message can be about: a name, a class, a base class and attributes."""

    def __init__(self, name: str, tags: tuple[str, ...] = ()) -> None:
        self.name = name
        self.tags = tags

    def GetName(self) -> str:
        return self.name

    def ClassName(self) -> str:
        return "RooGaussian"

    def InheritsFrom(self, name: str) -> bool:
        return name in ("RooGaussian", "RooAbsPdf")

    def getAttribute(self, name: str) -> bool:
        return name in self.tags


def test_the_service_starts_with_roots_three_streams(capsys: Any) -> None:
    """A tutorial's first lines of output depend on which topics the default streams print."""
    service = RooMsgService()
    service.Print()
    assert capsys.readouterr().out == "Active Message streams\n" + DEFAULT_STREAMS
    service.Print("v")
    assert capsys.readouterr().out == "All Message streams\n" + DEFAULT_STREAMS
    assert service.numStreams() == 3


def test_a_stream_about_one_object_lists_every_condition_and_whether_it_is_off(
    capsys: Any,
) -> None:
    """``Print("v")`` names what each stream is restricted to, as ROOT lists it."""
    service = RooMsgService()
    index = service.addStream(
        WARNING,
        RooCmdArg("Topic", TOPICS["Fitting"]),
        RooCmdArg("ClassName", "RooGaussian"),
        RooCmdArg("ObjectName", "g"),
        RooCmdArg("BaseClassName", "RooAbsPdf"),
        LabelName="tag",
    )
    assert index == 3
    service.setStreamStatus(index, False)
    assert service.getStreamStatus(index) is False
    service.Print("v")
    assert capsys.readouterr().out == (
        "All Message streams\n" + DEFAULT_STREAMS + "[3] MinLevel = WARNING Topic = Fitting  "
        "ObjectName = g ClassName = RooGaussian BaseClassName = RooAbsPdf TagLabel = tag "
        "(NOT ACTIVE)\n"
    )
    service.Print()
    assert capsys.readouterr().out == "Active Message streams\n" + DEFAULT_STREAMS


def test_a_stream_on_every_topic_says_any(capsys: Any) -> None:
    """A stream made without a topic takes every one, and is listed as ``Any``."""
    service = RooMsgService()
    service.addStream(DEBUG)
    service.Print()
    assert capsys.readouterr().out.splitlines()[-1] == "[3] MinLevel = DEBUG Topic =  Any "


def test_an_unknown_stream_number_is_refused_with_roots_error(capsys: Any) -> None:
    """ROOT prints an error and carries on when a stream number is out of range."""
    service = RooMsgService()
    service.setStreamStatus(9, False)
    assert service.getStreamStatus(9) is False
    service.setStreamStatus(-1, True)
    assert capsys.readouterr().out == (
        "RooMsgService::setStreamStatus() ERROR: invalid stream ID 9\n"
        "RooMsgService::getStreamStatus() ERROR: invalid stream ID 9\n"
        "RooMsgService::setStreamStatus() ERROR: invalid stream ID -1\n"
    )


def test_a_message_goes_to_the_first_stream_that_takes_it_with_its_number(capsys: Any) -> None:
    """The ``[#n]`` of a line is the number of the stream that printed it."""
    service = RooMsgService()
    service.log(None, INFO, "Fitting", "fitted")
    service.log(None, WARNING, "Integration", "warned")
    service.log(None, INFO, "Integration", "dropped")
    service.log(None, INFO, "HistFactory", "hf")
    assert capsys.readouterr().out == (
        "[#1] INFO:Fitting -- fitted\n[#0] WARNING:Integration -- warned\n"
        "[#2] INFO:HistFactory -- hf\n"
    )


def test_errors_are_counted_whether_or_not_they_are_printed(capsys: Any) -> None:
    """A fit's status looks at the error count, which counts silenced errors too."""
    service = RooMsgService()
    service.setGlobalKillBelow(messages.FATAL + 1)
    assert service.globalKillBelow() == 6
    service.log(None, ERROR, "Eval", "bad")
    service.log(None, messages.FATAL, "Eval", "worse")
    service.log(None, WARNING, "Eval", "not counted")
    assert service.errorCount() == 2
    assert capsys.readouterr().out == ""
    service.clearErrorCount()
    assert service.errorCount() == 0


def test_a_progress_line_is_ended_before_the_next_message(capsys: Any) -> None:
    """Progress dots stay on one line until a message of another level starts a new one."""
    service = RooMsgService()
    service.log(None, PROGRESS, "Fitting", ".")
    service.log(None, PROGRESS, "Fitting", ".")
    service.log(None, INFO, "Fitting", "done")
    assert capsys.readouterr().out == (
        "[#0] PROGRESS:Fitting -- .\n[#0] PROGRESS:Fitting -- .\n\n[#1] INFO:Fitting -- done\n"
    )


def test_a_stream_without_a_prefix_prints_the_bare_text() -> None:
    """``Prefix(false)`` streams write the message alone, to the stream they were given."""
    service = RooMsgService()
    sink = io.StringIO()
    first = service.addStream(DEBUG, Topic=TOPICS["Tracing"], Prefix=False, OutputStream=sink)
    assert service.getStream(first).stream() is sink
    service.log(None, DEBUG, "Tracing", "bare")
    assert sink.getvalue() == "bare\n"
    service.deleteStream(first)
    assert service.numStreams() == 3


def test_a_stream_can_write_to_a_file(tmp_path: Path) -> None:
    """``OutputFile("log.txt")`` opens the file, and the messages land there."""
    service = RooMsgService()
    path = tmp_path / "log.txt"
    index = service.addStream(DEBUG, Topic=TOPICS["Tracing"], OutputFile=str(path))
    service.log(None, DEBUG, "Tracing", "to a file")
    service.getStream(index).stream().close()
    assert path.read_text() == "[#3] DEBUG:Tracing -- to a file\n"


def test_a_stream_about_an_object_takes_only_messages_about_that_object() -> None:
    """Each of name, class, base class and tag must hold for the stream to take a message."""
    stream = StreamConfig(DEBUG, TOPICS["Eval"], objectName="g", className="RooGaussian")
    stream.baseClassName = "RooAbsPdf"
    stream.tagName = "watched"
    assert not stream.universal
    assert stream.match(INFO, TOPICS["Eval"], Tagged("g", ("watched",)))
    assert not stream.match(INFO, TOPICS["Eval"], Tagged("g"))
    assert not stream.match(INFO, TOPICS["Eval"], Tagged("h", ("watched",)))
    assert not stream.match(INFO, TOPICS["Eval"], None)
    stream.baseClassName = "RooAbsReal"
    assert not stream.match(INFO, TOPICS["Eval"], Tagged("g", ("watched",)))


def test_a_stream_turned_off_or_below_its_level_or_off_topic_takes_nothing() -> None:
    """Level, topic and the on switch are checked before anything about the object."""
    stream = StreamConfig(WARNING, TOPICS["Eval"])
    assert stream.universal
    assert stream.match(WARNING, TOPICS["Eval"], None)
    assert not stream.match(INFO, TOPICS["Eval"], None)
    assert not stream.match(WARNING, TOPICS["Fitting"], None)
    stream.addTopic(TOPICS["Fitting"])
    assert stream.match(WARNING, TOPICS["Fitting"], None)
    stream.removeTopic(TOPICS["Eval"])
    assert stream.topic == TOPICS["Fitting"]
    stream.active = False
    assert not stream.match(ERROR, TOPICS["Fitting"], None)


def test_a_stream_about_a_variable_matches_it_by_its_real_class() -> None:
    """A RooFit node answers the stream's questions about its name and class itself."""
    service = RooMsgService()
    x = RooRealVar("x", "x", 0.0, 1.0)
    service.addStream(DEBUG, Topic=TOPICS["Tracing"], ClassName="RooRealVar", ObjectName="x")
    assert service.activeStream(x, TOPICS["Tracing"], DEBUG) == 3
    assert service.isActive(x, TOPICS["Tracing"], DEBUG)
    assert not service.isActive(RooRealVar("y", "y", 0.0, 1.0), TOPICS["Tracing"], DEBUG)


def test_the_global_kill_level_silences_everything_below_it() -> None:
    """``setGlobalKillBelow(WARNING)`` is how tutorials quieten RooFit's INFO lines."""
    service = RooMsgService()
    assert service.isActive(None, TOPICS["Fitting"], INFO)
    service.setGlobalKillBelow(WARNING)
    assert service.activeStream(None, TOPICS["Fitting"], INFO) == -1
    assert service.activeStream(None, TOPICS["Fitting"], WARNING) == 0
    service.reset()
    assert service.globalKillBelow() == DEBUG


def test_a_saved_state_brings_back_the_streams_as_they_were() -> None:
    """``saveState`` copies each stream, so switching one off afterwards is undone."""
    service = RooMsgService()
    service.saveState()
    service.setStreamStatus(1, False)
    service.addStream(DEBUG)
    service.restoreState()
    assert service.numStreams() == 3
    assert service.getStreamStatus(1) is True


def test_silent_mode_is_a_flag_the_service_keeps() -> None:
    """Minimisers ask the service whether to be silent; it only remembers the answer."""
    service = RooMsgService()
    assert service.silentMode() is False
    service.setSilentMode(True)
    assert service.silentMode() is True


def test_there_is_one_service_whichever_way_it_is_asked_for() -> None:
    """``RooMsgService::instance()`` is the service every message goes through."""
    assert RooMsgService.instance() is SERVICE
    assert messages.service() is SERVICE


def test_the_module_functions_print_through_the_one_service(capsys: Any) -> None:
    """``log`` adds the prefix; ``log_plain`` writes only what it is given, as ``ooccoutW``."""
    SERVICE.saveState()
    try:
        messages.log(None, INFO, "Fitting", "with prefix")
        messages.log_plain(None, WARNING, "Fitting", "plain")
        messages.log_plain(None, DEBUG, "Fitting", "dropped")
    finally:
        SERVICE.restoreState()
    assert capsys.readouterr().out == "[#1] INFO:Fitting -- with prefix\nplain"


@pytest.mark.parametrize(("level", "name"), [(DEBUG, "DEBUG"), (messages.FATAL, "FATAL")])
def test_each_level_has_the_name_roots_prefix_gives_it(level: int, name: str) -> None:
    """The prefix spells a level as ``RooFit::MsgLevel`` names it."""
    assert messages.LEVELS[level] == name
