"""``RooHelpers``' guards on the message service, and ``RooFit::EvalBackend``."""

from __future__ import annotations

import pytest

import xrdroot.pyroot as ROOT
from xrdroot.roofit import messages


def _say(level: int, text: str) -> None:
    messages.service().log(None, level, "InputArguments", text)


def test_a_local_level_holds_only_while_the_guard_lives(capsys):
    held = messages.service()
    before = (held.globalKillBelow(), [stream.minLevel for stream in held._streams])
    guard = ROOT.RooHelpers.LocalChangeMsgLevel(ROOT.RooFit.FATAL, 0, ROOT.RooFit.InputArguments)
    assert held.globalKillBelow() == ROOT.RooFit.FATAL
    _say(ROOT.RooFit.ERROR, "silenced")
    guard.restore()
    guard.restore()  # once is all it does
    assert (held.globalKillBelow(), [stream.minLevel for stream in held._streams]) == before
    with ROOT.RooHelpers.LocalChangeMsgLevel(ROOT.RooFit.INFO, ROOT.RooFit.Tracing, 0, False):
        assert held.numStreams() == len(before[1]) + 1
    assert held.numStreams() == len(before[1])
    assert "silenced" not in capsys.readouterr().out


def test_a_hijacked_stream_keeps_the_messages_of_its_level_and_topics(capsys):
    def captured() -> str:
        hijack = ROOT.RooHelpers.HijackMessageStream(ROOT.RooFit.WARNING,
                                                     ROOT.RooFit.InputArguments)
        _say(ROOT.RooFit.WARNING, "caught")
        assert hijack.stream() is hijack._text
        return hijack.str()

    assert captured().endswith("WARNING:InputArguments -- caught\n")
    _say(ROOT.RooFit.WARNING, "printed")
    messages.service().setGlobalKillBelow(ROOT.RooFit.ERROR)
    named = ROOT.RooHelpers.HijackMessageStream(ROOT.RooFit.INFO, ROOT.RooFit.InputArguments, "x")
    assert named.str() == "" and messages.service().globalKillBelow() == ROOT.RooFit.INFO
    named.restore()
    assert messages.service().globalKillBelow() == ROOT.RooFit.ERROR
    messages.service().setGlobalKillBelow(ROOT.RooFit.DEBUG)
    assert "printed" in capsys.readouterr().out and "RooHelpers" in repr(ROOT.RooHelpers)


def test_an_eval_backend_is_a_command_with_a_name_and_a_number():
    cpu = ROOT.RooFit.EvalBackend.Cpu()
    assert (cpu.name(), cpu.value(), cpu.GetName()) == ("cpu", 1, "EvalBackend")
    assert cpu.value(0) == "cpu" and ROOT.RooFit.EvalBackend(cpu).name() == "cpu"
    names = [made().name() for made in (ROOT.RooFit.EvalBackend.Legacy, ROOT.RooFit.EvalBackend.Cuda,
                                        ROOT.RooFit.EvalBackend.Codegen,
                                        ROOT.RooFit.EvalBackend.CodegenNoGrad)]
    assert names == ["legacy", "cuda", "codegen", "codegen_no_grad"]
    assert ROOT.RooFit.EvalBackend.defaultValue() == 1 and ROOT.RooFit.EvalBackend("CPU").value() == 1
    with pytest.raises(ValueError, match="not one of RooFit's backends"):
        ROOT.RooFit.EvalBackend("gpu")


def test_std_get_takes_an_element_of_a_tuple_pair_or_vector_and_a_lone_value_is_its_own_first():
    pair = ROOT.std.pair["int", "double"](1, 2.5)
    assert ROOT.std.get[1](pair) == 2.5 and ROOT.std.get[0]((4, 5)) == 4
    assert ROOT.std.get[1](ROOT.std.vector["double"]([1.0, 3.0])) == 3.0
    lone = object()
    assert ROOT.std.get[0](lone) is lone and "get" in repr(ROOT.std.get)
    with pytest.raises(IndexError, match="lone object"):
        ROOT.std.get[1](lone)
