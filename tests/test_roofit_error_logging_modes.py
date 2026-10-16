"""ROOT's logging modes keep distinct counters and preserve logs when printing is restored."""

import io

import pytest

from xrdroot.roofit import evalerrors
from xrdroot.roofit.pdfs.basic import RooGaussian
from xrdroot.roofit.real import RooAbsReal as Real
from xrdroot.roofit.variables import RooRealVar


@pytest.fixture(autouse=True)
def restored_logging():
    previous = Real.evalErrorLoggingMode()
    evalerrors.collecting(False)
    evalerrors.clear()
    Real.setEvalErrorLoggingMode("CountErrors")
    Real.clearEvalErrorLog()
    Real.setEvalErrorLoggingMode("PrintErrors")
    yield
    evalerrors.collecting(False)
    evalerrors.clear()
    Real.setEvalErrorLoggingMode("CountErrors")
    Real.clearEvalErrorLog()
    Real.setEvalErrorLoggingMode(previous)


def gaussian():
    x = RooRealVar("x", "", 0.0, -1.0, 1.0)
    return RooGaussian("g", "", x, 0.0, 1.0)


@pytest.mark.parametrize(
    "name,number", [("PrintErrors", 0), ("CollectErrors", 1), ("CountErrors", 2), ("Ignore", 3)]
)
def test_mode_names_and_constants_select_the_same_policy(name, number):
    Real.setEvalErrorLoggingMode(name)
    assert Real.evalErrorLoggingMode() == number == getattr(Real, name)
    Real.setEvalErrorLoggingMode(number)
    assert Real.evalErrorLoggingMode() == number


def test_printing_reports_errors_immediately_without_retaining_them(capsys):
    gaussian().logEvalError("first")
    text = capsys.readouterr().out
    assert "RooAbsReal::logEvalError(g) evaluation error" in text
    assert "message      : first" in text
    assert "server values: x=x=0, mean=0=0, sigma=1=1" in text
    assert (Real.numEvalErrors(), Real.numEvalErrorItems()) == (0, 0)


def test_collecting_retains_messages_and_server_values_without_printing(capsys):
    Real.setEvalErrorLoggingMode("CollectErrors")
    g = gaussian()
    g.logEvalError("first")
    g.logEvalError("second", "custom=3")
    assert capsys.readouterr().out == ""
    assert (Real.numEvalErrors(), Real.numEvalErrorItems()) == (2, 1)
    stream = io.StringIO()
    Real.printEvalErrors(stream, 1)
    assert stream.getvalue() == (
        "RooGaussian::g[ x=x mean=0 sigma=1 ]\n"
        "     first @ x=x=0, mean=0=0, sigma=1=1\n"
        "     second @ custom=3\n"
    )


def test_counting_avoids_describing_the_failed_object(capsys):
    Real.setEvalErrorLoggingMode("CountErrors")

    def fail():
        pytest.fail("Counting must not evaluate the object description.")

    evalerrors.record("g", fail, "bad", fail, times=2)
    assert (Real.numEvalErrors(), Real.numEvalErrorItems()) == (2, 0)
    Real.printEvalErrors()
    assert capsys.readouterr().out == "2 errors counted\n"
    Real.clearEvalErrorLog()
    assert Real.numEvalErrors() == 0


def test_ignoring_errors_neither_prints_nor_records_them(capsys):
    Real.setEvalErrorLoggingMode("Ignore")
    gaussian().logEvalError("first")
    evalerrors.record("g", lambda: "g", "bad", lambda: "x=1")
    assert capsys.readouterr().out == ""
    assert (Real.numEvalErrors(), Real.numEvalErrorItems()) == (0, 0)


def test_switching_modes_does_not_clear_the_other_modes_storage():
    Real.setEvalErrorLoggingMode("CollectErrors")
    g = gaussian()
    g.logEvalError("first")
    g.logEvalError("second")
    Real.setEvalErrorLoggingMode("CountErrors")
    assert (Real.numEvalErrors(), Real.numEvalErrorItems()) == (0, 1)
    g.logEvalError("third")
    assert (Real.numEvalErrors(), Real.numEvalErrorItems()) == (1, 1)
    Real.clearEvalErrorLog()
    Real.setEvalErrorLoggingMode("PrintErrors")
    Real.clearEvalErrorLog()
    assert (Real.numEvalErrors(), Real.numEvalErrorItems()) == (2, 1)
    Real.setEvalErrorLoggingMode("Ignore")
    Real.clearEvalErrorLog()
    assert Real.numEvalErrors() == 2
    Real.setEvalErrorLoggingMode("CollectErrors")
    Real.clearEvalErrorLog()
    assert (Real.numEvalErrors(), Real.numEvalErrorItems()) == (0, 0)


def test_minimizer_collection_keeps_the_callers_public_policy_and_counter():
    Real.setEvalErrorLoggingMode("CountErrors")
    g = gaussian()
    g.logEvalError("first")
    g.logEvalError("second")
    evalerrors.collecting(True)
    try:
        g.logEvalError("fit")
        assert evalerrors.count() == 1
        assert "fit" in evalerrors.text(10)
    finally:
        evalerrors.collecting(False)
    assert Real.numEvalErrors() == 2
    evalerrors.clear()
    assert Real.numEvalErrors() == 2


def test_real_functions_inherit_error_logging(capsys):
    x = RooRealVar("x", "", 1.0)
    x.logEvalError("bad", "x=1")
    assert "message      : bad" in capsys.readouterr().out


@pytest.mark.parametrize("mode", ["unknown", -1, 2**32])
def test_invalid_enum_values_are_rejected(mode):
    with pytest.raises(ValueError):
        Real.setEvalErrorLoggingMode(mode)


def test_a_float_is_not_an_enum_value():
    with pytest.raises(TypeError):
        Real.setEvalErrorLoggingMode(1.5)


def test_an_unknown_unsigned_enum_value_can_be_stored_without_reporting(capsys):
    Real.setEvalErrorLoggingMode(4)
    gaussian().logEvalError("bad")
    assert Real.evalErrorLoggingMode() == 4
    assert capsys.readouterr().out == ""


def test_describing_an_error_does_not_report_a_second_error(monkeypatch):
    Real.setEvalErrorLoggingMode("CollectErrors")
    g = gaussian()
    original = g.printStream

    def describe(*args):
        g.logEvalError("nested")
        return original(*args)

    monkeypatch.setattr(g, "printStream", describe)
    g.logEvalError("outer")
    assert Real.numEvalErrors() == 1
    assert "outer" in evalerrors.text(10)
    assert "nested" not in evalerrors.text(10)


def test_a_failed_description_restores_the_collecting_policy():
    Real.setEvalErrorLoggingMode("CollectErrors")

    def fail():
        raise RuntimeError("cannot describe")

    with pytest.raises(RuntimeError, match="cannot describe"):
        evalerrors.record("g", fail, "bad", lambda: "x=1")
    assert evalerrors.active() and not evalerrors.quiet()
    gaussian().logEvalError("next")
    assert Real.numEvalErrors() == 1


def test_the_cpp_output_stream_receives_the_error_summary(capsys):
    from xrdroot.cint.runtime.streams import cout

    Real.setEvalErrorLoggingMode("CollectErrors")
    g = gaussian()
    g.logEvalError("first")
    Real.printEvalErrors(cout, 0)
    assert capsys.readouterr().out == "RooGaussian::g[ x=x mean=0 sigma=1 ] has 1 errors\n"
