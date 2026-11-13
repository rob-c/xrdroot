"""The registry of a program's tests, and ``RUN_ALL_TESTS()``: GoogleTest's runner and its output.

Each ``TEST`` registers itself as its module is run - the static
initialiser ``gtest/gtest.h``'s macro writes - naming the class its body
is a method of; a ``TEST_P`` registers a pattern, which each
``INSTANTIATE_TEST_SUITE_P`` turns into one test per value. The runner
then runs what ``--gtest_filter`` selects, suite by suite, printing what
GoogleTest prints, and answers 0 if and only if no test failed.

A test that raises - a C++ exception it does not catch, or xrdroot's own
refusal of something the test needs - fails, and the run goes on to the
next test, as GoogleTest catches an exception a test throws.
"""

from __future__ import annotations

import io
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from fnmatch import fnmatchcase
from typing import Any, TextIO

from .exceptions import internal
from .fixtures import PARAMETER, Environment, Test, TestParamInfo
from .state import STATE, Result, flush, report
from .values import show

__all__ = [
    "REGISTRY",
    "Register",
    "RegisterPattern",
    "Instantiate",
    "AddGlobalTestEnvironment",
    "InitGoogleTest",
    "RunAllTests",
    "reset",
    "registered",
    "run_all",
    "AFTER_TEST",
]

#: What ends with every test, after its teardown: a host's checks made in it and still alive.
AFTER_TEST: list[Callable[[], None]] = []

#: The tag GoogleTest puts before a test's name for each of its outcomes.
LABELS = {"OK": "[       OK ]", "FAILED": "[  FAILED  ]", "SKIPPED": "[  SKIPPED ]"}


@dataclass
class Case:
    """One test to run: its suite and name, and where its fixture class is to be found."""

    suite: str
    name: str
    fixture: str
    namespace: dict[str, Any]
    parameterised: bool = False
    param: Any = None

    @property
    def full(self) -> str:
        return f"{self.suite}.{self.name}"

    @property
    def disabled(self) -> bool:
        return self.suite.rpartition("/")[2].startswith("DISABLED_") or self.name.startswith(
            "DISABLED_"
        )

    @property
    def where(self) -> str:
        """``, where GetParam() = 3`` after a parameterised test's name in a failure's line."""
        return f", where GetParam() = {show(self.param)}" if self.parameterised else ""


@dataclass
class Instantiation:
    """One ``INSTANTIATE_TEST_SUITE_P``: the values a suite's patterns are run with."""

    prefix: str
    suite: str
    values: list[Any]
    namer: Callable[[TestParamInfo], Any] | None


@dataclass
class Registry:
    """Every test a program registered, and the flags its command line gave the runner."""

    cases: list[Case] = field(default_factory=list)
    patterns: list[Case] = field(default_factory=list)
    instantiations: list[Instantiation] = field(default_factory=list)
    environments: list[Environment] = field(default_factory=list)
    filter: str = "*"
    also_disabled: bool = False
    listing: bool = False
    #: Where in the C++ an exception was raised, as ``(file, line)``: the translation's map.
    locate: Callable[[BaseException], tuple[str, int] | None] = lambda why: None


#: The tests of the program being run.
REGISTRY = Registry()


def reset() -> None:
    """Forget every test and flag: a new program is about to be run."""
    fresh = Registry()
    for each in fields(fresh):
        setattr(REGISTRY, each.name, getattr(fresh, each.name))
    STATE.result, STATE.pending, STATE.stray = None, None, 0
    STATE.traces.clear()


def registered() -> bool:
    """Has the program registered any test - a ``TEST`` or a ``TEST_P``?"""
    return bool(REGISTRY.cases or REGISTRY.patterns)


def _case(suite: str, name: str, frame: Any) -> Case:
    return Case(str(suite), str(name), f"{suite}_{name}_Test", frame.f_globals)


def Register(suite: str, name: str, file: str, line: int) -> int:
    """A ``TEST`` or ``TEST_F``, registered by the module defining its class as it runs."""
    REGISTRY.cases.append(_case(suite, name, sys._getframe(1)))
    return 0


def RegisterPattern(suite: str, name: str, file: str, line: int) -> int:
    """A ``TEST_P``: run once for each value its suite is instantiated with."""
    REGISTRY.patterns.append(_case(suite, name, sys._getframe(1)))
    return 0


def Instantiate(prefix: str, suite: str, file: str, line: int, values: Any,
                namer: Any = None) -> int:  # fmt: skip
    """``INSTANTIATE_TEST_SUITE_P(prefix, suite, values[, namer])``."""
    REGISTRY.instantiations.append(Instantiation(str(prefix), str(suite), list(values), namer))
    return 0


def AddGlobalTestEnvironment(environment: Environment) -> Environment:
    """``::testing::AddGlobalTestEnvironment(new Env)``: set up first, torn down last."""
    REGISTRY.environments.append(environment)
    return environment


def _flag(argument: str) -> bool:
    """Take a ``--gtest_`` flag from the command line, as ``InitGoogleTest`` does: is it one?"""
    if not argument.startswith("--gtest_"):
        return False
    key, _, value = argument.partition("=")
    if key == "--gtest_filter":
        REGISTRY.filter = value
    REGISTRY.also_disabled |= key == "--gtest_also_run_disabled_tests"
    REGISTRY.listing |= key == "--gtest_list_tests"
    return True


def InitGoogleTest(argc: Any = None, argv: Any = None) -> None:
    """``::testing::InitGoogleTest(&argc, argv)``: take GoogleTest's flags out of the arguments."""
    if argv is None:
        return
    kept = [argument for argument in argv if not _flag(str(argument))]
    if isinstance(argv, list):
        argv[:] = kept
    if hasattr(argc, "value"):
        argc.value = len(kept)


def _expanded() -> list[Case]:
    """Every test: the plain ones, then each pattern once for each value it is instantiated with."""
    cases = list(REGISTRY.cases)
    for pattern in REGISTRY.patterns:
        for made in REGISTRY.instantiations:
            if made.suite == pattern.suite:
                cases.extend(_instances(pattern, made))
    return cases


def _instances(pattern: Case, made: Instantiation) -> list[Case]:
    suite = f"{made.prefix}/{made.suite}" if made.prefix else made.suite
    out = []
    for index, value in enumerate(made.values):
        label = index if made.namer is None else made.namer(TestParamInfo(value, index))
        name = f"{pattern.name}/{label}"
        out.append(Case(suite, name, pattern.fixture, pattern.namespace, True, value))
    return out


def _matches(name: str, globs: str) -> bool:
    return any(fnmatchcase(name, glob) for glob in globs.split(":") if glob)


def _selected(case: Case) -> bool:
    """Does ``--gtest_filter`` (``positive-negative``, ``:``-separated globs) select the test?"""
    positive, _, negative = REGISTRY.filter.partition("-")
    wanted = not positive or _matches(case.full, positive)
    return wanted and not _matches(case.full, negative)


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}{'' if number == 1 else 's'}"


def _say(line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def _milliseconds(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


def _list(cases: list[Case]) -> int:
    """``--gtest_list_tests``: each suite, and its tests indented under it."""
    suite = None
    for case in cases:
        if case.suite != suite:
            suite = case.suite
            _say(f"{suite}.")
        param = f"  # GetParam() = {show(case.param)}" if case.parameterised else ""
        _say(f"  {case.name}{param}")
    return 0


def RunAllTests() -> int:
    """``RUN_ALL_TESTS()``: run every selected test; 0 if and only if none failed."""
    flush()
    cases = [case for case in _expanded() if _selected(case)]
    if REGISTRY.listing:
        return _list(cases)
    chosen = [case for case in cases if REGISTRY.also_disabled or not case.disabled]
    outcomes = _ran(_suites(chosen))
    _disabled(len(cases) - len(chosen))
    return _status(outcomes)


def _status(outcomes: list[tuple[Case, str]]) -> int:
    """The program's exit status: 1 if any test - or any assertion outside one - failed."""
    failed = STATE.stray or any(status == "FAILED" for _, status in outcomes)
    return 1 if failed else 0


def _ran(suites: list[list[Case]]) -> list[tuple[Case, str]]:
    """Every suite run, between GoogleTest's header and its summary."""
    start = time.monotonic()
    _header(sum(map(len, suites)), len(suites))
    outcomes = _environed(suites)
    _footer(outcomes, len(suites), _milliseconds(start))
    return outcomes


def _environed(suites: list[list[Case]]) -> list[tuple[Case, str]]:
    """Every suite run, inside the global environments' ``SetUp`` and ``TearDown``."""
    for environment in REGISTRY.environments:
        environment.SetUp()
    outcomes = [outcome for suite in suites for outcome in _suite(suite)]
    for environment in reversed(REGISTRY.environments):
        environment.TearDown()
    return outcomes


def _suites(cases: list[Case]) -> list[list[Case]]:
    """The tests by suite, each suite where its first test is."""
    grouped: dict[str, list[Case]] = {}
    for case in cases:
        grouped.setdefault(case.suite, []).append(case)
    return list(grouped.values())


def _header(tests: int, suites: int) -> None:
    if REGISTRY.filter != "*":
        _say(f"Note: Google Test filter = {REGISTRY.filter}")
    _say(f"[==========] Running {_count(tests, 'test')} from {_count(suites, 'test suite')}.")
    _say("[----------] Global test environment set-up.")


def _suite(cases: list[Case]) -> list[tuple[Case, str]]:
    """One suite's tests, between its ``SetUpTestSuite`` and its ``TearDownTestSuite``."""
    first = cases[0]
    many = f"{_count(len(cases), 'test')} from {first.suite}"
    _say(f"[----------] {many}")
    start = time.monotonic()
    fixture = first.namespace.get(first.fixture)
    _suite_hook(fixture, "SetUp")
    outcomes = [(case, _one(case)) for case in cases]
    _suite_hook(fixture, "TearDown")
    _say(f"[----------] {many} ({_milliseconds(start)} ms total)\n")
    return outcomes


def _suite_hook(fixture: Any, verb: str) -> None:
    """``SetUpTestSuite`` (or its old name, ``SetUpTestCase``), whichever the fixture overrides."""
    if fixture is None:
        return
    hook = getattr(fixture, f"{verb}TestSuite")
    if hook is getattr(Test, f"{verb}TestSuite"):
        hook = getattr(fixture, f"{verb}TestCase")
    _guarded(hook, f"{verb}TestSuite()")
    flush()


def _one(case: Case) -> str:
    """Run one test, print its result line, and say what it came to: OK, FAILED or SKIPPED."""
    _say(f"[ RUN      ] {case.full}")
    result = STATE.result = Result(case.suite, case.name)
    STATE.traces.clear()
    PARAMETER.value = case.param
    start = time.monotonic()
    with _in_order():
        _phases(case, result)
        for ending in AFTER_TEST:
            ending()
    flush()
    STATE.result = None
    status = "FAILED" if result.failures else "SKIPPED" if result.skipped else "OK"
    where = case.where if status == "FAILED" else ""
    _say(f"{LABELS[status]} {case.full}{where} ({_milliseconds(start)} ms)")
    return status


class _InOrder(io.TextIOBase):
    """Standard output during a test: a failure being reported is printed before what follows.

    The failure's ``<< message`` is streamed onto it within the statement
    that failed; whatever the test writes next comes after that statement,
    so the failure, now complete, is printed first, where GoogleTest prints it.
    """

    def __init__(self, target: TextIO) -> None:
        super().__init__()
        self.target = target

    def write(self, text: str) -> int:
        flush()
        return self.target.write(text)

    def flush(self) -> None:
        self.target.flush()


@contextmanager
def _in_order() -> Iterator[None]:
    before = sys.stdout
    sys.stdout = _InOrder(before)
    try:
        yield
    finally:
        sys.stdout = before


def _phases(case: Case, result: Result) -> None:
    """Make the fixture, ``SetUp``, run the body unless that failed fatally, ``TearDown``."""
    fixture = _guarded(lambda: case.namespace[case.fixture](), "the test fixture's constructor")
    if fixture is None:
        return
    _guarded(fixture.SetUp, "SetUp()")
    if not (result.fatal or result.skipped):
        _guarded(fixture.TestBody, "the test body")
    _guarded(fixture.TearDown, "TearDown()")
    destructor = getattr(fixture, "_destruct", None)
    if destructor is not None:
        _guarded(destructor, "the test fixture's destructor")


def _guarded(call: Callable[[], Any], where: str) -> Any:
    """``call()``, an exception it raises made a failure of the test, as GoogleTest makes it."""
    flush()  # what failed before - in the body, before TearDown - is said before it runs
    try:
        return call()
    except Exception as why:
        kind = type(why).__name__ if internal(why) else "C++ exception"
        file, line = REGISTRY.locate(why) or ("unknown file", None)
        report(file, line, f'{kind} with description "{why}" thrown in {where}.', True)
        return None


def _footer(outcomes: list[tuple[Case, str]], suites: int, total: int) -> None:
    """GoogleTest's summary: how many ran, passed, were skipped, and which failed."""
    _say("[----------] Global test environment tear-down")
    ran = f"{_count(len(outcomes), 'test')} from {_count(suites, 'test suite')} ran."
    _say(f"[==========] {ran} ({total} ms total)")
    passed = sum(status == "OK" for _, status in outcomes)
    _say(f"[  PASSED  ] {_count(passed, 'test')}.")
    for status in ("SKIPPED", "FAILED"):
        _listed(status, [case for case, got in outcomes if got == status])
    failed = sum(status == "FAILED" for _, status in outcomes)
    if failed:
        _say(f"\n{failed:2d} FAILED TEST{'' if failed == 1 else 'S'}")


def _listed(status: str, cases: list[Case]) -> None:
    """The tests that ended ``status``, one a line, if there were any."""
    if not cases:
        return
    _say(f"{LABELS[status]} {_count(len(cases), 'test')}, listed below:")
    for case in cases:
        _say(f"{LABELS[status]} {case.full}{case.where if status == 'FAILED' else ''}")


def _disabled(count: int) -> None:
    if count:
        _say(f"\n  YOU HAVE {count} DISABLED TEST{'' if count == 1 else 'S'}\n")


def run_all(argv: list[str]) -> int:
    """A GoogleTest program without a ``main`` of its own: ``gtest_main``'s, run with ``argv``."""
    InitGoogleTest(None, list(argv))
    return RunAllTests()
