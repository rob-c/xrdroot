"""``::testing::Test`` and what a fixture or a parameterised test is built from.

A ``TEST`` is a class deriving from :class:`Test`, a ``TEST_F`` one deriving
from the fixture, which derives from :class:`Test`; the runner makes one per
test and calls ``SetUp``, ``TestBody`` and ``TearDown`` on it. A
``TEST_P``'s fixture derives from :class:`TestWithParam`, and ``GetParam()``
hands back the value the runner is running it with. The generators of
``INSTANTIATE_TEST_SUITE_P`` - ``Values``, ``Range``, ``Combine`` - are
lists of those values.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

from .state import STATE
from .values import show

__all__ = [
    "Test",
    "WithParamInterface",
    "TestWithParam",
    "Environment",
    "Values",
    "ValuesIn",
    "Range",
    "Bool",
    "Combine",
    "TestParamInfo",
    "PrintToStringParamName",
    "StaticAssertTypeEq",
    "UnitTest",
    "PARAMETER",
]


@dataclass
class Parameter:
    """The value of the parameterised test being run, for ``GetParam()``."""

    value: Any = None


#: What ``GetParam()`` gives: set by the runner before it makes each test's fixture.
PARAMETER = Parameter()


def _failures() -> int:
    return 0 if STATE.result is None else STATE.result.failures


class Test:
    """``::testing::Test``: a fixture's base, whose hooks do nothing until one overrides them."""

    def __init__(self) -> None:
        pass

    def SetUp(self) -> None:
        pass

    def TearDown(self) -> None:
        pass

    @staticmethod
    def SetUpTestSuite() -> None:
        pass

    @staticmethod
    def TearDownTestSuite() -> None:
        pass

    @staticmethod
    def SetUpTestCase() -> None:
        pass

    @staticmethod
    def TearDownTestCase() -> None:
        pass

    @staticmethod
    def HasFailure() -> bool:
        return _failures() > 0

    @staticmethod
    def HasFatalFailure() -> bool:
        return STATE.result is not None and STATE.result.fatal

    @staticmethod
    def HasNonfatalFailure() -> bool:
        return Test.HasFailure() and not Test.HasFatalFailure()

    @staticmethod
    def IsSkipped() -> bool:
        return STATE.result is not None and STATE.result.skipped

    @staticmethod
    def RecordProperty(key: Any, value: Any) -> None:
        pass


class WithParamInterface:
    """``::testing::WithParamInterface<T>``: what gives a fixture ``GetParam()``."""

    def __init__(self) -> None:
        pass

    def __class_getitem__(cls, kind: Any) -> type:
        return cls

    @staticmethod
    def GetParam() -> Any:
        return PARAMETER.value


class TestWithParam(Test, WithParamInterface):
    """``::testing::TestWithParam<T>``: a fixture of tests run once for each of a list of values."""


class Environment:
    """``::testing::Environment``: set up before every test of the run, torn down after."""

    def SetUp(self) -> None:
        pass

    def TearDown(self) -> None:
        pass


def Values(*values: Any) -> list[Any]:
    """``::testing::Values(a, b, c)``."""
    return list(values)


def ValuesIn(values: Any) -> list[Any]:
    """``::testing::ValuesIn(container)``."""
    return list(values)


def Range(start: Any, end: Any, step: Any = 1) -> list[Any]:
    """``::testing::Range(start, end, step)``: ``start``, ``start + step``, ... short of ``end``."""
    values = []
    value = start
    while value < end:
        values.append(value)
        value = value + step
    return values


def Bool() -> list[bool]:
    """``::testing::Bool()``: ``false``, then ``true``."""
    return [False, True]


def Combine(*generators: Any) -> list[tuple[Any, ...]]:
    """``::testing::Combine(g1, g2)``: one value from each in every way, the last fastest."""
    return list(itertools.product(*generators))


@dataclass
class TestParamInfo:
    """``::testing::TestParamInfo<T>``: what a name generator names a test from."""

    param: Any
    index: int

    def __class_getitem__(cls, kind: Any) -> type:
        return cls


def PrintToStringParamName() -> Any:
    """``::testing::PrintToStringParamName()``: each test named by its value, printed."""
    return lambda info: show(info.param)


class _TypeEquality:
    """``::testing::StaticAssertTypeEq<T1, T2>()``: a check the compiler made, so it holds."""

    def __getitem__(self, kinds: Any) -> Any:
        return lambda: True


StaticAssertTypeEq = _TypeEquality()


@dataclass
class TestInfo:
    """``::testing::TestInfo``: the names of the test running."""

    suite: str
    test: str

    def name(self) -> str:
        return self.test

    def test_suite_name(self) -> str:
        return self.suite

    def test_case_name(self) -> str:
        return self.suite


class UnitTest:
    """``::testing::UnitTest::GetInstance()``: asked which test is running."""

    @staticmethod
    def GetInstance() -> UnitTest:
        return UnitTest()

    def current_test_info(self) -> TestInfo | None:
        result = STATE.result
        return None if result is None else TestInfo(result.suite, result.name)
