"""GoogleTest for translated unit tests: the ``testing`` namespace.

Unit tests are GoogleTest programs. Their macros are defined by
:mod:`xrdroot.pycxx.gtest_macros` in terms of ``::testing::internal``, which
is here, with the parts of ``::testing`` the tests name themselves -
``Test``, ``TestWithParam``, ``Values``, ``HasSubstr``. A translation finds
them through its host's ``testing``
(:class:`~xrdroot.pycxx.runtime.namespace.HostNamespace`). A host whose
library has test support of its own adds its checks to ``testing::internal``
with :func:`provide`, and what ends with each test with
:data:`~.runner.AFTER_TEST`.

What GoogleTest has and this does not is refused by name where a test uses
it, with :func:`~.exceptions.refusal`'s exception, which fails that test alone.
"""

from __future__ import annotations

import io
import sys
from typing import Any

from . import checks, fixtures, matchers, runner, state, values
from .exceptions import refusal
from .runner import REGISTRY, registered, reset, run_all

__all__ = ["testing", "Namespace", "REGISTRY", "registered", "reset", "run_all", "provide"]


class Namespace:
    """A C++ namespace of GoogleTest's, which refuses by name what it does not have."""

    def __init__(self, name: str, members: dict[str, Any]) -> None:
        self._name = name
        self.__dict__.update(members)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        raise refusal(f"{self._name}::{name} is a part of GoogleTest that xrdroot does not provide")


#: What ``testing::internal::CaptureStdout()`` and its kin put in place, by stream.
CAPTURED: dict[str, tuple[Any, io.StringIO]] = {}


def _capture(stream: str) -> None:
    """``CaptureStdout()``: what is written to the stream from now on is kept, not printed."""
    taken = io.StringIO()
    CAPTURED[stream] = (getattr(sys, stream), taken)
    setattr(sys, stream, taken)


def _captured(stream: str) -> str:
    """``GetCapturedStdout()``: what was kept, the stream put back as it was."""
    before, taken = CAPTURED.pop(stream)
    setattr(sys, stream, before)
    return taken.getvalue()


def _members(module: Any) -> dict[str, Any]:
    return {name: getattr(module, name) for name in module.__all__}


internal = Namespace(
    "testing::internal",
    {
        **_members(checks),
        "Register": runner.Register,
        "RegisterPattern": runner.RegisterPattern,
        "Instantiate": runner.Instantiate,
        "RunAllTests": runner.RunAllTests,
        "Pending": state.pending,
        "Trace": state.trace,
        "MakeMatcher": matchers.MakeMatcher,
        "CaptureStdout": lambda: _capture("stdout"),
        "CaptureStderr": lambda: _capture("stderr"),
        "GetCapturedStdout": lambda: _captured("stdout"),
        "GetCapturedStderr": lambda: _captured("stderr"),
    },
)

#: ``::testing``: the fixtures, the generators, the matchers, and ``InitGoogleTest``.
testing = Namespace(
    "testing",
    {
        **{name: getattr(fixtures, name) for name in fixtures.__all__ if name != "PARAMETER"},
        **{name: getattr(matchers, name) for name in matchers.__all__[2:]},
        "PrintToString": values.show,
        "internal": internal,
        "InitGoogleTest": runner.InitGoogleTest,
        "AddGlobalTestEnvironment": runner.AddGlobalTestEnvironment,
    },
)


def provide(members: dict[str, Any]) -> None:
    """Add a host's test support to ``testing::internal``: what its own test macros call."""
    internal.__dict__.update(members)
