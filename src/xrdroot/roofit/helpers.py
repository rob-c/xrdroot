"""``RooHelpers``: the message-level and message-capture guards RooFit's own tests use.

``LocalChangeMsgLevel`` sets the level messages are printed from, on every
stream, for as long as it lives - a test silences an error it expects with
one - and ``HijackMessageStream`` takes the messages of a level and topics
into a string for as long as it lives, so a test can look at what RooFit
said. Each puts the message service back as it was when it goes: when
:meth:`restore` is called, or when the object is collected, which for a
local of a translated function is when the function returns, as a C++
destructor runs at the end of its scope.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from . import messages
from .messages import service

__all__ = ["LocalChangeMsgLevel", "HijackMessageStream", "RooHelpers"]


class _Guard:
    """The service's level and streams as they were, put back once."""

    def __init__(self) -> None:
        held = service()
        self._kill_below = held.globalKillBelow()
        self._streams = [replace(stream) for stream in held._streams]
        self._restored = False

    def restore(self) -> None:
        """The service as it was before; only the first call does anything."""
        if self._restored:
            return
        self._restored = True
        held = service()
        held.setGlobalKillBelow(self._kill_below)
        held._streams = self._streams

    def __del__(self) -> None:
        self.restore()

    def __enter__(self) -> _Guard:
        return self

    def __exit__(self, *exc: object) -> None:
        self.restore()


class LocalChangeMsgLevel(_Guard):
    """``LocalChangeMsgLevel(level=DEBUG, extraTopics=0, removeTopics=0, override=true)``:
    messages printed from ``level`` up, ``removeTopics`` taken off every stream, and a stream of
    ``extraTopics`` added."""

    def __init__(self, level: Any = messages.DEBUG, extraTopics: Any = 0, removeTopics: Any = 0,
                 overrideExternalLevel: Any = True) -> None:  # fmt: skip
        super().__init__()
        held = service()
        if overrideExternalLevel:
            held.setGlobalKillBelow(int(level))
        for stream in held._streams:
            if overrideExternalLevel:
                stream.minLevel = int(level)
            stream.removeTopic(int(removeTopics))
            stream.active = True
        if int(extraTopics):
            held.getStream(held.addStream(int(level))).topic = int(extraTopics)


class HijackMessageStream(_Guard):
    """``HijackMessageStream(level, topics[, objectName])``: the messages of ``level`` and up on
    ``topics`` - about ``objectName`` only, if given - written to a string, ``str()``, rather
    than printed."""

    def __init__(self, level: Any, topics: Any, objectName: Any = None) -> None:
        from ..cppstd.streams import ostringstream

        super().__init__()
        held = service()
        self._text = ostringstream()
        if held.globalKillBelow() > int(level):
            held.setGlobalKillBelow(int(level))
        taken = [stream for stream in held._streams
                 if stream.match(int(level), int(topics), None)]  # fmt: skip
        for stream in taken:
            stream.active = False
        named = {"ObjectName": str(objectName)} if objectName else {}
        held.addStream(int(level), Topic=int(topics), OutputStream=self._text, **named)
        for stream in taken:
            held.addStream(stream.minLevel, Topic=stream.topic, OutputStream=self._text,
                           ObjectName=stream.objectName)

    def str(self) -> str:
        return str(self._text.str())

    def stream(self) -> Any:
        return self._text


class _Namespace:
    """``RooHelpers``: the guards, by their C++ path."""

    LocalChangeMsgLevel = LocalChangeMsgLevel
    HijackMessageStream = HijackMessageStream

    def __repr__(self) -> str:
        return "<namespace RooHelpers>"


#: ``ROOT.RooHelpers``.
RooHelpers = _Namespace()
