"""``RooMsgService``: which of RooFit's messages are printed, and how they start.

Every message RooFit prints has a level - ``INFO``, ``WARNING`` - and a
topic - ``Fitting``, ``Plotting`` - and goes to the first of the service's
streams that takes both, prefixed with that stream's number: ``[#1]
INFO:Fitting -- ...``. A tutorial's output is these lines as much as its
numbers, so the streams are ROOT's, set up as ``RooMsgService::reset``
sets them up, and a macro that adds a stream or takes a topic off one
changes what is printed here exactly as it would in ROOT.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from . import cout

__all__ = [
    "LEVELS",
    "TOPICS",
    "DEBUG",
    "INFO",
    "PROGRESS",
    "WARNING",
    "ERROR",
    "FATAL",
    "RooMsgService",
    "StreamConfig",
    "log",
    "service",
]

#: ``RooFit::MsgLevel``.
DEBUG, INFO, PROGRESS, WARNING, ERROR, FATAL = range(6)
#: Each level's name, as the prefix of a message says it.
LEVELS = {
    DEBUG: "DEBUG",
    INFO: "INFO",
    PROGRESS: "PROGRESS",
    WARNING: "WARNING",
    ERROR: "ERROR",
    FATAL: "FATAL",
}
#: ``RooFit::MsgTopic``: each topic's bit, in the order ``Print`` lists them.
TOPICS = {
    "Generation": 1,
    "Minimization": 2,
    "Plotting": 4,
    "Fitting": 8,
    "Integration": 16,
    "LinkStateMgmt": 32,
    "Eval": 64,
    "Caching": 128,
    "Optimization": 256,
    "ObjectHandling": 512,
    "InputArguments": 1024,
    "Tracing": 2048,
    "Contents": 4096,
    "DataHandling": 8192,
    "NumericIntegration": 16384,
    "FastEvaluations": 1 << 15,
    "HistFactory": 1 << 16,
    "IO": 1 << 17,
}
#: The topic a stream takes when none is named: every one.
ANY = 0xFFFFF
#: The topics ``reset`` gives the second stream, which prints ``INFO``.
INFO_TOPICS = sum(
    TOPICS[name]
    for name in (
        "Eval",
        "Plotting",
        "Fitting",
        "Minimization",
        "Caching",
        "ObjectHandling",
        "NumericIntegration",
        "InputArguments",
        "DataHandling",
    )
)


@dataclass
class StreamConfig:
    """One reporting stream: from what level, on which topics, about what, and to where."""

    minLevel: int
    topic: int
    active: bool = True
    objectName: str = ""
    className: str = ""
    baseClassName: str = ""
    tagName: str = ""
    prefix: bool = True
    out: Any = field(default=None, repr=False)

    def addTopic(self, topic: int) -> None:
        self.topic |= int(topic)

    def removeTopic(self, topic: int) -> None:
        self.topic &= ~int(topic)

    @property
    def universal(self) -> bool:
        """Whether the stream takes messages about any object at all."""
        return not (self.objectName or self.className or self.baseClassName or self.tagName)

    def match(self, level: int, topic: int, obj: Any) -> bool:
        """``StreamConfig::match``: whether a message of ``level`` on ``topic`` about ``obj`` goes
        here."""
        if not self.active or level < self.minLevel or not self.topic & topic:
            return False
        if self.universal:
            return True
        return obj is not None and self._about(obj)

    def _about(self, obj: Any) -> bool:
        checks = (
            (self.objectName, lambda: obj.GetName() == self.objectName),
            (self.className, lambda: obj.ClassName() == self.className),
            (self.baseClassName, lambda: obj.InheritsFrom(self.baseClassName)),
            (self.tagName, lambda: obj.getAttribute(self.tagName)),
        )
        return all(check() for wanted, check in checks if wanted)

    def stream(self) -> Any:
        return self.out if self.out is not None else cout.STREAM


class RooMsgService:
    """The message service: its streams, in order, the first that takes a message printing it."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """The three streams RooFit starts with, and nothing silenced."""
        self._streams: list[StreamConfig] = []
        self._saved: list[list[StreamConfig]] = []
        self._kill_below = DEBUG
        self._silent = False
        self._last_level = DEBUG
        self.errors = 0
        self.addStream(PROGRESS, Topic=TOPICS["HistFactory"] - 1)
        self.addStream(INFO, Topic=INFO_TOPICS)
        self.addStream(INFO, Topic=TOPICS["HistFactory"])

    @staticmethod
    def instance() -> RooMsgService:
        return SERVICE

    # -- streams ------------------------------------------------------------------

    def addStream(self, level: int, *commands: Any, **options: Any) -> int:
        """A new stream from ``level`` up, with ``Topic``, ``ClassName``... as ``RooCmdArg``s or
        keywords."""
        settings = {command.name: command.value() for command in commands if command.name}
        settings.update(options)
        stream = StreamConfig(int(level), int(settings.get("Topic", ANY)))
        for key in ("ObjectName", "ClassName", "BaseClassName"):
            setattr(stream, key[0].lower() + key[1:], str(settings.get(key, "")))
        stream.tagName = str(settings.get("LabelName", ""))
        stream.prefix = bool(settings.get("Prefix", True))
        stream.out = self._output(settings)
        self._streams.append(stream)
        return len(self._streams) - 1

    @staticmethod
    def _output(settings: dict[str, Any]) -> Any:
        if "OutputStream" in settings:
            return settings["OutputStream"]
        name = settings.get("OutputFile")
        return open(str(name), "w") if name else None

    def deleteStream(self, index: int) -> None:
        del self._streams[int(index)]

    def getStream(self, index: int) -> StreamConfig:
        return self._streams[int(index)]

    def numStreams(self) -> int:
        return len(self._streams)

    def setStreamStatus(self, index: int, active: bool) -> None:
        if not 0 <= int(index) < len(self._streams):
            cout.line(f"RooMsgService::setStreamStatus() ERROR: invalid stream ID {index}")
            return
        self._streams[int(index)].active = bool(active)

    def getStreamStatus(self, index: int) -> bool:
        if not 0 <= int(index) < len(self._streams):
            cout.line(f"RooMsgService::getStreamStatus() ERROR: invalid stream ID {index}")
            return False
        return self._streams[int(index)].active

    def saveState(self) -> None:
        self._saved.append([replace(stream) for stream in self._streams])

    def restoreState(self) -> None:
        self._streams = self._saved.pop()

    def setGlobalKillBelow(self, level: int) -> None:
        self._kill_below = int(level)

    def globalKillBelow(self) -> int:
        return self._kill_below

    def setSilentMode(self, flag: bool) -> None:
        self._silent = bool(flag)

    def silentMode(self) -> bool:
        return self._silent

    def errorCount(self) -> int:
        return self.errors

    def clearErrorCount(self) -> None:
        self.errors = 0

    # -- what is printed ----------------------------------------------------------

    def activeStream(self, obj: Any, topic: int, level: int) -> int:
        """The number of the first stream that takes this message, or -1 for none."""
        if level < self._kill_below:
            return -1
        return next(
            (i for i, stream in enumerate(self._streams) if stream.match(level, topic, obj)), -1
        )

    def isActive(self, obj: Any, topic: int, level: int) -> bool:
        return self.activeStream(obj, topic, level) >= 0

    def log(self, obj: Any, level: int, topic: str, text: str) -> None:
        """Print ``text`` as a message of ``level`` on ``topic``, prefixed as its stream says."""
        if level >= ERROR:
            self.errors += 1
        found = self.activeStream(obj, TOPICS[topic], level)
        if found < 0:
            return
        stream = self._streams[found]
        out = stream.stream()
        if self._last_level == PROGRESS and level != PROGRESS:
            out.write("\n")
        self._last_level = level
        prefix = f"[#{found}] {LEVELS[level]}:{topic} -- " if stream.prefix else ""
        out.write(prefix + text + "\n")
        out.flush()

    def Print(self, options: str = "") -> None:
        every = "v" in str(options).lower()
        cout.line("All Message streams" if every else "Active Message streams")
        for index, stream in enumerate(self._streams):
            if stream.active or every:
                cout.line(self._described(index, stream, every))

    @staticmethod
    def _described(index: int, stream: StreamConfig, every: bool) -> str:
        if stream.topic == ANY:
            topics = " Any "
        else:
            topics = "".join(
                f"{name} " for name, bit in TOPICS.items() if bit & stream.topic and name != "IO"
            )
        text = f"[{index}] MinLevel = {LEVELS[stream.minLevel]} Topic = {topics}"
        for label, value in (
            ("ObjectName", stream.objectName),
            ("ClassName", stream.className),
            ("BaseClassName", stream.baseClassName),
            ("TagLabel", stream.tagName),
        ):
            if value:
                text += f" {label} = {value}"
        if every and not stream.active:
            text += " (NOT ACTIVE)"
        return text


#: The one message service, ``RooMsgService::instance()``.
SERVICE = RooMsgService()


def service() -> RooMsgService:
    return SERVICE


def log_plain(obj: Any, level: int, topic: str, text: str) -> None:
    """``ooccoutW(obj, topic) << text``: to the stream that takes it, without the prefix."""
    found = SERVICE.activeStream(obj, TOPICS[topic], level)
    if found >= 0:
        out = SERVICE.getStream(found).stream()
        out.write(text)
        out.flush()


def log(obj: Any, level: int, topic: str, text: str) -> None:
    """``oocoutI(obj, topic) << text``, and its kin by ``level``."""
    SERVICE.log(obj, level, topic, text)
