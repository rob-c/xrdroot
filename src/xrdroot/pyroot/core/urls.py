"""``TUrl``: a URL taken apart into ROOT's pieces - protocol, host, port, file, options, anchor.

ROOT reads a word with no protocol as a host to reach over ``http``, unless
told it names a file (``TUrl(url, true)``) or it starts with ``/``, which is
a local file. ``GetUrl`` puts the pieces back, leaving out a port that is
the protocol's own, and ``Compare`` orders URLs by that text, so a list of
them sorts.
"""

from __future__ import annotations

from typing import Any

from .objects import TObject

__all__ = ["TUrl"]

#: Each protocol's own port, which ``GetUrl`` leaves out.
PORTS = {"http": 80, "https": 443, "root": 1094, "roots": 1094, "xroot": 1094, "file": 0}


def _cut(text: str, mark: str) -> tuple[str, str]:
    """``text`` before ``mark`` and what follows it, or ``text`` and nothing."""
    head, _, tail = text.partition(mark)
    return head, tail


class TUrl(TObject):
    """``TUrl(url, defaultIsFile=false)``."""

    def __init__(self, url: Any = "", default_is_file: Any = False) -> None:
        super().__init__()
        self.SetUrl(url, default_is_file)

    def SetUrl(self, url: Any, default_is_file: Any = False) -> None:
        """Take ``url`` apart; what it leaves out is the default."""
        text = str(url)
        rest, self._anchor = _cut(text, "#")
        rest, self._options = _cut(rest, "?")
        protocol, sep, after = rest.partition("://")
        if not sep:
            local = default_is_file or rest.startswith("/")
            self._protocol = "file" if local else "http"
            self._host, self._file = ("", rest) if local else (rest, "")
        else:
            self._protocol = protocol
            self._host, self._file = self._split_host(after)
        self._port = PORTS.get(self._protocol, 0)
        self._host, port = _cut(self._host, ":")
        if port.isdigit():
            self._port = int(port)

    def _split_host(self, after: str) -> tuple[str, str]:
        if self._protocol == "file":
            return "", after
        host, _, path = after.partition("/")
        return host, path

    def GetProtocol(self) -> str:
        return self._protocol

    def GetHost(self) -> str:
        return self._host

    def GetHostFQDN(self) -> str:
        return self._host

    def GetPort(self) -> int:
        return self._port

    def GetFile(self) -> str:
        return self._file

    def GetOptions(self) -> str:
        return self._options

    def GetAnchor(self) -> str:
        return self._anchor

    def IsValid(self) -> bool:
        return bool(self._protocol)

    def GetFileAndOptions(self) -> str:
        """The file, with its ``?options`` and ``#anchor``."""
        options = f"?{self._options}" if self._options else ""
        anchor = f"#{self._anchor}" if self._anchor else ""
        return self._file + options + anchor

    def GetUrl(self) -> str:
        """The URL put back together, without a port that is the protocol's own."""
        if self._protocol == "file":
            return f"file://{self.GetFileAndOptions()}"
        port = "" if self._port == PORTS.get(self._protocol) else f":{self._port}"
        return f"{self._protocol}://{self._host}{port}/{self.GetFileAndOptions()}"

    def GetName(self) -> str:
        return self.GetUrl()

    def Compare(self, obj: Any) -> int:
        """``Compare``: by the URLs' text, as ROOT's sorts them."""
        mine, theirs = self.GetUrl(), obj.GetUrl()
        return int(mine > theirs) - int(mine < theirs)
