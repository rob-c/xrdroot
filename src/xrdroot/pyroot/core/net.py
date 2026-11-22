"""``TServerSocket`` and the address it is bound to: a TCP port a program opens for itself.

ROOT's socket classes are a thin layer over the C library's, and so are these:
:class:`TServerSocket` binds and listens, :class:`TInetAddress` is what
``GetLocalInetAddress`` answers with. A port of ``0`` asks the system for a free
one; ROOT instead searches upward from 5000 and records the one it found, which
is the same free port by another rule - nothing a program can tell apart, since
both give a port it can connect to.
"""

from __future__ import annotations

import socket as _socket
from typing import Any

from ...errors import UnsupportedFeatureError

__all__ = ["TServerSocket", "TInetAddress", "ESocketBindOption"]


class ESocketBindOption:
    """``ESocketBindOption``: which interfaces a server socket is bound to."""

    kInaddrAny = 0
    kInaddrLoopback = 1


class TInetAddress:
    """``TInetAddress``: the address a socket is bound to or connected to."""

    __slots__ = ("_address", "_port", "_name", "_valid")

    def __init__(
        self, address: str = "0.0.0.0", port: int = 0, name: str = "", valid: bool = False
    ) -> None:
        self._address = str(address)
        self._port = int(port)
        self._name = str(name)
        self._valid = bool(valid)

    def GetHostAddress(self) -> str:
        """The number the address is, as ``inet_ntoa`` writes it - ``127.0.0.1``."""
        return self._address

    def GetHostName(self) -> str:
        """The name the address has: ``localhost`` for a loopback one, else this host's."""
        return self._name

    def GetPort(self) -> int:
        return self._port

    def IsValid(self) -> bool:
        """Is this an address a socket gave, rather than the empty one a class starts with?"""
        return self._valid

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TInetAddress):
            return NotImplemented
        return (self._address, self._port) == (other._address, other._port)

    def __hash__(self) -> int:
        return hash((self._address, self._port))

    def __str__(self) -> str:
        return f"{self._address}:{self._port}"


#: The name a loopback address has, which ROOT answers for one bound to ``127.0.0.1``.
LOOPBACK_NAME = "localhost"

#: What a local socket answers for the address it has none of - measured on ROOT 6.40.
EMPTY_ADDRESS = "0.0.0.0"
UNKNOWN_HOST = "UnknownHost"
LOCAL_PORT = -1


class TServerSocket:
    """``TServerSocket``: a TCP port this program holds open, and the address it is bound to.

    The arguments are ROOT's: the port (``0`` for a free one, negative for no bind at all),
    whether the address may be reused by the next program to want it, the listen backlog, the
    window size - which the standard library has no call for and is kept - and the bind
    option, which says whether every interface is bound or the loopback alone.
    """

    #: ``TServerSocket::kDefaultBacklog``: how many connections may wait, as ROOT has it.
    kDefaultBacklog = 10

    __slots__ = ("_socket", "_address", "_option", "_windowsize")

    def __init__(
        self,
        port: Any = -1,
        reuse: Any = False,
        backlog: Any = None,
        windowsize: Any = -1,
        option: Any = None,
    ) -> None:
        self._socket: Any = None
        self._address = TInetAddress("")
        self._option = ESocketBindOption.kInaddrAny if option is None else int(option)
        self._windowsize = int(windowsize)
        backlog = self.kDefaultBacklog if backlog is None else int(backlog)
        if isinstance(port, str):
            self._bind_path(port, backlog, windowsize)
            return
        wanted = int(port)
        if wanted < 0:
            return  # ROOT's own: a server socket that binds nothing
        loopback = self._option == ESocketBindOption.kInaddrLoopback
        made = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        if reuse:
            made.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
        made.bind(("127.0.0.1" if loopback else "0.0.0.0", wanted))
        made.listen(backlog)
        self._socket = made
        bound, taken = made.getsockname()[:2]
        name = LOOPBACK_NAME if loopback else _hostname()
        self._address = TInetAddress(bound, taken, name, valid=True)

    def _bind_path(self, path: str, backlog: int, windowsize: Any) -> None:
        """``TServerSocket(const char *path)``: a local socket, which has no address of its own.

        ROOT binds ``AF_UNIX`` for the name and answers the empty address for it - port ``-1``,
        host ``0.0.0.0``, name ``UnknownHost``, and a window size of 0 unless one was asked for -
        which is what the oracle gives for ``TServerSocket("/tmp/x")``. Its peer connects to the
        path, and the path stays on disk until someone unlinks it, as ROOT's does.
        """
        if "/" not in path:
            raise UnsupportedFeatureError(
                f"TServerSocket({path!r}) is neither a port number nor a path to bind: "
                "xrdroot binds a TCP port from a number and a local socket from a path that has "
                'a / in it, so pass one of those - as TServerSocket(0) or TServerSocket("/tmp/x").'
            )
        made = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        made.bind(path)
        made.listen(backlog)
        self._socket = made
        self._address = TInetAddress(EMPTY_ADDRESS, LOCAL_PORT, UNKNOWN_HOST, valid=True)
        self._windowsize = 0 if windowsize is None or int(windowsize) < 0 else int(windowsize)

    def GetLocalInetAddress(self) -> TInetAddress:
        """The address this socket is bound to - the empty one where it binds nothing."""
        return self._address

    def GetLocalPort(self) -> int:
        return self._address.GetPort()

    def IsValid(self) -> bool:
        return self._socket is not None

    def GetOption(self) -> int:
        return self._option

    def GetTcpWindowSize(self) -> int:
        """The window size it was asked for - which the standard library leaves to the system."""
        return self._windowsize

    def Close(self) -> None:
        """``Close``: the port given up. A second call leaves it as it was, as ROOT's does."""
        if self._socket is not None:
            self._socket.close()
            self._socket = None
            self._address = TInetAddress("")

    def _destruct(self) -> None:
        """``~TServerSocket``, which C++ runs as the socket's scope ends."""
        self.Close()


def _hostname() -> str:
    """This machine's name, as an address bound to every interface is known by."""
    import socket

    return str(socket.gethostname())
