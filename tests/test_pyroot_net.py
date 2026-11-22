"""``TServerSocket`` and ``TInetAddress``: a TCP port a program opens for itself.

ROOT's ``TServerSocket(0, ...)`` asks for a free port and says which address it is bound to -
the loopback alone when the bind option says so, every interface otherwise. The numbers this
kit answers with are ROOT 6.40's own, measured against it: ``kDefaultBacklog`` is 10,
``kInaddrLoopback`` 1, a loopback bind's address ``127.0.0.1`` whose host name is
``localhost``, and a default bind's address ``0.0.0.0``.
"""

from __future__ import annotations

import socket

import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import expect
from xrdroot.pyroot.core.net import ESocketBindOption, TInetAddress, TServerSocket


def test_a_server_socket_asked_for_a_free_port_on_the_loopback_holds_one():
    with_server = TServerSocket(0, False, TServerSocket.kDefaultBacklog, -1,
                                ESocketBindOption.kInaddrLoopback)  # fmt: skip
    address = with_server.GetLocalInetAddress()
    expect(
        (address.GetHostAddress(), "127.0.0.1"),
        (address.GetHostName(), "localhost"),
        (bool(address.GetPort() > 0), True),
        (address.GetPort(), with_server.GetLocalPort()),
        (with_server.IsValid(), True),
        (with_server.GetOption(), ESocketBindOption.kInaddrLoopback),
        (with_server.GetTcpWindowSize(), -1),
    )
    with_server.Close()
    expect((with_server.IsValid(), False), (with_server.GetLocalInetAddress().IsValid(), False))


def test_a_server_socket_with_no_bind_option_is_bound_to_every_interface():
    with_server = TServerSocket(0, True)
    try:
        address = with_server.GetLocalInetAddress()
        expect(
            (address.GetHostAddress(), "0.0.0.0"),
            (address.GetHostName(), socket.gethostname()),
            (with_server.GetOption(), ESocketBindOption.kInaddrAny),
        )
    finally:
        with_server.Close()


def test_a_server_socket_of_a_negative_port_binds_nothing_as_roots_does():
    nothing = TServerSocket(-1)
    expect(
        (nothing.IsValid(), False),
        (nothing.GetLocalInetAddress().IsValid(), False),
        (nothing.GetLocalPort(), 0),
    )
    nothing.Close()  # nothing to give up, and ROOT's close of one is quiet too


def test_letting_a_server_socket_go_gives_its_port_back():
    made = TServerSocket(0, False, None, -1, ESocketBindOption.kInaddrLoopback)
    port = made.GetLocalPort()
    made._destruct()
    expect((made.IsValid(), False))
    # the port is free again, which is what a second bind to it proves
    again = TServerSocket(port, True, None, -1, ESocketBindOption.kInaddrLoopback)
    expect((again.GetLocalPort(), port))


def test_an_address_is_the_number_its_port_and_what_it_compares_by():
    one = TInetAddress("127.0.0.1", 5, "localhost")
    same = TInetAddress("127.0.0.1", 5)
    other = TInetAddress("127.0.0.1", 6)
    expect(
        (str(one), "127.0.0.1:5"),
        (bool(one == same), True),
        (bool(one == other), False),
        (hash(one), hash(same)),
        (one == 5, False),
        (TInetAddress().IsValid(), False),
    )


def test_the_namespace_offers_the_socket_class_and_its_bind_option():
    expect(
        (ROOT.TServerSocket, TServerSocket),
        (ROOT.TInetAddress, TInetAddress),
        (ROOT.ESocketBindOption.kInaddrAny, 0),
        (ROOT.ESocketBindOption.kInaddrLoopback, 1),
    )


@pytest.mark.parametrize(
    "option", [ESocketBindOption.kInaddrAny, ESocketBindOption.kInaddrLoopback]
)
def test_a_quiet_close_is_the_only_thing_a_socket_without_a_port_does(option):
    nothing = TServerSocket(-1, False, None, -1, option)
    nothing._destruct()
    expect((nothing.GetOption(), option), (nothing.GetTcpWindowSize(), -1))
