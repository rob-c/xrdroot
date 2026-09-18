"""Shared fixtures for the ROOT reader and writer.

Everything here is about keeping the suite off the network and off whoever is
running it. The format tests themselves are pure bytes; the handful that read
a tree over a URL stand up a :class:`~xrdclient.testing.FakeServer` of their own.
"""

from __future__ import annotations

import pytest
from xrdclient.config import Config


@pytest.fixture(autouse=True)
def _no_dotfile(tmp_path_factory, monkeypatch):
    """Keep whoever is running the tests out of them.

    ``Config.from_file`` reads ``~/.xrdrc`` when nothing says otherwise, so a
    developer with one would get different results from CI. Pointing
    ``$XRD_CONFIG`` at an empty file is the same as having no file at all.
    """
    empty = tmp_path_factory.mktemp("dotfile") / "config.ini"
    empty.write_text("")
    monkeypatch.setenv("XRD_CONFIG", str(empty))


@pytest.fixture(autouse=True)
def _a_short_data_stream_probe(monkeypatch):
    """Do not spend the suite's time learning what the fake server is.

    A file asks its server once per connection whether it serves a request
    that arrived on a data path, and :class:`~xrdclient.testing.FakeServer` is the
    standard push-only kind that never will - so every connection would sit
    out a whole ``data_stream_timeout`` to be told what this suite already
    knows. A quarter of a second is still an age on loopback.
    """
    monkeypatch.setenv("XRD_SUBSTREAMTIMEOUT", "0.25")


@pytest.fixture(autouse=True)
def _no_pooled_connections():
    """Never let one test's connection be handed to the next.

    Every :class:`~xrdclient.testing.FakeServer` gets an ephemeral port, and the
    kernel hands those out again: a connection left in the pool by a test
    whose server has since stopped would match a later test's server by
    address and be reused, dead.
    """
    from xrdclient.session import SESSIONS

    SESSIONS.clear()
    yield
    SESSIONS.clear()


@pytest.fixture
def config() -> Config:
    """A config that never reaches the network or the local filesystem."""
    return Config(
        username="tester", auth_order=("host",), require_tls=False, data_streams=0
    )
