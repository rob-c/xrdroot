"""Shared fixtures for the ROOT reader and writer.

Everything here is about keeping the suite off the network and off whoever is
running it. The format tests themselves are pure bytes; the handful that read
a tree over a URL stand up a :class:`~xrdclient.testing.FakeServer` of their own.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
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
    address and be reused, dead. A process that never loaded the transport has
    no pool to clear, and is left without importing it.
    """
    import sys

    session = sys.modules.get("xrdclient.session")
    if session is None:
        yield
        return
    session.SESSIONS.clear()
    yield
    session.SESSIONS.clear()


#: ROOT's globals a macro or a script may assign to, which live on the shared
#: ``xrdroot.pyroot`` namespace: ``gRandom = new TRandom3(0)`` in one test must
#: not decide what another draws.
ROOT_GLOBALS = ("gRandom", "gStyle", "gErrorIgnoreLevel", "gDebug", "gBenchmark")


@pytest.fixture(autouse=True)
def _root_globals_restored():
    """Put back whatever of ROOT's globals a test reassigned, as it found them."""
    import sys

    namespace = sys.modules.get("xrdroot.pyroot")
    if namespace is None:
        yield
        return
    saved = {name: namespace.__dict__[name] for name in ROOT_GLOBALS if name in namespace.__dict__}
    yield
    for name in ROOT_GLOBALS:
        if name in saved:
            namespace.__dict__[name] = saved[name]
        else:
            namespace.__dict__.pop(name, None)


@pytest.fixture
def config() -> Config:
    """A config that never reaches the network or the local filesystem."""
    from xrdclient.config import Config

    return Config(
        username="tester", auth_order=("host",), require_tls=False, data_streams=0
    )


@pytest.fixture(autouse=True)
def _fresh_statistics_defaults():
    """Keep one test's minimizer defaults and RooStats switches out of the next.

    ``MinimizerOptions``' defaults, RooStats' configuration and the
    asymptotic calculator's print level are process-wide, as ROOT's are, and
    a test that sets one - or asks for the algorithm before any minimizer was
    chosen, which ROOT then forgets - would change what later tests see. Only
    what the process has already loaded is saved and put back: a test that
    never touched the statistical engines has no state here to restore, and is
    left without importing them.
    """
    import sys

    defaults = sys.modules.get("xrdroot.fit.defaults")
    settings = sys.modules.get("xrdroot.roostats.config")
    asimov = sys.modules.get("xrdroot.roostats.asimov")
    saved = dict(defaults.DEFAULTS) if defaults is not None else {}
    switches = dict(settings.CONFIG.__dict__) if settings is not None else {}
    level = list(asimov._LEVEL) if asimov is not None else []
    yield
    if defaults is not None:
        defaults.DEFAULTS.clear()
        defaults.DEFAULTS.update(saved)
    if settings is not None:
        settings.CONFIG.__dict__.update(switches)
    if asimov is not None:
        asimov._LEVEL[:] = level


@pytest.fixture(autouse=True)
def _fresh_roofit_message_service():
    """Each test starts with fresh reporting streams and no preceding progress dots.

    Modules keep direct references to SERVICE, so retain the singleton itself.
    Reset builds new stream lists, leaving the saved state untouched until it
    is restored after the test. A process that never loaded RooFit's messages
    has no service to reset, and is left without importing them.
    """
    import sys

    messages = sys.modules.get("xrdroot.roofit.messages")
    if messages is None:
        yield
        return
    service = messages.SERVICE
    saved = dict(service.__dict__)
    service.reset()
    try:
        yield
    finally:
        service.__dict__.clear()
        service.__dict__.update(saved)
