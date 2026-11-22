"""The transport surface this package is written against, named in one place.

``xrdroot`` is a ROOT reader, not an XRootD client: it reaches the bytes of a
file through whatever :mod:`xrdclient` opens for it, and it uses a handful of
names to do so - :func:`~xrdclient.url.parse` and its
:class:`~xrdclient.url.XRootDURL`, :class:`~xrdclient.errors.XRootDError` as the
class an unreachable URL is raised as, :func:`~xrdclient.io.open_url` and
:class:`~xrdclient.config.Config`, plus the fake server the suite reads files
over. Those are asserted here so that a version bump of the transport fails in
one place, in a sentence about the transport, rather than inside a test about a
histogram.

The declared floor is ``xrdclient>=0.3.2`` rather than the newest 0.4.0 for one
reason: 0.4.0 requires Python 3.11 and this package still supports 3.10, where
pip resolves 0.3.2 instead. Both are held to this contract, so the two
behaviours below are what either version must keep.
"""

from __future__ import annotations

import inspect
import pickle

import pytest
from xrdclient.config import Config
from xrdclient.errors import XRootDError
from xrdclient.io import open_url
from xrdclient.url import XRootDURL, parse

from xrdroot.errors import FormatError, ROOTError, UnsupportedFeatureError


def test_a_url_is_split_into_the_parts_xrdroot_reaches_for():
    """``file.py`` decides local-or-remote from ``is_local`` and hands the rest on."""
    url = parse("root://user:pw@eos.example.ch:1095//store/f.root?sig=abc")
    assert (url.scheme, url.host, url.port, url.path) == (
        "root", "eos.example.ch", 1095, "/store/f.root",
    )  # fmt: skip
    assert (url.username, url.password) == ("user", "pw")
    assert url.query == {"sig": "abc"}
    assert (url.is_root, url.is_http, url.is_local) == (True, False, False)


def test_xrdd_s_double_slash_names_an_absolute_path():
    """ROOT's URLs are written ``root://host//abs`` and the second slash is the root."""
    assert parse("root://host:1094//abs/path.root").path == "/abs/path.root"
    assert parse("root://host/Abs/Path.root").path == "/Abs/Path.root"


def test_a_plain_path_is_local_and_keeps_its_slashes():
    """A bare path must not be taken for a URL: it is read with ``open()``."""
    local = parse("/tmp/dir/f.root")
    assert (local.is_local, local.path) == (True, "/tmp/dir/f.root")


def test_the_default_port_and_the_written_url_round_trip():
    """``str`` is the URL again, with the port spelled out, as a log line shows it."""
    assert str(parse("root://host//a/f.root")) == "root://host:1094//a/f.root"


def test_parsing_a_url_that_is_already_parsed_keeps_it():
    """A file reopens itself in a forked worker: ``parse`` is handed its own result."""
    url = parse("root://host//a/f.root")
    again = parse(url)
    assert isinstance(again, XRootDURL) and again == url


def test_every_error_xrdroot_raises_is_its_own_and_not_the_transport_s():
    """One ``except`` covers every file this package misreads, and the transport's errors are
    the transport's: ``ROOTError`` derives from no ``xrdclient`` class, so that eight hundred
    commands that read a local file never import the protocol behind a URL."""
    for err in (ROOTError("why"), FormatError("why"), UnsupportedFeatureError("why")):
        assert isinstance(err, ROOTError) and isinstance(err, Exception)
        assert not isinstance(err, XRootDError)
        assert str(err) == "why" and err.args == ("why",)


def test_an_error_crosses_to_a_worker_unchanged():
    """A distributed reader sends the failure back as bytes, not as a traceback."""
    back = pickle.loads(pickle.dumps(FormatError("the file is truncated")))
    assert (type(back), str(back)) == (FormatError, "the file is truncated")


def test_open_url_takes_a_url_a_mode_and_a_config_and_opens_through_them():
    """``file.py``, the writer and the updater all call it in exactly this shape."""
    parameters = inspect.signature(open_url).parameters
    assert list(parameters)[:2] == ["url", "mode"]
    assert {"config", "buffering"} <= set(parameters)


def test_a_config_carries_what_the_suite_and_a_reader_ask_it_for():
    """``tests/conftest.py`` builds one by hand; ``file.py`` passes it on again."""
    config = Config(username="tester", auth_order=("host",), require_tls=False, data_streams=0)
    assert (config.username, config.auth_order) == ("tester", ("host",))
    assert (config.require_tls, config.data_streams) == (False, 0)


def test_a_config_reads_the_data_stream_timeout_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The suite cuts this to a quarter second; it is what a short read waits for."""
    monkeypatch.setenv("XRD_SUBSTREAMTIMEOUT", "0.25")
    assert Config().data_stream_timeout == 0.25


def test_the_suite_can_stand_a_server_up_in_its_own_process():
    """Every remote test reads over ``FakeServer``; nothing in the suite uses a real one."""
    from xrdclient.testing import FakeServer

    with FakeServer(files={}) as server:
        assert isinstance(server.url, XRootDURL) and server.url.is_root
