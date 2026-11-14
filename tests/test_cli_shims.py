"""``import ROOT`` and ``import cppyy`` as a script run by ``xrdroot run`` sees them."""

from __future__ import annotations

import importlib
import json
import json.decoder
import sys
from collections.abc import Iterator

import pytest

from xrdroot.cli import shims

#: The names the tests import under an alias, taken out of ``sys.modules`` after each.
ALIASED = ("ROOT", "jsonalias", "cppyy")


@pytest.fixture
def isolated(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """The finders and aliased modules a test installs are taken away again after it."""
    importlib.import_module("xrdroot.pyroot")  # once, as itself, before any alias
    others = [finder for finder in sys.meta_path if not isinstance(finder, shims.Aliases)]
    monkeypatch.setattr(sys, "meta_path", others)
    kept = {name: module for name, module in sys.modules.items() if name.startswith(ALIASED)}
    yield
    for name in [name for name in sys.modules if name.startswith(ALIASED)]:
        del sys.modules[name]
    sys.modules.update(kept)


def test_an_alias_is_the_module_it_names_not_a_second_copy(isolated: None) -> None:
    sys.meta_path.insert(0, shims.Aliases("jsonalias", "json"))
    alias = importlib.import_module("jsonalias")
    decoder = importlib.import_module("jsonalias.decoder")
    assert alias is json
    assert decoder is json.decoder
    assert json.__spec__.name == "json" and json.decoder.__spec__.name == "json.decoder"


def test_an_alias_finds_nothing_for_other_names_or_missing_modules(isolated: None) -> None:
    finder = shims.Aliases("jsonalias", "json")
    assert finder.find_spec("jsonaliased") is None
    assert finder.find_spec("jsonalias.nowhere") is None
    assert finder.find_spec("jsonalias.decoder.deeper") is None  # a module is no package


def test_root_is_pyroot_and_its_modules_are_pyroots_own(isolated: None) -> None:
    root = shims.install_root()
    shims.install_root()  # twice is once
    pythonization = importlib.import_module("ROOT._pythonization")
    assert sys.modules["ROOT"] is root
    assert pythonization is importlib.import_module("xrdroot.pyroot._pythonization")
    assert sys.modules["ROOT.RDF"] is root.RDF
    assert sum(isinstance(finder, shims.Aliases) for finder in sys.meta_path) == 1


def test_cppyy_is_found_last_so_an_installed_cppyy_wins(isolated: None) -> None:
    shims.install_cppyy()
    shims.install_cppyy()
    aliases = [finder for finder in sys.meta_path if isinstance(finder, shims.Aliases)]
    assert [finder.alias for finder in aliases] == ["cppyy"]
    assert sys.meta_path[-1] is aliases[0]


def test_script_mode_has_nothing_to_tell_before_the_namespace_is_imported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "xrdroot.pyroot.core.messages", raising=False)
    assert shims.python_warnings(True) is False
