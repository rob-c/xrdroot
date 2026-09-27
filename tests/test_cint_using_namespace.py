"""A macro's unqualified RooFit names: ``using namespace RooFit`` makes ``Save`` RooFit's."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from xrdroot.cint.runtime.root import RootProxy


def test_a_name_root_lacks_is_looked_for_in_the_namespaces_a_macro_uses() -> None:
    """``Save`` is not ROOT's own, but ``RooFit``'s - found there, as the compiler finds it."""
    save = object()
    proxy = RootProxy()
    with proxy.bind(SimpleNamespace(RooFit=SimpleNamespace(Save=save), TH1F=int)):
        assert (proxy.Save, proxy.TH1F) == (save, int)


def test_a_name_no_used_namespace_has_either_is_refused_as_root_refuses_it() -> None:
    """The refusal is the namespace's own, not one made up here."""
    proxy = RootProxy()
    with proxy.bind(SimpleNamespace(RooFit=SimpleNamespace())), pytest.raises(AttributeError):
        proxy.Nowhere  # noqa: B018
