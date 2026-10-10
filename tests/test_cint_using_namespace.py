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


def test_a_class_found_in_a_used_namespace_is_remembered_but_an_object_or_a_declaration_is_not(
    monkeypatch,
) -> None:
    """``XYZVector(x, y, z)`` in a loop looks the class up once; what may change is looked up
    each time: an object (``RooFit.Save`` here stands for one), or a class a macro declared."""
    from xrdroot.cint.runtime import root

    math, roofit = SimpleNamespace(XYZVector=int), SimpleNamespace(Save=object())
    proxy = RootProxy()
    monkeypatch.setitem(root.DECLARED, "Declared", int)
    with proxy.bind(SimpleNamespace(Math=math, RooFit=roofit)):
        assert proxy.XYZVector is int and proxy.Declared is int
        first = proxy.Save
        math.XYZVector, roofit.Save = float, object()
        monkeypatch.setitem(root.DECLARED, "Declared", float)
        assert proxy.XYZVector is int  # remembered
        assert proxy.Save is not first and proxy.Declared is float  # looked up again
