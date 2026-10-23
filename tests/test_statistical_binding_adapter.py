"""Binding values stay in RooFit while the live ROOT namespace supplies macro names."""

from __future__ import annotations

import math

from statisticalproviders import install_provider
from xrdroot.roofit.bindings import bind_function
from xrdroot.roofit.variables import RooRealVar


def test_python_bindings_do_not_request_a_root_namespace_provider(monkeypatch):
    def unexpected(function):
        raise AssertionError("Python bindings keep their address instead of a registered name")

    install_provider(monkeypatch, "roofit.binding_name", unexpected)
    x = RooRealVar("x", "x", 0.5, -2, 2)
    binding = bind_function("f", math.erf, x)
    assert binding.getVal() == math.erf(0.5)
    assert "function=(0x" in binding.printArgs()


def test_macro_bindings_use_the_supplied_name_and_argument_contract(monkeypatch):
    received = []

    def named(function):
        received.append(function)
        return "custom::erf", ("input",)

    install_provider(monkeypatch, "roofit.binding_name", named)
    x = RooRealVar("x", "x", 0.5, -2, 2)
    scope = {"__name__": "__cint__", "bind": bind_function, "function": math.erf, "x": x}
    exec("binding = bind('f', function, x)", scope)
    binding = scope["binding"]
    assert received == [math.erf]
    assert binding.getVal() == math.erf(0.5)
    assert binding.printArgs() == "[ function=custom::erf input=x ]"


def test_the_root_binding_adapter_resolves_replaced_namespace_members_live(monkeypatch):
    import xrdroot.pyroot as ROOT
    from xrdroot.pyroot.roofit.bindings import registered_name

    original = ROOT.TMath.Gamma
    assert registered_name(original) == ("TMath::Gamma", ("z",))
    replacement = lambda value: value  # noqa: E731
    monkeypatch.setattr(ROOT.TMath, "Gamma", replacement)
    assert registered_name(replacement) == ("TMath::Gamma", ("z",))
    assert registered_name(original) is None
