"""Dynamic native proxy collections preserve the same strict ownership and reference checks."""

from __future__ import annotations

import copy

import pytest

from xrdroot.errors import FormatError
from xrdroot.roofit.io.links import validate
from xrdroot.roofit.io.stream import Streamed


def dynamic_record(proxy_class="RooTemplateProxy<RooAbsReal>"):
    owner = Streamed("RooSimultaneous")
    proxy = Streamed(proxy_class)
    proxy.m.update(TNamed={"fName": "signal", "TObject": {"fUniqueID": 1, "pid": 0}},
                   _owner=owner)
    collection = Streamed("TList")
    collection.m["items"] = [proxy]
    owner.m["_pdfProxyList"] = collection
    registry = Streamed("RooRefArray")
    registry.m.update(uids=[1], pid=0)
    owner.m["_proxyList"] = registry
    return owner, proxy


@pytest.mark.parametrize("proxy_class", ["RooTemplateProxy<RooAbsReal>", "RooRealProxy"])
def test_a_native_collection_declares_its_dynamic_proxies_without_following_other_models(
    proxy_class
):
    owner, proxy = dynamic_record(proxy_class)
    other, foreign = dynamic_record(proxy_class)
    assert foreign.get("_owner") is other
    owner.m["referenced_server"] = other
    assert validate(owner) == ["signal"]
    owner.m["_pdfProxyList"].m["items"].append(proxy)
    assert validate(owner) == ["signal"]


@pytest.mark.parametrize("corruption,expected", [("owner", "owner"),
                                                  ("missing", "no declared target"),
                                                  ("process", "different process"),
                                                  ("duplicate", "ambiguous")])
def test_dynamic_collection_references_cannot_bypass_any_existing_binding_check(
    corruption, expected
):
    owner, proxy = dynamic_record()
    if corruption == "owner":
        proxy.m["_owner"] = Streamed("RooSimultaneous")
    elif corruption == "missing":
        owner.m["_proxyList"].m["uids"] = [2]
    elif corruption == "process":
        proxy.m["TNamed"]["TObject"]["pid"] = 1
    else:
        duplicate = copy.deepcopy(proxy)
        duplicate.m["_owner"] = owner
        owner.m["_pdfProxyList"].m["items"].append(duplicate)
    with pytest.raises(FormatError, match=expected):
        validate(owner)


def test_explicit_null_references_and_unregistered_duplicates_keep_their_native_meaning():
    owner, proxy = dynamic_record()
    owner.m["_proxyList"].m["uids"] = [0, 1]
    assert validate(owner) == ["<null>", "signal"]
    zero = copy.deepcopy(proxy)
    zero.m["TNamed"]["TObject"]["fUniqueID"] = 0
    zero.m["_owner"] = owner
    owner.m["_pdfProxyList"].m["items"].extend([zero, copy.deepcopy(zero)])
    assert validate(owner) == ["<null>", "signal"]


def test_standard_nested_containers_and_cycles_do_not_create_or_lose_proxy_declarations():
    owner, proxy = dynamic_record()
    cycle = []
    cycle.append(cycle)
    owner.m["_pdfProxyList"].m["items"] = [{"nested": ([proxy], cycle)}]
    assert validate(owner) == ["signal"]
