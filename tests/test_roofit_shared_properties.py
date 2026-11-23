"""Native UUID sharing spans keys while weak ownership and failed reads stay isolated."""

from __future__ import annotations

import copy
import gc
import weakref

import pytest

import xrdroot.pyroot as ROOT
from xrdroot import open_root
from xrdroot.errors import FormatError, UnsupportedFeatureError
from xrdroot.roofit.io import variables  # noqa: F401 - register the native variable makers
from xrdroot.roofit.io.build import Builder
from xrdroot.roofit.io.records import named, put, record
from xrdroot.roofit.io.wnodes import Nodes
from xrdroot.roofit.shared import NULL_IDENTITY, PROPERTIES, UUID_FIELDS, SharedBinnings


def variable_record():
    value = record("RooRealVar")
    named(value, "x", "observable")
    put(value, "_value", 2.)
    shared = value.get("_sharedProp")
    put(shared, "_altBinning", {"side": Nodes().binning(ROOT.RooRangeBinning(1., 3., "side"))})
    return value


def identity(value):
    held = value.get("_sharedProp").get("_uuid")
    return tuple(tuple(held.m[name]) if name == "fNode" else held.m[name] for name in UUID_FIELDS)


def test_native_property_identity_is_interned_across_independent_builders():
    first_record = variable_record()
    second_record = copy.deepcopy(first_record)
    first = Builder().node(first_record)
    second = Builder().node(second_record)
    assert first is not second and first._shared is second._shared
    first.setRange("side", 2., 4.)
    assert (second.getMin("side"), second.getMax("side")) == (2., 4.)
    # ROOT discards incoming binnings when a live UUID already owns the properties.
    replacement = copy.deepcopy(first_record)
    put(replacement.get("_sharedProp"), "_altBinning", {})
    third = Builder().node(replacement)
    assert third._shared is first._shared
    assert (third.getMin("side"), third.getMax("side")) == (2., 4.)


def test_equal_ranges_with_independent_uuids_do_not_share_mutations():
    first_record, second_record = variable_record(), variable_record()
    assert identity(first_record) != identity(second_record)
    first, second = Builder().node(first_record), Builder().node(second_record)
    assert first._shared is not second._shared
    first.setRange("side", 2., 4.)
    assert (second.getMin("side"), second.getMax("side")) == (1., 3.)


def test_null_shared_property_sentinels_remain_independent_and_unregistered():
    values = [record("RooRealVar"), record("RooRealVar")]
    for index, value in enumerate(values):
        named(value, "x" + str(index), "observable")
        value.get("_sharedProp").get("_uuid").m.update(zip(UUID_FIELDS, NULL_IDENTITY, strict=True))
    first, second = (Builder().node(value) for value in values)
    assert first._shared is not second._shared
    assert first._shared.key() is None and second._shared.key() is None
    first.setRange("side", 1., 3.)
    assert not second.hasRange("side")
    assert NULL_IDENTITY not in PROPERTIES


def test_the_process_registry_does_not_keep_dead_properties_alive():
    properties = SharedBinnings({"side": ROOT.RooRangeBinning(1., 3., "side")})
    key = properties.key()
    assert key is not None
    PROPERTIES[key] = properties
    observed = weakref.ref(properties)
    del properties
    gc.collect()
    assert observed() is None and key not in PROPERTIES


def test_a_failed_graph_does_not_publish_its_new_property_uuid(monkeypatch):
    import xrdroot.roofit.io.build as build

    value = variable_record()
    key = identity(value)
    builder = Builder()
    original = build.dress

    def refused(node, record):
        raise FormatError("A postconstruction reference is invalid.")

    monkeypatch.setattr(build, "dress", refused)
    for _attempt in range(2):
        with pytest.raises(FormatError, match="postconstruction reference"):
            builder.node(value)
        assert key not in PROPERTIES
        assert (
            builder.made == {}
            and builder.shared_binnings == {}
            and builder.validated_schema == set()
        )
    monkeypatch.setattr(build, "dress", original)
    made = builder.node(value)
    assert PROPERTIES[key] is made._shared


def test_a_live_uuid_cannot_hide_unknown_incoming_persisted_fields():
    value = variable_record()
    made = Builder().node(value)
    invalid = copy.deepcopy(value)
    invalid.get("_sharedProp").m["unknown_native_state"] = 5.
    with pytest.raises(UnsupportedFeatureError, match="unknown_native_state"):
        Builder().node(invalid)
    assert PROPERTIES[identity(value)] is made._shared


def test_cloning_allocates_shared_properties_even_before_the_first_named_range():
    source = ROOT.RooRealVar("x", "x", 2., 0., 10.)
    assert source._shared.key() is None
    cloned = source.Clone("mean")
    assert source._shared is cloned._shared and source._shared.key() is not None
    source.setRange("side", 1., 3.)
    assert (cloned.getMin("side"), cloned.getMax("side")) == (1., 3.)
    assert copy.deepcopy(source)._shared is source._shared


def test_multiple_writes_retain_native_cross_key_named_range_aliases(tmp_path):
    x = ROOT.RooRealVar("x", "observable", 2., 0., 10.)
    x.setRange("side", 1., 3.)
    mean = x.Clone("mean")
    sigma = ROOT.RooRealVar("sigma", "width", 1., .1, 3.)
    workspace = ROOT.RooWorkspace("w", "workspace")
    workspace.Import(ROOT.RooGaussian("g", "signal", x, mean, sigma))
    key = workspace.var("x")._shared.key()
    path = tmp_path / "cross-key.root"
    with ROOT.TFile(str(path), "RECREATE"):
        assert workspace.Write("first") > 0
        assert workspace.Write("second") > 0
    assert workspace.var("x")._shared.key() == key
    with open_root(path) as file:
        first, second = file["first"], file["second"]
    assert first is not second
    assert first.var("x")._shared is second.var("mean")._shared
    assert first.var("x")._shared.key() == key
    first.var("x").setRange("side", 2., 4.)
    for held in (first, second):
        for name in ("x", "mean"):
            assert (held.var(name).getMin("side"), held.var(name).getMax("side")) == (2., 4.)


def test_plain_mapping_aliases_are_upgraded_without_losing_identity_between_writes():
    source = ROOT.RooRealVar("x", "x", 2., 0., 10.)
    other = source.Clone("other")
    supplied = {"side": ROOT.RooRangeBinning(1., 3., "side")}
    source._shared = supplied
    other._shared = supplied
    nodes = Nodes()
    first, second = nodes.node(source), nodes.node(other)
    assert source._shared is other._shared
    assert nodes.property_maps[id(supplied)][0] is supplied
    assert identity(first) == identity(second)
    assert identity(Nodes().node(source)) == identity(first)


def test_unallocated_properties_write_the_native_zero_uuid_sentinel():
    value = ROOT.RooRealVar("x", "x", 2., 0., 10.)
    assert identity(Nodes().node(value)) == NULL_IDENTITY


def test_cloning_legacy_plain_property_maps_retains_aliases_and_an_independent_default_binning():
    source = ROOT.RooRealVar("x", "x", 2., 0., 10.)
    supplied = {"side": ROOT.RooRangeBinning(1., 3., "side")}
    source._shared = supplied
    cloned = source.Clone("copy")
    assert source._shared is cloned._shared is supplied
    cloned.setRange("side", 2., 4.)
    assert (source.getMin("side"), source.getMax("side")) == (2., 4.)
    cloned.setRange(1., 9.)
    assert (source.getMin(), source.getMax()) == (0., 10.)
    assert (cloned.getMin(), cloned.getMax()) == (1., 9.)
