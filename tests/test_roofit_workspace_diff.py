"""File comparisons detect changed workspace physics and broken shared references."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from xrdroot import create, open_root
from xrdroot.cli.diff import Tolerance, compare
from xrdroot.roofit.io.records import named, record
from xrdroot.roofit.io.wworkspace import Payload

DATA = Path(__file__).parent / "data/roofit-write-workspace-6.40.root"


def pair():
    with open_root(DATA) as file:
        first = file["w"]
        second = file["w"]
    return first, second


def independent_named_properties(workspace, name):
    variable = workspace.var(name)
    # ROOT's independent constructor allocates a new property UUID; a copy shares it.
    fresh = ROOT.RooRealVar(name, variable.GetTitle(), variable.getVal(),
                           variable.getMin(), variable.getMax())
    for label in variable.getBinningNames():
        if label:
            fresh.setBinning(variable.getBinning(label), label)
    assert fresh._shared.key() != variable._shared.key()
    # Retain every within-workspace alias while substituting the independently made map.
    return copy.deepcopy(workspace, {id(variable._shared): fresh._shared})


def change(workspace, kind):
    if kind == "parameter":
        workspace.var("mean").setVal(7.)
    elif kind == "configuration":
        workspace.pdf("g").specialIntegratorConfig().setEpsAbs(1e-9)
    elif kind == "binning":
        workspace.var("extra").getBinning("variable").addBoundary(2.5)
    elif kind == "range":
        workspace.var("extra").setRange("side", 2., 4.)
    else:
        change_group(workspace, kind)


def change_group(workspace, kind):
    if kind == "data":
        data = workspace.data("modelData")
        data.add(data.get())
    elif kind == "snapshot":
        workspace.getSnapshot("initial").find("mean").setVal(6.)
    elif kind == "set":
        workspace.defineSet("observables", ROOT.RooArgSet(workspace.var("extra")))
    else:
        workspace.var("mean").setAttribute("changed")


@pytest.mark.parametrize("kind", ["parameter", "configuration", "binning", "range",
                                  "data", "snapshot", "set", "attribute"])
def test_changed_workspace_contents_are_reported_with_their_location(kind):
    first, second = pair()
    if kind in ("binning", "range"):
        second = independent_named_properties(second, "extra")
    assert list(compare(first, second, Tolerance())) == []
    change(second, kind)
    differences = list(compare(first, second, Tolerance(), path="workspace"))
    assert differences and all(line.startswith("workspace.") for line in differences)


def test_a_model_using_an_identically_named_copy_is_not_the_same_shared_graph():
    first, second = pair()
    model = second.pdf("g")
    detached = model.x.Clone()
    model.x = detached
    model._proxies[0].target = detached
    differences = list(compare(first, second, Tolerance()))
    assert any("owned_targets" in line for line in differences)


def test_invalid_process_identifiers_are_compared_as_written(tmp_path):
    paths = [tmp_path / "a.root", tmp_path / "b.root"]
    for path, title in zip(paths, ["invalid first identifier", "invalid second identifier"],
                           strict=True):
        identity = record("TProcessID")
        named(identity, "ProcessID0", title)
        with create(path) as file:
            file.write("ProcessID0", Payload(identity, title))
    with open_root(paths[0]) as first, open_root(paths[1]) as second:
        differences = list(compare(first, second, Tolerance()))
    assert len(differences) == 1 and "TNamed.fTitle" in differences[0]


def test_generic_model_config_bindings_are_part_of_the_workspace_definition():
    first, second = pair()
    for workspace in (first, second):
        config = ROOT.RooStats.ModelConfig("config", workspace)
        config.SetPdf(workspace.pdf("g"))
        workspace._generic["config"] = config
    assert list(compare(first, second, Tolerance())) == []
    other = pair()[0]
    other.var("mean").setVal(7.)
    second._generic["config"].ReplaceWS(other)
    assert second._generic["config"].GetPdf().mean.getVal() == 7.
    assert list(compare(first, second, Tolerance()))


@pytest.mark.parametrize("container", [lambda value: value, lambda value: [1, value],
                                       lambda value: {"persisted": value}])
def test_unknown_object_state_is_refused_instead_of_silently_discarded(container):
    from xrdroot.errors import UnsupportedFeatureError

    first, second = pair()
    second.pdf("g")._unrecognized_persisted_state = container(object())
    with pytest.raises(UnsupportedFeatureError, match="Comparing persisted workspace state"):
        list(compare(first, second, Tolerance()))


def test_equal_snapshot_values_do_not_hide_changed_alias_relationships():
    first, second = pair()
    for workspace in (first, second):
        workspace._snapshots["shared"] = workspace._snapshots["initial"]
    assert list(compare(first, second, Tolerance())) == []
    second._snapshots["shared"] = [one.Clone() for one in second._snapshots["initial"]]
    assert list(compare(first, second, Tolerance()))


def test_equal_named_binning_edges_do_not_hide_lost_shared_properties():
    first, second = pair()
    for workspace in (first, second):
        workspace.var("mean")._shared = workspace.var("x")._shared
    assert list(compare(first, second, Tolerance())) == []
    second.var("mean")._shared = dict(second.var("mean")._shared)
    assert list(compare(first, second, Tolerance()))


def test_a_detached_proxy_retains_its_actual_definition_as_well_as_its_name():
    first, second = pair()
    for workspace in (first, second):
        model = workspace.pdf("g")
        model.mean = model.mean.Clone()
        model._proxies[1].target = model.mean
    assert list(compare(first, second, Tolerance())) == []
    second.pdf("g").mean.setVal(7.)
    assert list(compare(first, second, Tolerance()))


def process_file(path, titles, classname="TProcessID"):
    with create(path) as file:
        for index, title in enumerate(titles):
            value = record(classname)
            named(value, "ProcessID" + str(index), title)
            file.write("ProcessID" + str(index), Payload(value, title))


def process_differences(paths):
    with open_root(paths[0]) as first, open_root(paths[1]) as second:
        return list(compare(first, second, Tolerance()))


def test_fresh_process_uuids_can_differ_when_the_same_identity_partition_is_preserved(tmp_path):
    paths = [tmp_path / "first.root", tmp_path / "second.root"]
    process_file(paths[0], ["00000000-0000-0000-0000-000000000001",
                            "00000000-0000-0000-0000-000000000002"])
    process_file(paths[1], ["00000000-0000-0000-0000-000000000003",
                            "00000000-0000-0000-0000-000000000004"])
    assert process_differences(paths) == []


def test_distinct_process_identifiers_cannot_be_collapsed_into_one_uuid(tmp_path):
    paths = [tmp_path / "first.root", tmp_path / "second.root"]
    process_file(paths[0], ["00000000-0000-0000-0000-000000000001",
                            "00000000-0000-0000-0000-000000000002"])
    process_file(paths[1], ["00000000-0000-0000-0000-000000000003"] * 2)
    assert any("ProcessID1" in line for line in process_differences(paths))


def test_a_valid_uuid_title_on_an_unrelated_named_object_is_compared_exactly(tmp_path):
    paths = [tmp_path / "first.root", tmp_path / "second.root"]
    for path, tail in zip(paths, [1, 2], strict=True):
        process_file(path, [f"00000000-0000-0000-0000-{tail:012d}"], "TNamed")
    assert any("fTitle" in line for line in process_differences(paths))


def test_a_new_persisted_native_member_cannot_be_dropped_by_the_reader():
    from xrdroot.errors import UnsupportedFeatureError
    from xrdroot.roofit.io import variables  # noqa: F401 - register the native reader maker
    from xrdroot.roofit.io.build import Builder

    value = record("RooRealVar")
    named(value, "x", "x")
    value.m["unrecognised_native_definition"] = 5.
    with pytest.raises(UnsupportedFeatureError, match="unrecognised_native_definition"):
        Builder().node(value)


@pytest.mark.parametrize("corruption", ["identifier", "process"])
def test_persisted_proxy_registry_corruption_is_not_hidden_by_direct_member_links(corruption):
    from xrdroot.errors import FormatError
    from xrdroot.roofit.io.links import validate
    from xrdroot.roofit.io.wworkspace import workspace

    source = pair()[0]
    payload = workspace(source, 0)
    nodes = payload.value.get("_allOwnedNodes").get("_list")
    gaussian = next(one for one in nodes if one.cls == "RooGaussian")
    registry = gaussian.get("_proxyList")
    if corruption == "identifier":
        registry.m["uids"][0] = 0x7fffffff
    else:
        registry.m["pid"] += 1
    with pytest.raises(FormatError, match="persisted workspace proxy"):
        validate(gaussian)


def test_changed_dataset_global_observable_snapshots_are_reported():
    first, second = pair()
    for workspace in (first, second):
        aux = ROOT.RooRealVar("aux", "aux", 3.)
        workspace.data("modelData").setGlobalObservables(ROOT.RooArgSet(aux))
    assert list(compare(first, second, Tolerance())) == []
    second.data("modelData").getGlobalObservables().find("aux").setVal(7.)
    assert list(compare(first, second, Tolerance()))


def test_nondefault_category_ranges_cannot_be_dropped_while_reading_a_workspace():
    from xrdroot.errors import UnsupportedFeatureError
    from xrdroot.roofit.io.build import Builder

    category = record("RooCategory")
    named(category, "category", "category")
    category.m["_rangesPointerForIO"] = {"selected": [2]}
    with pytest.raises(UnsupportedFeatureError, match="category named ranges"):
        Builder().node(category)


@pytest.mark.parametrize("error_name", ["UnsupportedFeatureError", "FormatError"])
def test_a_workspace_read_error_is_reported_as_a_difference_instead_of_equality(
    error_name, monkeypatch
):
    import xrdroot.errors as errors
    from xrdroot.file import Directory

    original = Directory._read

    def refused(directory, key):
        if key.classname == "RooWorkspace":
            raise getattr(errors, error_name)("Persisted definition cannot be decoded.")
        return original(directory, key)

    monkeypatch.setattr(Directory, "_read", refused)
    with open_root(DATA) as first, open_root(DATA) as second:
        differences = list(compare(first, second, Tolerance()))
    assert any(
        "not compared: Persisted definition cannot be decoded." in line for line in differences
    )


@pytest.mark.parametrize("corruption", ["duplicate", "owner"])
def test_ambiguous_or_misowned_persisted_proxy_targets_are_rejected(corruption):
    from xrdroot.errors import FormatError
    from xrdroot.roofit.io.links import identity, proxies, validate
    from xrdroot.roofit.io.records import put
    from xrdroot.roofit.io.wworkspace import workspace

    payload = workspace(pair()[0], 0)
    gaussian = next(one for one in payload.value.get("_allOwnedNodes").get("_list")
                    if one.cls == "RooGaussian")
    declared = proxies(gaussian)
    if corruption == "duplicate":
        identity(declared[1][1])["fUniqueID"] = identity(declared[0][1])["fUniqueID"]
    else:
        put(declared[0][1], "_owner", declared[1][1])
    with pytest.raises(FormatError, match="persisted workspace proxy"):
        validate(gaussian)


def test_zero_uid_unregistered_proxy_members_are_not_treated_as_ambiguous_references():
    from xrdroot.roofit.io.links import identity, proxies, validate
    from xrdroot.roofit.io.wworkspace import workspace

    payload = workspace(pair()[0], 0)
    gaussian = next(one for one in payload.value.get("_allOwnedNodes").get("_list")
                    if one.cls == "RooGaussian")
    for _name, proxy in proxies(gaussian):
        identity(proxy)["fUniqueID"] = 0
    gaussian.get("_proxyList").m["uids"] = []
    assert validate(gaussian) == []


@pytest.mark.parametrize("corruption", ["identifier", "process", "owner"])
def test_a_failed_build_cannot_be_reused_to_bypass_persisted_proxy_validation(corruption):
    from xrdroot.errors import FormatError
    from xrdroot.roofit.io.build import Builder
    from xrdroot.roofit.io.links import proxies
    from xrdroot.roofit.io.records import put
    from xrdroot.roofit.io.wworkspace import workspace

    payload = workspace(pair()[0], 0)
    gaussian = next(one for one in payload.value.get("_allOwnedNodes").get("_list")
                    if one.cls == "RooGaussian")
    valid = gaussian.get("x").get("_arg")
    builder = Builder()
    cached = builder.node(valid)
    saved = (builder.made.copy(), builder.validated_schema.copy(), builder.shared_binnings.copy())
    registry = gaussian.get("_proxyList")
    if corruption == "identifier":
        registry.m["uids"][0] = 0x7fffffff
    elif corruption == "process":
        registry.m["pid"] += 1
    else:
        put(proxies(gaussian)[0][1], "_owner", None)
    for _attempt in range(2):
        with pytest.raises(FormatError, match="persisted workspace proxy"):
            builder.node(payload.value)
        assert (builder.made, builder.validated_schema, builder.shared_binnings) == saved
        assert id(payload.value) not in builder.made and id(gaussian) not in builder.made
        assert builder.node(valid) is cached


def fixed_coefficient_record(kind):
    from xrdroot.roofit.io.records import put

    model = record("RooAddPdf")
    named(model, "model", "model")
    if kind == "normalization":
        normalization = record("RooCollectionProxy<RooArgSet>")
        put(normalization, "_list", [record("RooRealVar")])
        put(model, "_refCoefNorm", normalization)
    elif kind == "attribute":
        put(model, "_stringAttrib", {"ref_coef_norm": "x"})
    else:
        put(model, "_refCoefRangeName", {"fName": "fit", "fTitle": "fit"})
    return model


@pytest.mark.parametrize("kind", ["normalization", "attribute", "range"])
def test_fixed_coefficient_definitions_cannot_be_silently_discarded(kind):
    from xrdroot.errors import UnsupportedFeatureError
    from xrdroot.roofit.io import models  # noqa: F401 - register the native reader makers
    from xrdroot.roofit.io.build import Builder

    builder = Builder()
    model = fixed_coefficient_record(kind)
    for _attempt in range(2):
        with pytest.raises(UnsupportedFeatureError, match="RooAddPdf fixed coefficient"):
            builder.node(model)
        assert builder.made == {} and builder.validated_schema == set()


@pytest.mark.parametrize("kind", ["normalization", "attribute", "range"])
def test_a_refused_fixed_coefficient_definition_cannot_compare_equal(kind, monkeypatch):
    from xrdroot.file import Directory
    from xrdroot.roofit.io.build import Builder

    original = Directory._read

    def refused(directory, key):
        if key.classname == "RooWorkspace":
            return Builder().node(fixed_coefficient_record(kind))
        return original(directory, key)

    monkeypatch.setattr(Directory, "_read", refused)
    with open_root(DATA) as first, open_root(DATA) as second:
        differences = list(compare(first, second, Tolerance()))
    assert any("not compared: Reading RooAddPdf fixed coefficient" in line for line in differences)


@pytest.mark.parametrize("held", [lambda: np.array([object()], dtype=object),
                                  lambda: {object()}])
def test_unknown_objects_inside_arrays_and_sets_cannot_disappear_from_comparisons(held):
    from xrdroot.errors import UnsupportedFeatureError

    first, second = pair()
    second.pdf("g")._unrecognized_persisted_state = held()
    with pytest.raises(UnsupportedFeatureError, match="Comparing persisted workspace state"):
        list(compare(first, second, Tolerance()))


def test_numpy_scalar_definition_fields_are_compared_by_their_exact_value():
    first, second = pair()
    first.pdf("g")._measurement_index = np.int32(3)
    second.pdf("g")._measurement_index = np.int32(3)
    assert list(compare(first, second, Tolerance())) == []
    second.pdf("g")._measurement_index = np.int32(4)
    assert list(compare(first, second, Tolerance()))
