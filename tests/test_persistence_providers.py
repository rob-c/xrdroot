"""Persistence dispatch preserves file descriptions and mutable compatibility registries."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from xrdroot import canvas, interp, kinds, plugins
from xrdroot.canvas import CANVASES
from xrdroot.canvas import streamer as canvas_streamer
from xrdroot.canvas.streamer import STREAMED, streamer_for
from xrdroot.file import Directory


@pytest.fixture
def cold_kinds(monkeypatch):
    lower = {key: value for key, value in kinds._CLASSES.items()
             if key not in plugins.keys("canvas.dress")}
    # Record a restoration even when the lazy export has not existed yet.
    monkeypatch.setitem(vars(kinds), "CLASSES", lower)
    monkeypatch.delitem(vars(kinds), "CLASSES")
    monkeypatch.setattr(kinds, "_CLASSES", lower)
    monkeypatch.setattr(kinds, "_canvas_materialized", False)
    return kinds


def test_common_class_membership_and_dressing_do_not_resolve_graphics(cold_kinds, monkeypatch):
    def unexpected(name):
        raise AssertionError(f"Unexpected provider import: {name}")

    monkeypatch.setattr(plugins, "importlib", SimpleNamespace(import_module=unexpected))
    assert cold_kinds._known("TH1D")
    assert cold_kinds._known("TCanvas")
    assert not cold_kinds._known("TUnknown")
    assert cold_kinds.dress("TH1D", 1) == 1
    value = {"unread": 1}
    assert cold_kinds.dress("TUnknown", value) is value
    assert "CLASSES" in dir(cold_kinds) and "CLASSES" not in vars(cold_kinds)
    assert cold_kinds.__all__ == ["CLASSES", "dress"]
    with pytest.raises(AttributeError, match="has no attribute 'unknown'"):
        cold_kinds.__getattr__("unknown")


def test_public_classes_materializes_exact_class_objects_in_the_original_order(cold_kinds):
    lower = [key for key in cold_kinds._CLASSES if key not in cold_kinds._IMAGE_NAMES]
    classes = cold_kinds.CLASSES
    assert type(classes) is dict
    assert classes is cold_kinds.CLASSES is cold_kinds._CLASSES
    assert list(classes) == [*lower, *CANVASES, *cold_kinds._IMAGE_NAMES]
    assert all(classes[key] is value for key, value in CANVASES.items())


def test_public_class_overrides_and_removals_control_later_dressing(cold_kinds):
    classes = cold_kinds.CLASSES
    value, result = {}, object()
    classes["TCanvas"] = lambda classname, members: result
    assert cold_kinds.dress("TCanvas", value) is result
    del classes["TCanvas"]
    assert not cold_kinds._known("TCanvas")
    assert cold_kinds.dress("TCanvas", value) is value

    def read(buf):
        return value

    assert interp._shown("TCanvas", read) is read
    classes.clear()
    assert not cold_kinds._known("TH1D")
    assert cold_kinds.dress("TH1D", value) is value


def test_a_failed_public_class_resolution_does_not_partially_change_the_mapping(
        cold_kinds, monkeypatch):
    original = dict(cold_kinds._CLASSES)

    def missing(point, key):
        raise ImportError(f"Unavailable {point}: {key}")

    monkeypatch.setattr(cold_kinds, "lookup_loaded", lambda point: None)
    monkeypatch.setattr(cold_kinds, "require", missing)
    with pytest.raises(ImportError, match=r"Unavailable canvas.dress"):
        cold_kinds.__getattr__("CLASSES")
    assert cold_kinds._CLASSES == original
    assert not cold_kinds._canvas_materialized and "CLASSES" not in vars(cold_kinds)


def test_canvas_streamer_replacements_and_deletions_are_used_by_the_reader(monkeypatch):
    value = object()

    def replacement(described):
        return lambda buf: value

    source = SimpleNamespace(streamers=lambda: {})
    monkeypatch.setitem(STREAMED, "TCanvas", replacement)
    assert streamer_for("TCanvas") is replacement
    assert interp._by_hand("TCanvas", source, ())(None) is value
    monkeypatch.delitem(STREAMED, "TCanvas")
    assert streamer_for("TCanvas") is None
    assert interp._by_hand("TCanvas", source, ()) is None


def test_a_file_description_precedes_the_histfactory_fallback(monkeypatch):
    name, members = "RooStats::HistFactory::Sample", {"custom": object()}
    source = SimpleNamespace(streamers=lambda: {name: members})

    def unexpected(point, key):
        raise AssertionError(f"Unexpected fallback: {point}: {key}")

    monkeypatch.setattr(interp, "lookup", unexpected)
    assert interp._description(source, name) is members


def test_a_missing_description_uses_only_its_declared_layout_provider(monkeypatch):
    name, members, seen = "RooStats::HistFactory::Sample", {"member": object()}, []

    def lookup(point, key):
        seen.append((point, key))
        return (lambda: {name: members}) if key == name else None

    monkeypatch.setattr(interp, "lookup", lookup)
    source = SimpleNamespace(streamers=lambda: {})
    assert interp._description(source, name) is members
    assert interp._description(source, "Unknown") is None
    assert seen == [("io.layouts", name), ("io.layouts", "Unknown")]


def test_workspace_keys_pass_their_buffer_and_streamers_to_the_declared_reader(monkeypatch):
    buffer, infos, result, seen = object(), {}, object(), []
    directory = object.__new__(Directory)
    directory._source = SimpleNamespace(streamers=lambda: infos)
    key = SimpleNamespace(classname="RooWorkspace", buffer=lambda source: buffer)

    def require(point, name):
        seen.append((point, name))
        return lambda buf, streamers: result if buf is buffer and streamers is infos else None

    monkeypatch.setattr(plugins, "require", require)
    assert directory._read(key) is result
    assert seen == [("io.workspace", "RooWorkspace")]


def test_cold_class_metadata_can_recognize_graphics_without_loading_its_module(
        cold_kinds, monkeypatch):
    monkeypatch.setattr(cold_kinds, "lookup_loaded", lambda point: None)
    assert cold_kinds._known("TCanvas")
    assert not cold_kinds._known("Unknown")
    marker = object()
    monkeypatch.setattr(cold_kinds, "lookup", lambda point, key: lambda name, value: marker)
    assert cold_kinds.dress("TCanvas", {}) is marker
    assert all(cold_kinds.CLASSES[name] is target for name, target in CANVASES.items())


def test_loaded_canvas_registry_additions_are_copied_into_the_public_mapping(
        cold_kinds, monkeypatch):
    marker = object()

    def custom(name, members):
        return marker

    monkeypatch.setitem(CANVASES, "TCustomPad", custom)
    assert cold_kinds._known("TCustomPad")
    assert cold_kinds.dress("TCustomPad", {}) is marker
    classes = cold_kinds.CLASSES
    assert classes["TCustomPad"] is custom
    monkeypatch.delitem(CANVASES, "TCustomPad")
    assert classes["TCustomPad"] is custom


def test_loaded_canvas_registry_overrides_and_deletions_precede_materialization(
        cold_kinds, monkeypatch):
    marker = object()
    monkeypatch.setitem(CANVASES, "TCanvas", lambda name, members: marker)
    assert cold_kinds.dress("TCanvas", {}) is marker
    monkeypatch.delitem(CANVASES, "TCanvas")
    assert not cold_kinds._known("TCanvas")
    value = {}
    assert cold_kinds.dress("TCanvas", value) is value
    assert "TCanvas" not in cold_kinds.CLASSES


def test_a_custom_streamer_added_to_the_loaded_table_needs_no_new_registration(monkeypatch):
    marker = object()
    monkeypatch.setitem(STREAMED, "CustomStreamed", lambda described: lambda buf: marker)
    assert "CustomStreamed" not in plugins.keys("canvas.streamer")
    source = SimpleNamespace(streamers=lambda: {})
    assert interp._by_hand("CustomStreamed", source, ())(None) is marker


def test_an_unknown_streamer_does_not_load_the_graphics_component(monkeypatch):
    def expected(point):
        assert point == "canvas.streamer.custom"
        return None

    monkeypatch.setattr(interp, "lookup_loaded", expected)
    source = SimpleNamespace(streamers=lambda: {})
    assert interp._by_hand("UnknownStreamed", source, ()) is None


def _replace_binding(module, name, value, namespace):
    if namespace:
        vars(module)[name] = value
    else:
        setattr(module, name, value)


def _remove_binding(module, name, namespace):
    if namespace:
        del vars(module)[name]
    else:
        delattr(module, name)


def _unexpected_graphics(*args):
    raise AssertionError("An explicit public class binding must not resolve graphics.")


@pytest.mark.parametrize("materialized", [False, True])
@pytest.mark.parametrize("namespace", [False, True])
def test_public_class_rebindings_replace_the_whole_active_registry(
        cold_kinds, monkeypatch, materialized, namespace):
    if materialized:
        _ = cold_kinds.CLASSES
    marker = object()
    replacement = {"TReplacement": lambda name, members: marker}
    _replace_binding(cold_kinds, "CLASSES", replacement, namespace)
    monkeypatch.setattr(cold_kinds, "_canvas_classes", _unexpected_graphics)
    monkeypatch.setattr(cold_kinds, "lookup", _unexpected_graphics)
    assert cold_kinds._known("TReplacement")
    assert cold_kinds.dress("TReplacement", {}) is marker
    assert not cold_kinds._known("TH1D") and not cold_kinds._known("TCanvas")
    value = {}
    assert cold_kinds.dress("TCanvas", value) is value
    assert cold_kinds.CLASSES is replacement
    replacement.clear()
    assert not cold_kinds._known("TReplacement")


@pytest.mark.parametrize("namespace", [False, True])
def test_deleting_a_published_class_binding_keeps_python_missing_name_errors(
        cold_kinds, monkeypatch, namespace):
    _ = cold_kinds.CLASSES
    _remove_binding(cold_kinds, "CLASSES", namespace)
    monkeypatch.setattr(cold_kinds, "_canvas_classes", _unexpected_graphics)
    with pytest.raises(AttributeError, match="CLASSES"):
        _ = cold_kinds.CLASSES
    with pytest.raises(NameError, match="CLASSES"):
        cold_kinds._known("TCanvas")
    with pytest.raises(NameError, match="CLASSES"):
        cold_kinds.dress("TCanvas", {})


@pytest.mark.parametrize("namespace", [False, True])
def test_deleting_an_observed_replacement_never_resurrects_the_lower_registry(
        cold_kinds, monkeypatch, namespace):
    _replace_binding(cold_kinds, "CLASSES", {}, namespace)
    assert not cold_kinds._known("TH1D")
    _remove_binding(cold_kinds, "CLASSES", namespace)
    monkeypatch.setattr(cold_kinds, "_canvas_classes", _unexpected_graphics)
    with pytest.raises(AttributeError, match="CLASSES"):
        _ = cold_kinds.CLASSES
    with pytest.raises(NameError, match="CLASSES"):
        cold_kinds.dress("TH1D", {})


@pytest.mark.parametrize("namespace", [False, True])
@pytest.mark.parametrize("replacement", [None, object()])
def test_invalid_public_class_rebindings_keep_the_ordinary_python_failures(
        cold_kinds, monkeypatch, namespace, replacement):
    _replace_binding(cold_kinds, "CLASSES", replacement, namespace)
    monkeypatch.setattr(cold_kinds, "_canvas_classes", _unexpected_graphics)
    # Python's exact membership diagnostic changes between interpreter versions.
    with pytest.raises(TypeError) as expected:
        _ = "TH1D" in replacement
    with pytest.raises(TypeError) as actual:
        cold_kinds._known("TH1D")
    assert str(actual.value) == str(expected.value)
    with pytest.raises(AttributeError, match="get"):
        cold_kinds.dress("TH1D", {})
    assert cold_kinds.CLASSES is replacement


@pytest.mark.parametrize("namespace", [False, True])
def test_canvas_table_rebindings_are_live_until_public_classes_take_a_snapshot(
        cold_kinds, monkeypatch, namespace):
    monkeypatch.setattr(canvas, "CANVASES", canvas.CANVASES)
    marker = object()
    replacement = {"TReplacementPad": lambda name, members: marker}
    _replace_binding(canvas, "CANVASES", replacement, namespace)
    assert canvas.class_registry() is replacement
    assert cold_kinds._known("TReplacementPad")
    assert cold_kinds.dress("TReplacementPad", {}) is marker
    assert not cold_kinds._known("TCanvas")
    classes = cold_kinds.CLASSES
    assert classes["TReplacementPad"] is replacement["TReplacementPad"]
    _replace_binding(canvas, "CANVASES", {}, namespace)
    assert cold_kinds.CLASSES is classes
    assert cold_kinds._known("TReplacementPad")


@pytest.mark.parametrize("namespace", [False, True])
def test_streamer_table_rebindings_and_deletions_are_read_on_every_call(monkeypatch, namespace):
    monkeypatch.setattr(canvas_streamer, "STREAMED", canvas_streamer.STREAMED)
    marker = object()
    replacement = {"TCanvas": lambda described: lambda buf: marker}
    replacement["ReplacementStreamed"] = replacement["TCanvas"]
    _replace_binding(canvas_streamer, "STREAMED", replacement, namespace)
    assert streamer_for("TCanvas") is replacement["TCanvas"]
    source = SimpleNamespace(streamers=lambda: {})
    assert interp._by_hand("TCanvas", source, ())(None) is marker
    assert interp._by_hand("ReplacementStreamed", source, ())(None) is marker
    _replace_binding(canvas_streamer, "STREAMED", {}, namespace)
    assert interp._by_hand("TCanvas", source, ()) is None
    assert interp._by_hand("ReplacementStreamed", source, ()) is None
    _remove_binding(canvas_streamer, "STREAMED", namespace)
    with pytest.raises(NameError, match="STREAMED"):
        streamer_for("TCanvas")


def test_deleting_the_canvas_binding_does_not_change_an_existing_public_snapshot(
        cold_kinds, monkeypatch):
    classes = cold_kinds.CLASSES
    monkeypatch.delattr(canvas, "CANVASES")
    with pytest.raises(NameError, match="CANVASES"):
        canvas.class_registry()
    assert cold_kinds.CLASSES is classes
    assert cold_kinds._known("TCanvas")
