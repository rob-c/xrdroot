"""Promises survive missing results, and measurements remain traceable to the oracle."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from tools.tutorials import api_coverage, api_oracle, cli, ratchet, report, timing
from tools.tutorials.catalog import Tutorial
from tools.tutorials.harness import Harness, Settings

from xrdroot.errors import UnsupportedFeatureError


def record(path="a.C", status="PASS", root=2.0, mine=6.0):
    return {"path": path, "area": "a", "status": status, "reason": "",
            "oracle": {"runtime": root, "cached": True}, "xrdroot": {"runtime": mine}}


def document(*records):
    return report.document(records or [record()], {"root_version": "6.40.04"})


def test_the_ratchet_rejects_regressions_and_missing_tutorials():
    baseline = {"a.C": "PASS", "missing.C": "PASS", "new.C": "FAIL"}
    failed = ratchet.regressions(baseline, document(record(status="DIFF")))
    assert [(row["path"], row["status"]) for row in failed] == [
        ("a.C", "DIFF"), ("missing.C", "MISSING")]
    assert ratchet.regressions({"a.C": "PASS"}, document()) == []


def test_a_timeout_is_not_silently_excused(tmp_path, capsys):
    row = record(status="FAIL")
    row.update(reason="timeout", xrdroot={"runtime": 300, "timed_out": True})
    results, baseline = tmp_path / "results.json", tmp_path / "baseline.json"
    results.write_text(json.dumps(document(row)))
    baseline.write_text(json.dumps({"a.C": "PASS"}))
    assert cli.main(["ratchet", str(results), "--baseline", str(baseline)]) == 1
    assert "PASS -> FAIL: timeout" in capsys.readouterr().out


def test_only_a_passing_isolated_recheck_excuses_a_timeout(tmp_path, monkeypatch, capsys):
    results, baseline = tmp_path / "results.json", tmp_path / "baseline.json"
    results.write_text(json.dumps(document(record(status="FAIL"))))
    baseline.write_text(json.dumps({"a.C": "PASS"}))
    monkeypatch.setattr(ratchet, "recheck", lambda *args: {"a.C": record()})
    assert cli.main(["ratchet", str(results), "--baseline", str(baseline),
                     "--rerun-timeouts"]) == 0
    assert "passed when rerun alone" in capsys.readouterr().out


def test_baselines_store_statuses_and_the_corresponding_timings(tmp_path):
    results, baseline = tmp_path / "results.json", tmp_path / "baseline.json"
    results.write_text(json.dumps(document()))
    assert cli.main(["baseline", str(results), "--output", str(baseline)]) == 0
    assert json.loads(baseline.read_text()) == {"a.C": "PASS"}
    measured = json.loads((tmp_path / "baseline-timings.json").read_text())
    assert measured["tutorials"]["a.C"] == {
        "root_seconds": 2.0, "xrdroot_seconds": 6.0, "ratio": 3.0}


@pytest.mark.parametrize("meta", [{"only": ["a.C"], "root_version": "6"}, {},
                                  {"root_version": "6", "oracle_only": True},
                                  {"root_version": "6", "no_oracle": True}])
def test_partial_or_uncompared_runs_cannot_replace_the_baseline(tmp_path, meta):
    with pytest.raises(ValueError, match="full run"):
        ratchet.write_baseline({**document(), "meta": meta}, tmp_path / "baseline.json")


@pytest.mark.parametrize("records", [[], [record(status="unrecognised")]])
def test_empty_or_unknown_results_cannot_replace_the_baseline(tmp_path, records):
    with pytest.raises(ValueError, match="nonempty"):
        ratchet.write_baseline({**document(), "tutorials": records}, tmp_path / "baseline.json")


def test_reports_include_cached_oracle_wall_times_ratios_and_changes():
    previous = timing.snapshot(document(record(root=2, mine=4)))
    doc = report.document([record()], {}, previous)
    assert doc["timings"][0]["ratio_change"] == 1.5
    assert "| a.C | 2.000 | 6.000 | 3.000 | 1.500 |" in report.markdown(doc)
    assert "xrdroot/ROOT" in report.page(doc)


@pytest.mark.parametrize("root,mine", [(0, 2), (None, 2), (2, None), (float("inf"), 1)])
def test_unmeasured_or_zero_oracle_times_do_not_produce_invented_ratios(root, mine):
    assert timing.measured(record(root=root, mine=mine))["ratio"] is None


def test_slowest_thirty_are_ordered_by_xrdroot_wall_time():
    doc = document(*(record(f"{i}.C", mine=i) for i in range(40)))
    assert [row["path"] for row in doc["timings"][:2]] == ["39.C", "38.C"]
    assert "| 0.C |" not in report.markdown(doc)


def test_even_longtests_and_explicit_overrides_obey_the_timeout_cap(tmp_path):
    harness = Harness.__new__(Harness)
    harness.settings = Settings(tutorials=tmp_path, timeout=1000)
    assert harness._timeout(Tutorial("a.C", labels=("longtest",))) == 300
    harness.settings = Settings(tutorials=tmp_path, timeout=20)
    assert harness._timeout(Tutorial("a.C")) == 20


class Present:
    def Method(self):
        return 1

    def Conditional(self, refused=False):
        if refused:
            raise UnsupportedFeatureError("An optional path is unsupported.")
        return 2

    def Refused(self):
        raise UnsupportedFeatureError("This method is unsupported; use Method instead.")

    def Invalid(self):
        raise ValueError("This is an argument error, not a refusal.")


class Refused:
    def __init__(self):
        raise UnsupportedFeatureError("This GUI class is unsupported; use batch graphics instead.")


def inventory():
    return {"root_version": "6", "classes": [
        {"name": "Present", "library": "Core", "methods": [
            "Present", "Method", "Conditional", "Refused", "Invalid", "Missing"]},
        {"name": "Refused", "library": "Graf", "methods": ["Refused"]},
        {"name": "Unknown", "library": "Tree", "methods": ["GetEntry"]}]}


def test_api_coverage_counts_methods_and_classes_by_library_without_calling_them():
    namespace = SimpleNamespace(Present=Present, Refused=Refused)
    doc = api_coverage.compare(inventory(), namespace)
    assert doc["summary"]["Core"] == {
        "classes": {"present": 1}, "methods": {"present": 4, "refused": 1, "missing": 1}}
    assert doc["summary"]["Graf"]["classes"] == {"refused": 1}
    assert doc["summary"]["Tree"]["methods"] == {"missing": 1}
    assert "Structural presence" in api_coverage.markdown(doc)


def test_api_namespaces_and_builtin_implementations_can_be_inspected():
    namespace = SimpleNamespace(ROOT=SimpleNamespace(Thing=list))
    assert api_coverage.presence(namespace, "ROOT::Thing")[0] == "present"
    assert api_coverage.refusal(len) is None


def test_api_reports_can_be_regenerated_without_an_installed_oracle(tmp_path):
    source = tmp_path / "oracle.json"
    source.write_text(json.dumps(inventory()))
    assert cli.main(["api-coverage", "--inventory", str(source), "--out", str(tmp_path)]) == 0
    assert "Unknown" in (tmp_path / "api-coverage.json").read_text()
    assert (tmp_path / "api-coverage.md").is_file()


def test_dictionary_metadata_retains_library_names_and_unique_public_methods():
    cls = SimpleNamespace(Property=lambda: 0, GetSharedLibs=lambda: "libHist.so libCore.so",
                          GetListOfAllPublicMethods=lambda: [
                              SimpleNamespace(GetName=lambda: "Fill", ExtraProperty=lambda: 0),
                              SimpleNamespace(GetName=lambda: "Fill", ExtraProperty=lambda: 0)])
    root = SimpleNamespace(TClass=SimpleNamespace(GetClass=lambda name: cls), kIsNamespace=1,
                           kIsConstructor=2, kIsDestructor=4)
    assert api_oracle.describe(root, "TH1") == {
        "name": "TH1", "library": "Hist", "methods": ["Fill"]}
    assert api_oracle.library_name("") == "(unknown)"


def test_core_dictionary_headers_recover_library_names_without_autoload_metadata():
    headers = api_oracle.header_libraries('module "Core" {\n'
                                        '  module "TObject.h" { header "TObject.h" export * }\n'
                                        '}\nmodule "Hist" {\n  header "TH1.h"\n}')
    assert headers == {"TObject.h": "Core", "TH1.h": "Hist"}
    cls = SimpleNamespace(Property=lambda: 0, GetSharedLibs=lambda: "",
                          GetDeclFileName=lambda: "TObject.h", GetListOfAllPublicMethods=lambda: [])
    root = SimpleNamespace(TClass=SimpleNamespace(GetClass=lambda name: cls), kIsNamespace=1)
    assert api_oracle.describe(root, "TObject", headers)["library"] == "Core"


def test_scopes_inside_template_arguments_are_resolved_as_one_component():
    class Template:
        def __class_getitem__(cls, arguments):
            assert arguments == "ROOT::Math::Coordinates<double>"
            return Present

    namespace = SimpleNamespace(ROOT=SimpleNamespace(Math=SimpleNamespace(Vector=Template)))
    name = "ROOT::Math::Vector<ROOT::Math::Coordinates<double>>"
    assert api_coverage.presence(namespace, name)[0] == "present"
    assert api_coverage._parts(name) == ["ROOT", "Math", "Vector<ROOT::Math::Coordinates<double>>"]


def test_canonical_stl_names_and_multiple_template_arguments_find_their_classes():
    class Template:
        def __getitem__(self, arguments):
            assert arguments == ("string", "ROOT::Math::Vector<double>")
            return Present

    namespace = SimpleNamespace(std=SimpleNamespace(map=Template()))
    name = "map<string,ROOT::Math::Vector<double>>"
    assert api_coverage._class_presence(namespace, name)[0] == "present"
    assert api_coverage._class_presence(namespace, "absent")[0] == "missing"


def test_refusing_namespace_lookups_are_counted_without_constructing_objects():
    class Namespace:
        def __getattr__(self, name):
            raise UnsupportedFeatureError("The GUI namespace is unsupported; use batch graphics.")

    assert api_coverage.presence(Namespace(), "Widget")[0] == "refused"


def test_header_only_classes_are_included_and_enums_are_excluded():
    class Interpreter:
        def __init__(self):
            self.headers = iter(["ROOT/RNTupleModel.hxx", "ROOT/RNTupleModel.hxx", "vector"])
            self.properties = iter([1, 8, 1])
            self.deleted = False

        def ClassInfo_Factory(self):
            return None

        def ClassInfo_Next(self, info):
            self.header = next(self.headers, None)
            return self.header is not None

        def ClassInfo_FileName(self, info):
            return self.header

        def ClassInfo_Property(self, info):
            return next(self.properties)

        def ClassInfo_FullName(self, info):
            return "ROOT::RNTupleModel"

        def ClassInfo_Delete(self, info):
            self.deleted = True

    interpreter = Interpreter()
    root = SimpleNamespace(gInterpreter=interpreter, kIsClass=1, kIsStruct=2)
    assert api_oracle.declared_classes(root, {"ROOT/RNTupleModel.hxx": "ROOTNTuple"}) == [
        "ROOT::RNTupleModel"]
    assert interpreter.deleted


def test_constructors_destructors_and_operators_use_the_python_spellings():
    assert api_coverage._attribute("ROOT::Vector<double>", "Vector<double>") == "__init__"
    assert api_coverage._attribute("ROOT::Vector<double>", "~Vector") == "_destruct"
    assert api_coverage._attribute("TVector3", "operator+") == "__add__"
    method = SimpleNamespace(ExtraProperty=lambda: 2,
                             GetClass=lambda: SimpleNamespace(GetName=lambda: "TObject"))
    root = SimpleNamespace(kIsConstructor=2, kIsDestructor=4)
    assert api_oracle._included_method(root, method, "TObject")
    assert not api_oracle._included_method(root, method, "TH1")


def test_timeouts_are_rechecked_alone_with_the_original_settings(tmp_path, monkeypatch):
    from tools.tutorials import catalog as catalogue

    oracle = SimpleNamespace(version="6.40.04", features=frozenset(), python="python",
                             has_module=lambda name: True)
    monkeypatch.setattr(ratchet, "_oracle", lambda meta: oracle)
    monkeypatch.setattr(catalogue, "catalog", lambda *args: {})
    captured = []

    class Recheck:
        def __init__(self, settings, found, oracle):
            captured.append(settings)

        def xrdroot_commit(self):
            return "abc"

        def run(self):
            return [record()]

    monkeypatch.setattr(ratchet, "Harness", Recheck)
    doc = document()
    doc["meta"].update(tutorials_dir=str(tmp_path), xrdroot_python="python",
                       xrdroot_src=str(tmp_path), xrdroot_commit="abc", timeout=900)
    failed = [{**record(status="FAIL"), "xrdroot": {"timed_out": True}},
              record("other.C", status="DIFF")]
    retries = ratchet.recheck(doc, failed, tmp_path / "retry")
    assert retries["a.C"]["status"] == "PASS"
    settings, = captured
    assert (settings.jobs, settings.timeout, settings.only) == (1, 300, ("a.C",))
    assert (tmp_path / "retry/results.json").is_file()
    doc["meta"]["xrdroot_commit"] = "changed"
    with pytest.raises(ValueError, match="original xrdroot commit"):
        ratchet.recheck(doc, failed, tmp_path / "retry")


def test_unavailable_or_mismatched_oracles_cannot_excuse_a_timeout(tmp_path, monkeypatch):
    failed = [{**record(status="FAIL"), "xrdroot": {"timed_out": True}}]
    monkeypatch.setattr(ratchet, "_oracle", lambda meta: None)
    with pytest.raises(ValueError, match="same ROOT oracle"):
        ratchet.recheck(document(), failed, tmp_path)
    assert ratchet.recheck(document(), [], tmp_path) == {}


def test_cli_errors_are_reported_without_a_traceback(tmp_path, capsys):
    assert cli.main(["ratchet", str(tmp_path / "absent.json")]) == 2
    assert "absent.json" in capsys.readouterr().err
