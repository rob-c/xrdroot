"""Exercise contract failures on real source trees without importing the engines."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tools.component_contract.analysis import graph, inspect, inventory, ratchet
from tools.component_contract.cli import main
from tools.component_contract.imports import imports, modules
from tools.component_contract.isolation import Boundary, isolate
from tools.component_contract.model import ContractError, load_policy, policy_from
from tools.component_contract.tables import facade_targets


def _raw() -> dict:
    return {
        "schema": 1,
        "package": "demo",
        "source": "src/demo",
        "baseline": "debt.json",
        "provider_registry": "demo.plugins",
        "provider_functions": ["demo.plugins.lookup", "demo.plugins.require"],
        "groups": {"engines": ["cppstd", "core", "high"]},
        "ownership": {
            "cppstd": {"modules": ["demo.cppstd"], "hard": ["numpy"]},
            "core": {
                "modules": ["demo.core", "demo.plugins"],
                "depends": ["cppstd"],
                "optional": ["optional_lib"],
            },
            "high": {"modules": ["demo.high"], "depends": ["core"]},
            "app": {"exact": ["demo"], "modules": ["demo.cli"], "depends": ["@engines"]},
        },
    }


def _project(root: Path, sources: dict[str, str]):
    base = root / "src/demo"
    base.mkdir(parents=True)
    for module, source in {"__init__": "", "cppstd": "", "core": "", "high": "", **sources}.items():
        path = base / (module.replace(".", "/") + ".py")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    return policy_from(root, _raw())


def _review(root: Path, policy, report) -> Path:
    value = inventory(policy, report)
    value.update(reviewed=True, review="Test review of this exact synthetic source inventory.")
    path = root / "debt.json"
    path.write_text(json.dumps(value))
    return path


def _toml(root: Path) -> None:
    # Synthetic policy used by fresh interpreters; no installation or external dependency is needed.
    root.joinpath("pyproject.toml").write_text("""
[tool.xrdroot.components]
schema = 1
package = "demo"
source = "src/demo"
baseline = "debt.json"
[tool.xrdroot.components.ownership.cppstd]
modules = ["demo.cppstd"]
[tool.xrdroot.components.ownership.core]
modules = ["demo.core"]
depends = ["cppstd"]
[tool.xrdroot.components.ownership.high]
modules = ["demo.high"]
depends = ["core"]
[tool.xrdroot.components.ownership.app]
exact = ["demo"]
modules = ["demo.cli"]
depends = ["core", "high", "cppstd"]
""")


def test_lazy_relative_and_type_checking_imports_still_obey_the_graph(tmp_path: Path) -> None:
    policy = _project(
        tmp_path,
        {
            "core": """
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .high import VALUE
def deferred():
    from . import high
    return high.VALUE
"""
        },
    )
    report = inspect(policy)
    assert [(v.target, v.rule, v.scope) for v in report.violations] == [
        ("demo.high", "edge", ""),
        ("demo.high", "edge", "deferred"),
    ]
    assert {v.line for v in report.violations} == {4, 6}


def test_longest_prefix_overrides_support_extracted_submodules(tmp_path: Path) -> None:
    raw = _raw()
    raw["ownership"]["cppstd"]["modules"].append("demo.core.generic")
    policy = policy_from(tmp_path, raw)
    assert policy.owner("demo.core.generic.containers") == "cppstd"
    assert policy.owner("demo.core") == "core"
    assert policy.closure("high") == {"high", "core", "cppstd"}
    assert policy.closure("app") == {"app", "high", "core", "cppstd"}


@pytest.mark.parametrize(
    "change, message",
    [
        ("duplicate", "Ambiguous"),
        ("cycle", "cycle"),
        ("unknown", "Unknown dependency"),
        ("outside", "outside"),
        ("schema", "schema"),
    ],
)
def test_invalid_policies_fail_before_the_source_is_trusted(
    tmp_path: Path, change, message
) -> None:
    raw = _raw()
    changes = {
        "duplicate": lambda: raw["ownership"]["core"]["modules"].append("demo.high"),
        "cycle": lambda: raw["ownership"]["cppstd"].update(depends=["high"]),
        "unknown": lambda: raw["ownership"]["core"].update(depends=["absent"]),
        "outside": lambda: raw["ownership"]["core"].update(modules=["foreign"]),
        "schema": lambda: raw.update(schema=2),
    }
    changes[change]()
    with pytest.raises(ContractError, match=message):
        policy_from(tmp_path, raw)


def test_unowned_and_missing_modules_do_not_hide_under_the_root_facade(tmp_path: Path) -> None:
    policy = _project(tmp_path, {"core": "from .missing import value\n"})
    assert inspect(policy).violations[0].rule == "internal-missing"
    (tmp_path / "src/demo/new_area.py").write_text("")
    with pytest.raises(ContractError, match=r"No component owns demo.new_area"):
        inspect(policy)


def test_external_imports_distinguish_lazy_optional_and_private_dependencies(
    tmp_path: Path,
) -> None:
    policy = _project(
        tmp_path,
        {
            "core": """
import optional_lib
try:
    import optional_lib.safe
except ImportError:
    pass
def lazy():
    from optional_lib._private import helper
    import undeclared
"""
        },
    )
    assert [(v.rule, v.target) for v in inspect(policy).violations] == [
        ("optional-eager", "optional_lib"),
        ("external-private", "optional_lib._private"),
        ("external-undeclared", "undeclared"),
    ]


def test_literal_dynamic_imports_and_aliases_cannot_bypass_dependencies(tmp_path: Path) -> None:
    policy = _project(
        tmp_path,
        {
            "core": """
import importlib as il
from importlib import import_module as load
il.import_module('demo.high')
load('.high', __package__)
__import__('demo.high')
load(variable)
"""
        },
    )
    assert [v.rule for v in inspect(policy).violations] == [
        "edge",
        "edge",
        "edge",
        "dynamic-import",
    ]
    known = modules(policy)
    assert {r.target for r in imports(known["demo.core"], known, policy)} >= {"demo.high"}


def test_module_constants_resolve_but_local_shadowing_stays_dynamic(tmp_path: Path) -> None:
    policy = _project(
        tmp_path,
        {
            "core": """
import importlib
BASE = 'demo.high'
importlib.import_module(BASE)
def shadow(BASE):
    return importlib.import_module(BASE)
"""
        },
    )
    assert [v.rule for v in inspect(policy).violations] == ["edge", "dynamic-import"]


def test_function_import_aliases_do_not_overwrite_the_enclosing_loader(tmp_path: Path) -> None:
    policy = _project(
        tmp_path,
        {
            "core": """
from importlib import import_module as load
def unrelated():
    from json import loads as load
def sibling():
    return load('demo.high')
"""
        },
    )
    assert [(v.target, v.scope) for v in inspect(policy).violations] == [("demo.high", "sibling")]


def test_a_local_loader_does_not_become_a_loader_in_sibling_functions(tmp_path: Path) -> None:
    policy = _project(
        tmp_path,
        {
            "core": """
def local():
    from importlib import import_module as load
def sibling(load):
    return load('demo.high')
"""
        },
    )
    assert inspect(policy).violations == ()


def test_class_aliases_are_not_lexical_bindings_of_methods_or_module_siblings(
    tmp_path: Path,
) -> None:
    policy = _project(
        tmp_path,
        {
            "core": """
from importlib import import_module as load
class Example:
    from json import loads as load
    def method(self):
        return load('demo.high')
load('demo.high')
"""
        },
    )
    assert [(v.target, v.scope) for v in inspect(policy).violations] == [
        ("demo.high", "Example.method"),
        ("demo.high", ""),
    ]


def test_decorators_and_defaults_are_eager_even_though_the_function_body_is_lazy(tmp_path: Path):
    policy = _project(
        tmp_path,
        {
            "core": """
from importlib import import_module as load
@load('optional_lib').decorate
def deferred(value=load('optional_lib'), *, keyword=load('optional_lib')):
    return load('optional_lib')
"""
        },
    )
    failures = inspect(policy).violations
    assert len(failures) == 3
    assert {v.rule for v in failures} == {"optional-eager"}
    assert {v.scope for v in failures} == {""}


def test_nested_defaults_use_the_enclosing_function_scope_and_execution_time(
    tmp_path: Path,
) -> None:
    policy = _project(
        tmp_path,
        {
            "core": """
from importlib import import_module as load
def outer():
    def inner(value=load('optional_lib')):
        return value
    return load('demo.high')
"""
        },
    )
    known = modules(policy)
    calls = [r for r in imports(known["demo.core"], known, policy) if r.target == "optional_lib"]
    assert [(r.scope, r.lazy) for r in calls] == [("outer", True)]
    assert [(v.target, v.scope) for v in inspect(policy).violations] == [("demo.high", "outer")]


def test_imports_in_nested_function_bodies_do_not_bind_the_outer_scope(tmp_path: Path) -> None:
    policy = _project(
        tmp_path,
        {
            "core": """
from importlib import import_module as load
def outer():
    def inner():
        from json import loads as load
    return load('demo.high')
"""
        },
    )
    assert [(v.target, v.scope) for v in inspect(policy).violations] == [("demo.high", "outer")]


def _bounded(root: Path, caller: str = "demo.core", targets=None):
    raw = _raw()
    raw["dynamic"] = [
        {
            "caller": caller,
            "scope": "load",
            "statement": "importlib.import_module(name)",
            "targets": targets or ["demo.high"],
            "reason": "Reviewed synthetic site.",
        }
    ]
    return policy_from(root, raw)


def test_reviewed_dynamic_targets_still_obey_the_ordinary_graph(tmp_path: Path) -> None:
    _project(
        tmp_path,
        {
            "core": """
import importlib
def load(name):
    return importlib.import_module(name)
"""
        },
    )
    report = inspect(_bounded(tmp_path))
    assert [(v.target, v.rule) for v in report.violations] == [("demo.high", "edge")]
    policy = _bounded(tmp_path, targets=["optional_lib"])
    assert inspect(policy).violations == ()


def test_dynamic_declarations_are_exact_and_cannot_be_reused_after_a_site_moves(
    tmp_path: Path,
) -> None:
    _project(tmp_path, {"core": ""})
    with pytest.raises(ContractError, match="Unused dynamic import"):
        inspect(_bounded(tmp_path))


def test_internal_dynamic_prefixes_expand_to_every_source_owner(tmp_path: Path) -> None:
    _project(
        tmp_path,
        {
            "cppstd": """
import importlib
def load(name):
    return importlib.import_module(name)
"""
        },
    )
    raw = _raw()
    raw["dynamic"] = [
        {
            "caller": "demo.cppstd",
            "scope": "load",
            "statement": "importlib.import_module(name)",
            "prefixes": ["demo.high"],
            "reason": "A bound cannot permit cppstd to import higher modules.",
        }
    ]
    assert [v.rule for v in inspect(policy_from(tmp_path, raw)).violations] == ["edge"]
    raw["dynamic"][0]["prefixes"] = ["foreign"]
    with pytest.raises(ContractError, match="finite literal targets"):
        inspect(policy_from(tmp_path, raw))


def test_syntax_errors_are_reported_as_contract_failures(tmp_path: Path) -> None:
    policy = _project(tmp_path, {"core": "def broken(:\n"})
    with pytest.raises(ContractError, match=r"Cannot inspect src/demo/core.py"):
        inspect(policy)


def test_the_debt_ratchet_ignores_line_moves_but_counts_duplicate_imports(tmp_path: Path) -> None:
    policy = _project(tmp_path, {"core": "from .high import VALUE\n"})
    path = _review(tmp_path, policy, inspect(policy))
    source = tmp_path / "src/demo/core.py"
    source.write_text("\n\nfrom .high import VALUE\n")
    assert ratchet(policy, inspect(policy), path) == ({}, {})
    source.write_text("from .high import VALUE\nfrom .high import VALUE\n")
    added, removed = ratchet(policy, inspect(policy), path)
    assert sum(added.values()) == 1 and not removed
    source.write_text("")
    added, removed = ratchet(policy, inspect(policy), path)
    assert not added and sum(removed.values()) == 1


def test_unreviewed_or_policy_mismatched_debt_never_becomes_an_allowance(tmp_path: Path) -> None:
    policy = _project(tmp_path, {})
    path = tmp_path / "debt.json"
    path.write_text(json.dumps(inventory(policy, inspect(policy))))
    with pytest.raises(ContractError, match="explicit source review"):
        ratchet(policy, inspect(policy), path)
    path = _review(tmp_path, policy, inspect(policy))
    raw = _raw()
    raw["ownership"]["core"]["hard"] = ["numpy"]
    changed = policy_from(tmp_path, raw)
    with pytest.raises(ContractError, match="different component policy"):
        ratchet(changed, inspect(changed), path)


def _providers(tmp_path: Path, declaration: str):
    _project(
        tmp_path,
        {
            "plugins": declaration,
            "core": """
from .plugins import lookup
def draw():
    return lookup('draw', 'default')
""",
            "high": "def render():\n    pass\n",
        },
    )
    raw = _raw()
    raw["extensions"] = [
        {"caller": "demo.core", "point": "draw", "provider": "high", "target": "demo.high:render"}
    ]
    return policy_from(tmp_path, raw)


@pytest.mark.parametrize(
    "entry",
    [
        "('high', 'demo.high', 'render')",
        "{'component': 'high', 'module': 'demo.high', 'attribute': 'render'}",
    ],
)
def test_declared_provider_dispatch_does_not_grant_a_direct_import(tmp_path: Path, entry) -> None:
    policy = _providers(tmp_path, "PROVIDERS = {('draw', 'default'): " + entry + "}\n")
    assert inspect(policy).violations == ()
    with (tmp_path / "src/demo/core.py").open("a") as stream:
        stream.write("from .high import render\n")
    assert [v.rule for v in inspect(policy).violations] == ["edge"]
    assert "high" not in policy.closure("core")


def test_provider_declarations_cannot_mislabel_or_dynamically_resolve_targets(
    tmp_path: Path,
) -> None:
    policy = _providers(
        tmp_path, "PROVIDERS = {('draw', 'default'): ('core','demo.high','render')}\n"
    )
    assert {v.rule for v in inspect(policy).violations} == {"provider-owner", "provider-edge"}
    (tmp_path / "src/demo/plugins.py").write_text("PROVIDERS = make_table()\n")
    assert {v.rule for v in inspect(policy).violations} == {
        "provider-dynamic",
        "provider-point-undeclared",
    }


def test_extension_policy_never_allows_cppstd_to_dispatch_upwards(tmp_path: Path) -> None:
    raw = _raw()
    raw["extensions"] = [
        {"caller": "demo.cppstd", "point": "draw", "provider": "high", "target": "demo.high:render"}
    ]
    with pytest.raises(ContractError, match="cppstd cannot dispatch"):
        policy_from(tmp_path, raw)


def test_only_the_registry_lookup_may_import_declared_provider_targets(tmp_path: Path) -> None:
    policy = _providers(
        tmp_path,
        """
import importlib
PROVIDERS: dict = {('draw', 'default'): ('high','demo.high','render')}
def lookup(point, key='default'):
    _, module, attribute = PROVIDERS[point, key]
    return getattr(importlib.import_module(module), attribute)
""",
    )
    assert inspect(policy).violations == ()
    with (tmp_path / "src/demo/plugins.py").open("a") as stream:
        stream.write("def escape(module):\n    return importlib.import_module(module)\n")
    assert [v.rule for v in inspect(policy).violations] == ["dynamic-import"]


def test_facade_dynamic_targets_come_from_the_literal_export_table(tmp_path: Path) -> None:
    source = """
from importlib import import_module as _import_module
_EXPORTS: dict = {name: (module, name) for module, names in {'high': ('VALUE',)}.items()
                 for name in names}
_EXPORTS.update({name: (name, '') for name in ('core',)})
def __getattr__(name):
    module, _ = _EXPORTS[name]
    return _import_module(f'.{module}', __name__)
"""
    policy = _project(tmp_path, {"__init__": source})
    assert facade_targets(source, "demo") == ["demo.core", "demo.high"]
    assert inspect(policy).violations == ()
    with pytest.raises(ContractError, match="literal export-origin"):
        facade_targets("_EXPORTS = make_exports()", "demo")


def test_graphs_show_components_with_no_modules_without_claiming_isolation(tmp_path: Path) -> None:
    policy = _project(tmp_path, {})
    report = inspect(policy)
    assert "cppstd: 1 modules; depends on none" in graph(policy, report)
    assert '"high" -> "core";' in graph(policy, report, True)
    assert '"app" -> "cppstd";' in graph(policy, report, True)


def test_isolation_bootstrap_exception_never_allows_app_siblings_or_transport(
    tmp_path: Path,
) -> None:
    boundary = Boundary(_project(tmp_path, {}), "cppstd")
    boundary.check("demo", True)
    boundary.check("numpy", True)
    with pytest.raises(ImportError, match=r"demo.cli"):
        boundary.check("demo.cli", False)
    with pytest.raises(ImportError, match="xrdclient"):
        boundary.check("xrdclient", True)
    with pytest.raises(ContractError, match="Unknown component"):
        isolate(_project(tmp_path / "other", {}), ["absent"], [])


def test_fresh_process_isolation_catches_parent_and_function_local_leaks(tmp_path: Path) -> None:
    _project(tmp_path, {"cppstd": "VALUE = 1\n"})
    _toml(tmp_path)
    policy = load_policy(tmp_path)
    assert isolate(policy, ["cppstd"], []) == 0
    (tmp_path / "src/demo/__init__.py").write_text("from . import high\n")
    assert isolate(policy, ["cppstd"], []) == 1
    (tmp_path / "src/demo/__init__.py").write_text("")
    (tmp_path / "src/demo/cppstd.py").write_text("def leak():\n    from . import high\nleak()\n")
    assert isolate(policy, ["cppstd"], []) == 1


def test_cli_inventory_requires_review_and_cannot_overwrite_evidence(
    tmp_path: Path, capsys
) -> None:
    _project(tmp_path, {"core": "from . import high\n"})
    _toml(tmp_path)
    argv = ["--root", str(tmp_path), "check"]
    assert main([*argv, "--inventory", str(tmp_path / "debt.json")]) == 0
    assert json.loads((tmp_path / "debt.json").read_text())["reviewed"] is False
    assert main(argv) == 2
    assert "needs explicit source review" in capsys.readouterr().out
    assert main([*argv, "--inventory", str(tmp_path / "debt.json")]) == 2
    assert main([*argv, "--strict"]) == 1
    assert main(["--root", str(tmp_path), "graph", "--dot"]) == 0


def _scope(root: Path, source: str = "from demo.cppstd import value\n") -> dict:
    directory = root / "tests/pure"
    directory.mkdir(parents=True)
    (directory / "test_value.py").write_text(source)
    (directory / "conftest.py").write_text("import pytest\n")
    raw = _raw()
    raw["tests"] = {
        "generic": {
            "components": ["cppstd"],
            "paths": ["tests/pure"],
            "support": ["tests/pure/conftest.py"],
            "confcutdir": "tests/pure",
            "blocked": ["transport"],
        }
    }
    return raw


def test_declared_component_tests_use_their_explicit_fixture_cutoff(tmp_path: Path) -> None:
    from tools.component_contract.isolation import _pytest_args
    from tools.component_contract.testscope import select

    _project(tmp_path, {})
    policy = policy_from(tmp_path, _scope(tmp_path))
    selected = select(policy, "cppstd", ["generic"])
    assert selected.paths == ("tests/pure",) and selected.blocked == ("transport",)
    assert _pytest_args(policy, selected) == [
        "-q",
        "-p",
        "no:cacheprovider",
        "--confcutdir=" + str(tmp_path / "tests/pure"),
        "tests/pure",
    ]
    with pytest.raises(ContractError, match="declared scope names"):
        select(policy, "cppstd", ["tests/pure"])


@pytest.mark.parametrize(
    "source",
    [
        "import demo.high\n",
        "import transport\n",
        "import importlib\nimportlib.import_module(name)\n",
    ],
)
def test_pure_test_support_cannot_import_higher_or_undeclared_modules(tmp_path, source):
    _project(tmp_path, {})
    with pytest.raises(ContractError, match="outside its declared closure"):
        policy_from(tmp_path, _scope(tmp_path, source))


@pytest.mark.parametrize(
    "change, message",
    [("fixtures", "Every focused conftest"), ("escape", "escapes"), ("closure", "outside")],
)
def test_test_scope_paths_fixtures_and_components_are_checked(tmp_path, change, message):
    from tools.component_contract.testscope import select

    _project(tmp_path, {})
    raw = _scope(tmp_path)
    scope = raw["tests"]["generic"]
    if change == "fixtures":
        scope["support"] = []
    elif change == "escape":
        scope["paths"] = ["../elsewhere"]
    else:
        scope["components"] = ["high"]
    with pytest.raises(ContractError, match=message):
        policy = policy_from(tmp_path, raw)
        select(policy, "cppstd", ["generic"])


def _alias_policy(root: Path):
    _project(root, {})
    raw = _raw()
    raw["module_aliases"] = {
        "demo.high": {
            "target": "demo.cppstd",
            "reason": "Preserve the legacy module and globals dictionary identity.",
        }
    }
    return policy_from(root, raw)


def test_reviewed_module_alias_requires_exact_loaded_target_identity_and_origin(tmp_path):
    from types import SimpleNamespace

    from tools.component_contract.modulealiases import verify_alias

    policy = _alias_policy(tmp_path)
    target = SimpleNamespace(__file__=str(tmp_path / "src/demo/cppstd.py"))
    verify_alias(policy, "demo.high", target, {"demo.cppstd": target})
    with pytest.raises(ImportError, match="not the loaded"):
        verify_alias(policy, "demo.high", target, {"demo.cppstd": SimpleNamespace()})
    target.__file__ = str(tmp_path / "elsewhere.py")
    with pytest.raises(ImportError, match="another source tree"):
        verify_alias(policy, "demo.high", target, {"demo.cppstd": target})


def test_undeclared_module_alias_does_not_bypass_source_origin_validation(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from tools.component_contract.isolation import _load_source

    policy = _project(tmp_path, {})
    target = SimpleNamespace(__file__=str(tmp_path / "src/demo/cppstd.py"))
    monkeypatch.setattr(
        "tools.component_contract.isolation.importlib.import_module", lambda n: target
    )
    with pytest.raises(ImportError, match="another source tree"):
        _load_source(policy, "demo.high", tmp_path / "src/demo/high.py")


@pytest.mark.parametrize(
    "change, message",
    [
        ("upward", "downward"),
        ("missing", "concrete"),
        ("chain", "chain"),
        ("unreviewed", "review reason"),
    ],
)
def test_module_alias_policy_rejects_upward_missing_chained_and_unreviewed_targets(
    tmp_path, change, message
):
    _project(tmp_path, {})
    raw = _raw()
    raw["module_aliases"] = {"demo.high": {"target": "demo.cppstd", "reason": "Reviewed alias."}}
    if change == "upward":
        raw["module_aliases"] = {
            "demo.cppstd": {"target": "demo.high", "reason": "Invalid upward."}
        }
    elif change == "missing":
        raw["module_aliases"]["demo.high"]["target"] = "demo.missing"
    elif change == "chain":
        raw["module_aliases"]["demo.cppstd"] = {"target": "demo.core", "reason": "Invalid chain."}
    else:
        raw["module_aliases"]["demo.high"]["reason"] = ""
    with pytest.raises(ContractError, match=message):
        policy_from(tmp_path, raw)


def _private(root: Path, source: str | None = None):
    _project(
        root,
        {
            "core": source
            if source is not None
            else """
def calculation():
    from optional_lib._core import Engine
    return Engine()
"""
        },
    )
    raw = _raw()
    raw["private_imports"] = [
        {
            "caller": "demo.core",
            "scope": "calculation",
            "target": "optional_lib._core",
            "statement": "from optional_lib._core import Engine",
            "reason": "The public wrapper changes the established numerical calculation.",
        }
    ]
    return policy_from(root, raw)


def test_exact_reviewed_private_imports_remain_visible_in_the_report_and_inventory(tmp_path):
    policy = _private(tmp_path)
    report = inspect(policy)
    assert report.violations == ()
    assert [(v.target, v.rule, v.scope) for v in report.reviewed_private] == [
        ("optional_lib._core", "external-private", "calculation")
    ]
    saved = inventory(policy, report)
    assert saved["reviewed_private_imports"][0]["target"] == "optional_lib._core"
    assert saved["violations"] == []


def test_reviewing_one_private_module_does_not_allow_another_in_the_same_package(tmp_path):
    policy = _private(tmp_path)
    with (tmp_path / "src/demo/core.py").open("a") as stream:
        stream.write("\ndef another():\n    from optional_lib._other import Other\n")
    report = inspect(policy)
    assert [(v.target, v.rule) for v in report.violations] == [
        ("optional_lib._other", "external-private")
    ]
    assert len(report.reviewed_private) == 1


@pytest.mark.parametrize(
    "source",
    [
        "",
        """
def renamed():
    from optional_lib._core import Engine
""",
        """
def calculation():
    from optional_lib._core import Replacement
""",
        """
def calculation():
    from optional_lib._new import Engine
""",
        """
def calculation():
    from optional_lib._core import Engine
    from optional_lib._core import Engine
""",
    ],
)
def test_stale_moved_changed_and_duplicate_private_approvals_are_rejected(tmp_path, source):
    policy = _private(tmp_path, source)
    with pytest.raises(ContractError, match="exactly one current import"):
        inspect(policy)


def test_private_import_review_does_not_approve_an_eager_or_undeclared_dependency(tmp_path):
    policy = _private(tmp_path)
    raw = _raw()
    raw["private_imports"] = [
        {
            "caller": "demo.core",
            "scope": "",
            "statement": "from optional_lib._core import Engine",
            "target": "optional_lib._core",
            "reason": "Review cannot change optional dependency policy.",
        }
    ]
    (tmp_path / "src/demo/core.py").write_text("from optional_lib._core import Engine\n")
    policy = policy_from(tmp_path, raw)
    assert [v.rule for v in inspect(policy).violations] == ["optional-eager"]
    raw["ownership"]["core"]["optional"] = []
    assert [v.rule for v in inspect(policy_from(tmp_path, raw)).violations] == [
        "external-undeclared"
    ]


@pytest.mark.parametrize(
    "statement",
    [
        "run()",
        "from optional_lib._core import Engine; run()",
        "from optional_lib._core import  Engine",
    ],
)
def test_reviewed_private_imports_require_one_canonical_import_statement(tmp_path, statement):
    _project(tmp_path, {})
    raw = _raw()
    raw["private_imports"] = [
        {
            "caller": "demo.core",
            "scope": "",
            "target": "optional_lib._core",
            "statement": statement,
            "reason": "An exact source statement is required.",
        }
    ]
    with pytest.raises(ContractError, match=r"import statement|AST spelling"):
        policy_from(tmp_path, raw)


def _registration(root: Path, source: str | None = None):
    _project(
        root,
        {
            "cppstd": "def install(callback):\n    pass\n",
            "core": source
            or """
from .cppstd import install as register
def callback(value):
    return value
register(callback)
""",
        },
    )
    return policy_from(root, _registration_raw())


def _registration_raw():
    raw = _raw()
    raw["registration_functions"] = {"demo.cppstd.install": {"position": 0, "keyword": "callback"}}
    raw["registrations"] = [
        {
            "caller": "demo.core",
            "scope": "",
            "statement": "register(callback)",
            "callee": "demo.cppstd.install",
            "callback": "demo.core:callback",
            "reason": "A source-owned adapter supplies the explicit callback.",
        }
    ]
    return raw


def test_registration_aliases_and_callbacks_are_recorded_without_an_upward_import_allowance(
    tmp_path,
):
    policy = _registration(tmp_path)
    report = inspect(policy)
    assert report.violations == ()
    assert [(r.module, r.point, r.target) for r in report.registrations] == [
        ("demo.core", "demo.cppstd.install", "demo.core:callback")
    ]
    assert inventory(policy, report)["registrations"][0]["statement"] == "register(callback)"
    assert policy.closure("cppstd") == {"cppstd"}


@pytest.mark.parametrize(
    "call", ["register(callback)", "alias(callback)", "register(callback=callback)"]
)
def test_new_registration_calls_remain_violations_even_for_an_already_reviewed_target(
    tmp_path, call
):
    policy = _registration(tmp_path)
    with (tmp_path / "src/demo/core.py").open("a") as stream:
        stream.write("\ndef extra():\n    alias = register\n    " + call + "\n")
    report = inspect(policy)
    assert [(v.rule, v.scope) for v in report.violations] == [("registration-undeclared", "extra")]


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "from .cppstd import install as register\ndef callback(value):\n    return value\n",
            "Registration demo.core: must match exactly one current call; "
            "shrink stale declarations.",
        ),
        (
            "from .cppstd import install as register\ndef callback(value):\n    return value\n"
            "def moved():\n    register(callback)\n",
            "Registration demo.core: must match exactly one current call; "
            "shrink stale declarations.",
        ),
        (
            "from .cppstd import install as register\ndef replacement(value):\n    return value\n"
            "register(replacement)\n",
            "Registration target demo.core:callback has no source callable definition.",
        ),
        (
            "from .cppstd import install as register\ndef callback(value):\n    return value\n"
            "register(callback)\nregister(callback)\n",
            "Registration demo.core: must match exactly one current call; "
            "shrink stale declarations.",
        ),
    ],
)
def test_stale_moved_changed_and_duplicate_registration_declarations_fail(
    tmp_path, source, expected
):
    policy = _registration(tmp_path, source)
    with pytest.raises(ContractError) as error:
        inspect(policy)
    assert str(error.value) == expected


def test_registration_callback_ownership_and_concrete_definitions_are_checked(tmp_path):
    _registration(tmp_path)
    raw = _registration_raw()
    raw["registrations"][0]["callback"] = "demo.core:absent"
    with pytest.raises(ContractError, match="no source callable"):
        inspect(policy_from(tmp_path, raw))
    (tmp_path / "src/demo/high.py").write_text("def callback(value):\n    return value\n")
    raw["registrations"][0].update(caller="demo.cppstd", callback="demo.high:callback")
    with pytest.raises(ContractError, match="downward closure"):
        inspect(policy_from(tmp_path, raw))


def test_registration_function_metadata_must_match_its_real_source_signature(tmp_path):
    _registration(tmp_path)
    raw = _registration_raw()
    raw["registration_functions"]["demo.cppstd.install"]["position"] = 1
    with pytest.raises(ContractError, match="different callback signature"):
        inspect(policy_from(tmp_path, raw))
    raw["registration_functions"] = {"demo.core.callback": {"position": 0, "keyword": "value"}}
    with pytest.raises(ContractError, match="must belong to cppstd"):
        inspect(policy_from(tmp_path, raw))


def test_scoped_registration_aliases_do_not_leak_to_sibling_functions(tmp_path):
    policy = _registration(tmp_path)
    with (tmp_path / "src/demo/core.py").open("a") as stream:
        stream.write("""
def local():
    from .cppstd import install as temporary
    temporary(callback)
def sibling(temporary):
    temporary(callback)
""")
    report = inspect(policy)
    assert [(v.rule, v.scope) for v in report.violations] == [("registration-undeclared", "local")]


def test_pure_component_tests_can_inject_callbacks_through_a_declared_downward_api(tmp_path):
    _project(tmp_path, {"cppstd": "def install(callback):\n    pass\n"})
    raw = _scope(tmp_path, "from demo.cppstd import install\ninstall(lambda value: value)\n")
    raw["registration_functions"] = {"demo.cppstd.install": {"position": 0, "keyword": "callback"}}
    policy = policy_from(tmp_path, raw)
    assert inspect(policy).violations == ()


def test_registration_declarations_cannot_duplicate_a_site_under_different_review_reasons(tmp_path):
    _registration(tmp_path)
    raw = _registration_raw()
    raw["registrations"].append(
        dict(raw["registrations"][0], reason="Another review is still duplicate.")
    )
    with pytest.raises(ContractError, match="Duplicate registration"):
        policy_from(tmp_path, raw)


def _literal_facade(submodules: str) -> str:
    return (
        """
from importlib import import_module as _import_module
_EXPORTS: dict = {name: (module, name) for module, names in {'core': ('VALUE',)}.items()
                 for name in names}
_SUBMODULES = """
        + submodules
        + """
def __getattr__(name):
    if name in _SUBMODULES:
        return _import_module(f'.{name}', __name__)
    module, _ = _EXPORTS[name]
    return _import_module(f'.{module}', __name__)
"""
    )


def test_literal_facade_submodules_are_checked_against_the_same_component_graph(tmp_path):
    source = _literal_facade("frozenset({'core', 'high'})")
    _project(tmp_path, {"__init__": source})
    raw = _raw()
    raw["ownership"]["app"]["depends"] = ["core", "cppstd"]
    report = inspect(policy_from(tmp_path, raw))
    assert facade_targets(source, "demo") == ["demo.core", "demo.high"]
    assert [(v.target, v.rule) for v in report.violations] == [
        ("demo.high", "edge"),
        ("demo.high", "edge"),
    ]


@pytest.mark.parametrize(
    "expression",
    ["frozenset(names)", "frozenset({'foreign.module'})", "frozenset({'core', 1})", "discover()"],
)
def test_facade_submodule_declarations_reject_nonliteral_or_foreign_names(expression):
    with pytest.raises(ContractError, match="literal"):
        facade_targets(_literal_facade(expression), "demo")


def test_unknown_literal_facade_submodules_remain_missing_target_violations(tmp_path):
    source = _literal_facade("frozenset({'absent'})")
    policy = _project(tmp_path, {"__init__": source})
    report = inspect(policy)
    assert [(v.target, v.rule) for v in report.violations] == [
        ("demo.absent", "internal-missing"),
        ("demo.absent", "internal-missing"),
    ]


def _forwarding_registry() -> str:
    return """
PROVIDERS = {('draw', 'default'): ('high', 'demo.high', 'render')}
def lookup(point: str, key: str = 'default'):
    return PROVIDERS.get((point, key))
def require(point: str, key: str = 'default'):
    "Resolve a required target."
    found = lookup(point, key)
    if found is None:
        raise UnsupportedFeatureError(f"No provider is registered for {point!r} with key {key!r}.")
    return found
"""


def _forwarded_project(root: Path):
    policy = _providers(root, _forwarding_registry())
    (root / "src/demo/core.py").write_text(
        "from .plugins import require\ndef draw():\n    return require('draw')\n"
    )
    return policy


def test_registry_require_forwards_only_its_own_point_and_key_to_declared_lookup(tmp_path):
    policy = _forwarded_project(tmp_path)
    report = inspect(policy)
    assert report.violations == ()
    assert [(r.module, r.scope, r.statement, r.provider) for r in report.forwarders] == [
        ("demo.plugins", "require", "lookup(point, key)", "demo.plugins.lookup")
    ]
    assert inventory(policy, report)["forwarders"][0]["statement"] == "lookup(point, key)"
    assert "high" not in policy.closure("core")


@pytest.mark.parametrize(
    "replacement",
    [
        "found = lookup_loaded(point, key)",
        "found = undeclared(point, key)",
        "found = lookup('draw', key)",
        "found = lookup(point, key)\n    lookup(point, key)",
    ],
)
def test_changed_forwarding_targets_arguments_and_extra_dispatch_are_rejected(
    tmp_path, replacement
):
    policy = _forwarded_project(tmp_path)
    path = tmp_path / "src/demo/plugins.py"
    path.write_text(path.read_text().replace("found = lookup(point, key)", replacement))
    with pytest.raises(ContractError, match="reviewed point/key forwarding body"):
        inspect(policy)


def test_registry_forwarding_requires_the_declared_lookup_signature_and_stable_binding(tmp_path):
    policy = _forwarded_project(tmp_path)
    raw = _raw()
    raw["provider_functions"] = ["demo.plugins.require"]
    raw["extensions"] = [
        {"caller": "demo.core", "point": "draw", "provider": "high", "target": "demo.high:render"}
    ]
    with pytest.raises(ContractError, match="declared lookup"):
        inspect(policy_from(tmp_path, raw))
    path = tmp_path / "src/demo/plugins.py"
    source = path.read_text()
    path.write_text(
        source.replace("def lookup(point: str, key: str = 'default'):", "def lookup(point: str):")
    )
    with pytest.raises(ContractError, match="lookup/require signature"):
        inspect(policy)
    path.write_text(source + "\nlookup = other\n")
    with pytest.raises(ContractError, match="reviewed point/key forwarding body"):
        inspect(policy)


def test_require_forwarding_keeps_external_caller_and_dynamic_point_checks_active(tmp_path):
    policy = _forwarded_project(tmp_path)
    path = tmp_path / "src/demo/core.py"
    path.write_text("from .plugins import require\ndef draw(point):\n    return require(point)\n")
    assert [(v.scope, v.rule) for v in inspect(policy).violations] == [
        ("draw", "provider-point-undeclared")
    ]
    path.write_text("from .plugins import require\ndef draw():\n    return require('missing')\n")
    assert [v.rule for v in inspect(policy).violations] == ["provider-point-undeclared"]
    path.write_text("from .plugins import require\ndef draw():\n    return require('draw')\n")
    raw = _raw()
    raw["extensions"] = [
        {"caller": "demo.high", "point": "draw", "provider": "high", "target": "demo.high:render"}
    ]
    assert [v.rule for v in inspect(policy_from(tmp_path, raw)).violations] == ["provider-edge"]


def test_other_registry_functions_do_not_gain_dynamic_point_forwarding_permission(tmp_path):
    policy = _forwarded_project(tmp_path)
    path = tmp_path / "src/demo/plugins.py"
    path.write_text(path.read_text() + "\ndef escape(point, key):\n    return lookup(point, key)\n")
    assert [(v.scope, v.rule) for v in inspect(policy).violations] == [
        ("escape", "provider-point-undeclared")
    ]



@pytest.mark.parametrize("source", [
    "class lookup: pass", "class require: pass",
    "async def lookup(point, key='default'): pass",
    "async def require(point, key='default'): pass",
    "try: risky()\nexcept Exception as lookup: pass",
    "match value:\n    case {'item': lookup}: pass",
    "match value:\n    case [*lookup]: pass",
    "match value:\n    case {'item': captured, **require}: pass",
    "def helper(value=(lookup := other)): pass",
    "class Helper((lookup := other)): pass",
    "@(require := decorator)\ndef helper(): pass",
    "from .other import *",
    "values = [(lookup := other) for item in items]",
    "class Helper:\n    global lookup\n    lookup = other",
])
def test_registry_forwarding_rejects_every_competing_module_binding_form(tmp_path, source):
    policy = _forwarded_project(tmp_path)
    path = tmp_path / "src/demo/plugins.py"
    path.write_text(path.read_text() + "\n" + source + "\n")
    with pytest.raises(ContractError, match="reviewed point/key forwarding body"):
        inspect(policy)


@pytest.mark.parametrize("source", [
    "def helper():\n    lookup = None\n    return lookup",
    "class Helper:\n    lookup = None\n    def method(self):\n"
    "        require = None\n        return require",
    "values = [lookup for lookup in ()]",
    "values = {require: require for require in ()}",
    "helper = lambda lookup: lookup",
])
def test_registry_binding_validation_preserves_local_function_class_and_comprehension_scopes(
        tmp_path, source):
    policy = _forwarded_project(tmp_path)
    path = tmp_path / "src/demo/plugins.py"
    path.write_text(path.read_text() + "\n" + source + "\n")
    assert inspect(policy).violations == ()


def _rewritten_scope(root: Path, source: str, production: str):
    _project(root, {"cppstd": production})
    _toml(root)
    _scope(root, source)
    with (root / "pyproject.toml").open("a") as stream:
        stream.write("""
[tool.xrdroot.components.tests.generic]
components = ["cppstd"]
paths = ["tests/pure"]
support = ["tests/pure/conftest.py"]
confcutdir = "tests/pure"
blocked = ["transport"]
""")
    return load_policy(root)


def test_fresh_component_tests_run_with_default_pytest_assertion_rewriting(tmp_path, capfd):
    policy = _rewritten_scope(
        tmp_path,
        "from demo.cppstd import VALUE\n\ndef test_value():\n    assert VALUE == 7\n",
        "VALUE = 7\n",
    )
    assert isolate(policy, ["cppstd"], ["generic"]) == 0
    assert "1 passed" in capfd.readouterr().out


def test_rewritten_tests_cannot_grant_production_a_cached_private_pytest_import(tmp_path, capfd):
    policy = _rewritten_scope(
        tmp_path,
        "from demo.cppstd import leak\n\ndef test_value():\n    assert leak() is None\n",
        "def leak():\n    import _pytest.assertion.rewrite\n",
    )
    assert isolate(policy, ["cppstd"], ["generic"]) == 1
    output = capfd.readouterr()
    message = "Component isolation blocked external import _pytest.assertion.rewrite."
    assert message in output.out + output.err
    assert "1 failed" in output.out


def _rewrite_boundary(root: Path, testing: bool = True, blocked: tuple[str, ...] = ()):
    _project(root, {})
    policy = policy_from(root, _scope(root))
    return Boundary(policy, "cppstd", tests=testing, blocked=blocked)


def _import_at_file(boundary, filename, module, target, mechanism):
    from tools.component_contract.isolation import _guarded_import

    namespace = {"__name__": module, "__package__": "", "boundary": boundary}
    namespace["importer"] = _guarded_import(boundary, lambda *args: "permitted")
    operation = "importer" if mechanism == "guarded" else "boundary.find_spec"
    exec(compile("result = " + operation + "(" + repr(target) + ")", str(filename), "exec"),
         namespace)
    return namespace["result"]


@pytest.mark.parametrize("mechanism", ["guarded", "finder"])
def test_only_the_exact_rewrite_helper_is_allowed_from_an_existing_test_file(tmp_path, mechanism):
    boundary = _rewrite_boundary(tmp_path)
    result = _import_at_file(
        boundary, tmp_path / "tests/pure/test_value.py", "test_value",
        "_pytest.assertion.rewrite", mechanism,
    )
    assert result == ("permitted" if mechanism == "guarded" else None)


@pytest.mark.parametrize("mechanism", ["guarded", "finder"])
@pytest.mark.parametrize(
    "target", ["_pytest", "_pytest.assertion", "_pytest.config", "_pytest.assertion.rewrite.extra"]
)
def test_other_private_pytest_imports_remain_blocked_from_test_files(tmp_path, mechanism, target):
    boundary = _rewrite_boundary(tmp_path)
    with pytest.raises(ImportError) as error:
        _import_at_file(
            boundary, tmp_path / "tests/pure/test_value.py", "test_value", target, mechanism
        )
    assert str(error.value) == f"Component isolation blocked external import {target}."


@pytest.mark.parametrize("mechanism", ["guarded", "finder"])
@pytest.mark.parametrize("module", ["demo", "demo.cppstd"])
def test_a_test_filename_cannot_grant_the_rewrite_exception_to_production_names(
    tmp_path, mechanism, module
):
    boundary = _rewrite_boundary(tmp_path)
    with pytest.raises(ImportError) as error:
        _import_at_file(
            boundary, tmp_path / "tests/pure/test_value.py", module,
            "_pytest.assertion.rewrite", mechanism,
        )
    assert str(error.value) == (
        "Component isolation blocked external import _pytest.assertion.rewrite."
    )


def _non_test_caller(root: Path, kind: str) -> Path:
    tests = root / "tests/pure"
    paths = {
        "missing": tests / "missing.py",
        "directory": tests / "directory.py",
        "non-python": tests / "notes.txt",
        "symlink": tests / "escape.py",
        "parent-escape": tests / "../../src/demo/cppstd.py",
    }
    (tests / "directory.py").mkdir()
    (tests / "notes.txt").write_text("")
    (tests / "escape.py").symlink_to(root / "src/demo/cppstd.py")
    return paths[kind]


@pytest.mark.parametrize("mechanism", ["guarded", "finder"])
@pytest.mark.parametrize(
    "kind", ["missing", "directory", "non-python", "symlink", "parent-escape"]
)
def test_the_rewrite_exception_requires_a_real_python_file_resolved_under_tests(
    tmp_path, mechanism, kind
):
    boundary = _rewrite_boundary(tmp_path)
    filename = _non_test_caller(tmp_path, kind)
    with pytest.raises(ImportError) as error:
        _import_at_file(boundary, filename, "test_value", "_pytest.assertion.rewrite", mechanism)
    assert str(error.value) == (
        "Component isolation blocked external import _pytest.assertion.rewrite."
    )


@pytest.mark.parametrize("mechanism", ["guarded", "finder"])
def test_the_rewrite_exception_is_unavailable_outside_test_mode(tmp_path, mechanism):
    boundary = _rewrite_boundary(tmp_path, testing=False)
    with pytest.raises(ImportError) as error:
        _import_at_file(
            boundary, tmp_path / "tests/pure/test_value.py", "demo.cppstd",
            "_pytest.assertion.rewrite", mechanism,
        )
    assert str(error.value) == (
        "Component isolation blocked external import _pytest.assertion.rewrite."
    )


@pytest.mark.parametrize("mechanism", ["guarded", "finder"])
def test_an_explicitly_blocked_rewrite_helper_stays_blocked_for_real_test_files(
    tmp_path, mechanism
):
    boundary = _rewrite_boundary(tmp_path, blocked=("_pytest.assertion.rewrite",))
    with pytest.raises(ImportError) as error:
        _import_at_file(
            boundary, tmp_path / "tests/pure/test_value.py", "test_value",
            "_pytest.assertion.rewrite", mechanism,
        )
    assert str(error.value) == "Component test scope blocked _pytest.assertion.rewrite."


def test_handwritten_private_pytest_imports_remain_outside_the_declared_test_closure(tmp_path):
    _project(tmp_path, {})
    with pytest.raises(ContractError, match="outside its declared closure"):
        policy_from(tmp_path, _scope(tmp_path, "import _pytest.assertion.rewrite\n"))
