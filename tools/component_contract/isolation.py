"""Run imports behind a fresh-process boundary, with no cached package escape."""

from __future__ import annotations

import builtins
import importlib
import importlib.abc
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import FrameType
from typing import Any

from .imports import modules
from .model import Policy, load_policy
from .modulealiases import verify_alias, verify_origin
from .testscope import Selection, select


class Boundary(importlib.abc.MetaPathFinder):
    def __init__(
        self, policy: Policy, component: str, tests: bool = False, blocked: tuple[str, ...] = ()
    ):
        self.policy = policy
        self.allowed = policy.closure(component)
        self.external = {
            name
            for c in self.allowed
            for name in policy.components[c].hard + policy.components[c].optional
        }
        self.testing = tests
        self.blocked = blocked

    def internal(self, fullname: str) -> None:
        if fullname == self.policy.package:
            return  # The app-owned parent bootstrap, never other app modules.
        if self.policy.owner(fullname) not in self.allowed:
            raise ImportError(f"Component isolation blocked {fullname}.")

    def check(
        self, fullname: str, project_caller: bool, test_caller: bool = False
    ) -> None:
        self._blocked_import(fullname)
        if fullname == self.policy.package or fullname.startswith(self.policy.package + "."):
            self.internal(fullname)
            return
        if project_caller and not self._permitted_external(fullname, test_caller):
            raise ImportError(f"Component isolation blocked external import {fullname}.")

    def _blocked_import(self, fullname: str) -> None:
        if any(fullname == p or fullname.startswith(p + ".") for p in self.blocked):
            raise ImportError(f"Component test scope blocked {fullname}.")

    def _permitted_external(self, fullname: str, test_caller: bool = False) -> bool:
        root = fullname.split(".", 1)[0]
        return (
            root in self.policy.stdlib or root in self.external
            or (self.testing and root == "pytest")
            # Pytest injects this exact import into rewritten tests and conftests.
            or (self.testing and test_caller and fullname == "_pytest.assertion.rewrite")
        )

    def project(self, frame: FrameType) -> bool:
        name = str(frame.f_globals.get("__name__", ""))
        filename = Path(frame.f_code.co_filename)
        return (
            name == self.policy.package
            or name.startswith(self.policy.package + ".")
            or (self.testing and filename.is_relative_to(self.policy.root / "tests"))
        )

    def _test_file(self, frame: FrameType | None) -> bool:
        if not self.testing or frame is None:
            return False
        name = str(frame.f_globals.get("__name__", ""))
        if name == self.policy.package or name.startswith(self.policy.package + "."):
            return False
        return self._test_path(Path(frame.f_code.co_filename))

    def _test_path(self, filename: Path) -> bool:
        tests = self.policy.root / "tests"
        if filename.suffix != ".py" or not filename.is_relative_to(tests):
            return False
        resolved = filename.resolve()
        return (
            resolved.suffix == ".py" and resolved.is_relative_to(tests.resolve())
            and resolved.is_file()
        )

    def _requesting_frame(self) -> FrameType | None:
        frame = sys._getframe(2)
        while frame:
            name = str(frame.f_globals.get("__name__", ""))
            if not name.startswith(("importlib", "_frozen_importlib", __name__)):
                return frame
            frame = frame.f_back
        return None

    def requester(self) -> bool:
        frame = self._requesting_frame()
        return frame is not None and self.project(frame)

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
        frame = self._requesting_frame()
        project = frame is not None and self.project(frame)
        self.check(fullname, project, self._test_file(frame))
        return None


def _guarded_import(boundary: Boundary, original: Any) -> Any:
    def guarded(
        name: str, globals: Any = None, locals: Any = None, fromlist: Any = (), level: int = 0
    ) -> Any:
        package = (globals or {}).get("__package__", "")
        absolute = importlib.util.resolve_name("." * level + name, package) if level else name
        frame = sys._getframe(1)
        boundary.check(absolute, boundary.project(frame), boundary._test_file(frame))
        return original(name, globals, locals, fromlist, level)

    return guarded


def child_main(root: str, component: str, tests: list[str]) -> int:
    policy = load_policy(Path(root))
    known = modules(policy)
    selected = _selection(policy, known, component)
    selection = select(policy, component, tests)
    source_parent = (policy.root / policy.source).parent
    sys.path.insert(0, str(source_parent))
    boundary = Boundary(policy, component, bool(tests), selection.blocked)
    original = builtins.__import__
    builtins.__import__ = _guarded_import(boundary, original)
    sys.meta_path.insert(0, boundary)
    try:
        for name in selected:
            _load_source(policy, name, policy.root / known[name].path)
        for name in sys.modules:
            boundary.check(name, False)
        if tests:
            return _run_tests(policy, selection, boundary)
    finally:
        builtins.__import__ = original
        sys.meta_path.remove(boundary)
    print(json.dumps({"component": component, "imported": selected, "tests": tests}))
    return 0


def _run_tests(policy: Policy, selection: Selection, boundary: Boundary) -> int:
    pytest = importlib.import_module("pytest")
    code = int(pytest.main(_pytest_args(policy, selection)))
    for name in sys.modules:
        boundary.check(name, False)
    return code


def _pytest_args(policy: Policy, selection: Selection) -> list[str]:
    return [
        "-q",
        "-p",
        "no:cacheprovider",
        "--confcutdir=" + str(policy.root / str(selection.confcutdir)),
        *selection.paths,
    ]


def _selection(policy: Policy, known: dict[str, Any], component: str) -> list[str]:
    selected = sorted(n for n in known if policy.owner(n) == component)
    if not selected:
        raise ImportError(f"Component {component} has no source modules to isolate.")
    cached = [n for n in sys.modules if n == policy.package or n.startswith(policy.package + ".")]
    if cached:
        raise ImportError(f"Component isolation requires a fresh package namespace: {cached}.")
    return selected


def _load_source(policy: Policy, name: str, source: Path) -> None:
    module = importlib.import_module(name)
    if name in policy.module_aliases:
        verify_alias(policy, name, module, sys.modules)
    else:
        verify_origin(module, source, name)


def isolate(policy: Policy, components: list[str], tests: list[str]) -> int:
    tool_root = Path(__file__).resolve().parents[2]
    bootstrap = (
        "import sys; sys.path.insert(0, " + repr(str(tool_root)) + "); "
        "from tools.component_contract.isolation import child_main; "
        "raise SystemExit(child_main(sys.argv[1], sys.argv[2], sys.argv[3:]))"
    )
    status = 0
    for component in components:
        policy.closure(component)
        select(policy, component, tests)
        argv = [sys.executable, "-I", "-c", bootstrap, str(policy.root), component, *tests]
        env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME"}}
        try:
            result = subprocess.run(argv, cwd=policy.root, env=env, check=False, timeout=120)
            status = max(status, int(result.returncode != 0))
        except subprocess.TimeoutExpired:
            print(f"Component isolation of {component} exceeded 120 seconds.")
            status = 1
    return status
