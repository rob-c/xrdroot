"""API presence is measured without constructing physics objects or calling methods.

Presence is a structural measure, not a claim of behavioural parity. Only an
unconditional refusal in a Python implementation is marked refused; conditional
refusals require tutorial/runtime evidence, and remain present in this inventory.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import os
import subprocess
import textwrap
from collections import Counter, defaultdict
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Any

from .environment import Oracle

#: Libraries covering the tutorial engines; unavailable libraries remain visible as errors.
LIBRARIES = ("Core", "RIO", "Hist", "Graf", "Gpad", "Tree", "TreePlayer", "MathCore",
             "MathMore", "GenVector", "ROOTDataFrame", "ROOTNTuple", "RooFit", "RooStats",
             "HistFactory", "TMVA", "Geom")

#: Python's spelling of common C++ operators; unmapped overloads remain visible as missing.
OPERATORS = {"operator()": "__call__", "operator[]": "__getitem__", "operator+": "__add__",
             "operator-": "__sub__", "operator*": "__mul__", "operator/": "__truediv__",
             "operator+=": "__iadd__", "operator-=": "__isub__", "operator*=": "__imul__",
             "operator/=": "__itruediv__", "operator==": "__eq__", "operator!=": "__ne__",
             "operator<": "__lt__", "operator<=": "__le__", "operator>": "__gt__",
             "operator>=": "__ge__", "operator bool": "__bool__", "operator=": "_assign"}


def export(oracle: Oracle, destination: Path, libraries: list[str]) -> dict[str, Any]:
    if oracle.python is None:
        raise ValueError("The API inventory requires the oracle's PyROOT interpreter.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("api_oracle.py")
    subprocess.run([oracle.python, str(script), str(destination), *libraries], check=True,
                   env=dict(os.environ, PYTHONPATH=oracle.pythonpath))
    return json.loads(destination.read_text())


@cache
def refusal(value: Any) -> str | None:
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(value)))
    except (OSError, TypeError, IndentationError, SyntaxError):
        return None
    function = tree.body[0]
    if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    body = function.body
    statements = [node for node in body if not isinstance(node, ast.Expr)]
    if statements and isinstance(statements[0], ast.Raise) and _refusing(value, statements[0]):
        # Resolve a helper-created exception without invoking the helper or constructor.
        return ast.unparse(statements[0])
    return None


def _refusing(value: Any, raised: ast.Raise) -> bool:
    names = {node.id for node in ast.walk(raised) if isinstance(node, ast.Name)}
    if "UnsupportedFeatureError" in names or "NotImplementedError" in names:
        return True
    namespace = getattr(value, "__globals__", {})
    for name in names:
        helper = namespace.get(name)
        if inspect.isfunction(helper):
            source = inspect.getsource(helper)
            if "UnsupportedFeatureError(" in source:
                return True
    return False


def resolve(namespace: Any, name: str) -> Any:
    value = namespace
    for part in _parts(name):
        base, template, arguments = part.partition("<")
        value = getattr(value, base)
        if template:
            try:
                given = tuple(part.strip() for part in _split(arguments[:-1], ","))
                value = value[given[0] if len(given) == 1 else given]
            except (TypeError, ValueError, KeyError) as why:
                raise AttributeError(f"Unavailable template {part}") from why
    return value


def _parts(name: str) -> list[str]:
    """A scope separator inside a template argument belongs to that argument."""
    return _split(name, "::")


def _split(name: str, separator: str) -> list[str]:
    parts, start, depth = [], 0, 0
    for index, char in enumerate(name):
        depth += (char == "<") - (char == ">")
        if depth == 0 and name.startswith(separator, index):
            parts.append(name[start:index])
            start = index + len(separator)
    parts.append(name[start:])
    return parts


def presence(namespace: Any, name: str) -> tuple[str, str, Any]:
    from xrdroot.errors import UnsupportedFeatureError

    try:
        value = resolve(namespace, name)
    except AttributeError:
        return "missing", "", None
    except UnsupportedFeatureError as why:
        return "refused", str(why), None

    if isinstance(value, type):
        refused = refusal(value.__init__)
    else:
        refused = refusal(value) if callable(value) else None
    if refused:
        return "refused", refused, value
    if isinstance(value, UnsupportedFeatureError):
        return "refused", str(value), value
    return "present", "", value


def compare(inventory: Mapping[str, Any], namespace: Any) -> dict[str, Any]:
    classes = []
    for entry in inventory["classes"]:
        status, reason, cls = _class_presence(namespace, entry["name"])
        methods = []
        for name in entry["methods"]:
            attribute = _attribute(entry["name"], name)
            state, why, _ = presence(cls, attribute) if cls is not None else (status, reason, None)
            methods.append({"name": name, "status": state, "reason": why})
        classes.append({**entry, "status": status, "reason": reason, "methods": methods})
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in classes:
        grouped[entry["library"]].append(entry)
    summary = {library: totals(entries) for library, entries in sorted(grouped.items())}
    return {"schema": "xrdroot-api/1", "root_version": inventory["root_version"],
            "libraries": inventory.get("libraries", []),
            "failed_libraries": inventory.get("failed_libraries", []),
            "errors": inventory.get("errors", []), "summary": summary, "classes": classes,
            "measurement": "Structural presence; present APIs require behavioural validation."}


def _class_presence(namespace: Any, name: str) -> tuple[str, str, Any]:
    found = presence(namespace, name)
    if found[0] == "missing" and len(_parts(name)) == 1:
        # ROOT's dictionary drops std:: from canonical STL names.
        standard = presence(namespace, "std::" + name)
        if standard[0] != "missing":
            return standard
    return found


def _attribute(owner: str, method: str) -> str:
    constructor = _parts(owner)[-1].partition("<")[0]
    if method.partition("<")[0] == constructor:
        return "__init__"
    if method.startswith("~"):
        return "_destruct"
    return OPERATORS.get(method, method)


def totals(entries: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    classes = Counter(entry["status"] for entry in entries)
    methods = Counter(method["status"] for entry in entries for method in entry["methods"])
    return {"classes": dict(classes), "methods": dict(methods)}


def markdown(doc: Mapping[str, Any]) -> str:
    lines = ["# ROOT API coverage", "", doc["measurement"], "",
             f"ROOT {doc['root_version']}", "",
             "| library | classes present / refused / missing |"
             " methods present / refused / missing |",
             "|---|---|---|"]
    for library, summary in doc["summary"].items():
        cells = [" / ".join(str(summary[kind].get(status, 0)) for status in
                            ("present", "refused", "missing")) for kind in ("classes", "methods")]
        lines.append(f"| {library} | {' | '.join(cells)} |")
    lines += ["", f"Unavailable libraries: {', '.join(doc['failed_libraries']) or '(none)' }.",
              f"Dictionary errors: {len(doc['errors'])} (see api-coverage.json).", ""]
    return "\n".join(lines)


def run(args: Any) -> int:
    from .environment import find_oracle, root_config

    args.out.mkdir(parents=True, exist_ok=True)
    if args.inventory:
        inventory = json.loads(args.inventory.read_text())
    else:
        oracle = find_oracle(root_config(args.oracle_prefix), args.root_python)
        if oracle is None:
            raise ValueError("The API inventory requires a ROOT oracle installation.")
        inventory = export(oracle, args.out / "api-oracle.json", args.libraries)
    doc = compare(inventory, importlib.import_module("xrdroot.pyroot"))
    (args.out / "api-coverage.json").write_text(json.dumps(doc, indent=1) + "\n")
    (args.out / "api-coverage.md").write_text(markdown(doc))
    print(args.out / "api-coverage.md")
    return int(bool(doc["failed_libraries"] or doc["errors"]))
