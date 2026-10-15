"""Export ROOT dictionary metadata in the oracle's interpreter, never in xrdroot."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


def library_name(shared: str) -> str:
    name = Path(shared.split()[0]).name if shared else "(unknown)"
    return name.removeprefix("lib").split(".")[0]


def header_libraries(source: str) -> dict[str, str]:
    module, found = "(unknown)", {}
    for line in source.splitlines():
        declaration = re.match(r'^module\s+"?([\w.]+)"?\s*\{', line)
        if declaration:
            module = declaration[1]
        library = "Core" if module.startswith("ROOT_") else module
        for header in re.findall(r'header\s+"([^"]+)"', line):
            found[header] = library
    return found


def describe(root: Any, name: str, headers: dict[str, str] | None = None) -> dict[str, Any] | None:
    cls = root.TClass.GetClass(name)
    if not cls or cls.Property() & root.kIsNamespace:
        return None
    methods = sorted({str(method.GetName()) for method in cls.GetListOfAllPublicMethods()
                      if _included_method(root, method, name)})
    library = library_name(str(cls.GetSharedLibs()))
    if library == "(unknown)":
        library = (headers or {}).get(str(cls.GetDeclFileName()), library)
    return {"name": name, "library": library, "methods": methods}


def _included_method(root: Any, method: Any, owner: str) -> bool:
    if method.ExtraProperty() & (root.kIsConstructor | root.kIsDestructor):
        return str(method.GetClass().GetName()) == owner
    return True


def declared_classes(root: Any, headers: dict[str, str]) -> list[str]:
    """Header-only APIs such as RNTuple have no TClassTable registration."""
    interpreter = root.gInterpreter
    info = interpreter.ClassInfo_Factory()
    names = []
    try:
        while interpreter.ClassInfo_Next(info):
            header = str(interpreter.ClassInfo_FileName(info))
            properties = interpreter.ClassInfo_Property(info)
            if header in headers and properties & (root.kIsClass | root.kIsStruct):
                names.append(str(interpreter.ClassInfo_FullName(info)))
    finally:
        interpreter.ClassInfo_Delete(info)
    return names


def inventory(root: Any, libraries: list[str], headers: dict[str, str]) -> dict[str, Any]:
    failed = [name for name in libraries if root.gSystem.Load("lib" + name) < 0]
    root.gClassTable.Init()
    names = []
    while name := root.gClassTable.Next():
        names.append(str(name))
    names += declared_classes(root, headers)
    classes, errors = [], []
    for name in sorted(set(names)):
        try:
            row = describe(root, name, headers)
            if row is not None:
                classes.append(row)
        except Exception as why:
            errors.append({"name": name, "error": str(why)})
    return {"root_version": str(root.gROOT.GetVersion()), "libraries": libraries,
            "failed_libraries": failed, "errors": errors, "classes": classes}


def main() -> None:
    import ROOT

    ROOT.gROOT.SetBatch(True)
    destination, *libraries = sys.argv[1:]
    modulemap = Path(str(ROOT.gROOT.GetIncludeDir())) / "ROOT.modulemap"
    headers = header_libraries(modulemap.read_text()) if modulemap.is_file() else {}
    Path(destination).write_text(json.dumps(inventory(ROOT, libraries, headers), indent=1) + "\n")


if __name__ == "__main__":
    main()
