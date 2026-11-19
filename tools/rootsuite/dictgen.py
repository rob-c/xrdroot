"""genreflex and rootcling command lines as ``xrdroot build dictionary``.

Some tests run ROOT's dictionary generators themselves. Their words carry
what ``xrdroot build dictionary`` needs, spelled each tool's way:

* ``genreflex HEADER... [-o OUT] [-s|--selection_file|--select SEL] [-l LIB]
  [--rootmap MAP] [--rootmap-lib LIB] [-I DIR]...`` - the library is ``-l``
  (else ``--rootmap-lib``), else the dictionary is named after its output,
  else after the first header with ``_rflx``;
* ``rootcling [-v2] [-f] OUT [-s LIB] [-rml LIB] [-rmf MAP] [-I DIR]...
  HEADER... [LinkDef.h]`` - the library is ``-s`` (else ``-rml``), else the
  output's stem; the LinkDef, as rootcling takes it, is the last header when
  its name says ``linkdef``.

A rootmap is written only when one is asked for (``--rootmap``, ``-rmf``);
otherwise ``--no-rootmap``. genreflex's ``--fail_on_warnings`` is
``--fail-on-warnings``; rootcling run with no LinkDef and no C++ module asked
for (``-cxxmodule``) is ``--selection-required``, as it then refuses a
dictionary nothing selects for. The output's directory is ``--out``. Options
that change nothing a translated dictionary holds (``-v``, ``--verbose``,
``--interpreteronly``, ``-D``...) are dropped.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import PurePosixPath

__all__ = ["GENERATORS", "dictionary_words"]

#: The dictionary generators ROOT ships, run by name.
GENERATORS = ("genreflex", "rootcling", "rootcint")

#: genreflex's options taking a value, by the key it is kept under.
_GENREFLEX = {"-o": "out", "--output": "out", "-s": "selection", "--selection_file": "selection",
              "--select": "selection", "-l": "library", "--library": "library",
              "--rootmap": "rootmap", "--rootmap-lib": "rootmap-lib", "-I": "include",
              "-m": "pcm"}

#: rootcling's options taking a value, by the key it is kept under.
_ROOTCLING = {"-f": "out", "-s": "library", "-rml": "rootmap-lib", "-rmf": "rootmap",
              "-I": "include", "-m": "pcm", "-moduleMapFile": "pcm"}


def _parsed(words: Sequence[str], options: dict[str, str]) -> dict[str, list[str]]:
    """Each option's values under its key, ``-Ix`` and ``--x=y`` spellings included; the
    other words that are not options as ``files``."""
    found: dict[str, list[str]] = {"files": []}
    pending = ""
    for word in words:
        if pending:
            found.setdefault(pending, []).append(word)
            pending = ""
            continue
        key, _, value = word.partition("=")
        if word.startswith("-I") and len(word) > 2:
            key, value = "-I", word[2:]
        if key in options:
            if value:
                found.setdefault(options[key], []).append(value)
            else:
                pending = options[key]
        elif not word.startswith("-"):
            found["files"].append(word)
    return found


def _stem(path: str) -> str:
    """A library or output file's name less its directory, ``lib`` and suffixes."""
    name = PurePosixPath(path).name.split(".")[0]
    return name[3:] if name.startswith("lib") and len(name) > 3 else name


def _build(name: str, found: dict[str, list[str]], headers: list[str],
           extra: list[str]) -> list[str]:
    out = PurePosixPath(found.get("out", [""])[0]).parent.as_posix()
    words = ["dictionary", "--name", name, *extra]
    words += [w for directory in found.get("include", []) for w in ("-I", directory)]
    words += ["--out", out] if out not in ("", ".") else []
    words += [] if found.get("rootmap") else ["--no-rootmap"]
    return [*words, *headers]


def _genreflex(words: Sequence[str]) -> list[str]:
    found = _parsed(words, _GENREFLEX)
    headers = found["files"]
    library = (found.get("library") or found.get("rootmap-lib") or [""])[0]
    output = found.get("out", [""])[0]
    name = _stem(library or output) or f"{_stem(headers[0]) if headers else 'dict'}_rflx"
    selection = found.get("selection", [])
    extra = ["--selection", selection[0]] if selection else []
    extra += ["--fail-on-warnings"] if "--fail_on_warnings" in words else []
    extra += ["--selection-syntax-only"] if "--selSyntaxOnly" in words else []
    return _build(name, found, headers, extra)


def _linkdef(files: list[str]) -> tuple[str, list[str]]:
    """rootcling's LinkDef - its last file, when named so - and the headers before it."""
    if files and "linkdef" in PurePosixPath(files[-1]).name.lower():
        return files[-1], files[:-1] or files[-1:]  # a LinkDef alone #includes its headers
    return "", files


def _rootcling(words: Sequence[str]) -> list[str]:
    found = _parsed(words, _ROOTCLING)
    files = found["files"]
    if "out" not in found and files:  # rootcling OUT ..., without -f
        found["out"], files = files[:1], files[1:]
    linkdef, headers = _linkdef(files)
    library = (found.get("library") or found.get("rootmap-lib") or [""])[0]
    name = _stem(library or found.get("out", ["dict"])[0])
    module = any(word.startswith("-cxxmodule") for word in words)
    extra = ["--linkdef", linkdef] if linkdef else [] if module else ["--selection-required"]
    return _build(name, found, headers, extra)


def dictionary_words(tool: str, words: Sequence[str]) -> list[str]:
    """The words after ``xrdroot build`` standing in for ``tool`` run with ``words``."""
    return _genreflex(words) if tool == "genreflex" else _rootcling(words)
