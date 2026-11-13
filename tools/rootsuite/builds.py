"""ROOT's build steps as xrdroot's: dictionaries, libraries, executables and ACLiC.

ROOT's tests lean on things the build compiles: dictionaries rootcling or
genreflex generate, test libraries, test executables, macros ACLiC compiles.
xrdroot stands in for each with one command, run in the directory ROOT
would build in, writing its product there::

    xrdroot build dictionary --name NAME [--linkdef FILE] [--selection FILE] [-I DIR]... HEADER...
    xrdroot build library --name NAME [-I DIR]... SOURCE...           # lib<NAME>.xrdlib
    xrdroot build executable --name EXE [--library NAME]... [-I DIR]... SOURCE...   # ./EXE
    xrdroot build aclic MACRO.C[+opts]                                # <macro>_C.xrdlib

``NAME`` is the file ROOT would write without ``lib`` and its suffix:
``ROOTTEST_GENERATE_DICTIONARY(libEvDict ...)`` writes ``libEvDict.so``, so
``--name EvDict``. A library with its own dictionary - a LinkDef package, a
``G__`` source among its sources - is a dictionary step and then a library
step of the same name, the second adding the compiled sources to it.

A build is a tuple: the directory it runs in (below ROOT's tree), then the
words after ``xrdroot build``, its paths spelled ``<build>/...`` so that the
plan rewrites them to the working copy.
"""

from __future__ import annotations

from collections.abc import Iterable

from .model import Dictionary, Registry, Target

__all__ = ["Build", "library_name", "dictionary", "split", "library", "executable", "aclic",
           "closure"]

#: One ``xrdroot build`` command: its directory, then its arguments.
Build = tuple[str, ...]


def library_name(file_name: str) -> str:
    """``--name`` for a library ROOT writes as ``file_name``: ``libEvent`` is ``Event``."""
    return file_name[3:] if file_name.startswith("lib") and len(file_name) > 3 else file_name


def _path(relative: str) -> str:
    return f"<build>/{relative}"


def _includes(directories: Iterable[str]) -> list[str]:
    return [word for found in dict.fromkeys(directories) for word in ("-I", _path(found))]


def dictionary(entry: Dictionary, name: str) -> Build:
    """``xrdroot build dictionary``: one dictionary's headers, LinkDef or selection, includes."""
    words = ["dictionary", "--name", name]
    if entry.linkdef:
        words += ["--linkdef", _path(entry.linkdef)]
    if entry.selection:
        words += ["--selection", _path(entry.selection)]
    words += _includes([entry.directory, *entry.includes])
    headers = entry.headers or [entry.linkdef]  # a LinkDef alone #includes its headers
    return (entry.directory, *words, *(_path(header) for header in headers if header))


def split(registry: Registry, target: Target) -> tuple[list[Dictionary], list[str]]:
    """The dictionaries generated into a target, and the sources compiled into it."""
    generated = [registry.dictionaries[name] for name in target.files
                 if name in registry.dictionaries]
    sources = [name for name in target.files if name not in registry.dictionaries]
    return [*target.dictionaries, *generated], sources


def library(registry: Registry, target: Target) -> list[Build]:
    """A test library: its dictionaries, then its sources, under the library's one name."""
    dictionaries, sources = split(registry, target)
    steps = [dictionary(entry, target.output) for entry in dictionaries]
    if sources:
        words = ["library", "--name", target.output,
                 *_includes([target.directory, *target.includes])]
        steps.append((target.directory, *words, *(_path(source) for source in sources)))
    return steps


def closure(registry: Registry, target: Target) -> list[str]:
    """The test-local libraries a target links, and those they link, each once."""
    found: list[str] = []
    pending = list(target.libraries)
    while pending:
        name = pending.pop(0)
        if name in registry.libraries and name not in found:
            found.append(name)
            pending.extend(registry.libraries[name].libraries)
    return found


def executable(registry: Registry, target: Target, name: str,
               libraries: bool = True) -> list[Build]:
    """An executable: the libraries it links (unless built by tests of their own), then it.

    A dictionary generated straight into the executable's sources - a gtest
    with its ``G__`` file - becomes a library of the dictionary's name.
    """
    steps: list[Build] = []
    linked: list[str] = []
    for found in (registry.libraries[entry] for entry in closure(registry, target)):
        steps += library(registry, found) if libraries else []
        linked.append(found.output)
    dictionaries, sources = split(registry, target)
    for entry in dictionaries:
        steps.append(dictionary(entry, library_name(entry.name)))
        linked.append(library_name(entry.name))
    words = ["executable", "--name", name, *(w for lib in linked for w in ("--library", lib)),
             *_includes([target.directory, *target.includes])]
    steps.append((target.directory, *words, *(_path(source) for source in sources)))
    return steps


def aclic(directory: str, macro: str, options: str = "kc") -> Build:
    """``xrdroot build aclic``: a macro as ``gSystem->CompileMacro`` with ``options`` builds it."""
    return (directory, "aclic", f"{_path(macro)}+{options}")
