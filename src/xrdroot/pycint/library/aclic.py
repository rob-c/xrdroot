"""ACLiC: ``.L macro.C+``, ``.x macro.C+(args)``, ``gSystem->CompileMacro`` - translated, kept.

ROOT's ACLiC compiles a macro into ``macro_C.so`` beside it (or in
``gSystem->GetBuildDir()``), loads it, and compiles it again only when the
macro or a header it includes has changed, or ``++`` (or option ``f``)
forces it. Here the shared library is ``macro_C.xrdlib``, the macro's
translation (:mod:`.manifest`): built as ROOT builds one - saying ``Info in
<TUnixSystem::ACLiC>: creating shared library ...`` on standard error - kept
until the macro's text, a header's or the translator changes, and loaded
like any other library, so ``gSystem->Load("macro_C")`` finds it later.

The options after the ``+`` are ACLiC's: ``g`` and ``O`` (debug, optimised)
change nothing a translation does; ``k`` keeps the library, which it always
is; ``f`` forces a build; ``c`` builds without loading.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

from ...pycxx.errors import CxxError
from ...pycxx.execute import entry, placed
from ...pycxx.translation import Translation
from ..dialect import ROOT as DIALECT
from .build import BuildError, build_aclic, explained, translator_version
from .loading import HEADERS, LOADED, Loaded, find, load_library
from .manifest import SUFFIX, Library, read, stem_of
from .translate import digest

__all__ = [
    "ACLIC",
    "split",
    "library_name",
    "library_place",
    "compiled",
    "load",
    "call",
    "compile_macro",
    "is_compiled",
    "is_library",
]

#: ``macro.C+``, ``macro.C++g``: the file, ACLiC's one or two pluses, and its options.
ACLIC = re.compile(r"^(?P<path>.*?[^+])(?P<plus>\+\+?)(?P<options>[gOkfcs-]*)$", re.S)


def split(spec: str) -> tuple[str, str]:
    """``macro.C++g`` as the file and ACLiC's options, ``++g``; a plain macro's are ``""``."""
    found = ACLIC.match(spec.strip())
    if found is None:
        return spec.strip(), ""
    return found["path"], found["plus"] + found["options"]


def library_name(macro: Path) -> str:
    """The library ACLiC makes of ``macro``: ``hsimple.C`` is ``hsimple_C``."""
    return f"{macro.stem}_{macro.suffix.lstrip('.')}"


def _build_dir() -> tuple[str, bool]:
    """``gSystem->GetBuildDir()``, and whether it was set flat - if pyroot's gSystem is there."""
    system = sys.modules.get("xrdroot.pyroot.core.system")
    if system is None:
        return "", False
    return str(system.gSystem.GetBuildDir()), bool(system.gSystem.IsBuildDirFlat())


def library_place(macro: Path, out: Path | None = None) -> Path:
    """Where ACLiC puts ``macro``'s library: ``out``, else the build directory - itself if it
    was set flat, else the macro's own path under it - else beside the macro."""
    if out is not None:
        return out.resolve()
    build, flat = _build_dir()
    if not build:
        return macro.resolve().parent
    place = Path(build).resolve()
    return place if flat else place / str(macro.resolve().parent).lstrip("/")


def _current(library: Library | None) -> bool:
    """Whether ``library`` is up to date: built by this translator from the files as they are."""
    if library is None or library.translator != translator_version():
        return False
    return library.digest == digest(library.resolved(library.sources)
                                    + library.resolved(library.headers))  # fmt: skip


def _system() -> str:
    return "TMacOSXSystem" if sys.platform == "darwin" else "TUnixSystem"


def compiled(
    macro: Path,
    *,
    force: bool = False,
    name: str | None = None,
    out: Path | None = None,
    explain: bool = True,
) -> Library:
    """``macro``'s library, built unless an up-to-date one is there and nothing forces it.

    With ``explain``, what the translator refuses is said as a compiler's error and
    ``Compilation failed!`` raised, as a session's ACLiC says it; else it is raised as is.
    """
    from ...pyroot.core.messages import message

    place = library_place(macro, out)
    named = name or library_name(macro)
    existing = read(place / f"{named}{SUFFIX}")
    if not force and existing is not None and _current(existing):
        return existing
    message("Info", f"{_system()}::ACLiC", f"creating shared library {place / named}{SUFFIX}")
    try:
        return build_aclic(macro, named, place)
    except BuildError:
        raise
    except CxxError as why:
        if not explain:
            raise
        raise explained(why) from None


def _loaded(library: Library) -> Loaded:
    """``library`` loaded - again if what is loaded from its file was built from older text."""
    key = str(library.path.resolve()) if library.path is not None else library.name
    old = LOADED.get(key)
    if old is not None and old.library.digest != library.digest:
        del LOADED[key]
        for header in old.headers:
            HEADERS.pop(header, None)
    return load_library(library)


def load(spec: str) -> Loaded:
    """``.L macro.C+``: the macro's library built if it has to be, and loaded."""
    path, options = split(spec)
    return _loaded(compiled(Path(path), force="++" in options or "f" in options))


def is_library(path: str) -> bool:
    """Whether ``path`` names a library - ``x_C.so``, ``libFoo.dylib`` - rather than a macro."""
    return stem_of(path) != os.path.basename(path)


def is_compiled(spec: str) -> bool:
    """Whether ``spec`` runs from a library: ACLiC's ``x.C+``, or a library, ``x_C.so``."""
    path, options = split(spec)
    return bool(options) or is_library(path)


def _library(name: str) -> Loaded:
    """The library ``name`` names, loaded - as ``.x x_C.so`` loads it before calling it."""
    found = find(name)
    library = read(found) if found is not None else None
    if library is None:
        raise FileNotFoundError(f"{name} is no library xrdroot built, on the library path")
    return load_library(library)


def call(spec: str, args: tuple[Any, ...] = ()) -> tuple[Any, Translation]:
    """``.x macro.C+(args)`` - or ``.x macro_C.so(args)``: the library built if it has to be,
    loaded, its function called; what it returned, and the translation that says how to
    print it."""
    loaded = load(spec) if split(spec)[1] else _library(spec)
    made = loaded.library.translation()
    function = entry(made, loaded.module, made.file, args)
    if function is None:
        return None, made
    with DIALECT.session(), placed([made]):
        return function(*args), made


def compile_macro(
    filename: Any, option: Any = "", library: Any = "", build_dir: Any = "", dirmode: Any = 0
) -> int:
    """``gSystem->CompileMacro``: ``1`` when the macro is built (and, without ``c``, loaded),
    ``0`` when it could not be."""
    path, plus = split(str(filename))
    options = f"{plus}{option}"
    out = Path(str(build_dir)) if build_dir else None
    try:
        made = compiled(Path(path), force="f" in options or "++" in options,
                        name=stem_of(str(library)) or None, out=out)  # fmt: skip
        if "c" not in options:
            _loaded(made)
    except (CxxError, OSError) as why:
        from ...pyroot.core.messages import message

        message("Error", "ACLiC", str(why))
        return 0
    return 1
