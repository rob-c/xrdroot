"""ROOT's build model, translated: dictionaries, libraries, ACLiC, and loading them by name.

ROOT builds C++ into shared libraries and loads them later: rootcling's
dictionary of a header (``libFoo.so`` with its rootmap), a library linked
from sources, ACLiC's ``macro_C.so``, and ``gSystem->Load("libFoo")``,
``R__LOAD_LIBRARY(Foo)`` or ``.L libFoo.so`` to load one. xrdroot has no
compiler and needs none: a library is the translation of what it was built
from, kept in a ``.xrdlib`` file (:mod:`.manifest`), built by
:mod:`.build` (``xrdroot build``), ACLiC's by :mod:`.aclic`, and loaded -
or autoloaded by a name it provides, as a rootmap allows - by
:mod:`.loading`.
"""

from __future__ import annotations

from .aclic import call, compile_macro, compiled
from .aclic import load as load_compiled
from .build import BuildError, build_aclic, build_dictionary, build_library
from .loading import autoload, find, load, load_library, loaded_names, rebind
from .manifest import SUFFIX, Library, read, stem_of

__all__ = [
    "BuildError",
    "Library",
    "SUFFIX",
    "autoload",
    "build_aclic",
    "build_dictionary",
    "build_library",
    "call",
    "compile_macro",
    "compiled",
    "find",
    "load",
    "load_compiled",
    "load_library",
    "loaded_names",
    "read",
    "rebind",
    "stem_of",
]
