"""``TSelector``, and ``TTree::MakeSelector``'s skeleton, run the way ROOT runs one.

    >>> t.Process("Tree.C")                                        # doctest: +SKIP
    >>> t.MakeSelector("Tree")                                     # doctest: +SKIP

ROOT's ``MakeSelector`` writes a class deriving from ``TSelector`` whose
``Process(entry)`` reads the tree's branches through a ``TTreeReader``, and
``TTree::Process`` builds that class from the macro and calls it entry by entry
- ``Init``, ``Begin``, ``SlaveBegin``, ``Process`` for each entry,
``SlaveTerminate``, ``Terminate``. The class it writes is C++ here as it is in
ROOT: the translator reads it, so a macro that fills in ``Process`` runs
unchanged. What the skeleton names its readers is ROOT's rule - the leaf's
name with its dots turned into underscores, an array reader for a fixed-size
array or a collection, a value reader for everything else - and the option
picks which of them to write: nothing for every leaf of the tree, ``"@"`` for
the objects each top-level branch holds, a branch's name for the leaves under
it, ``"@branch"`` for that branch as an object, and several of them joined by
``;``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ... import plugins
from ...errors import UnsupportedFeatureError
from .layout import element_class

__all__ = ["TSelector"]

#: The C++ spelling of a leaf of each numpy dtype, as ROOT's own leaf types are spelled.
_LEAF_TYPE = {
    ("b", 1): "Bool_t",
    ("f", 4): "Float_t",
    ("f", 8): "Double_t",
    ("i", 1): "Char_t",
    ("i", 2): "Short_t",
    ("i", 4): "Int_t",
    ("i", 8): "Long64_t",
    ("u", 1): "UChar_t",
    ("u", 2): "UShort_t",
    ("u", 4): "UInt_t",
    ("u", 8): "ULong64_t",
}

#: The C++ spelling of a ``std::vector``'s element, which ROOT writes in lower case.
_VECTOR_TYPE = {
    ("b", 1): "bool",
    ("f", 4): "float",
    ("f", 8): "double",
    ("i", 4): "int",
    ("i", 8): "long",
    ("u", 4): "unsigned int",
    ("u", 8): "unsigned long",
}

#: ROOT's leaf type names themselves: a leaf that has one needs no reading to be typed.
_NAMED = frozenset({"Bool_t", "Char_t", "Double_t", "Float_t", "Int_t", "Long64_t",
                    "Long_t", "Short_t", "UChar_t", "UInt_t", "ULong64_t", "ULong_t",
                    "UShort_t"})  # fmt: skip

#: The classes whose branch holds many objects: ROOT's two arrays, and any ``vector<T>``.
_COLLECTIONS = frozenset({"TClonesArray", "TObjArray"})


class TSelector:
    """``TSelector``: the option, the lists and the reader a ``Process`` call runs with.

    ROOT's generated class derives from this one and overrides ``Init``, ``Begin``,
    ``SlaveBegin``, ``Process``, ``SlaveTerminate`` and ``Terminate``; a macro that
    overrides some of them keeps these for the rest, as ROOT's does.
    """

    def __init__(self, tree: Any = None) -> None:
        self.fOption = ""
        self.fObject: Any = None
        self.fInput: Any = None
        self.fOutput: Any = None
        self.fTree: Any = tree
        self.fReader: Any = None

    def GetOption(self) -> Any:
        return self.fOption

    def SetOption(self, option: Any) -> None:
        self.fOption = str(option)

    def SetObject(self, obj: Any) -> None:
        self.fObject = obj

    def SetInputList(self, inputs: Any) -> None:
        self.fInput = inputs

    def GetOutputList(self) -> Any:
        return self.fOutput

    def Version(self) -> int:
        return 2

    def Init(self, tree: Any = None) -> None:
        """Point the reader the skeleton declares at ``tree``: what every entry is read by."""
        reader = getattr(self, "fReader", None)
        if reader is not None:
            reader.SetTree(tree)
        self.fTree = tree

    def Begin(self, tree: Any = None) -> None:
        self.fTree = tree

    def SlaveBegin(self, tree: Any = None) -> None:
        self.fTree = tree

    def Notify(self) -> bool:
        return True

    def Process(self, entry: int) -> bool:
        return True

    def SlaveTerminate(self) -> None:
        """The end of the loop; nothing to do in a plain selector."""

    def Terminate(self) -> None:
        """The end of the query; nothing to do in a plain selector."""


def _observed(tree: Any, column: str) -> Any:
    """What ``column`` holds in the first entry, which is what its reader has to be.

    ROOT types a reader from the dictionary of the class it split a branch of; xrdroot
    reads the entry instead, so a branch of a class, a collection of them and a number
    are told apart by what the file holds rather than by a name that may be missing.
    """
    before = tree._read_entry
    try:
        tree._read_entry = 0
        return tree._current(column)
    except Exception:  # the column is refused or absent: fall back on its leaf's type
        return None
    finally:
        tree._read_entry = before


def _numeric(leaf: Any, value: Any) -> str:
    """The C++ type of a numeric leaf: its own name, or the dtype it was read as."""
    if leaf.typename in _NAMED:
        return str(leaf.typename)
    dtype = np.asarray(value).dtype
    return _LEAF_TYPE.get((dtype.kind, dtype.itemsize), "Double_t")


def _element(tree: Any, leaf: Any) -> str:
    """The C++ type one item of a collection leaf holds: its ``vector<T>``, or what it read."""
    spelled = leaf.typename
    if spelled.startswith("vector<") and spelled.endswith(">"):
        return str(spelled[len("vector<") : -1])
    dtype = np.asarray(_observed(tree, leaf.column)).dtype
    return _VECTOR_TYPE.get((dtype.kind, dtype.itemsize), "int")


def _many(branch: Any) -> bool:
    """Does this branch hold a collection of objects, or one object per entry?

    ROOT declares ``TTreeReaderArray<Element>`` for a branch of a collection - a
    ``TClonesArray``, a ``TObjArray`` or a ``vector<T>`` - and ``TTreeReaderValue<Class>``
    for one of a class, whose own members may hold collections without the branch
    holding more than one object.
    """
    if branch is None:
        return False
    spelled = str(branch.classname)
    return bool(branch.clones) or spelled in _COLLECTIONS or spelled.startswith("vector<")


def _member_type(tree: Any, branch: Any, leaf: Any) -> str | None:
    """The type the file's streamer information gives the member a branch stands for.

    A branch split out of a class carries the class it was split from - ``ClassB`` for
    the branch of its member ``fC``, whose own type is ``ClassC`` - so a reader of it
    typed from the branch alone would hand a macro a ``ClassB``. ROOT types it from the
    class's dictionary, which is the same account of the member as the file's streamer
    information: what a branch names is its own class for one holding a whole object.
    """
    backing = getattr(tree, "_backing", None)
    held = getattr(backing(), "branches", {}).get(branch.name) if backing else None
    source = getattr(held, "_source", None)
    if source is None:
        return None
    found = source.streamers().get(branch.classname, {}).get(leaf.name.rpartition(".")[2])
    return None if found is None else str(found.typename)


def _reader(tree: Any, leaf: Any) -> tuple[str, str]:
    """``Value`` or ``Array``, and the C++ type: how ROOT declares this leaf's reader."""
    value = _observed(tree, leaf.column)
    branch = tree._branch_info(leaf.branch)
    if isinstance(value, Mapping):
        held = _member_type(tree, branch, leaf) or element_class(branch, leaf)
        return ("Array" if _many(branch) else "Value"), held
    if leaf.vector:
        element = _element(tree, leaf)
        if element in ("bool", "Bool_t"):
            return "Value", "vector<bool>"
        return "Array", element
    if leaf.size > 1 or isinstance(value, np.ndarray):
        return "Array", _numeric(leaf, value)
    return "Value", _numeric(leaf, value)


def _leaves(tree: Any) -> list[Any]:
    """Every leaf ROOT declares a reader for: the ``TObject`` base of an object is none."""
    return [leaf for leaf in tree._leaves() if not leaf.name.endswith("TObject")]


def _top(tree: Any) -> list[Any]:
    """The tree's top-level branches, which ``@`` names as objects."""
    return list(tree._layout())


def _under(tree: Any, name: str) -> list[Any]:
    """The leaves of the branch ``name``: those whose path is that branch and its members."""
    return [leaf for leaf in _leaves(tree)
            if leaf.column == name or leaf.column.startswith(f"{name}.")]


def _chosen(tree: Any, option: str) -> list[Any]:
    """What the readers of ``option`` are, in ROOT's reading of it.

    Nothing is every leaf of the tree, ``@`` the objects its top-level branches hold, a
    branch's name the leaves under it, ``@branch`` that branch as an object; several of
    those are joined by ``;``. A name the tree does not have is ROOT's warning, and the
    other parts are still read.
    """
    if not option:
        return _leaves(tree)
    found: list[Any] = []
    for part in option.split(";"):
        found += _part(tree, part)
    return found


def _part(tree: Any, part: str) -> list[Any]:
    """What one ``;``-separated part of an option names, and a warning if it names nothing."""
    if part == "@":
        return _objects(tree, _top(tree))
    if part.startswith("@"):
        branch = part[1:]
        matched = [top for top in tree._every_branch() if top.name == branch]
        if matched:
            return _objects(tree, matched)
        _no_branch(tree, branch)
        return []
    leaves = _under(tree, part)
    if not leaves:
        _no_branch(tree, part)
    return leaves


def _objects(tree: Any, branches: list[Any]) -> list[Any]:
    """One leaf per branch, reading the branch whole: what an ``@`` reader of it reads."""
    found = []
    for branch in branches:
        whole = [leaf for leaf in _leaves(tree) if leaf.column == branch.name]
        found += whole or [leaf for leaf in branch.leaves if not leaf.name.endswith("TObject")]
    return found


def _no_branch(tree: Any, name: str) -> None:
    from ..core.messages import Warning

    Warning("ParseOptions", "Tree %s does not contain a branch named %s.", tree.GetName(), name)


def _declarations(tree: Any, option: str) -> list[str]:
    """The reader declarations of the skeleton, one per name, the first of a name winning."""
    lines: list[str] = []
    seen: set[str] = set()
    for leaf in _chosen(tree, option):
        name = leaf.name.replace(".", "_").replace("[", "_").replace("]", "")
        if name in seen:
            continue
        seen.add(name)
        kind, held = _reader(tree, leaf)
        lines.append(f'   TTreeReader{kind}<{held}> {name} = {{fReader, "{leaf.column}"}};')
    return lines


#: What ROOT's skeleton declares, less the readers that :func:`_declarations` adds.
_HEAD = """//////////////////////////////////////////////////////////
// This class has been automatically generated by
// xrdroot's TTree::MakeSelector for {tree}/{title}
//////////////////////////////////////////////////////////

#ifndef {name}_h
#define {name}_h

#include <TSelector.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>
#include <TTreeReaderArray.h>

class {name} : public TSelector {{
public :
   TTreeReader     fReader;  ///<!the tree reader

   // Readers to access the data (delete the ones you do not need).
{readers}

   {name}(TTree * /*tree*/ =0) {{ }}
   ~{name}() {{ }}
   Int_t   Version() const {{ return 2; }}
   void    Begin(TTree *tree);
   void    SlaveBegin(TTree *tree);
   void    Init(TTree *tree);
   bool    Notify();
   bool    Process(Long64_t entry);
   void    SetOption(const char *option) {{ fOption = option; }}
   void    SetObject(TObject *obj) {{ fObject = obj; }}
   void    SetInputList(TList *input) {{ fInput = input; }}
   TList  *GetOutputList() const {{ return fOutput; }}
   void    SlaveTerminate();
   void    Terminate();
}};

// A macro that fills a skeleton in defines this class's methods out of line and never
// names the class, and the translator emits a header's class only once the macro refers
// to it: this names it, so that the bodies the macro writes belong to the class.
{name} *xrdroot_selector_keep({name} *selector) {{ return selector; }}

#endif
"""

#: The skeleton's own ``.C``: the methods of a selector that reads nothing yet.
_BODY = """// The class definition in {name}.h has been generated automatically by xrdroot's
// TTree::MakeSelector(). This class is derived from the ROOT class TSelector.
//
// To use this file, try the following session on your Tree T:
//
// root> T->Process("{name}.C")
// root> T->Process("{name}.C","some options")
//
#define {name}_cxx

#include "{name}.h"

void {name}::Begin(TTree * /*tree*/)
{{
   TString option = GetOption();
}}

void {name}::SlaveBegin(TTree * /*tree*/)
{{
   TString option = GetOption();
}}

bool {name}::Process(Long64_t entry)
{{
   fReader.SetEntry(entry);

   return true;
}}

void {name}::SlaveTerminate() {{ }}

void {name}::Terminate() {{ }}
"""


def _title(tree: Any) -> str:
    return str(tree.GetTitle()).replace("\n", " ")


def make_selector(tree: Any, name: str = "", option: str = "") -> int:
    """``TTree::MakeSelector``: write ``Name.h`` and ``Name.C`` here; ``1`` when they are.

    ROOT writes the class into the current directory, which a macro moves into with
    ``gSystem->cd`` before calling it, so the names are relative paths.
    """
    written = str(name) or tree.GetName()
    readers = "\n".join(_declarations(tree, str(option))) or "   // (no reader)"
    header = _HEAD.format(tree=tree.GetName(), title=_title(tree), name=written, readers=readers)
    Path(f"{written}.h").write_text(header, encoding="utf-8")
    Path(f"{written}.C").write_text(_BODY.format(name=written), encoding="utf-8")
    return 1


#: A quoted ``#include`` of a file of the macro's own, which ROOT looks for from the
#: current directory and the translator from beside the including file.
_LOCAL_INCLUDE = re.compile(r'^([ \t]*#[ \t]*include[ \t]+)"([^"]+)"', re.MULTILINE)


def _resolved(text: str, path: Path) -> str:
    """A macro's text with each of its own quoted includes made a full path.

    ROOT finds ``#include "generated_selectors/Tree.h"`` from the directory the macro runs
    in - which is where ``MakeSelector`` wrote the skeleton - while the translator searches
    beside the including file and then its own directories. A skeleton written into a
    directory of its own is therefore spelled out here, so that the macro reads the class
    ROOT would read.
    """
    def full(found: re.Match[str]) -> str:
        name = found.group(2)
        if Path(name).is_absolute() or (path.parent / name).is_file():
            return found.group(0)
        if Path(name).is_file():
            return f'{found.group(1)}"{Path(name).resolve()}"'
        return found.group(0)

    return _LOCAL_INCLUDE.sub(full, text)


def _macro(source: str, path: Path) -> dict[str, Any]:
    """A selector macro's names, built as ROOT builds them when ``Process`` loads the file.

    The translation is the interpreter's, through the provider a host installs, so that a
    tree needs no more of :mod:`xrdroot.pycint` than ``gROOT`` itself does.
    """
    run_source = plugins.lookup("interpreter.run_source")
    if run_source is None:
        raise UnsupportedFeatureError(
            "TTree::Process translates and runs the selector macro with xrdroot's C++ "
            "interpreter, and none is installed here; install xrdroot with its translator, "
            "or call the selector's methods from Python instead"
        )
    return dict(run_source(_resolved(source, path), str(path), call=False))


def _built(names: Mapping[str, Any], where: str, tree: Any) -> Any:
    """An instance of the selector the macro at ``where`` declares, named after its file."""
    stem = Path(where).name.split(".")[0]
    made = names.get(stem)
    if not isinstance(made, type) or not issubclass(made, TSelector):
        raise UnsupportedFeatureError(
            f"the selector {where} declares no class {stem} deriving from TSelector, which "
            f"is what TTree::Process runs; write one with TTree::MakeSelector, or name the "
            f"macro after the class it declares"
        )
    return made(tree)


def process(
    tree: Any, filename: Any, option: Any = "", nentries: int = -1, firstentry: int = 0
) -> int:
    """``TTree::Process``: run a selector macro over the tree; how many entries it took.

    ROOT strips a trailing ``+``, which builds the file with ACLiC; a macro is translated
    here instead, so the compiled and the interpreted file are the same one. The
    selector's ``Begin``, ``SlaveBegin``, ``Process``, ``SlaveTerminate`` and
    ``Terminate`` are called as ROOT calls them, ``Init`` first to point its reader at
    this tree.
    """
    spelled = str(filename)
    path = Path(spelled.rstrip("+"))  # ROOT's ``name.C+``: built with ACLiC, a macro here
    names = _macro(path.read_text(encoding="utf-8", errors="replace"), path)
    selector = _built(names, spelled, tree)
    selector.SetOption(str(option))
    selector.Init(tree)
    selector.Begin(tree)
    selector.SlaveBegin(tree)
    last = tree.GetEntries() if int(nentries) < 0 else int(firstentry) + int(nentries)
    taken = 0
    for entry in range(int(firstentry), min(last, tree.GetEntries())):
        selector.fReader.SetEntry(entry)
        selector.Process(entry)
        taken += 1
    selector.SlaveTerminate()
    selector.Terminate()
    return taken
