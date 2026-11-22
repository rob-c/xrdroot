"""ROOT's reflection: ``TClass``, its streamer information, data members and bases.

A ``TClass`` is what ROOT knows of a class: its name and version, what it
derives from, its data members, and how its objects are streamed - its
``TStreamerInfo``, one ``TStreamerElement`` a base or member. Here that
knowledge comes from three places: a class C++ declared, whose translation
describes its members (:mod:`.declared`, :mod:`xrdroot.wdeclared`); one of
ROOT's own, as the donor files describe it (:data:`xrdroot.winfo.INFOS`); and
otherwise the namespace's own Python class, which knows its name and bases.

A file read describes classes too. Each description it holds is noted
against its class (:func:`note_file_infos`), so ``GetStreamerInfos`` lists
the layouts met so far beside the one in memory, as ROOT's does.
"""

from __future__ import annotations

import re
import sys
from typing import Any, ClassVar
from weakref import WeakKeyDictionary

from ...wdeclared import ROOT_TYPEDEFS as TYPEDEFS
from ...winfo import INFOS, Element
from . import datatypes as _datatypes
from .classedit import TClassEdit
from .messages import message
from .objects import TNamed, TObject

#: How ROOT shortens a type name - ``TClassEdit::ShortType`` - which is what a class is
#: called by, since ``TClass::GetClass`` is asked for a name in any of its spellings.
short_type = TClassEdit.ShortType

__all__ = [
    "ClassInfo",
    "TClass",
    "TClassTable",
    "gClassTable",
    "TStreamerInfo",
    "TVirtualStreamerInfo",
    "TStreamerElement",
    "TStreamerBase",
    "TStreamerBasicType",
    "TStreamerBasicPointer",
    "TStreamerString",
    "TStreamerSTL",
    "TStreamerSTLstring",
    "TStreamerObject",
    "TStreamerObjectAny",
    "TStreamerObjectPointer",
    "TDataType",
    "kNumDataTypes",
    "TStreamerObjectAnyPointer",
    "TStreamerLoop",
    "TDataMember",
    "TBaseClass",
    "TMethod",
    "TFunctionTemplate",
    "kIsClass",
    "kIsStruct",
    "kIsUnion",
    "kIsEnum",
    "kIsTypedef",
    "kIsFundamental",
    "kIsAbstract",
    "kIsVirtual",
    "kIsPureVirtual",
    "kIsUnionMember",
    "kIsPublic",
    "kIsProtected",
    "kIsPrivate",
    "kIsPointer",
    "kIsArray",
    "kIsStatic",
    "kIsDefault",
    "kIsReference",
    "kIsDirectInherit",
    "kIsCCompiled",
    "kIsCPPCompiled",
    "kIsCompiled",
    "kIsNotReacheable",
    "kIsConstant",
    "kIsVirtualBase",
    "kIsConstPointer",
    "kIsScopedEnum",
    "kIsConstexpr",
    "kIsExplicit",
    "kIsNamespace",
    "kIsConstMethod",
    "kIsUsing",
    "kIsDefinedInStd",
    "kIsConstructor",
    "kIsConversion",
    "kIsDestructor",
    "kIsOperator",
    "kIsInlined",
    "kIsTemplateSpec",
    "kExactMatch",
    "kConversionMatch",
]

#: ``TDictionary::EProperty`` and ``EFunctionProperty``, and ``ROOT::EFunctionMatchMode``: the
#: bits and modes ``TMethod::ExtraProperty`` and ``TClass::GetMethodWithPrototype`` answer
#: with. C++ declares them twice - the two enums overlap on purpose - so the ones both name
#: have one value here, and each is a name of the namespace, as ``ROOT::kIsInlined`` is.
kIsClass = 0x00000001
kIsStruct = 0x00000002
kIsUnion = 0x00000004
kIsEnum = 0x00000008
kIsTypedef = 0x00000010
kIsFundamental = 0x00000020
kIsAbstract = 0x00000040
kIsVirtual = 0x00000080
kIsPureVirtual = 0x00000100
kIsUnionMember = 0x00000100
kIsPublic = 0x00000200
kIsProtected = 0x00000400
kIsPrivate = 0x00000800
kIsPointer = 0x00001000
kIsArray = 0x00002000
kIsStatic = 0x00004000
kIsDefault = 0x00008000
kIsReference = 0x00010000
kIsDirectInherit = 0x00020000
kIsCCompiled = 0x00040000
kIsCPPCompiled = 0x00040000
kIsCompiled = 0x00040000
kIsNotReacheable = 0x00080000
kIsConstant = 0x00100000
kIsVirtualBase = 0x00200000
kIsConstPointer = 0x00400000
kIsScopedEnum = 0x00800000
kIsConstexpr = 0x02000000
kIsExplicit = 0x04000000
kIsNamespace = 0x08000000
kIsConstMethod = 0x10000000
kIsUsing = 0x20000000
kIsDefinedInStd = 0x40000000
kIsConstructor = 0x00000001
kIsConversion = 0x00000002
kIsDestructor = 0x00000004
kIsOperator = 0x00000008
kIsInlined = 0x00000010
kIsTemplateSpec = 0x00000020
kExactMatch = 0
kConversionMatch = 1

#: One layout of one class: its checksum, version and elements.
Layout = tuple[int, int, tuple[Element, ...]]
#: The layouts files read so far describe each class with, in the order they were met.
FILE_INFOS: dict[str, list[Layout]] = {}

#: The classes a file's layout was reported as disagreeing with the class in memory about -
#: which ROOT warns of once per class, however many files its layouts are read from.
DISAGREED: set[str] = set()

#: The flat lists of persistent data members built so far, by class: ``BuildRealData`` builds
#: one and ``GetListOfRealData`` hands it back, as ROOT caches it with the class.
REAL_INFOS: dict[str, list[Any]] = {}

#: ``TStreamerElement``'s ``kMissingOffset``: the offset ROOT gives an element of a file's
#: layout it cannot set against the class in memory.
_MISSING_OFFSET = 99999

#: The numeric types a member can be declared with, whose width - and not whose spelling -
#: is what ROOT compares: they are as wide on every machine ROOT builds on.
NUMBERS = frozenset({
    "char", "short", "int", "long", "long long", "float", "double", "bool",
    "unsigned char", "unsigned short", "unsigned int", "unsigned long",
    "unsigned long long",
})  # fmt: skip

#: The dictionaries generated while the program ran, which ``gInterpreter->GenerateDictionary``
#: names: their classes have a ``TClass`` bootstrap from then on, as a loaded library's would.
GENERATED: set[str] = set()

#: The ``TMethod`` objects made for each class of a program's, by the class, and its member
#: function templates: made once, so the one a search answers with is still the one its list
#: holds after a ``GenerateDictionary`` replaces what the interpreter described of it.
METHODS: WeakKeyDictionary[type, list[Any]] = WeakKeyDictionary()
TEMPLATES: WeakKeyDictionary[type, list[Any]] = WeakKeyDictionary()

#: The functions looked up in a class only the interpreter describes, by class: ROOT asks the
#: interpreter for the one a program names and keeps the ``TMethod`` it made, which is what lets
#: a program hold that method across the dictionary that replaces the interpreter's description.
INTERPRETED_METHODS: dict[str, list[TNamed]] = {}

#: What a class's translation says of each of its functions, by position - which is what
#: ``TMethod`` is made of: the name, the return type as written, each parameter's type, whether
#: it is const, static, inline, virtual and a template, and its kind (``function``,
#: ``constructor``, ``destructor`` or ``operator``). A kit class that knows nothing of its C++
#: signatures keeps none, and its methods are found only where no prototype is asked for.
Signature = tuple[str, str, tuple[str, ...], bool, bool, bool, bool, bool, str]

#: The standard containers, whose classes ROOT knows without a dictionary of their own.
CONTAINERS = ("vector", "list", "deque", "map", "multimap", "set", "multiset",
              "unordered_map", "unordered_set", "pair", "array", "forward_list",
              "unordered_multimap", "unordered_multiset", "bitset")  # fmt: skip

#: The members every standard container declares with no argument, which are the functions a
#: lookup in a container - a class only the interpreter describes here - can be answered with:
#: nothing here knows a container's C++ signatures, so a function that takes one is ``nullptr``.
CONTAINER_METHODS = ("clear", "empty", "max_size", "size")

#: The fundamental types, which have no ``TClass``.
FUNDAMENTAL = frozenset({
    "bool", "char", "short", "int", "long", "long long", "float", "double", "unsigned char",
    "unsigned short", "unsigned int", "unsigned long", "unsigned long long", "signed char",
    "void", "Int_t", "Float_t", "Double_t", "Long64_t", "ULong64_t", "Bool_t", "Char_t",
    "Short_t",
})  # fmt: skip

#: ``EDataType::kNumDataTypes``: one past the last code ``TDataType`` describes, which ROOT
#: keeps in the namespace so that a program can ask whether a code it holds is a type at all.
kNumDataTypes = 23

#: The fundamental types ``TDataType`` describes, by their ``EDataType`` code: the name ROOT
#: gives each, how many bytes it is and how it is aligned, which on the 64-bit oracle is its
#: size for these. The codes with nothing here - ``kNoType_t`` 0, ``kOther_t`` -1 and
#: ``kNumDataTypes`` itself - describe no type, and ROOT answers with ``nullptr`` for them.
DATA_TYPES: dict[int, tuple[str, int]] = {
    1: ("char", 1), 2: ("short", 2), 3: ("int", 4), 4: ("long", 8),
    5: ("float", 4), 6: ("counter", 4), 7: ("char*", 8), 8: ("double", 8),
    9: ("Double32_t", 4), 10: ("char", 1), 11: ("unsigned char", 1),
    12: ("unsigned short", 2), 13: ("unsigned int", 4), 14: ("unsigned long", 8),
    15: ("bits", 8), 16: ("long long", 8), 17: ("unsigned long long", 8),
    18: ("bool", 1), 19: ("Float16_t", 2), 20: ("void", 0),
}  # fmt: skip

#: The typedefs that name how many bits a float is stored in - ``Double32_t`` is a ``double``
#: written as a float, ``Float16_t`` a ``float`` written in 16 bits - spelled as the type
#: they are, which is the only spelling the translation keeps of them.
PRECISION = {"Double32_t": "double", "Float16_t": "float"}

#: ``kIsTObject`` and the other bits ``TClass::Property`` and ``GetState`` answer with.
K_IS_TOBJECT = 0x00008000
#: ``TClass::EState``: ``kInterpreted``, a class the interpreter describes and no dictionary
#: does, and ``kHasTClassInit`` - ROOT's ``kLoaded`` too - one a dictionary describes.
K_INTERPRETED = 3
K_HAS_TCLASS_INIT = 4


def note_generated_dictionary(classes: Any) -> None:
    """Note the classes ``gInterpreter->GenerateDictionary`` was asked for - a semicolon
    separated list of names - as having a dictionary from then on.

    Nothing is compiled here, so what a dictionary adds to a class this kit already builds is
    that ``TClass`` reports it as one a dictionary describes, which is what a program asking
    for a dictionary and then asking the state of the class checks for.

    A dictionary arriving for a class a file has already described is ROOT's
    ``TClass::PostLoadCheck``: :func:`_post_load_check` says what the two layouts disagree
    about, which the file could not have said itself - the class had no dictionary when it
    was read.
    """
    for name in str(classes).split(";"):
        if name.strip():
            known = _known_as(name)
            GENERATED.add(known)
            _post_load_check(known)


def _post_load_check(name: str) -> None:
    """``TClass::PostLoadCheck``: the class's dictionary has arrived, so the layouts a file
    gave it can at last be held against the class in memory.

    A class a file described while it had no dictionary kept the file's layout as that
    version's - it was the only description there was - so a dictionary of the same version
    with another checksum is the class having been changed since, and ROOT says so once per
    class: ``roottest``'s ``checkWarn0`` reads ``data1.root`` and then ``.L data2.C+`` and is
    warned, where ``checkWarn2`` loads the class first and is warned as the file opens.

    Only a class a file has described is looked at, so loading a dictionary computes nothing
    about the classes no file has read: what a class's layout costs is its writer's knowledge
    of every type it holds, which is asked for when it is written, not when it is loaded.
    """
    mine = TClass(name)._layout() if name in FILE_INFOS else None
    if mine is None or name in DISAGREED:
        return
    on_file = _layout_of_version(name, mine)
    if on_file is None or not TClass(name)._versioned():
        return
    _post_load_warnings(name, mine, on_file)


def _layout_of_version(name: str, mine: Layout) -> Layout | None:
    """The layout a file gave ``name`` of the version the class in memory has, if its
    checksum is not the class's own - the disagreement ``PostLoadCheck`` is about."""
    return next((info for info in FILE_INFOS.get(name, [])
                 if info[1] == mine[1] and info[0] != mine[0]), None)


def _post_load_warnings(name: str, mine: Layout, on_file: Layout) -> None:
    """The two warnings ROOT prints for a dictionary that arrives after the class's file
    layouts were read: ``TClass::PostLoadCheck``, that the file's version of the class is not
    the one in memory, and ``TStreamerInfo::CompareContent``, which lists the members the
    file's layout has and the class in memory does not declare - ``int myvar_one; //`` beside
    ``double mydouble_two; //``.

    The file's layout stays the version's current one, as ROOT keeps it, so what the class's
    streamer information lists afterwards is the file's, which is why ``checkWarn0`` prints
    ``myvar_one`` after loading the class that declares ``mydouble_two``.
    """
    DISAGREED.add(name)
    message("Warning", "TClass::PostLoadCheck",
            "\n   The StreamerInfo version %d for the class %s which was read\n"
            "   from a file previously opened has the same version as the active class\n"
            "   but a different checksum. You should update the version to ClassDef(%s,%d).\n"
            "   Do not try to write objects with the current class definition,\n"
            "   the files will not be readable.\n",
            on_file[1], name, name, on_file[1] + 1)
    message("Warning", "TStreamerInfo::CompareContent",
            "The following data member of\nthe in-memory layout version %d of class '%s' "
            "is missing from\nthe on-file layout version %d:\n   %s",
            on_file[1], name, mine[1],
            "\n   ".join(_declaration(element) for element in _lost(mine, on_file)))


def _lost(mine: Layout, on_file: Layout) -> list[Element]:
    """The members a file's layout records that the class in memory declares by other names,
    which is what ``CompareContent`` lists as missing."""
    declared = {element[1] for element in mine[2]}
    return [element for element in on_file[2] if element[1] not in declared]


def _known_as(name: str) -> str:
    """The name a class is known by: the type shortened, without ``std::``, as ROOT keeps the
    one name for a class whatever spelling it was asked for with."""
    return short_type(name).replace("std::", "").strip()


def forget_dictionaries() -> None:
    """A fresh session's state: the dictionaries a library gave are not there any more."""
    GENERATED.clear()


def note_file_infos(infos: list[tuple[str, Layout]], source: Any = None) -> None:
    """Note the layouts a file describes its classes with, as opening it makes ROOT do.

    A layout already met, by its checksum, is not noted again, nor is one the class in memory
    has: a class with ``ClassDef`` by its version, one without by its checksum. A class
    nothing in memory describes is emulated from its files, and ROOT numbers its layouts as
    they come, so a second layout of a version already taken gets the next number.

    ``source`` is the file they were read from - the source itself, or its name - which the
    ``BuildCheck`` warning names, so a layout of a version the class has in memory but of a
    different checksum is reported where ROOT reports it: as the file is opened. Opening a
    file notes its layouts without naming it, so the name is then the one
    :mod:`xrdroot.streamers` read them from.
    """
    origin = _origin(source)
    for name in _as_root_lists_them(infos):
        if name not in FILE_INFOS and _undictionaried(name):
            message("Warning", "TClass::Init", "no dictionary for class %s is available", name)
    for name, (checksum, version, elements) in infos:
        met = _met(name, (checksum, version, elements), origin)
        if met or _in_memory(name, checksum, version):
            continue
        held = FILE_INFOS[name]
        taken = {found[1] for found in held}
        held.append((checksum, _free_version(name, version, taken), elements))


def _as_root_lists_them(infos: list[tuple[str, Layout]]) -> list[str]:
    """The classes a file describes, in the order ROOT warns of them: a class before the
    classes its layout is built on.

    ROOT registers a class's streamer information as it first streams an object of it, and a
    base's after the derived class's - ``roottest/root/io/tree``'s ``write.C`` streams a
    ``TEmcl``, whose base is ``TPhysObj``, so the file ROOT wrote lists ``TEmcl`` first and
    ``missingClass.ref`` names it first. The writer here lays a base down before its derived
    class, which is ROOT's order read backwards, so the order ROOT would have read is worked
    out from the layouts: a class whose base the file lists before it is warned of first, and
    the base with it.
    """
    bases = {}
    for name, (_sum, _version, elements) in infos:
        bases[name] = _bases_of(elements)
    deferred: set[str] = set()
    for found in bases.values():
        deferred.update(found)
    order: list[str] = []
    warned: set[str] = set()
    for name, _layout in infos:
        if name not in deferred:
            _warn_order(name, bases, order, warned)
    for name, _layout in infos:
        _warn_order(name, bases, order, warned)
    return order


def _bases_of(elements: tuple[Element, ...]) -> list[str]:
    """The classes a layout is built on: the names of its ``TStreamerBase`` elements."""
    found = []
    for element in elements:
        if element[0] == "TStreamerBase":
            found.append(element[1])
    return found


def _warn_order(name: str, bases: dict[str, list[str]], order: list[str],
                warned: set[str]) -> None:  # fmt: skip
    """``name``, then the classes its layout is built on, each once - which is the order a
    file ROOT wrote lists them in."""
    if name in warned:
        return
    warned.add(name)
    order.append(name)
    for base in bases.get(name, ()):
        _warn_order(base, bases, order, warned)


def _origin(source: Any) -> str:
    """Which file the layouts came from, for the warning that names it."""
    if source is not None:
        return str(getattr(source, "name", source) or "")
    from ...streamers import LAST_SOURCE

    return LAST_SOURCE


def _met(name: str, on_file: Layout, origin: str) -> bool:
    """Whether a layout a file describes is one already met - reported here, or already held.

    ROOT warns about a class once, so a layout it has reported, and one whose checksum a
    layout already held has, are both met; the disagreement with the class in memory is
    reported as the file is opened, which is where ROOT reports it.
    """
    held = FILE_INFOS.setdefault(name, [])
    met = any(found[0] == on_file[0] for found in held) or name in DISAGREED
    mine = _disagreement_with(name, on_file, met)
    if mine is not None:
        DISAGREED.add(name)
        _disagreement(name, mine, on_file, origin)
    return met


def _differing(mine: Layout, on_file: Layout) -> list[tuple[Element, Element]]:
    """The members of the file's layout ROOT says the class in memory declares differently,
    each with the class's own member it is set against.

    A member of a number differs when the file records it at another *width*: ROOT reports
    ``AddVersion-1-6``'s ``int a`` against ``Long64_t a`` and stays silent about
    ``AddVersion-3-6``'s ``long a`` against ``Long64_t a``, which is as wide. Anything else
    differs when its type is spelled as another type, a typedef resolved: an old file spells
    ``Int_t`` under a type code today's ROOT no longer writes for it
    (``io/evolution/ClonesArrayEvo`` reads one and prints nothing), so the code an element
    carries is not a difference, and a pointer or a ``TString`` is as wide on the machine that
    wrote the file as on this one.
    """
    here = {element[1]: element for element in mine[2]}
    named = {element[1] for element in on_file[2]}
    found = []
    for index, element in enumerate(on_file[2]):
        ours = here.get(element[1]) or _renamed(mine, named, index)
        if ours is not None and _differs(element, ours):
            found.append((element, ours))
    return found


def _renamed(mine: Layout, named: set[str], index: int) -> Element | None:
    """The class's member a file's member is set against when the two spell it differently.

    ROOT pairs the members of two layouts of a version by where they come rather than by
    name, so ``roottest``'s ``checkWarn`` reads a file whose ``int myvar_one`` the class in
    memory declares as ``double mydouble_two`` and is told the two differ. A member the file
    names is only ever set against the class's member of that name, so a name both layouts
    have is never paired here.
    """
    if index >= len(mine[2]) or mine[2][index][1] in named:
        return None
    return mine[2][index]


def _differs(on_file: Element, mine: Element) -> bool:
    """Is one member, as the file records it and as the class in memory declares it, one ROOT
    would report?"""
    theirs, ours = _spelled(on_file[8]), _spelled(mine[8])
    if theirs in NUMBERS and ours in NUMBERS:
        return on_file[4] != mine[4]
    return theirs != ours


def _spelled(typename: str) -> str:
    """A type name with ROOT's numeric typedefs resolved, ``UChar_t*`` being
    ``unsigned char*``."""
    base = typename.rstrip("*")
    return TYPEDEFS.get(base, base) + typename[len(base):]


def _disagreement_with(name: str, on_file: Layout, met: bool) -> Layout | None:
    """The layout the class has in memory, when the one a file describes disagrees with it in
    the way ROOT reports as the file is opened - a version the class versions, a checksum of
    its own, and a member it declares differently.

    A class already reported is not reported again: ``AddVersion-1-6`` reads
    ``myclass1.root`` and then ``myclass2.root``, whose own layout of the class differs from
    the class in memory just as much, and warns about the first file only.
    """
    current = TClass(name)._layout()
    if met or current is None or not TClass(name)._versioned():
        return None
    same_version, differs = current[1] == on_file[1], current[0] != on_file[0]
    if same_version and differs and _differing(current, on_file):
        return current
    return None


def _disagreement(name: str, mine: Layout, on_file: Layout, origin: str) -> None:
    """The two warnings ROOT prints for a file whose layout of a versioned class disagrees
    with the one in memory: ``BuildCheck``, which says the checksums differ and names the
    file, and ``CompareContent``, which sets out each member the two record differently.

    ROOT prints them when it opens the file, and the in-memory layout stays the current one -
    which is why the class's layouts are listed afterwards as they were before the file was
    read, the file's own for that version dropped.
    """
    if origin:
        message("Warning", "TStreamerInfo::BuildCheck",
                "\n   The StreamerInfo for version %d of class %s read from the file %s\n"
                "   has a different checksum than the previously loaded StreamerInfo.\n"
                "   Reading objects of type %s from the file %s\n"
                "   (and potentially other files) might not work correctly.\n"
                "   Most likely the version number of the class was not properly\n"
                "   updated [See ClassDef(%s,%d)].",
                on_file[1], name, origin, name, origin, name, on_file[1])
    _compare_content(name, mine, on_file)


def _declaration(element: Element) -> str:
    """How ``CompareContent`` spells a member: its type and name, then the comment after it -
    ``int a; //``."""
    member = TStreamerElement(element)
    return f"{member.GetTypeName()} {member.GetFullName()}; //{member.GetTitle()}"


def _compare_content(name: str, mine: Layout, on_file: Layout) -> None:
    """``TStreamerInfo::CompareContent``: each member of a versioned class whose type the file
    records differently from the class in memory, the two declarations one above the other -
    called for a file whose layout differs, which is what :func:`_differing` says."""
    differ = _differing(mine, on_file)
    message("Warning", "TStreamerInfo::CompareContent",
            "The following data member of\nthe on-file layout version %d of class '%s' "
            "differs from\nthe in-memory layout version %d:\n   %s\nvs\n   %s",
            on_file[1], name, mine[1],
            "\n   ".join(_declaration(theirs) for theirs, _ours in differ),
            "\n   ".join(_declaration(ours) for _theirs, ours in differ))


def _free_version(name: str, version: int, taken: set[int]) -> int:
    """The version to list ``name``'s next layout at, given the ones already taken.

    A file written by a class the library has a version of its own for says version 1 - ROOT
    records what the class was before ``RootClassVersion`` gave it a number - so the layout
    belongs above the in-memory one. A class nothing describes is emulated from its files and
    ROOT numbers its layouts as they come, so a version already taken gets the next number.
    """
    current = TClass(name)._layout()
    if current is not None and current[1] > 1 and version <= 1:
        taken.add(current[1])
        return next(number for number in range(1, len(taken) + 2) if number not in taken)
    return max(taken) + 1 if version in taken and _emulated(name) else version


def _version_of(layout: tuple[Layout, bool]) -> int:
    """The version a layout is listed at, which is the index ROOT keeps it at."""
    return layout[0][1]


def _in_memory(name: str, checksum: int, version: int) -> bool:
    """Is a layout of ``name`` the one the class in memory has - its version for a class with
    ``ClassDef``, its checksum for one without?"""
    known = TClass(name)
    current = known._layout()
    if current is None:
        return False
    return current[0] == checksum if known.IsForeign() else current[1] == version


def _emulated(name: str) -> bool:
    return TClass(name)._layout() is None


#: How a class's enum constant is spelled: ``kSomething`` or ``ALL_CAPS``, and a number.
_CONSTANT = re.compile(r"^(?:k[A-Z]\w*|[A-Z][A-Z0-9_]*)$")


def _is_constant(name: str, value: Any) -> bool:
    """Is this class attribute one of C++'s enum constants - a name and a number?"""
    return bool(_CONSTANT.match(name)) and isinstance(value, int) and not isinstance(value, bool)


def _undictionaried(name: str) -> bool:
    """Has nothing here a dictionary of the class ``name`` - neither the program nor ROOT -
    which ROOT warns of when a file it opens describes one?"""
    if _template(name) or _synthetic_pair(name):
        return False
    if _pair_base(name) or _instantiable(name) or _modelled(name):
        return False
    return TClass(name)._class() is None


#: Classes ROOT's own libraries carry a dictionary for that this kit implements nothing of
#: and so has no layout for - it never reads one into an object of its own. They are named
#: here because ROOT names no warning for them either: a geometry file describes ``TGeoAtt``,
#: ``TGeoVoxelFinder`` and ``TGeoParallelWorld``, and ROOT opens it in silence, which
#: ``tests/test_pyroot_geom_rootfile.py`` pins line by line.
ROOT_CLASSES = frozenset({"TGeoAtt", "TGeoVoxelFinder", "TGeoParallelWorld"})


def _modelled(name: str) -> bool:
    """Is ``name`` a class of ROOT's own, one of those ROOT's libraries carry a dictionary
    for, so that opening a file which describes it says nothing?

    A name's spelling is no help: a user's class is called ``TEmcl`` or ``TCedarEvent`` as
    readily as ROOT's own is called ``TBasket``, and ROOT warns of the program's four
    emulated classes in ``roottest/root/io/tree``'s ``missingClass`` while staying silent
    about the ``TTree`` beside them. What tells them apart is knowledge of the class, not of
    its name, and what the kit knows of ROOT's classes is what it implements - as a Python
    class of the namespace's, ``TCanvas`` and ``TUUID`` too, which derive from no ``TObject``
    here - and what it streams: the layouts of :data:`xrdroot.winfo.INFOS`, and the readers
    of :mod:`xrdroot.objects` and :mod:`xrdroot.known` for the classes the kit writes itself.
    A name none of those knows is emulated from the file with ROOT's warning.
    """
    from ...known import KNOWN
    from ...objects import CLASSES
    from ...sparse import SPARSE

    if name in INFOS or name in GENERATED or name in ROOT_CLASSES:
        return True
    if name in CLASSES or name in KNOWN or name in SPARSE:
        return True
    return _implemented(name)


def _implemented(name: str) -> bool:
    """Does the namespace hold a class called ``name``, whether or not it derives from
    ``TObject`` - which is how ``TEventList``, ``TUUID``, ``TStyle`` and ``TColor`` are known
    to be ROOT's own, whose files ROOT describes without a warning."""
    for module in list(sys.modules.values()):
        if getattr(module, "__name__", "").startswith("xrdroot.pyroot"):
            if isinstance(getattr(module, name, None), type):
                return True
    return False


def _pair_base(name: str) -> bool:
    """Is ``name`` the ``__pair_base<A,B>`` a standard library's ``pair`` derives from?

    ROOT never reports that class, which is not one a program can ask for: its
    ``TClass::Init`` skips the warning for a ``pair``'s base (``TClassEdit::IsStdPairBase``),
    and a file which describes one - ``roottest/root/io/stdpair``'s ``ab02.root`` - opens with
    no warning about it, as the 6.40 oracle does.
    """
    return name.removeprefix("std::").startswith("__pair_base<")


def _instantiable(name: str) -> bool:
    """Is ``name`` an instance of a template whose class a dictionary describes?

    ROOT asks the interpreter for the class of a name a file describes, and the interpreter
    answers a template's instance from the template itself: with a dictionary for ``Ref`` in
    the session, ``Ref<vector<...>>`` is a type cling can declare, so ``TClass::Init`` warns
    about no missing dictionary for it - which is why ``pairEnumEvo``'s runs that load the
    library before the file print none of the warnings the runs that read the file first do.
    """
    head = name.split("<", 1)[0]
    return bool(head) and _last(_known_as(head)) in GENERATED


def _synthetic_pair(name: str) -> bool:
    """Is ``name`` a ``pair<A,B>``, which no ``ClassDef`` ever versions? ROOT tells the
    layouts of one apart by checksum - state ``kEmulated``, ``IsForeign`` - so a listing of
    them names no version, while a plain class only a file describes prints the one it gave.
    """
    return name.removeprefix("std::").startswith("pair<")


def _class_named(name: str) -> type | None:
    """The namespace's class called ``name``, wherever it was defined in the kit."""
    for module in list(sys.modules.values()):
        if getattr(module, "__name__", "").startswith("xrdroot.pyroot"):
            found = getattr(module, name, None)
            if isinstance(found, type) and issubclass(found, TObject):
                return found
    return None


def _declared(name: str) -> type | None:
    """The class declared as ``name``, or - for a template instance whose argument is one of
    :data:`PRECISION` - the class declared under the plain spelling of that argument.

    A macro's linkdef names the instance as ROOT's dictionary does, ``MyVector<Double32_t>``,
    but a declaration writes the typedef's type, so that is the one class here: the
    translation makes ``Double32_t`` a ``double`` and the instance it records is
    ``MyVector<double>``. roottest's ``io/double32`` asks for the class by the linkdef's
    spelling, and ROOT answers it because its dictionary is under that spelling too.
    """
    from .declared import find_class

    return find_class(name) or find_class(_plain_spelling(name))


def _plain_spelling(name: str) -> str:
    """``name`` with each :data:`PRECISION` typedef in it written as the type it is."""
    spelled = name
    for typedef, plain in PRECISION.items():
        spelled = spelled.replace(typedef, plain)
    return spelled


def _template(name: str) -> bool:
    head = name.removeprefix("std::").split("<", 1)[0]
    return "<" in name and head in CONTAINERS


class ClassInfo:
    """``ClassInfo_t``: what the interpreter knows of a class, as ``GetClassInfo`` gives it.

    ROOT's is an opaque handle and its parts are read through ``TInterpreter``'s
    ``ClassInfo_*`` functions; what is kept here is the ``TClass`` it describes,
    which is what those questions are about.
    """

    def __init__(self, klass: TClass) -> None:
        self._klass = klass

    def __repr__(self) -> str:
        return f"<ClassInfo_t of {self._klass.GetName()}>"

    def IsValid(self) -> bool:
        return True

    def GetName(self) -> str:
        """``ClassInfo_Name``: the class's name, as C++ spells it."""
        return self._klass.GetName()

    def Size(self) -> int:
        """``ClassInfo_Size``: how many bytes one object of the class takes."""
        return self._klass.Size()

    def Tagnum(self) -> int:
        """``ClassInfo_Tagnum``: the interpreter's own number for the class, which is 0 here
        since nothing is tagged: a macro that memoises by it memoises once."""
        return 0


class TDataType(TNamed):
    """``TDataType``: one of ROOT's fundamental types, named and sized as ROOT has it.

    An enumeration answers ``TEnum::GetUnderlyingType`` with an ``EDataType`` code, and ROOT
    turns that code into a type with ``TDataType::GetDataType`` - which is how roottest's
    ``enumSize`` checks that ``enum E : short`` is stored as a ``short``. The question the
    other way round is ``GetType``: a C++ type's ``type_info`` answers with the ``EDataType``
    code ROOT keeps for it, which is how a branch, a leaf and a streamer name what they hold.
    """

    def __init__(self, name: str = "", code: int = 0, size: int = 0) -> None:
        super().__init__(str(name), str(name))
        self._code = int(code)
        self._size = int(size)

    def GetType(self, info: Any = None) -> int:
        """``GetType``: ROOT declares this twice, and Python has the one name for both.

        On a ``TDataType`` - ``TDataType::GetDataType(kShort_t)->GetType()`` - it is the
        type's own ``EDataType`` code; called on the class with a C++ type's ``type_info``,
        as ``TDataType::GetType(typeid(T))`` is, it is the code ROOT gives that type.
        """
        if isinstance(self, TDataType):
            return self._code
        return _code_of_type(self)  # ROOT's static overload, asked with a ``type_info``

    def Size(self) -> int:
        """``Size``: how many bytes a value of the type takes, as ROOT's ``sizeof`` does."""
        return self._size

    def GetAlignOf(self) -> int:
        """``GetAlignOf``: the alignment ROOT reports for the type, which for these is its
        size - ``short`` 2, ``double`` 8 - and 0 for a type it knows nothing of."""
        return self._size

    def GetTypeName(self) -> str:
        """``GetTypeName``: the type's name, as ROOT's ``TString`` gives it."""
        return self.GetName()

    def GetFullTypeName(self) -> str:
        """``GetFullTypeName``: the type's name, which for a fundamental type is its name."""
        return self.GetName()

    @staticmethod
    def GetDataType(code: Any) -> TDataType | None:
        """``GetDataType(code)``: the type that code names, or ``nullptr`` for no type.

        ROOT types the argument ``EDataType``, which is an integer; a program that passes
        something else - as a macro passing a string does when it means one of the names -
        gets no type back rather than a guess.
        """
        try:
            number = int(code)
        except (TypeError, ValueError):
            return None
        found = DATA_TYPES.get(number)
        return None if found is None else TDataType(found[0], number, found[1])


class TClassTable(TObject):
    """``TClassTable``: the classes that have a dictionary, as ROOT's table of them.

    A program asks ``GetDict`` to decide whether it has to load a library first
    (``if (!TClassTable::GetDict("Event")) gSystem->Load("libEvent")``), so what answers is
    a class a dictionary describes - one of ROOT's own, or one a compiled library declared -
    and never a class a macro's own ``#include`` described, which no dictionary is behind.
    """

    @staticmethod
    def GetDict(name: Any, load: bool = True, silent: bool = False) -> TClass | None:
        """``GetDict(name)``: the class's ``TClass`` where a dictionary describes it.

        A name may be given qualified - ``geo::Shape`` - as ROOT's own table takes one: the
        class is the last part of it, which is the name the dictionary is under.
        """
        text = _last(_known_as(str(name)))
        return TClass.GetClass(text, False, silent) if text in _with_dictionary() else None

    @staticmethod
    def GetDictAny(name: Any) -> TClass | None:
        """``GetDictAny(name)``: ``GetDict``, which has nothing to autoload here."""
        return TClassTable.GetDict(name, False)

    @staticmethod
    def Add(name: Any, version: Any = 0, factory: Any = None) -> None:
        """``Add``: a dictionary's class, noted as one from now on."""
        GENERATED.add(_last(_known_as(str(name))))

    @staticmethod
    def Classes() -> int:
        """``Classes()``: how many classes the table holds."""
        return len(_with_dictionary())

    @staticmethod
    def At(index: Any) -> str:
        """``At(index)``: the name of the class the table lists at ``index``."""
        return _with_dictionary()[int(index)]

    @staticmethod
    def GetTable() -> list[str]:
        """``GetTable()``: every name the table holds, which ROOT gives as ``const char**``."""
        return _with_dictionary()

    @staticmethod
    def Print(option: Any = "") -> None:
        """``Print()``: the names, one a line, as ROOT's own ``Print`` lists them."""
        for name in _with_dictionary():
            print(name)


def _last(name: str) -> str:
    """A class's name without the namespace it is in, which is how a table holds it."""
    return name.split("::")[-1].strip()


def _with_dictionary() -> list[str]:
    """The classes a dictionary here describes: ROOT's own the kit carries, and a library's."""
    return sorted({*INFOS, *GENERATED})


#: ROOT's global table of the classes that have a dictionary - ``gClassTable``.
gClassTable = TClassTable("TClassTable", "")


class TClass(TNamed):
    """``TClass``: what ``IsA`` answers - a class's name, version, bases, members and layout."""

    #: What ``ClassDef`` says of the class.
    CLASS_TITLE = "Dictionary of a class"
    kIsTObject = K_IS_TOBJECT
    kHasTClassInit = K_HAS_TCLASS_INIT
    kInterpreted = K_INTERPRETED
    kEmulated = 2
    kNoInfo = 0
    kLoaded = K_HAS_TCLASS_INIT

    #: What ``TClass::GetClass`` is handed back for a class nobody knows: nothing.
    _MISSING: ClassVar[None] = None

    def __init__(self, name: Any = "", python: type | None = None) -> None:
        TNamed.__init__(self, str(name), "")
        self._python = python

    # -- what it is -----------------------------------------------------------------------

    def _class(self) -> type | None:
        """The Python class standing for this one: the one it was made with, one C++
        declared, else the namespace's."""
        if self._python is not None:
            return self._python
        return _declared(self._name) or _class_named(self._name)

    def _declared(self) -> type | None:
        from .declared import declared_class

        cls = self._class()
        return cls if declared_class(cls) else _declared(self._name)

    @staticmethod
    def GetClass(name: Any, load: bool = True, silent: bool = False) -> TClass | None:
        """``TClass::GetClass(name)`` - or of a ``typeid`` - the class, or ``nullptr`` for a
        fundamental type, which has none. A class a dictionary the kit cannot see declares is
        taken to be there, as a loaded library would make it."""
        if isinstance(name, TClass):
            return name
        if isinstance(name, type):
            return TClass(getattr(name, "__cpp_name__", name.__name__), name)
        spelled = getattr(name, "_spelled", None)
        # ROOT keeps a class under the name it shortens the type to - GetClass("string")
        # and GetClass("basic_string<char,char_traits<char> >") are the one class, called
        # "string" - so the same shortening is what this is called by.
        text = _known_as(str(spelled if spelled is not None else name))
        return None if text in FUNDAMENTAL else TClass(text)

    @staticmethod
    def GetDict(name: Any, load: bool = True, silent: bool = False) -> TClass | None:
        """``TClass::GetDict(name)``: the class's ``TClass`` where a dictionary describes it.

        ROOT keeps this older spelling of ``TClassTable::GetDict`` - a name in any of its
        spellings, and for a namespace the ``TClass`` describing it - because programs ask it
        to decide whether they must load a library first (roottest's ``execfwdDeclarations``
        asks for every class its dictionary was built for and for one it was not).

        ROOT 6.40's signature is ``static DictFuncPtr_t GetDict(const char*)``: the answers
        here are ``TClass``es, and ``load`` and ``silent`` are ``TClassTable::GetDict``'s
        arguments, which no program in the suite passes. Nothing calls the dictionary
        function the pointer would be, so the object that answers truthiness is enough.
        """
        return TClassTable.GetDict(name, load, silent)

    def HasDictionary(self) -> bool:
        """``HasDictionary``: does anything here describe the class's members?"""
        return self._class() is not None or self._name in INFOS or _template(self._name)

    def GetClassInfo(self) -> ClassInfo | None:
        """``GetClassInfo()``: the interpreter's descriptor of the class - ``nullptr`` when no
        dictionary describes it.

        A macro asks this to decide whether it has to load a library first
        (``if (cl->GetClassInfo() == 0) gROOT->ProcessLine(".L lib.cxx+")``), so a
        class that is here - one PyROOT has, one the interpreter declared, or a
        standard container this kit builds - answers with one, and one nobody
        describes answers ``nullptr``.
        """
        return ClassInfo(self) if self.HasDictionary() else None

    def HasInterpreterInfo(self) -> bool:
        """``HasInterpreterInfo``: is the class one the interpreter describes?

        ROOT asks it before it builds a ``TStreamerInfo`` from the class's own
        declaration - a class only a file describes has none, and what is known of
        such a class is what that file says. A standard container is described by
        the interpreter from its template's definition, as ROOT's is.
        """
        return self._class() is not None or self._name in INFOS or _template(self._name)

    def IsLoaded(self) -> bool:
        return self.HasDictionary()

    def GetState(self) -> int:
        """``GetState``: how much is known of the class.

        A standard container is one the interpreter builds from the template's own definition,
        with no dictionary of its own - ``kInterpreted`` - until a dictionary for it is
        generated, which makes it ``kHasTClassInit``. A class only a file describes is
        ``kEmulated``.
        """
        if _template(self._name) and self._name not in GENERATED:
            return K_INTERPRETED
        return K_HAS_TCLASS_INIT if self.HasDictionary() else self.kEmulated

    def IsForeign(self) -> bool:
        """``IsForeign``: declared without ``ClassDef``, so its objects carry a checksum."""
        from .declared import foreign

        cls = self._declared()
        return cls is not None and foreign(cls)

    def IsVersioned(self) -> bool:
        return not (self.GetClassVersion() <= 1 and self.IsForeign())

    def IsTObject(self) -> bool:
        cls = self._class()
        return isinstance(cls, type) and issubclass(cls, TObject)

    def Property(self) -> int:
        return K_IS_TOBJECT if self.IsTObject() else 0

    def InheritsFrom(self, other: Any) -> bool:
        """Is this class ``other`` - a name or a ``TClass`` - or derived from it?"""
        wanted = other.GetName() if isinstance(other, TClass) else str(other)
        if wanted == self._name:
            return True
        python = self._class()
        if python is not None and wanted in (base.__name__ for base in python.mro()):
            return True
        return any(base.InheritsFrom(wanted) for base in self._base_classes())

    def GetClassVersion(self) -> int:
        """``GetClassVersion``: ``ClassDef``'s - -1 for a class declared without one, whose
        layouts are told apart by checksum - and ROOT's own as the donors say.

        A class nothing in memory describes has the version the files that describe it give
        it, which is the class's version for the rest of the session: ``roottest``'s
        ``io/tree/read`` unloads ``classes_C`` and asks a class only ``copy.root`` describes
        for its ``ClassDef``, and ROOT answers with the file's - ``0`` for a class whose
        file says so - where the class in memory would be no answer at all.
        """
        info = self._layout() or self._file_layout()
        if info is not None:
            return -1 if self.IsForeign() and info[1] <= 1 else info[1]
        version = getattr(self._class(), "Class_Version", None)
        return int(version()) if callable(version) else 1

    def GetCheckSum(self, code: int = 0) -> int:
        info = self._layout()
        return info[0] if info is not None else 0

    def GetDeclFileName(self) -> str:
        return ""

    def GetImplFileName(self) -> str:
        return ""

    def GetDeclFileLine(self) -> int:
        return -1

    def Size(self) -> int:
        return 8 * max(1, len(self._fields()))

    GetClassSize = Size

    def CanSplit(self) -> bool:
        return True

    def GetCollectionProxy(self) -> Any:
        """``GetCollectionProxy``: for a standard container, what iterates it."""
        if not _template(self._name):
            return None
        from .collectionproxy import TVirtualCollectionProxy

        return TVirtualCollectionProxy(self._name)

    def GetActualClass(self, obj: Any) -> TClass:
        """``GetActualClass``: the class the object really is, one derived from this or this."""
        return obj.IsA() if isinstance(obj, TObject) else self

    def GetBaseClassOffset(self, base: Any) -> int:
        """``GetBaseClassOffset``: where ``base``'s part of this class starts - 0 here, where
        an object is all its classes at once - or -1 when ``base`` is not one of them."""
        return 0 if self.InheritsFrom(base) else -1

    # -- layout -----------------------------------------------------------------------------

    def _layout(self) -> tuple[int, int, tuple[Element, ...]] | None:
        """The class's checksum, version and streamer elements, as it is in memory now."""
        cls = self._declared()
        if cls is not None:
            from .declared import catalogue_of, declaration_of

            name = declaration_of(cls).name
            return catalogue_of(cls).info(name)
        return INFOS.get(self._name)

    def _file_layout(self) -> tuple[int, int, tuple[Element, ...]] | None:
        """The layout the first file to describe the class gave it, if any.

        A class nothing in memory describes is emulated from the first description a file it
        reads holds - ``kEmulated``, which the class to ``pair<reco::Muon::MuonTrackType,
        edm::Ref<...>>`` roottest writes for shows - and that is the class's current layout
        for the rest of the session.
        """
        known = FILE_INFOS.get(self._name, [])
        return known[0] if known else None

    def _versioned(self) -> bool:
        """Does a listing of one of the class's layouts name their version?

        ROOT names it for the layouts a ``ClassDef`` versions - a class declared with one,
        and one only files describe, which its files version - and not for those it tells
        apart by checksum: a class declared without ``ClassDef``, and a ``pair<A,B>``, which
        is never given one.
        """
        return self.IsVersioned() and not _synthetic_pair(self._name)

    def _current_layout(self) -> Layout | None:
        """The layout ROOT keeps as the class's current one: the class's own as it is in
        memory, or, where a file has already given a layout of that version a checksum of its
        own, the file's - which ROOT keeps as that version's and warns over instead of
        replacing, saying the class should be given the next version.
        """
        current = self._layout()
        if current is None:
            return self._file_layout()
        return next((info for info in FILE_INFOS.get(self._name, [])
                     if info[1] == current[1] and info[0] != current[0]), current)

    def GetStreamerInfo(self, version: int = 0) -> TStreamerInfo | None:
        """``GetStreamerInfo``: the layout of the class in memory - or, given a version, the
        layout of that version, from a file read if it is not the one in memory.

        A class nothing in memory describes is emulated from the first layout a file it read
        gave it, and ROOT builds what it hands back - so its elements are listed with the
        action each is read with - where a layout it merely lists stays unbuilt.
        """
        mine = self._layout()
        emulated = mine is None
        current = self._current_layout()
        if current is not None and version in (0, current[1]):
            return TStreamerInfo(self._name, current, built=True, versioned=self._versioned(),
                                 from_file=current is not mine)
        found = [i for i in FILE_INFOS.get(self._name, []) if i[1] == version]
        if not found:
            return None
        return TStreamerInfo(self._name, found[0], built=emulated, versioned=self._versioned(),
                             from_file=True, memory=mine[2] if mine else None)

    def FindStreamerInfo(self, checksum: int) -> TStreamerInfo | None:
        """``FindStreamerInfo``: the layout with that checksum, in memory or read."""
        current = self._layout()
        if current is not None and current[0] == checksum:
            return self.GetStreamerInfo()
        found = [i for i in FILE_INFOS.get(self._name, []) if i[0] == checksum]
        return TStreamerInfo(self._name, found[0], versioned=self._versioned()) if found else None

    def GetStreamerInfos(self) -> Any:
        """``GetStreamerInfos``: every layout known - the one in memory, and each a file read
        describes it with that is not that one.

        ROOT keeps a versioned class's layouts in the order of their versions, which is not
        the order the files were read in: ``roottest``'s ``AddVersion-1-4`` reads a file
        holding versions 1 and 3 beside a library at version 2 and lists them 1, 2, 3. A class
        told apart by checksum has no version to be sorted by, so there the layout in memory
        comes first and each file's follows as it was met.
        """
        from .collections import TObjArray

        made = TObjArray()
        current = self._layout()
        versioned = self._versioned()
        held = [(current, True)] if current is not None else []
        held.extend((info, False) for info in FILE_INFOS.get(self._name, []))
        if versioned:
            held.sort(key=_version_of)
        for info, in_memory in held:
            made.Add(TStreamerInfo(self._name, info, built=in_memory, versioned=versioned,
                                   memory=None if in_memory or current is None
                                   else current[2]))
        return made

    # -- members and bases --------------------------------------------------------------------

    def _fields(self) -> list[tuple[str, str, str, tuple[Any, ...]]]:
        cls = self._declared()
        if cls is not None:
            return [tuple(field) for field in cls.__dict__["_cxx_layout_"][2]]
        elements = INFOS.get(self._name, (0, 0, ()))[2]
        members = [e for e in elements if e[0] != "TStreamerBase"]
        return [(e[1], e[8], e[2], tuple(e[7][: e[6]])) for e in members]

    def _base_names(self) -> list[str]:
        cls = self._declared()
        if cls is not None:
            return [str(base) for base in cls.__dict__["_cxx_layout_"][1]]
        info = INFOS.get(self._name)
        if info is not None:
            return [e[1] for e in info[2] if e[0] == "TStreamerBase"]
        python = self._class()
        bases = python.__bases__ if python is not None else ()
        return [b.__name__ for b in bases if issubclass(b, TObject) and b is not python]

    def _base_classes(self) -> list[TClass]:
        return [TClass(name) for name in self._base_names()]

    def GetListOfBases(self) -> Any:
        from .collections import TList

        made = TList()
        for name in self._base_names():
            made.Add(TBaseClass(name, self))
        return made

    def GetBaseClass(self, name: Any) -> TClass | None:
        """``GetBaseClass``: ``name`` if this class derives from it, at any depth."""
        wanted = name.GetName() if isinstance(name, TClass) else str(name)
        for base in self._base_classes():
            if base.GetName() == wanted:
                return base
            deeper = base.GetBaseClass(wanted)
            if deeper is not None:
                return deeper
        return None

    def GetListOfEnums(self, load: bool = False) -> Any:
        """``GetListOfEnums``: the enumerations the class - or the namespace - declares.

        ROOT keeps one table of the interpreter's enumerations and answers a class with the
        part of it that class owns, so this is a filter of the session's table and not a
        list of its own: a class declaring no enum answers with an empty ``TListOfEnums``,
        never with ``nullptr`` (roottest's ``execEnumsTest`` dereferences it).

        ``load`` is the autoloading ROOT would do to fill the list in by reading the class's
        payload, which no library here is read for - so it changes nothing, as an enum the
        interpreter has not declared is one nothing here can find.
        """
        from .troot import class_enums

        return class_enums(self._name)

    def GetListOfDataMembers(self, load: bool = True) -> Any:
        """``GetListOfDataMembers``: the class's members - and its enum constants, which C++
        lists as members too (``TObject::kZombie`` is one of ``TNamed``'s)."""
        from .collections import TList

        made = TList()
        for field in self._fields():
            made.Add(TDataMember(self, *field))
        for name in self._constants():
            made.Add(TDataMember(self, name, "int", "", ()))
        return made

    def _constants(self) -> list[str]:
        """The enum constants the class declares itself: the names a dictionary lists beside
        its data members."""
        cls = self._class()
        if cls is None:
            return []
        return [name for name, value in vars(cls).items()
                if _is_constant(name, value)]  # fmt: skip

    def GetListOfAllPublicDataMembers(self, load: bool = True) -> Any:
        """Every data member, this class's and its bases'."""
        made = self.GetListOfDataMembers()
        for base in self._base_classes():
            for member in base.GetListOfAllPublicDataMembers():
                made.Add(member)
        return made

    def GetDataMember(self, name: Any) -> TDataMember | None:
        """``GetDataMember``: a member by name - this class's or a base's, as ROOT finds one."""
        wanted = str(name).split("[")[0]
        return next((m for m in self.GetListOfAllPublicDataMembers() if m.GetName() == wanted),
                    None)

    def GetMissingDictionaries(self, result: Any, recurse: bool = False) -> None:
        """``GetMissingDictionaries``: fill ``result`` - a ``THashTable`` - with the ``TClass``es
        of the classes no dictionary describes.

        The class itself where nothing describes it, and the class of each of its data
        members - a member of ``std::vector<Elem>`` adds ``vector<Elem>`` and, a standard
        container's elements being what the interpreter describes of it, ``Elem``. With
        ``recurse`` those classes' own members are walked too; without it a class it found is
        listed and not walked, as ROOT lists what a class refers to and leaves the descent to
        the walk into it. A class a dictionary here describes - ROOT's own, or one a built
        dictionary was loaded for - is not among them.
        """
        self._missing_into(result, bool(recurse), set(), members=True)

    def _missing_into(self, result: Any, recurse: bool, seen: set[str], members: bool) -> None:
        """Add the classes ``self`` refers to that no dictionary describes, walking their
        members when ``members`` - or a recursion, or a description - says to."""
        if self._name in seen:
            return
        seen.add(self._name)
        described = not self._dictless()
        if not described:
            result.Add(self)  # ``seen`` has each class once, however many members are of it
            for base in self._base_names():
                TClass(base)._missing_into(result, recurse, seen, members=False)
        if recurse or members or described:
            for klass in self._referenced(recurse):
                klass._missing_into(result, recurse, seen, members=recurse)

    def _referenced(self, recurse: bool) -> list[TClass]:
        """The classes the class's data members are, as ``GetMissingDictionaries`` walks them:
        each member's own class, a standard container's elements', and with ``recurse`` the
        arguments of any template instance.

        An instance of a standard container or of a wrapper has no data members to read here -
        the interpreter describes ``vector<Elem>`` from its element - so what it refers to is
        what it holds.
        """
        if _template(self._name) or _wrapper(self._name):
            return _classes_of(_arguments(self._name))
        found = self._member_classes(recurse)
        if recurse:
            found.extend(_classes_of(_arguments(self._name)))
        return found

    def _member_classes(self, recurse: bool) -> list[TClass]:
        """Each data member's class, and - a standard container's or a wrapper's, whose
        elements the interpreter describes as its members - the classes it holds."""
        found: list[TClass] = []
        for _, spelled, _, _ in self._fields():
            name = _class_spelling(str(spelled))
            if name is None:
                continue
            found.append(TClass(name))
            if recurse or _template(name) or _wrapper(name):
                found.extend(_classes_of(_arguments(name)))
        return found

    def _dictless(self) -> bool:
        """Is this a class no dictionary describes?

        ROOT's own classes are described by the kit's layouts, and one a loaded dictionary
        selected is described by it - a dictionary's other classes, translated with it, have
        no dictionary of their own. A standard container is described where ROOT's own
        dictionaries instantiate it, for the built-in types, and otherwise only where one was
        generated for it: a member of ``std::vector<Elem>`` is ``vector<Elem>`` even when
        ``Elem`` itself has a dictionary. A wrapper that streams what it holds -
        ``std::unique_ptr<T>``, ``std::array<T, N>``, ``std::tuple<...>`` - is described by
        what it holds, and is never itself a class a dictionary is missing for.
        """
        name = self._name
        if name in INFOS or name in GENERATED or _last(name) in GENERATED:
            return False
        if _instance(name):
            if _wrapper(name):
                return False
            if _template(name):
                return not all(_built_in(each) for each in _arguments(name))
            return True
        return _declared(name) is not None or _class_named(name) is None

    def BuildRealData(self, name: Any = None) -> None:
        """``BuildRealData``: work out the flat list of the class's persistent members.

        ROOT builds it from the interpreter's description of the class and caches it, which
        is what makes it right to ask once ``gInterpreter->Declare`` has made the class known:
        asking before that would have been too soon, and ROOT's own comment on
        ``execInitOrder`` says so. ``name`` is a ``TClass`` to build the real data of instead,
        as ROOT lets a caller hand one in.
        """
        target = name if isinstance(name, TClass) else self
        REAL_INFOS[target.GetName()] = _real_data_of(target)

    def GetListOfRealData(self) -> Any:
        """``GetListOfRealData``: the persistent members, each at the offset it is held at.

        A class's members as an object of it holds them: the class's own, then the ones it
        inherits with nothing in front of their names, and a member that is an object of a
        class of its own followed by that object's members named after it. A class nothing
        has asked the real data of yet is built here, as ROOT builds one on demand.
        """
        from .collections import TList

        if self._name not in REAL_INFOS:
            REAL_INFOS[self._name] = _real_data_of(self)
        made = TList()
        for data in REAL_INFOS[self._name]:
            made.Add(data)
        return made

    def GetListOfMethods(self, load: bool = True) -> Any:
        """``GetListOfMethods``: the class's own functions, by name.

        A class C++ declared lists the functions its translation describes, and after them the
        ones only the namespace's class carries - ``Class``, ``IsA``, the ``ClassDef`` helpers.
        A class only the interpreter describes lists the functions a search has made so far, as
        ROOT's list of them holds what the interpreter has been asked for.
        """
        from .collections import TList

        made = TList()
        for method in self._methods():
            made.Add(method)
        return made

    def _methods(self) -> list[Any]:
        """The class's functions as :meth:`GetListOfMethods` lists them.

        A class's are made once and kept, so the ``TMethod`` a search answers with is still the
        list's after - and whatever a dictionary does to the class later.
        """
        cls = self._class()
        if cls is None:
            return INTERPRETED_METHODS.get(self._name, [])
        kept = METHODS.get(cls)
        if kept is None:
            kept = METHODS[cls] = self._made_methods(cls)
        return kept

    def _made_methods(self, cls: type) -> list[Any]:
        """The ``TMethod`` of each function ``cls`` describes, then one a Python name alone made."""
        found: list[Any] = []
        spelled: set[str] = set()
        for signature in cls.__dict__.get("_cxx_methods_", ()):
            found.append(TMethod(signature[0], self, tuple(signature)))
            spelled.add(_python_name(signature[0]))
        for name in sorted(vars(cls)):
            if name in spelled or name.startswith("_") or not callable(getattr(cls, name)):
                continue
            found.append(TMethod(name, self))
        return found

    def GetMethodAny(self, name: Any) -> Any:
        return next((m for m in self.GetListOfMethods() if m.GetName() == str(name)), None)

    GetMethodAllAny = GetMethodAny

    def GetMethod(self, method: Any, params: Any = "", isconst: bool = False) -> Any:
        """``GetMethod``: the function called ``method`` that ``params`` - its parameters'
        types, comma separated - can be given, as ``GetMethodWithPrototype`` finds it."""
        return self.GetMethodWithPrototype(method, params, isconst, kConversionMatch)

    def GetMethodWithPrototype(self, name: Any, proto: Any = "", isconst: bool = False,
                               mode: int = kConversionMatch) -> Any:  # fmt: skip
        """``GetMethodWithPrototype``: the function called ``name`` that ``proto`` - its
        parameters' types - can be given, or ``nullptr`` when the class has none.

        ``isconst`` asks for one a const object can call: a const member or a static one. A
        non-const object prefers the non-const overload, as C++ overload resolution does. A
        function looked up in a class only the interpreter describes is made here and stays in
        its list, as ROOT asks the interpreter for it and keeps the ``TMethod`` it made: that is
        what lets a program hold the method across the dictionary a later
        ``GenerateDictionary`` gives the class (roottest's ``ReuseMethod``).
        """
        wanted = _parameters(proto)
        candidates = [
            found for found in self._methods()
            if found.GetName() == str(name) and found._given(wanted, int(mode), bool(isconst))
        ]  # fmt: skip
        if candidates:
            return _preferred(candidates, bool(isconst))
        if not self._made_here(str(name), str(proto)):
            return None
        made = TMethod(str(name), self)
        INTERPRETED_METHODS.setdefault(self._name, []).append(made)
        return made

    def GetFunctionTemplate(self, name: Any) -> Any:
        """``GetFunctionTemplate``: the member function template called ``name``, or ``nullptr``."""
        cls = self._class()
        if cls is None:
            return None
        kept = TEMPLATES.get(cls)
        if kept is None:
            kept = TEMPLATES[cls] = [
                TFunctionTemplate(each[0], self, tuple(each))
                for each in cls.__dict__.get("_cxx_methods_", ()) if each[7]
            ]
        return next((each for each in kept if each.GetName() == str(name)), None)

    def _made_here(self, name: str, proto: str) -> bool:
        """Is the function one this kit makes for a class only the interpreter describes - a
        member every container declares with no argument, and nothing else?"""
        return self.GetState() == K_INTERPRETED and not proto and name in CONTAINER_METHODS

    # -- objects ---------------------------------------------------------------------------

    def New(self, *args: Any) -> Any:
        """``TClass::New``: a default-made object of the class - a container of the kit's for
        a standard container's."""
        if _template(self._name):
            from .declared import new_value

            return new_value(self._name)
        cls = self._class()
        if cls is None:
            raise TypeError(f"xrdroot.pyroot has no class {self._name} to make one of")
        return cls()

    def DynamicCast(self, base: TClass, obj: Any, up: bool = True) -> Any:
        """``DynamicCast``: ``obj``, of this class, seen as of ``base`` - or, with ``up``
        false, ``obj`` of ``base`` seen as of this class again - as PyROOT gives it back:
        an object of the other type over the same state, and the original on the way back."""
        wanted = base._class() if up else self._class()
        if wanted is None or type(obj) is wanted:
            return obj
        original = obj.__dict__.get("_cast_of")
        if type(original) is wanted:
            return original
        view: Any = object.__new__(wanted)
        view.__dict__ = obj.__dict__.copy()
        view.__dict__["_cast_of"] = obj
        return view

    def Streamer(self, obj: Any, buf: Any, onfile_class: Any = None) -> None:  # type: ignore[override]
        """``Streamer``: ``obj`` written into ``buf`` - or read from it, as it goes.

        C++ overloads this name in ``TClass``: ``Streamer(obj, buf)`` here hides
        ``TObject::Streamer(buf)`` there, an object streamed by its class, and Python's one
        name for both is why the signature differs from the base's.
        """
        buf.StreamObject(obj, self)

    def Destructor(self, obj: Any, dtorOnly: bool = False) -> None:
        """``Destructor``: what the object's destructor does, which Python's collector frees."""
        destruct = getattr(obj, "_destruct", None)
        if callable(destruct):
            destruct()

    def IgnoreTObjectStreamer(self, ignore: bool = True) -> None:
        """``IgnoreTObjectStreamer``: objects of the class written without their ``TObject``."""
        from ...wobjects import IGNORED

        if ignore:
            IGNORED.add(self._name)
        else:
            IGNORED.discard(self._name)

    def CanIgnoreTObjectStreamer(self) -> bool:
        from ...wobjects import IGNORED

        return self._name in IGNORED

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TClass) and other._name == self._name

    def __hash__(self) -> int:
        return hash(self._name)

    def __repr__(self) -> str:
        return f"<TClass {self._name}>"


class TBaseClass(TNamed):
    """``TBaseClass``: one base of a class, as ``GetListOfBases`` lists it."""

    def __init__(self, name: str, owner: TClass) -> None:
        TNamed.__init__(self, name, "")
        self._owner = owner

    def GetClassPointer(self, load: bool = True) -> TClass:
        return TClass(self._name)

    def GetDelta(self) -> int:
        return 0

    def IsSTLContainer(self) -> int:
        return 0


class TDataMember(TNamed):
    """``TDataMember``: one data member of a class, as declared - transient ones included."""

    def __init__(self, owner: TClass, name: str, spelled: str, comment: str,
                 dims: tuple[Any, ...]) -> None:  # fmt: skip
        TNamed.__init__(self, str(name), str(comment))
        self._owner = owner
        self._spelled = str(spelled)
        self._dims = tuple(int(d) if d is not None else 0 for d in dims)

    def GetClass(self) -> TClass:
        return self._owner

    def GetTypeName(self) -> str:
        """The type as the dictionary names it, less the stars of a pointer: ``Double32_t``,
        ``Int_t``, ``vector<int>``."""
        return self.GetFullTypeName().rstrip("*")

    def GetFullTypeName(self) -> str:
        from ...wdeclared import normalised

        return normalised(self._spelled, resolve=False)

    def GetTrueTypeName(self) -> str:
        from ...wdeclared import normalised

        return normalised(self._spelled)

    def IsPersistent(self) -> bool:
        return not self._title.startswith("!")

    def IsaPointer(self) -> bool:
        return self._spelled.endswith("*")

    def IsBasic(self) -> bool:
        from ...wclasses import TYPES
        from ...wdeclared import parse_type

        found = parse_type(self._spelled)
        return not found.args and found.name in TYPES

    def IsEnum(self) -> bool:
        return False

    def IsSTLContainer(self) -> int:
        from ...wdeclared import STL_KINDS, parse_type

        return STL_KINDS.get(parse_type(self._spelled).name, 0)

    def GetArrayDim(self) -> int:
        return len(self._dims)

    def GetMaxIndex(self, dim: int) -> int:
        return self._dims[dim] if 0 <= dim < len(self._dims) else -1

    def GetArrayIndex(self) -> str:
        """``GetArrayIndex``: the counter a ``//[n]`` comment names, or nothing."""
        from ...wdeclared import Field, _counter_of

        counter = _counter_of(Field(self._name, self._spelled, self._title, self._dims))
        return counter or ""

    def GetUnitSize(self) -> int:
        from ...wclasses import TYPES
        from ...wdeclared import parse_type

        found = TYPES.get(parse_type(self._spelled).name)
        return 8 if self.IsaPointer() or found is None else found[2]

    def GetOffset(self) -> int:
        return 0

    def GetComment(self) -> str:
        return self._title


#: The bytes and alignment of the types whose width nothing else here needs - the kit writes
#: a container member by member and never had to know how wide one is. ROOT's ``std::vector``
#: is three pointers and ``std::string`` a pointer and a small buffer, the same on every
#: machine this runs on.
_FIXED_SIZES: dict[str, tuple[int, int]] = {
    "vector": (24, 8), "string": (32, 8), "map": (48, 8), "set": (48, 8), "list": (24, 8),
}  # fmt: skip


def _aligned(offset: int, align: int) -> int:
    """The first place at or after ``offset`` a value aligned to ``align`` starts."""
    return (offset + align - 1) // align * align


def _dims_of(spelled: str, dims: tuple[Any, ...]) -> tuple[int, ...]:
    """The dimensions a member declared as ``spelled`` with ``dims`` holds.

    ``std::array<T, N>`` is an array of ``N`` as far as a real-data listing goes, which ROOT
    names ``m_a[3]`` rather than ``m_a``, and which is the one container whose width the
    declaration itself gives.
    """
    from ...wdeclared import parse_type

    found = tuple(int(dim) for dim in dims if dim is not None)
    parsed = parse_type(str(spelled).strip())
    if parsed.name == "array" and len(parsed.args) == 2:
        found += (int(parsed.args[1].name),)
    return found


def _size_align(spelled: str, seen: tuple[str, ...] = ()) -> tuple[int, int]:
    """``sizeof`` and ``alignof`` of one value of a member declared as ``spelled``, which is
    what places the member after it: a type nothing here knows - a class only a file describes,
    a template instantiation with no layout - takes no room. ``std::array<T, N>`` is one value
    of ``T`` here, so that a caller counting its ``N`` does not count them twice."""
    from ...wclasses import TYPES
    from ...wdeclared import parse_type

    text = str(spelled).strip()
    if text.endswith("*"):
        return (8, 8)
    parsed = parse_type(text)
    if parsed.name == "array" and len(parsed.args) == 2:
        return _size_align(parsed.args[0].name, seen)
    if not parsed.args and parsed.name in TYPES:
        return (TYPES[parsed.name][2], TYPES[parsed.name][2])
    if parsed.name in _FIXED_SIZES:
        return _FIXED_SIZES[parsed.name]
    return _class_size(TClass(parsed.name), seen)


def _class_size(cls: TClass, seen: tuple[str, ...] = ()) -> tuple[int, int]:
    """The bytes an object of ``cls`` takes and what it is aligned to: each base's and then
    each own member's place, every one of them aligned to what it holds."""
    if cls.GetName() in seen:
        return (0, 1)
    seen += (cls.GetName(),)
    offset, align = 0, 1
    for base in cls._base_names():
        size, base_align = _class_size(TClass(base), seen)
        offset, align = _aligned(offset, base_align) + size, max(align, base_align)
    for _name, spelled, comment, dims in cls._fields():
        if comment.startswith("!"):
            continue
        size, member_align = _size_align(spelled, seen)
        count = 1
        for dim in _dims_of(spelled, dims):
            count *= dim
        offset, align = _aligned(offset, member_align) + size * count, max(align, member_align)
    return (_aligned(offset, align), align)


def _object_class(spelled: str) -> TClass | None:
    """The class a member declared as ``spelled`` is an object of, whose own members are that
    member's - a pointer's are not held by the object, and a container's are a listing of its
    own.

    A member of a class template's instantiation - ``MyVector<Double32_t> v1`` - is such an
    object too, and ROOT lists its members as ``v1.d32``: the instantiation is asked for by
    the whole spelling, since it is the instantiation and not the template that holds the
    layout. A standard container declares none, so a member of one has none either.
    """
    from ...wdeclared import parse_type

    text = str(spelled).strip()
    if text.endswith("*"):
        return None
    parsed = parse_type(text)
    if parsed.name in _FIXED_SIZES or parsed.name == "array":
        return None
    deeper = TClass(text)
    return deeper if deeper._fields() else None


def _bases_at(cls: TClass, prefix: str, base: int,
              seen: tuple[str, ...]) -> tuple[list[TRealData], int]:  # fmt: skip
    """Each base's members at the place the base starts, and the first offset after them."""
    made: list[TRealData] = []
    offset = 0
    for name in cls._base_names():
        size, align = _class_size(TClass(name), seen)
        offset = _aligned(offset, align)
        made.extend(_real_members(TClass(name), prefix, base + offset, seen))
        offset += size
    return made, offset


def _members_from(cls: TClass, prefix: str, base: int, offset: int,
                  seen: tuple[str, ...]) -> list[TRealData]:  # fmt: skip
    """Each persistent member of the class at the place it is held from ``offset`` on, and the
    members of the object it holds after it."""
    made: list[TRealData] = []
    listed = {member.GetName(): member for member in cls.GetListOfDataMembers()}
    for name, spelled, comment, dims in cls._fields():
        member = listed.get(name)
        if comment.startswith("!") or member is None:
            continue
        size, align = _size_align(spelled, seen)
        offset = _aligned(offset, align)
        found = _dims_of(spelled, dims)
        where = prefix + name + "".join(f"[{dim}]" for dim in found)
        made.append(TRealData(where, cls, member, base + offset))
        deeper = _object_class(spelled)
        if deeper is not None:
            made.extend(_real_members(deeper, prefix + name + ".", base + offset, seen))
        offset += size * max(1, _count(found))
    return made


def _real_members(cls: TClass, prefix: str, base: int,
                  seen: tuple[str, ...] = ()) -> list[TRealData]:  # fmt: skip
    """``TClass::BuildRealData``'s walk of one class: each persistent member at the place it
    is held, and, for a member that is an object of a class of its own, that class's members
    after it, named the member's name and then their own after a dot.

    A base's members are the object's own with no name in front of them. ROOT lists them
    after the class's own, though they are held before them, so the offset a base's members
    are given is the one the base really starts at and the order is ROOT's."""
    if cls.GetName() in seen:
        return []
    seen += (cls.GetName(),)
    inherited, offset = _bases_at(cls, prefix, base, seen)
    # ROOT lists a class's own members before the ones it inherits - ``Derived``'s ``fValue``
    # is its first real data and its base ``TObject``'s two come after - which is not the
    # order they are held in, as the base's bytes are the object's first.
    return _members_from(cls, prefix, base, offset, seen) + inherited


def _count(dims: tuple[int, ...]) -> int:
    """How many values a member of those dimensions holds."""
    count = 1
    for dim in dims:
        count *= dim
    return count


def _real_data_of(cls: TClass) -> list[TRealData]:
    """The flat list of a class's persistent members, as ROOT builds one on demand."""
    return _real_members(cls, "", 0)


class TRealData(TNamed):
    """``TRealData``: one persistent data member of an object, where the object holds it."""

    CLASS_TITLE = "Description of persistent data members"

    def __init__(self, name: str, owner: TClass, member: TDataMember, this_offset: int) -> None:
        TNamed.__init__(self, name, self.CLASS_TITLE)
        self._owner, self._member, self._this_offset = owner, member, this_offset

    def GetClass(self) -> TClass:
        return self._owner

    def GetDataMember(self) -> TDataMember:
        return self._member

    def GetThisOffset(self) -> int:
        return self._this_offset

    def GetOffset(self) -> int:
        return self._this_offset

    def IsPointer(self) -> bool:
        return self._member.IsaPointer()


class TStreamerInfo(TNamed):
    """``TStreamerInfo``: one layout of one class - its checksum, version and elements."""

    CLASS_TITLE = "Streamer information for one class version"

    def __init__(
        self, name: Any = "", info: tuple[int, int, tuple[Element, ...]] | None = None,
        built: bool = False, versioned: bool = True, from_file: bool = False,
        memory: tuple[Element, ...] | None = None,
    ) -> None:  # fmt: skip
        TNamed.__init__(self, str(name), "")
        self._checksum, self._version, elements = info if info is not None else (0, 1, ())
        self._memory = memory
        self._elements = [element_of(e) for e in elements]
        for element in self._elements:
            element._on_file = not built
        self._built = built
        self._versioned = versioned
        self._from_file = from_file

    def GetClass(self) -> TClass:
        return TClass(self._name)

    def GetClassVersion(self) -> int:
        return self._version

    def GetCheckSum(self, code: int = 0) -> int:
        return self._checksum

    def GetElements(self) -> Any:
        from .collections import TObjArray

        made = TObjArray()
        for element in self._elements:
            made.Add(element)
        return made

    def GetElement(self, index: int) -> TStreamerElement | None:
        return self._elements[index] if 0 <= index < len(self._elements) else None

    def GetNelement(self) -> int:
        return len(self._elements)

    def IsBuilt(self) -> bool:
        return self._built

    def Build(self, isTransient: bool = False) -> None:
        self._built = True

    def IsCompiled(self) -> bool:
        return self._built

    def _unmatched(self) -> bool:
        """Are the elements listed as ROOT lists those of a file's layout it cannot set
        against the class in memory - the class having a layout of that version now, with a
        checksum the file's does not have?

        ROOT then marks each element with no counterpart as a pointer to what the file
        recorded - a type a hundred above the file's own - its offset missing, and no new
        type, which is what ``checkWarn`` reads back.
        """
        mine = TClass(self._name)._layout()
        return self._from_file and mine is not None and mine[0] != self._checksum

    def _new_types(self) -> list[int]:
        """Each element's *new* type: the code the same member has in the layout in memory,
        where a file spelled it another way - an enum a file wrote as the ``int`` it was
        declared as, which today's ROOT reads as ``kEnum``.

        ROOT keeps that code on the element - ``fNewType``, which its listing names and which
        adds two hundred to the action the element is read with - and it is what
        ``BuildCheck`` reads back when it compares a file's layout with the one in memory.
        """
        if self._memory is None:
            return [0] * len(self._elements)
        mine = {element[1]: element for element in self._memory}
        found: list[int] = []
        for element in self._elements:
            other = mine.get(element._name)
            spelled_alike = other is not None and other[8] == element.GetTypeName()
            new_type = other[3] if other is not None else element.GetType()
            found.append(new_type if spelled_alike and new_type != element.GetType() else 0)
        return found

    def ls(self, option: str = "") -> None:
        """``ls``: ROOT's ``StreamerInfo for class`` line, each element, and - for the layout
        in memory, which ROOT builds - the actions each element is read with."""
        version = f", version={self._version}" if self._versioned else ""
        unmatched = self._unmatched()
        new_types = self._new_types()
        print(f"\nStreamerInfo for class: {self._name}{version}, checksum={self._checksum:#x}")
        for element, new_type in zip(self._elements, new_types, strict=True):
            element.ls(option, unmatched, new_type)
        if not self._built:
            return
        pairs = zip(self._elements, new_types, strict=True)
        for index, (element, new_type) in enumerate(pairs):
            kind = element.GetType() + (200 if new_type else 100 if unmatched else 0)
            offset = _MISSING_OFFSET if unmatched else 0
            length = max(1, element.GetArrayLength())
            print(f"   i={index:2d}, {element.GetName():<15} type={kind:3d}, "
                  f"offset={offset:3d}, len={length}, method=0")  # fmt: skip


#: ``TVirtualStreamerInfo``: the interface ``TStreamerInfo`` fills in.
class TVirtualStreamerInfo(TStreamerInfo):
    """``TVirtualStreamerInfo``: its static switches, which this writer's choices answer."""

    _member_wise = True

    @staticmethod
    def SetStreamMemberWise(enable: bool = True) -> bool:
        """``SetStreamMemberWise``: containers of objects written member by member, as ROOT 6
        writes them - the previous setting handed back."""
        previous = TVirtualStreamerInfo._member_wise
        TVirtualStreamerInfo._member_wise = bool(enable)
        return previous

    @staticmethod
    def GetStreamMemberWise() -> bool:
        return TVirtualStreamerInfo._member_wise

    @staticmethod
    def SetCanDelete(opt: bool = True) -> None:
        """``SetCanDelete``: who deletes what is read, which Python's collector decides."""

    @staticmethod
    def SetFactory(factory: Any) -> None:
        """``SetFactory``: there is one way to describe classes here."""


class TStreamerElement(TNamed):
    """``TStreamerElement``: one base or data member of a class's layout."""

    #: Whether it was read from a file, whose elements carry the bits that say how what is
    #: read is to be kept - ``kDoNotDelete``, which a listing shows.
    _on_file = False

    def __init__(self, element: Element | None = None) -> None:
        _kind, name, title, stype, size, alen, adim, maxidx, typename, extras = (
            element if element is not None else ("", "", "", 0, 0, 0, 0, (0,) * 5, "", ())
        )
        TNamed.__init__(self, name, title)
        self._stype, self._size, self._alen, self._adim = stype, size, alen, adim
        self._maxidx, self._typename, self._extras = tuple(maxidx), typename, tuple(extras)

    def GetType(self) -> int:
        return self._stype

    def GetTypeName(self) -> str:
        return self._typename

    def GetTypeNameBasic(self) -> str:
        return self._typename

    def GetSize(self) -> int:
        return self._size

    def GetArrayLength(self) -> int:
        return self._alen

    def GetArrayDim(self) -> int:
        return self._adim

    def GetMaxIndex(self, index: int) -> int:
        return self._maxidx[index] if 0 <= index < len(self._maxidx) else 0

    def GetOffset(self) -> int:
        return 0

    def IsBase(self) -> bool:
        return False

    def IsaPointer(self) -> bool:
        return self._typename.endswith("*")

    def GetFullName(self) -> str:
        """The name with its array dimensions: ``arr[2][3]``."""
        return self._name + "".join(f"[{d}]" for d in self._maxidx[: self._adim])

    def GetClassPointer(self) -> TClass | None:
        return TClass.GetClass(self._typename.rstrip("*"))

    def _sequence(self) -> str:
        return ""

    def ls(self, option: str = "", unmatched: bool = False, newtype: int = 0) -> None:
        """``ls``: the element's line of a ``TStreamerInfo`` listing.

        An element of a file's layout the class in memory cannot be set against - one
        ``unmatched`` says so of the whole layout - is listed at ROOT's missing offset, with
        no new type; one the class in memory types another way names that code instead.
        """
        where = f"offset={_MISSING_OFFSET}" if unmatched else f"offset={0:3d}"
        new = "newtype=-1 " if unmatched else f"newtype={newtype:2d} " if newtype else ""
        print(f"  {self._typename:<14} {self.GetFullName():<15} {where} "
              f"type={self._stype:2d} {new}{self._sequence()}{self._title:<20}")  # fmt: skip


class TStreamerBase(TStreamerElement):
    def IsBase(self) -> bool:
        return True

    def GetBaseVersion(self) -> int:
        return int(str(self._extras[0])) if self._extras else 0

    def ls(self, option: str = "", unmatched: bool = False, newtype: int = 0) -> None:
        where = f"offset={_MISSING_OFFSET}" if unmatched else f"offset={0:3d}"
        new = "newtype=-1 " if unmatched else f"newtype={newtype:2d} " if newtype else ""
        print(f"  {self._name:<14} {'BASE':<15} {where} "
              f"type={self._stype:2d} {new}{self._title:<20}")  # fmt: skip


class TStreamerBasicType(TStreamerElement):
    pass


class TStreamerBasicPointer(TStreamerElement):
    def GetCountName(self) -> str:
        return str(self._extras[1]) if len(self._extras) > 1 else ""


class TStreamerLoop(TStreamerBasicPointer):
    pass


class TStreamerString(TStreamerElement):
    pass


class TStreamerObject(TStreamerElement):
    pass


class TStreamerObjectAny(TStreamerElement):
    pass


class TStreamerObjectPointer(TStreamerElement):
    pass


class TStreamerObjectAnyPointer(TStreamerElement):
    pass


class TStreamerSTL(TStreamerElement):
    """``TStreamerSTL``: a standard container member - which container, and of what."""

    def GetSTLtype(self) -> int:
        return int(str(self._extras[0])) if self._extras else 0

    def GetCtype(self) -> int:
        return int(str(self._extras[1])) if len(self._extras) > 1 else 0

    def _sequence(self) -> str:
        # ROOT's "(nodelete)", which a file sets on a map and on a container of pointers.
        nodelete = self.GetSTLtype() in (4, 5, 12, 13) or self.GetCtype() == 63
        return " (nodelete) " if nodelete and self._on_file else ""

    def ls(self, option: str = "", unmatched: bool = False, newtype: int = 0) -> None:
        where = f"offset={_MISSING_OFFSET}" if unmatched else f"offset={0:3d}"
        new = "newtype=-1 " if unmatched else f"newtype={newtype:2d} " if newtype else ""
        print(f"  {self._typename:<14} {self.GetFullName():<15} {where} "
              f"type={self._stype:2d} {new}stl={self.GetSTLtype()} ctype={self.GetCtype()} "
              f"{self._sequence()}{self._title:<20}")  # fmt: skip


class TStreamerSTLstring(TStreamerSTL):
    pass


#: Each element's class, by the ``TStreamerElement`` subclass that wrote it.
_KINDS: tuple[type[TStreamerElement], ...] = (
    TStreamerBase, TStreamerBasicType, TStreamerBasicPointer, TStreamerLoop, TStreamerString,
    TStreamerObject, TStreamerObjectAny, TStreamerObjectPointer, TStreamerObjectAnyPointer,
    TStreamerSTL, TStreamerSTLstring,
)  # fmt: skip
ELEMENT_CLASSES: dict[str, type[TStreamerElement]] = {cls.__name__: cls for cls in _KINDS}


def element_of(element: Element) -> TStreamerElement:
    """The ``TStreamerElement`` an element of a layout is, of its own subclass - a container's
    ``kStreamer``, which a file says in its place, read back as the ``kSTL`` it is."""
    if element[0] in ("TStreamerSTL", "TStreamerSTLstring") and element[3] == 500:
        element = (*element[:3], 300, *element[4:])
    return ELEMENT_CLASSES.get(element[0], TStreamerElement)(element)


#: The ``EFunctionProperty`` bit each kind of C++ function is, as ``TDictionary`` numbers them.
FUNCTION_KINDS = {"constructor": kIsConstructor, "destructor": kIsDestructor,
                  "operator": kIsOperator}  # fmt: skip


def _parameters(proto: Any) -> tuple[str, ...]:
    """``GetMethod``'s parameter list as the types it is made of: ``"int, char*"`` is two.

    ``""`` is a function taking none, and a ``void`` written out is taken as none, as C++
    takes it. A type is compared with its spaces and its ``std::`` out - the spellings a
    caller may differ in - and a default it wrote out is not part of the type.
    """
    text = str(proto).strip()
    if not text:
        return ()
    found = tuple(_squashed(part) for part in _split_parameters(text))
    return () if found == ("void",) else found


def _split_parameters(text: str) -> list[str]:
    """``text`` split at the commas outside brackets, so ``map<int, char>, bool`` is two."""
    parts: list[str] = []
    depth, start = 0, 0
    for at, letter in enumerate(text):
        if letter in "<([":
            depth += 1
        elif letter in ">)]":
            depth -= 1
        elif letter == "," and not depth:
            parts.append(text[start:at])
            start = at + 1
    parts.append(text[start:])
    return parts


def _squashed(one: str) -> str:
    """A type as two spellings of it are compared: no spaces, no ``std::``, no default."""
    text = one.split("=", 1)[0]
    return text.replace(" ", "").replace("std::", "")


def _matches(wanted: str, declared: str, mode: int) -> bool:
    """Can an argument of type ``wanted`` be passed to a parameter declared ``declared``?

    An exact match takes the type as declared; a conversion match also takes the ones C++
    would - a ``char*`` for a ``const char*``, which is a pointer to more than it promises.
    """
    if wanted == declared:
        return True
    return mode == kConversionMatch and wanted == declared.replace("const", "")


def _preferred(candidates: list[Any], isconst: bool) -> Any:
    """The overload of ``candidates`` a search answers with.

    A const object's is a const - or a static - function; a non-const object's is the non-const
    overload, as C++ prefers it, and a const one where that is all the class declares.
    """
    def rank(found: Any) -> int:
        signature = found._signature
        const = signature is not None and signature[3]
        static = signature is not None and signature[4]
        if isconst:
            return 0 if const else (2 if static else 1)
        return 2 if const else (3 if static else 0)

    return min(candidates, key=rank)


def _python_name(name: str) -> str:
    """The Python attribute a C++ function's name is written as in a class of the kit's."""
    from ...pycxx.symbols import member_name

    return member_name(name)


def _instance(name: str) -> bool:
    """Is ``name`` a template made a class by its arguments - ``vector<Elem>``, ``Tmplt<int>``?"""
    return "<" in name and name.endswith(">")


def _wrapper(name: str) -> bool:
    """Is ``name`` an instance of one of the standard ways of holding one thing?"""
    return name.split("<", 1)[0] in WRAPPERS


def _arguments(name: str) -> list[str]:
    """The template arguments ``name`` is an instance with: ``map<int,char>`` gives two."""
    if "<" not in name or not name.endswith(">"):
        return []
    inside = name[name.index("<") + 1 : -1]
    return [part.strip() for part in _split_parameters(inside)]


def _class_spelling(spelled: str) -> str | None:
    """The class a member's type as written names - ``TNamed*`` is ``TNamed`` - or none for a
    type that is no class: a number, a pointer to nothing, the ``void`` of a member that is
    not there."""
    text = re.sub(r"\[[^\]]*\]", "", spelled).strip()
    while text.endswith(("*", "&")):
        text = text[:-1].strip()
    text = text.removeprefix("const ").strip()
    if not text or text in FUNDAMENTAL or text in NUMBERS or text == "void":
        return None
    if text[0].isdigit() or text[0] == "-":  # ``std::array<Elem, 4>``'s length
        return None
    return _known_as(text)


#: The standard ways of holding one thing that ROOT streams by what they hold - no dictionary
#: of their own describes them, and one is never a class a dictionary is missing for.
WRAPPERS = frozenset({"unique_ptr", "shared_ptr", "array", "tuple", "optional", "variant",
                      "atomic", "weak_ptr"})  # fmt: skip


def _classes_of(spellings: list[str]) -> list[TClass]:
    """The classes the template arguments ``spellings`` name, the ones that name a class."""
    return [TClass(name) for name in map(_class_spelling, spellings) if name is not None]


def _built_in(spelled: str) -> bool:
    """Is a template argument one ROOT's own container dictionaries instantiate - a built-in
    number or a string - rather than a class of a program's own that needs one of its own?"""
    text = _squashed(spelled)
    while text.endswith("*"):
        text = text[:-1]
    return text in NUMBERS or text in ("string", "bool", "char")


def _code_of_type(info: Any) -> int:
    """``TDataType::GetType``: the ``EDataType`` code ROOT gives the C++ type ``info``.

    The identity a ``type_info`` answers with is the Itanium mangled name, which is the key
    :mod:`.datatypes` - the family that keeps the table of codes - looks it up by.
    """
    return _datatypes.BY_MANGLED.get(str(info.name()), _datatypes.kOther_t)


class TMethod(TNamed):
    """``TMethod``: one function of a class, as ``GetListOfMethods`` lists it.

    A class C++ declared describes its functions in its translation, so a method made for it
    knows its signature - what ``GetReturnTypeName``, ``GetPrototype`` and ``ExtraProperty``
    answer with. One made from a Python name alone knows none of that, and is found only where
    no prototype is asked for.
    """

    CLASS_TITLE = "Method or function descriptor"

    def __init__(self, name: Any, owner: TClass | None = None,
                 signature: Signature | None = None) -> None:  # fmt: skip
        TNamed.__init__(self, str(name), "")
        self._owner = owner
        self._signature = signature

    def GetClass(self) -> TClass | None:
        """``GetClass``: the class the function is a member of."""
        return self._owner

    def GetReturnTypeName(self) -> str:
        """``GetReturnTypeName``: the type it returns, as it was declared - ``int&``, ``void``."""
        return "" if self._signature is None else self._signature[1]

    def GetPrototype(self) -> str:
        """``GetPrototype``: the declaration ROOT prints it as, ``int Holder::data() const``."""
        if self._signature is None:
            return f"{self.GetName()}(...)"
        returns, params, const = self._signature[1], self._signature[2], self._signature[3]
        owner = self._owner.GetName() if self._owner is not None else ""
        spelled = ", ".join(params)
        return f"{returns} {owner}::{self.GetName()}({spelled})" + (" const" if const else "")

    def ExtraProperty(self) -> int:
        """``ExtraProperty``: the ``EFunctionProperty`` bits the function is - an inlined one,
        a constructor, a destructor, an operator, a conversion."""
        if self._signature is None:
            return 0
        inlined, template, kind = self._signature[5], self._signature[7], self._signature[8]
        bits = FUNCTION_KINDS.get(kind, 0)
        if kind == "operator" and self._converts():
            bits = kIsConversion  # a conversion is no operator to ROOT, but a cast
        if inlined:
            bits |= kIsInlined
        if template:
            bits |= kIsCompiled
        return bits

    def _converts(self) -> bool:
        """Is it a conversion operator - ``operator int``, and not ``operator+``?"""
        spelled = self.GetName()[len("operator") :].strip()
        return bool(spelled) and (spelled[0].isalpha() or spelled[0] == "_")

    def _given(self, wanted: tuple[str, ...], mode: int, isconst: bool) -> bool:
        """Can it be called with arguments of ``wanted`` on a const object, or not?"""
        if self._signature is None:
            return not wanted
        const, static = self._signature[3], self._signature[4]
        declared = tuple(_squashed(each) for each in self._signature[2])
        kind = self._signature[8]
        if isconst and not (const or static) and kind not in ("constructor", "destructor"):
            return False
        if len(declared) != len(wanted):
            return False
        return all(_matches(one, other, mode)
                   for one, other in zip(wanted, declared, strict=True))  # fmt: skip


class TFunctionTemplate(TMethod):
    """``TFunctionTemplate``: a member function template, as ``GetFunctionTemplate`` finds it.

    No instance of it is made here, so what it says of its parameters is what the template
    declared them as; that it is inline and a template is what a search of it asks.
    """

    CLASS_TITLE = "Template function descriptor"


# ROOT has one ``TDataType``, and this merge found it written twice: the class a dictionary
# reports - ``GetDataType``, the name, the size - is the one above, and the codes a C++
# type's ``type_info`` is looked up in landed in :mod:`.datatypes`. The two independent
# additions are one class, and it is left under that name too, so that whether a program asks
# the namespace or the family the codes came from it gets the one class that answers both of
# ROOT's questions.
_datatypes.TDataType = TDataType  # type: ignore[attr-defined]  # one class, one name
