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

from ...wdeclared import ROOT_TYPEDEFS as TYPEDEFS
from ...winfo import INFOS, Element
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
]

#: One layout of one class: its checksum, version and elements.
Layout = tuple[int, int, tuple[Element, ...]]
#: The layouts files read so far describe each class with, in the order they were met.
FILE_INFOS: dict[str, list[Layout]] = {}

#: The classes a file's layout was reported as disagreeing with the class in memory about -
#: which ROOT warns of once per class, however many files its layouts are read from.
DISAGREED: set[str] = set()

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

#: The functions looked up in a class only the interpreter describes, by class: ROOT asks the
#: interpreter for the one a program names and keeps the ``TMethod`` it made, which is what lets
#: a program hold that method across the dictionary that replaces the interpreter's description.
INTERPRETED_METHODS: dict[str, list[TNamed]] = {}

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
    """
    for name in str(classes).split(";"):
        if name.strip():
            GENERATED.add(_known_as(name))


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
    for name, (checksum, version, elements) in infos:
        if name not in FILE_INFOS and _undictionaried(name):
            message("Warning", "TClass::Init", "no dictionary for class %s is available", name)
        met = _met(name, (checksum, version, elements), origin)
        if met or _in_memory(name, checksum, version):
            continue
        held = FILE_INFOS[name]
        taken = {found[1] for found in held}
        held.append((checksum, _free_version(name, version, taken), elements))


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


def _differing(mine: Layout, on_file: Layout) -> list[Element]:
    """The members of the file's layout ROOT says the class in memory declares differently.

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
    return [e for e in on_file[2] if e[1] in here and _differs(e, here[e[1]])]


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
    here = {element[1]: element for element in mine[2]}
    differ = _differing(mine, on_file)
    message("Warning", "TStreamerInfo::CompareContent",
            "The following data member of\nthe on-file layout version %d of class '%s' "
            "differs from\nthe in-memory layout version %d:\n   %s\nvs\n   %s",
            on_file[1], name, mine[1],
            "\n   ".join(_declaration(e) for e in differ),
            "\n   ".join(_declaration(here[e[1]]) for e in differ))


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


#: What the names of ROOT's own classes look like: each has a dictionary in ROOT.
ROOT_NAMES = re.compile(r"^(T[A-Z]|Roo|ROOT::|TMVA::|RooStats::|HistFactory::)")


def _undictionaried(name: str) -> bool:
    """Has nothing here a dictionary of the class ``name`` - neither the program nor ROOT -
    which ROOT warns of when a file it opens describes one?"""
    if ROOT_NAMES.match(name) or _template(name) or _synthetic_pair(name):
        return False
    return TClass(name)._class() is None


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
    from .declared import find_class

    return find_class(name)


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
    ``enumSize`` checks that ``enum E : short`` is stored as a ``short``.
    """

    def __init__(self, name: str = "", code: int = 0, size: int = 0) -> None:
        super().__init__(str(name), str(name))
        self._code = int(code)
        self._size = int(size)

    def GetType(self) -> int:
        """``GetType``: the ``EDataType`` code of the type."""
        return self._code

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
        layouts are told apart by checksum - and ROOT's own as the donors say."""
        info = self._layout()
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

    def GetStreamerInfo(self, version: int = 0) -> TStreamerInfo | None:
        """``GetStreamerInfo``: the layout of the class in memory - or, given a version, the
        layout of that version, from a file read if it is not the one in memory.

        A class nothing in memory describes is emulated from the first layout a file it read
        gave it, and ROOT builds what it hands back - so its elements are listed with the
        action each is read with - where a layout it merely lists stays unbuilt.
        """
        current = self._layout()
        emulated = current is None
        if emulated:
            current = self._file_layout()
        if current is not None and version in (0, current[1]):
            return TStreamerInfo(self._name, current, built=True, versioned=self._versioned())
        found = [i for i in FILE_INFOS.get(self._name, []) if i[1] == version]
        if not found:
            return None
        return TStreamerInfo(self._name, found[0], built=emulated, versioned=self._versioned())

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
            made.Add(TStreamerInfo(self._name, info, built=in_memory, versioned=versioned))
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

    def GetListOfMethods(self, load: bool = True) -> Any:
        """``GetListOfMethods``: the class's own functions, by name.

        A class only the interpreter describes lists the functions a search has made so far,
        as ROOT's list of them holds what the interpreter has been asked for.
        """
        from .collections import TList

        made = TList()
        cls = self._class()
        for found in sorted(vars(cls) if cls is not None else ()):
            if callable(getattr(cls, found)) and not found.startswith("_"):
                made.Add(TNamed(found, ""))
        if cls is None:
            for looked_up in INTERPRETED_METHODS.get(self._name, ()):
                made.Add(looked_up)
        return made

    def GetMethodAny(self, name: Any) -> Any:
        return next((m for m in self.GetListOfMethods() if m.GetName() == str(name)), None)

    GetMethodAllAny = GetMethodAny

    def GetMethodWithPrototype(self, name: Any, proto: Any = "", isconst: bool = False,
                               mode: int = 0) -> Any:  # fmt: skip
        """``GetMethodWithPrototype``: the function called ``name`` with the prototype
        ``proto``, or ``nullptr`` when the class has none.

        A function looked up in a class only the interpreter describes is made here and stays
        in its list, as ROOT asks the interpreter for it and keeps the ``TMethod`` it made: that
        is what lets a program hold the method across the dictionary a later
        ``GenerateDictionary`` gives the class (roottest's ``ReuseMethod``), where the
        dictionary would otherwise have replaced the interpreter's description.
        """
        for found in self.GetListOfMethods(False):
            if found.GetName() == str(name) and found.GetTitle() == str(proto):
                return found
        if not self._made_here(str(name), str(proto)):
            return None
        made = TNamed(str(name), str(proto))
        INTERPRETED_METHODS.setdefault(self._name, []).append(made)
        return made

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

    def Streamer(self, obj: Any, buf: Any, onfile_class: Any = None) -> None:
        """``Streamer``: ``obj`` written into ``buf`` - or read from it, as it goes."""
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


class TStreamerInfo(TNamed):
    """``TStreamerInfo``: one layout of one class - its checksum, version and elements."""

    CLASS_TITLE = "Streamer information for one class version"

    def __init__(self, name: Any = "", info: tuple[int, int, tuple[Element, ...]] | None = None,
                 built: bool = False, versioned: bool = True) -> None:  # fmt: skip
        TNamed.__init__(self, str(name), "")
        self._checksum, self._version, elements = info if info is not None else (0, 1, ())
        self._elements = [element_of(e) for e in elements]
        for element in self._elements:
            element._on_file = not built
        self._built = built
        self._versioned = versioned

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

    def ls(self, option: str = "") -> None:
        """``ls``: ROOT's ``StreamerInfo for class`` line, each element, and - for the layout
        in memory, which ROOT builds - the actions each element is read with."""
        version = f", version={self._version}" if self._versioned else ""
        print(f"\nStreamerInfo for class: {self._name}{version}, checksum={self._checksum:#x}")
        for element in self._elements:
            element.ls(option)
        if not self._built:
            return
        for index, element in enumerate(e for e in self._elements):
            print(f"   i={index:2d}, {element.GetName():<15} type={element.GetType():3d}, "
                  f"offset={0:3d}, len={max(1, element.GetArrayLength())}, method=0")  # fmt: skip


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

    def ls(self, option: str = "") -> None:
        """``ls``: the element's line of a ``TStreamerInfo`` listing."""
        print(f"  {self._typename:<14} {self.GetFullName():<15} offset={0:3d} "
              f"type={self._stype:2d} {self._sequence()}{self._title:<20}")  # fmt: skip


class TStreamerBase(TStreamerElement):
    def IsBase(self) -> bool:
        return True

    def GetBaseVersion(self) -> int:
        return int(str(self._extras[0])) if self._extras else 0

    def ls(self, option: str = "") -> None:
        print(f"  {self._name:<14} {'BASE':<15} offset={0:3d} "
              f"type={self._stype:2d} {self._title:<20}")  # fmt: skip


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

    def ls(self, option: str = "") -> None:
        print(f"  {self._typename:<14} {self.GetFullName():<15} offset={0:3d} "
              f"type={self._stype:2d} stl={self.GetSTLtype()} ctype={self.GetCtype()} "
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
