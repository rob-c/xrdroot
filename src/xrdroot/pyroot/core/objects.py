"""``TObject`` and ``TNamed``, and the ``TAtt`` mixins every drawable one carries.

Every class in the namespace is a :class:`TObject`: it has ROOT's name and
title, ``ClassName``, ``IsA``, ``InheritsFrom``, ``Clone``, ``Draw`` through
the hook in :mod:`.hooks`, ``Write`` to the current directory, and ``Print``
and ``ls`` in ROOT's words. A wrapper of an xrdroot object keeps it in
``._xrd`` and keeps its state there too - its name in its ``TNamed`` members,
its line colour in its ``TAttLine`` - so what a script sets is what is written.

The ``TAtt`` setters and getters are made from tables, one pair per member,
rather than written out: ``SetLineColor`` is ``fLineColor`` in the object's
``TAttLine`` members, found where the xrdroot object keeps them or kept on
the wrapper for an object that has none.
"""

from __future__ import annotations

import copy
import sys
from typing import Any

from . import hooks
from .colors import BITS

__all__ = [
    "TObject",
    "TNamed",
    "TClass",
    "TAttLine",
    "TAttFill",
    "TAttMarker",
    "TAttText",
    "TAtt3D",
    "SetOwnership",
    "addressof",
    "nullptr",
]

#: ROOT's ``nullptr``, which PyROOT spells as Python's ``None``.
nullptr = None


class Indent:
    """``TROOT::IndentLevel``: how deep ``ls`` is, one space per level."""

    level = 0

    @classmethod
    def text(cls) -> str:
        """The spaces an ``ls`` line starts with at this depth."""
        return " " * cls.level

    @classmethod
    def deeper(cls, by: int = 1) -> None:
        """``IncreaseDirLevel``, or ``DecreaseDirLevel`` with a negative ``by``."""
        cls.level = max(0, cls.level + by)


def SetOwnership(obj: Any, owned: bool) -> None:
    """``ROOT.SetOwnership``: who deletes an object, which Python's collector decides here."""


def addressof(obj: Any) -> int:
    """``ROOT.addressof``: a number standing for the object, unique while it lives."""
    return id(obj)


def _class_named(name: str) -> type | None:
    """The namespace's class called ``name``, wherever it was defined in the kit."""
    for module in list(sys.modules.values()):
        if getattr(module, "__name__", "").startswith("xrdroot.pyroot"):
            found = getattr(module, name, None)
            if isinstance(found, type) and issubclass(found, TObject):
                return found
    return None


class TClass:
    """``TClass``: what ``IsA`` answers - a class's name, and what it inherits from."""

    def __init__(self, name: str, python: type | None = None) -> None:
        self._name = str(name)
        self._python = python if python is not None else _class_named(self._name)

    def GetName(self) -> str:
        """The class's name, such as ``TH1F``."""
        return self._name

    def InheritsFrom(self, other: Any) -> bool:
        """Is this class ``other`` - a name or a ``TClass`` - or derived from it?"""
        wanted = other.GetName() if isinstance(other, TClass) else str(other)
        if wanted == self._name:
            return True
        return self._python is not None and wanted in (base.__name__ for base in self._python.mro())

    def New(self) -> Any:
        """``TClass::New``: a default-made object of the class."""
        if self._python is None:
            raise TypeError(f"xrdroot.pyroot has no class {self._name} to make one of")
        return self._python()

    @staticmethod
    def GetClass(name: Any) -> TClass:
        """``TClass::GetClass(name)``: the class called ``name``."""
        return TClass(str(name))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TClass) and other._name == self._name

    def __hash__(self) -> int:
        return hash(self._name)

    def __repr__(self) -> str:
        return f"<TClass {self._name}>"


class TObject:
    """``TObject``: the root of ROOT's classes, and of every class here."""

    #: What ``ClassDef`` says of the class, which is a ``TObject``'s title.
    CLASS_TITLE = "Basic ROOT object"

    def __init__(self, *args: Any) -> None:
        self._unique_id = 0
        if args and isinstance(args[0], TObject):
            args[0].Copy(self)

    # -- what it is -------------------------------------------------------------

    def ClassName(self) -> str:
        """The ROOT class this is, such as ``TH1F``."""
        return type(self).__name__

    def IsA(self) -> TClass:
        """``IsA``: the :class:`TClass` of this object."""
        return TClass(self.ClassName(), type(self))

    def InheritsFrom(self, other: Any) -> bool:
        """Is this object's class ``other``, or derived from it?"""
        return self.IsA().InheritsFrom(other)

    @staticmethod
    def Class() -> TClass:
        """``TObject::Class()``: the class of the base, which every class here is."""
        return TClass("TObject", TObject)

    def GetName(self) -> str:
        """A ``TObject``'s name is its class's."""
        return self.ClassName()

    def GetTitle(self) -> str:
        """A ``TObject``'s title is its class's description, as ``ClassDef`` gave it."""
        return self.CLASS_TITLE

    def GetIconName(self) -> str:
        return self.GetName()

    def GetObjectInfo(self, px: int, py: int) -> str:
        return ""

    def IsFolder(self) -> bool:
        return False

    def IsZombie(self) -> bool:
        """``IsZombie``: made but not to be used, which nothing here is left as."""
        return False

    def Hash(self) -> int:
        return hash(self.GetName())

    # -- bits --------------------------------------------------------------------

    def _bitword(self) -> int:
        """The object's bits; a wrapper keeps them in its xrdroot object's ``fBits``."""
        return int(self.__dict__.get("_bits", 0))

    def _store_bits(self, bits: int) -> None:
        self.__dict__["_bits"] = int(bits)

    def SetBit(self, bit: int, value: bool = True) -> None:
        """``SetBit(f, set)``: set - or with ``set`` false, clear - the bits ``f``."""
        bits = self._bitword()
        self._store_bits(bits | int(bit) if value else bits & ~int(bit))

    def ResetBit(self, bit: int) -> None:
        self._store_bits(self._bitword() & ~int(bit))

    def TestBit(self, bit: int) -> bool:
        return bool(self._bitword() & int(bit))

    def TestBits(self, bits: int) -> int:
        return self._bitword() & int(bits)

    def InvertBit(self, bit: int) -> None:
        self._store_bits(self._bitword() ^ int(bit))

    def GetUniqueID(self) -> int:
        return self._unique_id

    def SetUniqueID(self, uid: int) -> None:
        self._unique_id = int(uid)

    # -- copies -----------------------------------------------------------------

    def Copy(self, obj: Any) -> None:
        """``Copy(obj)``: make ``obj`` this object over again, sharing nothing."""
        obj.__dict__.update(copy.deepcopy(self.__dict__))

    def Clone(self, newname: str = "") -> Any:
        """``Clone``: a copy of this object sharing nothing, renamed if ``newname`` is given."""
        made = copy.deepcopy(self)
        if newname and hasattr(made, "SetName"):
            made.SetName(newname)
        return made

    # -- drawing, printing and writing -------------------------------------------------

    def Draw(self, option: str = "") -> None:
        """``Draw``: onto the current pad, by whatever hook the graphics installed."""
        hooks.draw_hook(self, str(option))

    def DrawClone(self, option: str = "") -> Any:
        """``DrawClone``: draw a copy, which stays as drawn whatever happens to this one."""
        made = self.Clone()
        made.Draw(option)
        return made

    def Paint(self, option: str = "") -> None:
        """``Paint``: what a pad calls to draw, which the graphics part of the kit does."""

    def Pop(self) -> None:
        """``Pop``: bring to the front of its pad, which needs a pad to mean anything."""

    def Print(self, option: str = "") -> None:
        """``Print``: ``OBJ: class<TAB>name<TAB>title``."""
        print(f"OBJ: {self.ClassName()}\t{self.GetName()}\t{self.GetTitle()}")

    def ls(self, option: str = "") -> None:
        """``ls``: the ``OBJ:`` line, indented as deep as the listing is."""
        line = f"{Indent.text()}OBJ: {self.ClassName()}\t{self.GetName()}\t{self.GetTitle()} : "
        line += str(int(self.TestBit(BITS["kCanDelete"])))
        if "noaddr" not in str(option):
            line += f" at: {hex(id(self))}"
        print(line)

    def Dump(self) -> None:
        """``Dump``: every data member, one to a line."""
        for name, value in sorted(vars(self).items()):
            print(f"{name:<20}{value!r}")

    def Write(self, name: Any = None, option: int = 0, bufsize: int = 0) -> int:
        """``Write``: write this object to the current directory, under ``name`` or its own."""
        from .directories import current_directory

        return int(current_directory().WriteTObject(self, name, option))

    def SaveAs(self, filename: str = "", option: str = "") -> None:
        """``SaveAs``: a ``.root`` file holding this object alone."""
        from .files import TFile

        target = TFile(filename or f"{self.GetName()}.root", "RECREATE")
        target.WriteTObject(self)
        target.Close()
        print(f"Info in <TObject::SaveAs>: ROOT file {target.GetName()} has been created")

    def Delete(self, option: str = "") -> None:
        """``Delete``: forget this object where it is listed, which is all Python lets go of."""
        from .directories import forget

        forget(self)

    def RecursiveRemove(self, obj: Any) -> None:
        """``RecursiveRemove``: what a container does when something in it is deleted."""

    def AppendPad(self, option: str = "") -> None:
        """``AppendPad``: onto the current pad, as ``Draw`` puts it there."""
        self.Draw(option)

    def FindObject(self, name: Any) -> Any:
        """A plain object holds nothing to find."""
        return None

    def Error(self, method: str, fmt: str, *args: Any) -> None:
        """``Error``: ROOT's message, from this object's class and ``method``."""
        from .messages import message

        message("Error", f"{self.ClassName()}::{method}", fmt, *args)

    def Warning(self, method: str, fmt: str, *args: Any) -> None:
        """``Warning``: ROOT's message, from this object's class and ``method``."""
        from .messages import message

        message("Warning", f"{self.ClassName()}::{method}", fmt, *args)

    def Info(self, method: str, fmt: str, *args: Any) -> None:
        """``Info``: ROOT's message, from this object's class and ``method``."""
        from .messages import message

        message("Info", f"{self.ClassName()}::{method}", fmt, *args)

    def __repr__(self) -> str:
        return f'<cppyy.gbl.{self.ClassName()} object ("{self.GetName()}") at {hex(id(self))}>'


class TNamed(TObject):
    """``TNamed``: an object with a name and a title."""

    def __init__(self, name: Any = "", title: Any = "") -> None:
        super().__init__()
        if isinstance(name, TNamed):
            name.Copy(self)
            return
        self._name = str(name)
        self._title = str(title)

    def GetName(self) -> str:
        return self._name

    def GetTitle(self) -> str:
        return self._title

    def SetName(self, name: Any) -> None:
        self._name = str(name)

    def SetTitle(self, title: Any = "") -> None:
        self._title = str(title)

    def SetNameTitle(self, name: Any, title: Any) -> None:
        self.SetName(name)
        self.SetTitle(title)

    def Compare(self, other: Any) -> int:
        """``Compare``: by name, as a sorted list orders them."""
        mine, theirs = self.GetName(), other.GetName()
        return int(mine > theirs) - int(mine < theirs)

    def Sizeof(self) -> int:
        return len(self.GetName()) + len(self.GetTitle())


# -- the attribute mixins ---------------------------------------------------------------------

#: Each attribute group's members and the value each starts at, as ROOT's
#: default constructors leave them.
ATTRIBUTES: dict[str, dict[str, Any]] = {
    "TAttLine": {"fLineColor": 1, "fLineStyle": 1, "fLineWidth": 1},
    "TAttFill": {"fFillColor": 1, "fFillStyle": 0},
    "TAttMarker": {"fMarkerColor": 1, "fMarkerStyle": 1, "fMarkerSize": 1.0},
    "TAttText": {
        "fTextAngle": 0.0,
        "fTextSize": 0.05,
        "fTextAlign": 11,
        "fTextColor": 1,
        "fTextFont": 62,
    },
}

#: What each member is kept as: a colour or a style is an int, a size a float.
_FLOATS = {"fMarkerSize", "fTextAngle", "fTextSize", "fLineWidth"}


def attribute_home(obj: Any, group: str) -> dict[str, Any]:
    """Where ``obj`` keeps one group's members: its xrdroot object's, or its own."""
    held = obj._attribute_holder()
    home = held.get(group) if isinstance(held, dict) else None
    if isinstance(home, dict):
        for member, value in ATTRIBUTES[group].items():
            home.setdefault(member, value)
        return home
    own: dict[str, dict[str, Any]] = obj.__dict__.setdefault("_atts", {})
    return own.setdefault(group, dict(ATTRIBUTES[group]))


def _kept(member: str, value: Any) -> Any:
    """A value as the member keeps it: a colour cast to int, a size to float."""
    if member in _FLOATS:
        return float(value)
    return int(value)


def _setter(group: str, member: str) -> Any:
    def setter(self: Any, value: Any = ATTRIBUTES[group][member]) -> None:
        attribute_home(self, group)[member] = _kept(member, value)

    setter.__doc__ = f"``Set{member[1:]}``: ``{member}`` in the object's ``{group}``."
    return setter


def _getter(group: str, member: str) -> Any:
    def getter(self: Any) -> Any:
        return attribute_home(self, group)[member]

    getter.__doc__ = f"``Get{member[1:]}``: ``{member}`` in the object's ``{group}``."
    return getter


def _alpha_setter(group: str, member: str) -> Any:
    def setter(self: Any, color: Any, alpha: float) -> None:
        attribute_home(self, group)[member] = int(color)
        attribute_home(self, group)[member.replace("Color", "Alpha")] = float(alpha)

    setter.__doc__ = f"``Set{member[1:]}Alpha``: a colour, and how opaque to draw it."
    return setter


def _dressed(cls: type, group: str) -> None:
    """Give an attribute mixin a ``Set`` and a ``Get`` for each of its group's members."""
    for member in ATTRIBUTES[group]:
        setattr(cls, f"Set{member[1:]}", _setter(group, member))
        setattr(cls, f"Get{member[1:]}", _getter(group, member))
        if member.endswith("Color"):
            setattr(cls, f"Set{member[1:]}Alpha", _alpha_setter(group, member))
    setattr(cls, f"Reset{group[1:]}", _reset(group))
    setattr(cls, f"Copy{group}", _copy_to(group))


class _Attributes:
    """What the attribute mixins share: where the object keeps its members."""

    def _attribute_holder(self) -> Any:
        """The xrdroot members holding the groups, or ``None`` to keep them on the wrapper."""
        return None


class TAttLine(_Attributes):
    """``TAttLine``: a line's colour, style and width."""


class TAttFill(_Attributes):
    """``TAttFill``: a fill's colour and style."""


class TAttMarker(_Attributes):
    """``TAttMarker``: a marker's colour, style and size."""


class TAttText(_Attributes):
    """``TAttText``: text's angle, size, alignment, colour and font."""


def _reset(group: str) -> Any:
    def reset(self: Any, option: str = "") -> None:
        attribute_home(self, group).update(ATTRIBUTES[group])

    return reset


def _copy_to(group: str) -> Any:
    def copied(self: Any, other: Any) -> None:
        attribute_home(other, group).update(attribute_home(self, group))

    return copied


for _mixed in (TAttLine, TAttFill, TAttMarker, TAttText):
    _dressed(_mixed, _mixed.__name__)


class TAtt3D:
    """``TAtt3D``: the mark of a class drawn in three dimensions, with nothing in it."""
