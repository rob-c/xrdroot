"""``TFile``, ``TDirectoryFile`` and ``TKey``: ROOT files, read, written and updated.

``TFile("out.root", "RECREATE")`` is :func:`xrdroot.create`, ``"UPDATE"``
:func:`xrdroot.update` over the file already there, and ``"READ"``
:func:`xrdroot.open_root`, which reads any URL the library can; the file
opened becomes the current directory, so the histograms a script books
after it are kept in it and ``Write()`` writes them. ``Get`` hands back the
namespace's classes - a ``TH1F`` read is a ``TH1F`` - and keeps a histogram
read in the directory's memory, as ROOT does, so a second ``Get`` is the
same object. ``Close`` finishes the file; one still open when the program
ends is closed then, as ROOT closes it.

``ls`` prints ROOT's listing: the file, what is in memory, then every key,
the newest cycle of a name marked ``[current cycle]`` when an older is
kept too.
"""

from __future__ import annotations

import atexit
import fnmatch
import io
import os
import shutil
import struct
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...compression import spelled
from ...cppstd.containers import std
from ...errors import ROOTError
from ...wdeclared import COMPLEX_TEMPLATES, Catalogue, parse_type
from ...wdobjects import write_value
from ...writer import WBuffer, info_entry
from .collections import THashList
from .declared import catalogue_for, made_from, payload_of, plain_value
from .directories import TDirectory, current_directory, forget_file, set_current
from .messages import from_script, message
from .objects import Indent, TNamed, TObject, templated
from .records import recorded
from .references import file_process, settle
from .wrapping import from_members, wrap

__all__ = ["TFile", "TFileOpenHandle", "TMemFile", "TDirectoryFile", "TKey"]

#: ``TFile``'s modes, as ROOT reads the option: upper-cased, with its synonyms.
MODES = {
    "": "READ",
    "READ": "READ",
    "NEW": "CREATE",
    "CREATE": "CREATE",
    "RECREATE": "RECREATE",
    "UPDATE": "UPDATE",
    "READ_WITHOUT_GLOBALREGISTRATION": "READ",
}
#: The setting ROOT writes a file with when it is not told one: zlib at level 1.
DEFAULT_COMPRESSION = 101
#: ``TFile::fgReadaheadSize``: the bytes a batch of blocks is read ahead by - ROOT's 256 kB,
#: shared by every file because ROOT's is a static.
READAHEAD_SIZE = 256000
#: The classes a directory keeps in memory once read, as ``TH1`` and ``TTree`` add themselves.
KEPT = ("TH1", "TTree")


class TOther(TObject):
    """An object of a class the namespace has no wrapper for: its class, name and members.

    ``._xrd`` is what xrdroot read - its members, or its own object - so a
    script can still ask its name, and the rest is there to be looked at.
    """

    def __init__(self, classname: str = "TObject", xrd: Any = None) -> None:
        super().__init__()
        self._classname, self._xrd = str(classname), xrd

    def _named(self) -> dict[str, Any]:
        named = self._xrd.get("TNamed") if isinstance(self._xrd, dict) else None
        return named if isinstance(named, dict) else {}

    def ClassName(self) -> str:
        return self._classname

    def GetName(self) -> str:
        return str(self._named().get("fName", self._classname))

    def GetTitle(self) -> str:
        return str(self._named().get("fTitle", ""))


class TKey(TNamed):
    """``TKey``: the label in front of a record - its class, name, title and cycle."""

    CLASS_TITLE = "Header description of a logical record on file"

    def __init__(self, name: Any = "", title: Any = "", classname: str = "", cycle: int = 1,
                 directory: Any = None, key: Any = None) -> None:  # fmt: skip
        super().__init__(name, title)
        self._classname, self._cycle = str(classname), int(cycle)
        self._directory, self._key = directory, key

    def GetClassName(self) -> str:
        return self._classname

    def GetCycle(self) -> int:
        return self._cycle

    def GetMotherDir(self) -> Any:
        return self._directory

    def GetNbytes(self) -> int:
        return int(getattr(self._key, "nbytes", 0))

    def GetObjlen(self) -> int:
        return int(getattr(self._key, "objlen", 0))

    def GetKeylen(self) -> int:
        return int(getattr(self._key, "keylen", 0))

    def GetSeekKey(self) -> int:
        return int(getattr(self._key, "seek_key", 0))

    def GetDatime(self) -> Any:
        from .timing import TDatime

        made = TDatime()
        when = getattr(self._key, "time", None)
        if when is not None:
            made.Set(when.year, when.month, when.day, when.hour, when.minute, when.second)
        return made

    def IsFolder(self) -> bool:
        return self._classname in ("TDirectory", "TDirectoryFile")

    @templated
    def ReadObj(self) -> Any:
        """``ReadObj``: the object this key labels, this cycle of it."""
        return self._directory.Get(f"{self.GetName()};{self._cycle}")

    ReadObjectAny = ReadObject = ReadObj

    def Print(self, option: str = "") -> None:
        print(f"TKey Name = {self.GetName()}, Title = {self.GetTitle()}, Cycle = {self._cycle}")

    def ls(self, current: Any = None) -> None:
        """``ls``: ``KEY: class<TAB>name;cycle<TAB>title``, and which cycle, when told."""
        line = f"{Indent.text()}KEY: {self._classname}\t{self.GetName()};{self._cycle}"
        line += f"\t{self.GetTitle()}"
        if isinstance(current, bool):
            line += " [current cycle]" if current else " [backup cycle]"
        print(line)


def _listing_marks(keys: list[TKey]) -> list[Any]:
    """What ``ls`` says of each key: current or backup, when a name has more than one cycle."""
    marks: list[Any] = []
    for at, key in enumerate(keys):
        first = at == 0 or keys[at - 1].GetName() != key.GetName()
        backup = at + 1 < len(keys) and keys[at + 1].GetName() == key.GetName()
        marks.append((True if backup else None) if first else False)
    return marks


class TDirectoryFile(TDirectory):
    """``TDirectoryFile``: a directory in a file - its keys on file, and what is in memory."""

    CLASS_TITLE = "Describe directory structure in a ROOT file"

    def __init__(
        self, name: Any = "", title: Any = "", classname: str = "", mother: Any = None
    ) -> None:
        super().__init__(name, title, mother=mother)
        self._keys: list[TKey] = []
        self._subdirs: dict[str, TDirectoryFile] = {}
        self._read: dict[str, Any] = {}
        self._writer: Any = None
        if mother is not None:
            self._keys_from(self._reader())

    # -- where it is on file ------------------------------------------------------------

    def _path(self) -> str:
        """Where this directory is from the top of its file: ``""`` for the file itself."""
        if self._mother is None or not isinstance(self._mother, TDirectoryFile):
            return ""
        above = self._mother._path()
        return f"{above}/{self.GetName()}" if above else self.GetName()

    def GetFile(self) -> Any:
        return self._mother.GetFile() if isinstance(self._mother, TDirectoryFile) else None

    def _reader(self) -> Any:
        """The xrdroot directory this is on file, or ``None`` for one only being written."""
        top = self.GetFile()._reading if self.GetFile() is not None else None
        if top is None:
            return None
        path = self._path()
        if not path:
            return top
        try:
            return top[path]
        except KeyError:
            return None  # a directory made in this session, not on file yet

    def _writable(self) -> Any:
        """The xrdroot directory this writes into, or ``None`` for a file opened to read.

        Another part of the kit - a tree being written - asks for this to put
        its records where this directory is.
        """
        held = self.GetFile()
        top = held._writing if held is not None else None
        if top is None:
            return None
        if self._writer is None:
            path = self._path()
            self._writer = top.mkdir(path) if path else top
        return self._writer

    @property
    def _xrd(self) -> Any:
        """What another part of the kit writes a tree into: the xrdroot directory, keys noted."""
        writable = self._writable()
        return None if writable is None else _Noting(writable, self)

    def _keys_from(self, reader: Any) -> None:
        if reader is None:
            return
        names = list(dict.fromkeys(key.name for key in reader.all_keys()))
        found = sorted(reader.all_keys(), key=lambda key: (names.index(key.name), -key.cycle))
        for key in found:
            classname = "TDirectoryFile" if key.classname == "TDirectory" else key.classname
            self._keys.append(TKey(key.name, key.title, classname, key.cycle, self, key))

    def IsWritable(self) -> bool:
        return self.GetFile() is not None and self.GetFile()._writing is not None

    # -- keys and reading -----------------------------------------------------------------

    def GetListOfKeys(self) -> THashList:
        made = THashList()
        for key in self._keys:
            made.Add(key)
        return made

    def FindKey(self, keyname: Any) -> Any:
        """``FindKey("name;cycle")``: the key of that cycle, or the newest without one."""
        name, _, cycle = str(keyname).partition(";")
        return self.GetKey(name, int(cycle) if cycle else 9999)

    def FindKeyAny(self, keyname: Any) -> Any:
        """``FindKeyAny``: the key here, or in the first directory below that has it."""
        found = self.FindKey(keyname)
        if found is not None:
            return found
        for key in self._keys:
            below = self._named_directory(key.GetName()) if key.IsFolder() else None
            found = below.FindKeyAny(keyname) if below is not None else None
            if found is not None:
                return found
        return None

    def GetNkeys(self) -> int:
        return len(self._keys)

    def GetKey(self, name: Any, cycle: int = 9999) -> Any:
        """``GetKey``: the key of ``name`` - its newest cycle, or ``cycle`` if given."""
        found = [key for key in self._keys if key.GetName() == str(name)]
        chosen = [key for key in found if key.GetCycle() == cycle] or found
        return max(chosen, key=lambda key: key.GetCycle()) if chosen else None

    @templated
    def Get(self, namecycle: Any) -> Any:
        """``Get("name")``, ``Get("name;2")`` or ``Get("dir/name")``: an object here or below."""
        text = str(namecycle)
        head, _, rest = text.partition("/")
        if rest:
            below = self.GetDirectory(head)
            return None if below is None else below.Get(rest)
        name, _, cycle = text.partition(";")
        if not cycle:
            here = self._list.FindObject(name)
            if here is not None:
                return here
        return self._read_key(name, cycle)

    def _read_key(self, name: str, cycle: str) -> Any:
        """The object a key labels, read once, kept in memory if ROOT keeps its kind."""
        key = self.GetKey(name, int(cycle) if cycle else 9999)
        if key is None:
            return None
        if key.IsFolder():
            return self._subdirectory(key.GetName(), key.GetTitle())
        label = f"{key.GetName()};{key.GetCycle()}"
        if label not in self._read:
            self._read[label] = self._fetched(label, key)
        return self._read[label]

    def _fetched(self, label: str, key: TKey) -> Any:
        """Read one record; a histogram or tree of the newest cycle stays in memory here."""
        if key.GetClassName() == "ROOT::RNTuple":  # an RNTuple is read by its own reader
            return self._reading[key.GetName()]
        read = self._record_of(label, key)
        found = _object_of(key, read)
        settle(found, read, self.GetFile())
        newest = key is self.GetKey(key.GetName())
        if newest and any(_inherits(found, kind) for kind in KEPT):
            self.Append(found, True)
        return found

    def _record_of(self, label: str, key: TKey) -> Any:
        """A key's record as read: from the file - or, from one still being written, from
        what this session wrote under it."""
        reader = self._reader()
        again = getattr(key._key, "read", None) if reader is None else None
        return again() if again is not None else reader[label]

    def _subdirectory(self, name: str, title: str = "") -> TDirectoryFile:
        if name not in self._subdirs:
            self._subdirs[name] = TDirectoryFile(name, title, mother=self)
        return self._subdirs[name]

    def GetDirectory(self, path: Any, printError: bool = False, funcname: str = "") -> Any:
        """``GetDirectory``: ``"a/b"`` below, ``".."`` up, ``"file.root:/a"`` from the top."""
        text = str(path)
        if ":" in text:
            text = text.split(":", 1)[1]
            here: Any = self.GetFile()
        else:
            here = self
        for part in [p for p in text.split("/") if p and p != "."]:
            here = here._mother if part == ".." else here._named_directory(part)
            if here is None:
                return None
        return here

    def _named_directory(self, name: str) -> Any:
        found = self._list.FindObject(name)
        if isinstance(found, TDirectory):
            return found
        key = self.GetKey(name)
        return (
            self._subdirectory(name, key.GetTitle()) if key is not None and key.IsFolder() else None
        )

    def __getattr__(self, name: str) -> Any:
        """``f.hpx``: an object by name as an attribute, as PyROOT's files give them."""
        if name.startswith("_"):
            raise AttributeError(name)
        found = self.Get(name)
        if found is None:
            raise AttributeError(f"{self.ClassName()} {self.GetName()!r} has no object {name!r}")
        return found

    # -- writing ----------------------------------------------------------------------------

    def WriteTObject(self, obj: Any, name: Any = None, option: Any = "", bufsize: int = 0) -> int:
        """``WriteTObject``: ``obj`` written here under ``name`` or its own, as the next cycle."""
        writer = self._writable()
        called = str(name) if name else obj.GetName()
        if writer is None:
            message(
                "Error",
                "TDirectoryFile::WriteTObject",
                "Directory %s is not writable",
                self.GetName(),
            )
            return 0
        own = getattr(obj, "_write_into", None)
        if own is not None:
            return int(own(self, called))
        declared = payload_of(obj)
        if declared is not None:  # an object of a class C++ declared, by its declaration
            writer.write(called, declared, title=declared.title)
            key = self._note_key(called, declared.title, declared.classname)
            key._key = _Written(declared)
            return 1
        writer.write(called, recorded(obj, writer), title=obj.GetTitle())
        self._note_key(called, obj.GetTitle(), obj.ClassName())
        return 1

    def WriteObjectAny(
        self, obj: Any, classname: Any, name: Any, option: Any = "", bufsize: int = 0
    ) -> int:
        """``WriteObjectAny(obj, "MyClass", name)``: ``obj`` written here as an object of
        ``classname``, which is the name its key carries.

        An object of a class that says what it is - one C++ declared, or one of ROOT's that
        knows its own name - is written as itself, the class name given agreeing with it. A
        value of no such class, a ``complex<float>`` written as one, has only the name it is
        handed, so the class name given is the one both its key and its record carry.
        """
        if payload_of(obj) is not None or hasattr(obj, "GetTitle"):
            return self.WriteTObject(obj, name, option)
        return self._write_any(str(classname), obj, name)

    def _write_any(self, classname: str, value: Any, name: Any) -> int:
        """``value`` written here under ``classname``, which no class of its Python type is."""
        writer = self._writable()
        called = str(name) if name else classname
        if writer is None:
            message(
                "Error",
                "TDirectoryFile::WriteObjectAny",
                "Directory %s is not writable",
                self.GetName(),
            )
            return 0
        writer.write(called, _ValueObject(classname, value))
        self._note_key(called, "", classname)
        return 1

    def _note_key(self, name: str, title: str, classname: str) -> TKey:
        """A key for a record just written: the next cycle, listed before the older ones."""
        older = [key for key in self._keys if key.GetName() == name]
        cycle = max((key.GetCycle() for key in older), default=0) + 1
        made = TKey(name, title, classname, cycle, self)
        at = self._keys.index(older[0]) if older else len(self._keys)
        self._keys.insert(at, made)
        return made

    def Write(self, name: Any = None, option: int = 0, bufsize: int = 0) -> int:
        """``Write``: every object in memory here written, and every directory below its own.

        ROOT writes the file's bookkeeping as it goes, so what has been written is a file on
        disk that a program can open and read straight away - ``testTChainSaveAsCxx`` and
        ROOT's own ``w.C`` probe both write a file and read it back in one run. The file stays
        open for writing and what is written next is appended.
        """
        if self._writable() is None:
            message(
                "Error", "TDirectoryFile::Write", "file %s not opened in write mode", self.GetName()
            )
            return 0
        total = super().Write(name, option, bufsize)
        self._write_trees()
        held = self.GetFile()
        assert held is not None, "a directory that writes is one with a file under it"
        held._writing.checkpoint()
        return total

    def mkdir(self, name: Any, title: Any = "", returnExistingDirectory: bool = False) -> Any:
        """``mkdir``: a directory below this one - on file too - and ``a/b`` makes both."""
        if self.GetFile() is None:  # a directory in no file: its directories are in memory too
            return super().mkdir(name, title, returnExistingDirectory)
        head, _, rest = str(name).strip("/").partition("/")
        found = self._named_directory(head)
        if found is None:
            if self._writable() is None:
                message(
                    "Error",
                    "TDirectoryFile::mkdir",
                    "Can not create directory %s in a file opened to read",
                    head,
                )
                return None
            found = self._subdirectory(head, str(title) or head)
            self.Append(found)
            found._writable()
            self._note_key(head, found.GetTitle(), "TDirectoryFile")
        elif not rest and not returnExistingDirectory:
            message("Error", "TDirectoryFile::mkdir", "An object with name %s exists already", head)
            return None
        return found.mkdir(rest, title, returnExistingDirectory) if rest else found

    # -- listing -----------------------------------------------------------------------------

    def ls(self, option: str = "") -> None:
        """``ls``: this directory, what is in memory here, then every key, one level in."""
        print(f"{Indent.text()}{self.ClassName()}*\t\t{self.GetName()}\t{self.GetTitle()}")
        text = str(option).strip()
        Indent.deeper()
        try:
            if not text.startswith("-d"):
                self._ls_objects(option)
            if not text.startswith("-m"):
                self._ls_keys(text[2:] if text.startswith("-d") else text)
        finally:
            Indent.deeper(-1)

    def _ls_objects(self, option: str) -> None:
        from .directories import _chosen

        for obj in _chosen(self._list, option):
            obj.ls(option)

    def _ls_keys(self, pattern: str) -> None:
        import fnmatch

        keys = [
            key for key in self._keys if not pattern or fnmatch.fnmatchcase(key.GetName(), pattern)
        ]
        for key, mark in zip(keys, _listing_marks(keys), strict=False):
            key.ls(mark)

    def MakeProject(self, dirname: Any, classes: Any = "*", option: Any = "new") -> None:
        """``TFile::MakeProject``: what classes the file describes written out as a project.

        ROOT writes, under ``dirname``, a header per class no loaded dictionary already
        describes, and the project header, LinkDef and ``G__`` source that make a library of
        them; a reader then includes ``<dirname>/<Class>.h`` and gets the class back. What
        the file describes is what is written - its streamer infos, and the C++ type of an
        RNTuple record's fields - so a file that describes a class in neither way writes no
        header for it, rather than one guessed from the values.

        ``classes`` selects by name, as ``*`` selects everything; ``option`` with
        ``recreate`` in it empties ``dirname`` first. ROOT compiles the library when the
        option ends in ``+``; nothing is compiled here, because a translation reads the
        header it is given, so the project is usable as soon as it is written.
        """
        write_project(self, str(dirname), str(classes), str(option))

    def _write_trees(self) -> None:
        """The trees not yet written - but not one with no name, as ``new TTree()`` makes."""
        for tree in [obj for obj in self._list if _inherits(obj, "TTree")]:
            if not getattr(tree, "_written", True) and tree.GetName():
                tree.Write()

    def Close(self, option: str = "") -> None:
        """``Close``: trees not yet written written, then what is in memory here forgotten."""
        for below in self._subdirs.values():
            below.Close(option)
        if self._writable() is not None:
            self._write_trees()
        self.Clear()
        here = current_directory()
        if here is self:
            set_current(self._mother if self._mother is not None else _top())


def _compression(settings: int) -> tuple[str | None, int | None]:
    """The writer's algorithm and level for one of ROOT's settings; level 0 stores raw."""
    return spelled(settings)


def _top() -> Any:
    from .troot import gROOT

    return gROOT


#: The classes of tree a key may label, which the trees part of the kit reads.
TREES = ("TTree", "TNtuple", "TNtupleD")


def _tree(read: Any, key: TKey) -> Any:
    """A tree read, as the trees part of the kit hands one back - ``None`` for anything else."""
    import importlib
    import importlib.util

    if key.GetClassName() not in TREES or importlib.util.find_spec("xrdroot.pyroot.trees") is None:
        return None
    trees = importlib.import_module("xrdroot.pyroot.trees")
    return trees.wrap(read, key.GetClassName(), key._key)


class _Written:
    """A record of an object of a declared class this session wrote, kept so that the file,
    still open for writing, can read it back - and say how long it is."""

    def __init__(self, declared: Any) -> None:
        self.classname, self.catalogue = declared.classname, declared.catalogue
        self.payload = declared.root_payload(0)[0]
        self.objlen = self.nbytes = len(self.payload)

    def read(self) -> Any:
        from ...buffer import Buffer
        from ...rdobjects import reader_for

        return reader_for(self.classname, self.catalogue)(Buffer(self.payload))


def _note_layouts(reading: Any) -> None:
    """Note the layouts a file opened describes its classes with, as ROOT notes them against
    each class on opening it - a file whose description cannot be read noting none."""
    from ...errors import FormatError
    from ...streamers import read_infos
    from .reflection import note_file_infos

    try:
        note_file_infos(read_infos(reading._source))
    except (FormatError, ROOTError):
        return


def _object_of(key: TKey, read: Any) -> Any:
    """The object a key's record read is: a tree made of it, the pyroot class standing for it -
    or, a record nothing made an object of yet, the class its members and name describe."""
    found = _tree(read, key) or wrap(read)
    if found is read and not isinstance(found, TObject) and not _engine_own(found):
        # Nothing made the record into an object, so it is one read as its members: a value
        # the kit makes instead - a std::complex - is an object already, and is left as it is.
        named = str(getattr(found, "classname", "") or key.GetClassName())  # as it was read
        return _dressed(key.GetClassName(), found, named)
    return found


def _dressed(classname: str, members: Any, named: str) -> Any:
    """A record read as its members, made into its class: one the kit makes again, one C++
    declared, a value such as a ``std::complex``, or else an object that only says what it is
    - by ``named``, the class it was read as."""
    made = from_members(classname, members)  # a null TRef is false, and made
    if made is None:
        made = made_from(classname, members)
    if made is None:
        made = _value_of(classname, members)
    return made if made is not None else TOther(named, members)


def _value_of(classname: str, members: Any) -> Any:
    """A class the kit has a value of rather than a class of ours: a ``std::complex<T>``,
    whose record in a file is its ``_real`` and ``_imag`` - the two numbers of that width
    made the complex they are - and ``None`` for every other class."""
    found = parse_type(classname)
    if found.name not in COMPLEX_TEMPLATES or found.stars or not found.args:
        return None
    if not isinstance(members, Mapping):
        return None
    made = std.complex[found.args[0].text()]
    return made(members.get("_real", 0.0), members.get("_imag", 0.0))


class _Noting:
    """A writable xrdroot directory that notes, as a key of its owner, each tree made in it."""

    def __init__(self, writable: Any, owner: TDirectoryFile) -> None:
        self._writable, self._owner = writable, owner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._writable, name)

    def tree(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """``tree``: the tree made as the writable directory makes it, and its key noted."""
        made = self._writable.tree(name, *args, **kwargs)
        held = self._owner._list.FindObject(name)
        classname = held.ClassName() if held is not None else "TTree"
        self._owner._note_key(name, str(kwargs.get("title") or ""), classname)
        return made


def _inherits(obj: Any, kind: str) -> bool:
    return bool(hasattr(obj, "InheritsFrom") and obj.InheritsFrom(kind))


def _file_title(reading: Any) -> str:
    """The title a file was written with, from the key in front of its own record."""
    from ...buffer import Buffer
    from ...file import Key

    source = reading._source
    begin = struct.unpack(">i", source.read(8, 4))[0]
    nbytes = struct.unpack(">i", source.read(begin, 4))[0]
    return str(Key(Buffer(source.read(begin, min(nbytes, 512)))).title)


def _exists(name: str) -> bool:
    """Is there a file at ``name`` already? Asked only of a local one; a URL is taken to be."""
    if not _local(name):
        return True
    return os.path.exists(name[7:] if name.startswith("file://") else name)


def _local(name: str) -> bool:
    return "://" not in name or name.startswith("file://")


def _fetched(name: str) -> Any:
    """What a file is read from: its path, or - for a URL - the bytes fetched from it."""
    from ...remote import fetch, is_remote

    return io.BytesIO(fetch(name)) if is_remote(name) else name


class TFileOpenHandle(TNamed):
    """``TFileOpenHandle``: a file ``AsyncOpen`` was asked to open, opened by ``Open``."""

    def __init__(
        self, name: Any = "", option: Any = "READ", ftitle: Any = "", compress: int = 101
    ) -> None:
        super().__init__(str(name), str(ftitle))
        self._args = (str(name), option, ftitle, compress)


class TFile(TDirectoryFile):
    """``TFile``: a ROOT file - read, made anew, or added to - and the directory at its top."""

    CLASS_TITLE = "ROOT file"

    #: ``TFile::fgReadaheadSize``: one setting for every file, as ROOT's static is.
    _readahead_size = READAHEAD_SIZE

    def __init__(
        self,
        fname: Any = "",
        option: Any = "READ",
        ftitle: Any = "",
        compress: int = DEFAULT_COMPRESSION,
    ) -> None:
        name = str(fname)
        super().__init__(name, str(ftitle))
        self._option = MODES.get(str(option).upper().strip(), "READ")
        self._compress = int(compress)
        self._reading: Any = None
        self._writing: Any = None
        self._cache_read: Any = None
        self._cache_reads: list[tuple[Any, Any]] = []
        self._zombie = False
        self._open(name)
        if self._zombie and from_script():  # PyROOT's pythonization of the constructor
            raise OSError(f"Failed to open file {name}")
        if not self._zombie:
            from .troot import gROOT

            gROOT.GetListOfFiles().Add(self)
            _OPEN.append(self)
            self.cd()

    def _open(self, name: str) -> None:
        """Open the file the way the mode says, or leave a zombie with ROOT's error."""
        exists = _exists(name)
        mode = self._option
        if mode == "CREATE" and exists:
            message("Error", "TFile::TFile", "file %s already exists", name)
            self._zombie = True
            return
        if mode == "READ" or (mode == "UPDATE" and exists):
            self._read_file(name)
        if not self._zombie and mode != "READ":
            self._write_file(name, exists and mode == "UPDATE")

    def _read_file(self, name: str) -> None:
        from ...file import open_root

        try:
            self._reading = open_root(_fetched(name))
        except (OSError, ValueError, ROOTError) as why:
            missing = isinstance(why, FileNotFoundError)
            text = "file %s does not exist" if missing else "file %s is not a ROOT file"
            message("Error", "TFile::TFile", text, name)
            self._zombie = True
            return
        if not self.GetTitle():
            self.SetTitle(_file_title(self._reading))
        self._keys_from(self._reading)
        _note_layouts(self._reading)

    def _write_file(self, name: str, updating: bool) -> None:
        from ...writer import create
        from ...wupdate import update

        if updating:
            self._writing = update(name)
        else:
            algorithm, level = _compression(self._compress)
            self._writing = create(
                name, title=self.GetTitle(), compression=algorithm, level=level
            )
        self._writer = self._writing

    @staticmethod
    def Open(
        name: Any, option: Any = "READ", ftitle: Any = "", compress: int = 101, netopt: int = 0
    ) -> Any:
        """``TFile::Open``: the file, or ``None`` - a null pointer - if it would not open.

        ``name`` may be what ``AsyncOpen`` gave, which opens now as it was asked to then."""
        if isinstance(name, TFileOpenHandle):
            name, option, ftitle, compress = name._args
        made = TFile(name, option, ftitle, compress)
        return None if made.IsZombie() else made

    @staticmethod
    def AsyncOpen(
        name: Any, option: Any = "READ", ftitle: Any = "", compress: int = 101, netopt: int = 0
    ) -> TFileOpenHandle:
        """``TFile::AsyncOpen``: a handle to open the file with later, by ``TFile::Open``."""
        return TFileOpenHandle(name, option, ftitle, compress)

    @staticmethod
    def Cp(src: Any, dst: Any, progressbar: Any = True, buffersize: Any = 1000000) -> bool:
        """``TFile::Cp(src, dst)``: a file - here, or at an ``http`` or ``https`` URL - copied
        to ``dst``; ``false``, with ROOT's error, when it cannot be read; the progress bar
        ROOT draws on the standard error, at its end."""
        from ...remote import fetch

        name = str(src)
        try:
            data = fetch(name)
        except OSError as why:  # a connection refused or a file missing alike
            message("Error", "TFile::Cp", "cannot open source file %s: %s", name, str(why))
            return False
        Path(str(dst)).write_bytes(data)
        if progressbar:
            sys.stderr.write(f"[TFile::Cp] Total {len(data) / 1048576:.2f} MB\t|"
                             f"{'=' * 20}| 100.00 % \r\n")  # fmt: skip
        return True

    # -- what it is -----------------------------------------------------------------------

    def GetFile(self) -> Any:
        return self

    def WriteProcessID(self, pid: Any) -> int:
        """``WriteProcessID``: the file's index for the process ``pid`` - its ``ProcessID``
        record written now if the file has none - or -1 for a file not being written.

        ``pid`` is the process itself, or the UUID another file titled its record with: a
        clone copies references whose process the new file has only ever seen from that
        side, and what it has to write is the record the reference was numbered against.
        """
        writer = self._writable()
        if writer is None:
            return -1
        from .references import TProcessID

        held = pid if pid is not None else TProcessID.GetPID()
        return int(writer.process_index(held if isinstance(held, str) else held.GetTitle()))

    def ReadProcessID(self, index: int) -> Any:
        """``ReadProcessID``: the process the file's ``ProcessID{index}`` names, as the session
        has it, or ``None`` if the file has none of that index."""
        return file_process(self, int(index))

    def IsZombie(self) -> bool:
        return self._zombie

    def IsOpen(self) -> bool:
        return not self._zombie and (self._reading is not None or self._writing is not None)

    def GetOption(self) -> str:
        """``GetOption``: ``READ``, ``UPDATE`` - or ``CREATE``, which ``RECREATE`` is once done."""
        return "CREATE" if self._option == "RECREATE" else self._option

    def GetSize(self) -> int:
        """``GetSize``: the file's size in bytes, as it is now."""
        name = self.GetName()
        return os.path.getsize(name) if _local(name) and os.path.exists(name) else -1

    def GetEND(self) -> int:
        return self.GetSize()

    def GetVersion(self) -> int:
        return int(getattr(self._reading, "version", 64101))

    def GetCompressionSettings(self) -> int:
        if self._writing is not None:
            return int(self._writing._codes)
        return int(getattr(self._reading, "compression", DEFAULT_COMPRESSION))

    def GetCompressionLevel(self) -> int:
        return self.GetCompressionSettings() % 100

    def GetCompressionAlgorithm(self) -> int:
        return self.GetCompressionSettings() // 100

    def SetCompressionLevel(self, level: int = 1) -> None:
        """``SetCompressionLevel``: what is written from now on, at this level."""
        algorithm = self.GetCompressionAlgorithm()
        self.SetCompressionSettings(algorithm * 100 + max(0, min(int(level), 99)))

    def SetCompressionSettings(self, settings: int = DEFAULT_COMPRESSION) -> None:
        """``SetCompressionSettings``: what is written from now on, by algorithm and level."""
        self._compress = int(settings)
        if self._writing is not None:
            file = self._writing._file
            file._algorithm, file._level = _compression(self._compress)

    @staticmethod
    def GetReadaheadSize() -> int:
        """``GetReadaheadSize``: how far ahead a batch of blocks is read, for every file."""
        return TFile._readahead_size

    @staticmethod
    def SetReadaheadSize(bufsize: int = READAHEAD_SIZE) -> None:
        """``SetReadaheadSize``: how far ahead a batch of blocks is read, for every file.

        A cache that asks for several blocks at once has them read in one go while they fit
        in this many bytes; 0 turns that batching off, which a test measuring single reads
        asks for.
        """
        TFile._readahead_size = int(bufsize)

    def GetCacheRead(self, tree: Any = None) -> Any:
        """``GetCacheRead``: the cache this file reads blocks through, or ``None`` as ROOT's
        null pointer when it has none.

        A tree cache belongs to one tree and answers only for it; the last cache set on the
        file is the default that every other tree reads through, as in ROOT.
        """
        if tree is not None:
            for held, cache in self._cache_reads:
                if held is tree:
                    return cache
            return self._cache_read
        if self._cache_read is None and len(self._cache_reads) == 1:
            return self._cache_reads[0][1]
        return self._cache_read

    def SetCacheRead(self, cache: Any, tree: Any = None, action: Any = None) -> None:
        """``SetCacheRead``: read blocks through ``cache`` from now on, for ``tree`` alone when
        one is named, else for the file - and that cache becomes the file's default one.

        ``action`` is ROOT's ``ECacheAction``, which tells a displaced cache whether to let go
        of the file; xrdroot's caches read through the file they were given, so it is kept for
        the signature and changes nothing.
        """
        if tree is not None:
            self._cache_reads = [
                (held, held_cache) for held, held_cache in self._cache_reads if held is not tree
            ]
            if cache is not None:
                self._cache_reads.append((tree, cache))
        self._cache_read = cache

    def Flush(self) -> None:
        """``Flush``: records go out as they are written."""

    def ls(self, option: str = "") -> None:
        """``TFile::ls``: the file, and what it holds; a file that would not open lists nothing."""
        if self._zombie:
            return
        print(f"{Indent.text()}{self.ClassName()}**\t\t{self.GetName()}\t{self.GetTitle()}")
        Indent.deeper()
        try:
            super().ls(option)
        finally:
            Indent.deeper(-1)

    def Close(self, option: str = "") -> None:
        """``Close``: the file finished and let go; what was in memory here is forgotten."""
        if not self.IsOpen():
            return
        super().Close(option)
        for held in (self._writing, self._reading):
            if held is not None:
                held.close()
        self._writing = self._reading = None
        from .troot import gROOT

        gROOT.GetListOfFiles().Remove(self)
        if self in _OPEN:
            _OPEN.remove(self)
        forget_file(self)

    def _destruct(self) -> None:
        """A file on a macro's stack, as its scope ends: closed, as ROOT's destructor does."""
        self.Close()

    def __enter__(self) -> TFile:
        return self

    def __exit__(self, *exc: object) -> None:
        self.Close()

    def __bool__(self) -> bool:
        return not self._zombie


#: The files still open, to be closed when the program ends as ROOT closes them.
_OPEN: list[TFile] = []


@atexit.register
def _close_all() -> None:
    for opened in list(_OPEN):
        opened.Close()


class _ValueObject:
    """A value handed to a file with the name of its class, as ``WriteObjectAny`` hands one:
    it streams itself by that name, ROOT's layout of the value - a ``complex<float>``'s two
    numbers - which nothing of its Python type could be described by."""

    def __init__(self, classname: str, value: Any) -> None:
        self.classname = classname
        self.value = value
        self.title = ""

    def root_payload(
        self, keylen: int
    ) -> tuple[bytes, tuple[str, ...], dict[tuple[str, int], bytes]]:
        """The value's bytes as ROOT writes them, the ROOT classes they reach, and the
        descriptions of the class they are written by."""
        catalogue: Catalogue = catalogue_for(self.classname)
        buf = WBuffer()
        plain = plain_value(self.value, self.classname, catalogue)
        write_value(buf, self.classname, plain, catalogue)
        carried = {
            (name, version): info_entry(name, sum_, version, elements)
            for name, (sum_, version, elements) in catalogue.entries().items()
        }
        return bytes(buf.data), tuple(catalogue.used), carried


def _engine_own(obj: Any) -> bool:
    """Whether ``obj`` is one of the RooFit or RooStats engine's objects - a workspace read from
    the file - which speak ROOT's API themselves."""
    return type(obj).__module__.startswith(("xrdroot.roofit", "xrdroot.roostats"))


class TMemFile(TFile):
    """``TMemFile(name, option)``: a ROOT file kept in memory, never on disk.

    It is written as a file on disk is - a script that gives one its
    histograms and trees, as a scratch directory, or hands its bytes on, does
    not leave files behind - and ``TMemFile(name, buffer, size)`` reads one
    from bytes already in memory. ``GetSize`` and ``CopyTo`` give its bytes.
    """

    CLASS_TITLE = "A ROOT file in memory"

    def __init__(self, fname: Any, *args: Any) -> None:
        self._memory = io.BytesIO()
        if args and isinstance(args[0], (bytes, bytearray, memoryview)):
            size = int(args[1]) if len(args) > 1 else len(args[0])
            self._memory = io.BytesIO(bytes(args[0])[:size])
            args = ("READ", *args[2:])
        super().__init__(fname, *args)

    def _open(self, name: str) -> None:
        """Read the bytes given, or write into memory; there is nothing to read without them."""
        if self._option == "READ" and not self._memory.getbuffer().nbytes:
            message("Error", "TMemFile::TMemFile", "no buffer to read %s from", name)
            self._zombie = True
            return
        if self._option == "READ":
            self._read_file(name)
            return
        from ...writer import create

        algorithm, level = _compression(self._compress)
        self._writing = self._writer = create(self._memory, compression=algorithm, level=level)

    def _read_file(self, name: str) -> None:
        from ...file import open_root

        self._reading = open_root(self._memory)
        self._keys_from(self._reading)

    def GetSize(self) -> int:
        """``GetSize``: the bytes written into memory so far."""
        return self._memory.getbuffer().nbytes

    def CopyTo(self, target: Any) -> int:
        """``CopyTo(buffer)``: the file's bytes, as many as fit, into a ``bytearray`` or array."""
        data = self._memory.getvalue()
        view = memoryview(target).cast("B")
        count = min(len(view), len(data))
        view[:count] = data[:count]
        return count


# -- `TFile::MakeProject`: the file's classes written out as a project a reader includes ------

#: The class of the key an RNTuple is under, whose fields name the classes it holds.
RNTUPLE = "ROOT::RNTuple"

#: The streamer element kinds that are a class's base, which a declaration names instead of
#: holding as a member.
BASES = ("TStreamerBase",)

#: The standard library's class templates, which no project header declares: the file's
#: spelling of one is what a reader's own header already says.
STANDARD = frozenset(
    "array bitset complex deque function list map multimap optional pair set shared_ptr "
    "string tuple unique_ptr unordered_map unordered_multimap unordered_set unordered_multiset "
    "valarray variant vector weak_ptr".split()
)

#: The include a member's standard type needs, so a compiler that reads the header has it.
INCLUDES = {
    "bitset": "<bitset>",
    "complex": "<complex>",
    "deque": "<deque>",
    "list": "<list>",
    "map": "<map>",
    "multimap": "<map>",
    "multiset": "<set>",
    "pair": "<utility>",
    "set": "<set>",
    "string": "<string>",
    "tuple": "<tuple>",
    "unordered_map": "<unordered_map>",
    "unordered_multimap": "<unordered_map>",
    "unordered_multiset": "<unordered_set>",
    "unordered_set": "<unordered_set>",
    "vector": "<vector>",
}  # fmt: skip


@dataclass(frozen=True)
class _Member:
    """One member a project header declares: the C++ type it was written with, its name, and
    the ``[2][3]`` a fixed array was declared with."""

    spelling: str
    name: str
    dims: str = ""


@dataclass(frozen=True)
class _Declared:
    """One class a project header declares: the classes it derives from, and its members."""

    bases: tuple[str, ...]
    members: tuple[_Member, ...]


def write_project(file: TDirectoryFile, dirname: str, classes: str, option: str) -> None:
    """A file's classes written under ``dirname`` as ROOT's project generator writes them.

    ROOT makes a directory of headers, a project header that includes them, the LinkDef that
    gives each a dictionary entry and the ``G__`` source the dictionary is compiled from.
    """
    place = Path(dirname)
    if "recreate" in option.lower() and place.exists():
        shutil.rmtree(place)
    place.mkdir(parents=True, exist_ok=True)
    chosen = {
        name: what
        for name, what in project_classes(file).items()
        if fnmatch.fnmatch(name, classes)
    }
    base = place.name
    for name, what in chosen.items():
        (place / f"{_short(name)}.h").write_text(_header(base, name, what), encoding="utf-8")
    (place / f"{base}Project.h").write_text(_project(base, list(chosen)), encoding="utf-8")
    (place / f"{base}LinkDef.h").write_text(_linkdef(list(chosen)), encoding="utf-8")
    (place / f"G__{base}.cxx").write_text(_generated(base), encoding="utf-8")


def project_classes(file: TDirectoryFile) -> dict[str, _Declared]:
    """Every class the file describes, by name.

    What the file describes is what can be declared: its streamer infos, and the C++ type each
    RNTuple record field was written with. A class the session already has - one the namespace
    implements, one a loaded dictionary declared, a standard container - is left out, as ROOT
    leaves out a class whose dictionary it finds, since a header for it would declare a second
    class of that name.
    """
    reading = file._reading
    if reading is None:
        return {}
    found: dict[str, _Declared] = dict(_described(reading))
    for ntuple in _ntuples(reading):
        for field in ntuple.fields.values():
            found.setdefault(field.cxx_type, _record(field))
    return {name: what for name, what in found.items() if not _known(name)}


def _described(reading: Any) -> dict[str, _Declared]:
    """The classes the file's streamer infos describe; a file whose descriptions cannot be read
    describes none, as ROOT's generator finds none rather than refusing the project."""
    from ...errors import FormatError
    from ...streamers import read_infos

    try:
        infos = read_infos(reading._source)
    except (FormatError, ROOTError):
        return {}
    return {name: _layout(layout) for name, layout in infos}


def _ntuples(reading: Any) -> list[Any]:
    """The RNTuples in one directory of the file, and those in the directories below it."""
    found: list[Any] = []
    for key in reading.all_keys():
        if key.classname == RNTUPLE:
            found.append(reading[key.name])
        elif key.classname in ("TDirectory", "TDirectoryFile"):
            found.extend(_ntuples(reading[key.name]))
    return found


def _layout(layout: Any) -> _Declared:
    """One class as its streamer info describes it: its bases, and each member's C++ type."""
    bases, members = [], []
    for element in layout[2]:
        if element[0] in BASES:
            bases.append(element[1])
        else:
            members.append(_member(element))
    return _Declared(tuple(bases), tuple(members))


def _member(element: Any) -> _Member:
    """One member as the streamer element describes it: its C++ type - with the ``*`` a pointer
    to a class the file does not spell one for - and the ``[2][3]`` a fixed array was declared
    with."""
    dims = "".join(f"[{size}]" for size in tuple(element[7])[: element[6]])
    pointed = element[0].endswith("Pointer") and not element[8].endswith("*")
    return _Member(element[8] + "*" if pointed else element[8], element[1], dims)


def _record(field: Any) -> _Declared:
    """One class as an RNTuple record field describes it: the members the file names, with the
    C++ types each was written with."""
    node = field.node
    members = tuple(
        _Member(member.field.type, name)
        for name, member in zip(node.names, node.members, strict=False)
    )
    return _Declared((), members)


def _known(name: str) -> bool:
    """Whether the namespace implements ``name`` itself, so that a project header for it would
    declare a second class of that name.

    ROOT also leaves out a class whose dictionary its root map finds along its library path. A
    ``.xrdlib`` in the working directory is found here too, but a project is what a reader
    includes to get the class as the file describes it, so only what the namespace itself
    implements - a ROOT class, a standard container - is left out.
    """
    bare = name.removeprefix("std::").split("<", 1)[0]
    if not bare or bare in STANDARD or bare.startswith(("ROOT::", "ROOT.")):
        return True
    return _implemented(bare.split("::")[-1])


def _implemented(short: str) -> bool:
    """Whether a module of the namespace defines a class of this name - one it wrote itself,
    not one a dictionary loaded into it."""
    for module in list(sys.modules.values()):
        name = getattr(module, "__name__", "")
        found = getattr(module, short, None)
        if name.startswith("xrdroot.pyroot") and getattr(found, "__module__", None) == name:
            return True
    return False


def _short(name: str) -> str:
    """The name a class is written by: its own, without the namespace it is declared in."""
    return name.split("::")[-1]


def _guard(base: str, name: str) -> str:
    """The macro a header's declarations are kept behind, named for the class and the project."""
    spelled = "".join(char if char.isalnum() else "_" for char in f"R__{base}_{name}")
    return spelled.upper()


def _header(base: str, name: str, what: _Declared) -> str:
    """One class's header: what it needs included, then its declaration."""
    guard = _guard(base, name)
    lines = [f"#ifndef {guard}", f"#define {guard}", ""]
    needs = _needs(what)
    lines += [f"#include {each}" for each in needs]
    if needs:  # a streamer spells a standard type by its own name, ``vector<int>``
        lines += ["", "using namespace std;"]
    lines += ["", *_declaration(name, what), "", f"#endif // {guard}", ""]
    return "\n".join(lines)


def _needs(what: _Declared) -> list[str]:
    """The standard headers the members' types need, each once, in a stable order."""
    found: set[str] = set()
    for member in what.members:
        found.update(INCLUDES[word] for word in _typed(member.spelling) if word in INCLUDES)
        if _fixed_width(member.spelling):
            found.add("<cstdint>")
    return sorted(found)


def _typed(spelling: str) -> set[str]:
    """The types a member's spelling names without their namespaces or template arguments:
    ``unordered_map<std::int32_t,std::string>`` names three."""
    pieces = spelling.replace("<", " ").replace(">", " ").replace(",", " ").split()
    return {piece.removeprefix("std::").strip("*") for piece in pieces}


def _fixed_width(spelling: str) -> bool:
    """Whether a type only ``<cstdint>`` declares names a member: ``std::int32_t``."""
    return "std::" in spelling and any(
        word in spelling for word in ("int8_t", "int16_t", "int32_t", "int64_t", "uint8_t")
    )


def _declaration(name: str, what: _Declared) -> list[str]:
    """A class's declaration, inside the namespaces its own name gives it."""
    derives = f" : public {', public '.join(what.bases)}" if what.bases else ""
    body = [
        f"class {_short(name)}{derives} {{",
        "public:",
        *(f"   {member.spelling} {member.name}{member.dims};" for member in what.members),
        "};",
    ]
    return _namespaced(name, body)


def _namespaced(name: str, body: list[str]) -> list[str]:
    """``body`` inside the namespaces of ``name``'s qualification - ``ns::Deep`` in ``ns``."""
    above = name.split("::")[:-1]
    return [
        *(f"namespace {part} {{" for part in above),
        *body,
        *(f"}} // namespace {part}" for part in reversed(above)),
    ]


def _project(base: str, names: list[str]) -> str:
    """The project's own header: every class of the project included."""
    guard = _guard(base, "project")
    lines = [f"#ifndef {guard}", f"#define {guard}", ""]
    lines += [f'#include "{_short(name)}.h"' for name in names]
    lines += ["", f"#endif // {guard}", ""]
    return "\n".join(lines)


def _linkdef(names: list[str]) -> str:
    """The LinkDef that gives each class of the project a dictionary entry."""
    lines = ["#ifdef __CLING__", ""]
    lines += [f"#pragma link C++ class {name}+;" for name in names]
    lines += ["", "#endif", ""]
    return "\n".join(lines)


def _generated(base: str) -> str:
    """The source ROOT has the project's dictionary compiled from."""
    return "\n".join(
        [
            f'#include "{base}Project.h"',
            "",
            "#ifdef __CLING__",
            f'#include "{base}LinkDef.h"',
            "#endif",
            "",
        ]
    )
