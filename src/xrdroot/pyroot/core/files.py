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
import os
import struct
from typing import Any

from ...errors import ROOTError
from .collections import TList
from .directories import TDirectory, current_directory, set_current
from .messages import message
from .objects import Indent, TNamed, TObject, templated
from .wrapping import from_members, unwrap, wrap

__all__ = ["TFile", "TDirectoryFile", "TKey"]

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

    ReadObjectAny = ReadObj

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
        top = self.GetFile()._writing
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

    def GetListOfKeys(self) -> TList:
        made = TList()
        for key in self._keys:
            made.Add(key)
        return made

    def GetNkeys(self) -> int:
        return len(self._keys)

    def FindKey(self, name: Any) -> Any:
        return self.GetKey(name)

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
        read = self._reader()[label]
        found = _tree(read, key) or wrap(read)
        if not isinstance(found, TObject) and not _engine_own(found):
            found = from_members(key.GetClassName(), found) or TOther(key.GetClassName(), found)
        newest = key is self.GetKey(key.GetName())
        if newest and any(_inherits(found, kind) for kind in KEPT):
            self.Append(found, True)
        return found

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

    def __getitem__(self, name: str) -> Any:
        """``f["hpx"]``: an object by name, as PyROOT's files give them; ``KeyError`` if none."""
        found = self.Get(name)
        if found is None:
            raise KeyError(f"{self.ClassName()} {self.GetName()!r} has no object {name!r}")
        return found

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
        writer.write(called, unwrap(obj), title=obj.GetTitle())
        self._note_key(called, obj.GetTitle(), obj.ClassName())
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
        """``Write``: every object in memory here written, and every directory below its own."""
        if self._writable() is None:
            message(
                "Error", "TDirectoryFile::Write", "file %s not opened in write mode", self.GetName()
            )
            return 0
        return super().Write(name, option, bufsize)

    def mkdir(self, name: Any, title: Any = "", returnExistingDirectory: bool = False) -> Any:
        """``mkdir``: a directory below this one - on file too - and ``a/b`` makes both."""
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
        for key, mark in zip(keys, _listing_marks(keys)):
            key.ls(mark)

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


class TFile(TDirectoryFile):
    """``TFile``: a ROOT file - read, made anew, or added to - and the directory at its top."""

    CLASS_TITLE = "ROOT file"

    def __init__(
        self, fname: Any = "", option: Any = "READ", ftitle: Any = "", compress: int = 101
    ) -> None:
        name = str(fname)
        super().__init__(name, str(ftitle))
        self._option = MODES.get(str(option).upper().strip(), "READ")
        self._reading: Any = None
        self._writing: Any = None
        self._zombie = False
        self._open(name)
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
        from ... import open_root

        try:
            self._reading = open_root(name)
        except (OSError, ValueError, ROOTError) as why:
            missing = isinstance(why, FileNotFoundError)
            text = "file %s does not exist" if missing else "file %s is not a ROOT file"
            message("Error", "TFile::TFile", text, name)
            self._zombie = True
            return
        if not self.GetTitle():
            self.SetTitle(_file_title(self._reading))
        self._keys_from(self._reading)

    def _write_file(self, name: str, updating: bool) -> None:
        from ... import create, update

        self._writing = update(name) if updating else create(name)
        self._writer = self._writing

    @staticmethod
    def Open(
        name: Any, option: Any = "READ", ftitle: Any = "", compress: int = 101, netopt: int = 0
    ) -> Any:
        """``TFile::Open``: the file, or ``None`` - a null pointer - if it would not open."""
        made = TFile(name, option, ftitle, compress)
        return None if made.IsZombie() else made

    # -- what it is -----------------------------------------------------------------------

    def GetFile(self) -> Any:
        return self

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
        return int(getattr(self._reading, "compression", 101))

    def GetCompressionLevel(self) -> int:
        return self.GetCompressionSettings() % 100

    def GetCompressionAlgorithm(self) -> int:
        return self.GetCompressionSettings() // 100

    def SetCompressionLevel(self, level: int = 1) -> None:
        """``SetCompressionLevel``: the file is compressed as xrdroot's writer compresses it."""

    SetCompressionSettings = SetCompressionLevel

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


def _engine_own(obj: Any) -> bool:
    """Whether ``obj`` is one of the RooFit or RooStats engine's objects - a workspace read from
    the file - which speak ROOT's API themselves."""
    return type(obj).__module__.startswith(("xrdroot.roofit", "xrdroot.roostats"))
