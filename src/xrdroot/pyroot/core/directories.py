"""``TDirectory`` and ``gDirectory``: where objects made in a script are kept.

ROOT puts every histogram it makes in the current directory - memory, when
no file is open, and otherwise the file opened last - and ``TFile::Write``
writes what that directory holds. A script depends on this without saying
so: ``TFile("out.root", "RECREATE")``, book, fill, ``Write()``, and the
histograms are in the file. So here too: a directory holds a list of the
objects in memory, :func:`current_directory` is the one ``gDirectory`` means,
and ``cd`` changes it.

``gDirectory`` is a stand-in that is always the current directory, so a
script that took it with ``from ROOT import gDirectory`` before opening a
file still sees the file after.
"""

from __future__ import annotations

from typing import Any

from .collections import TList
from .messages import message
from .objects import Indent, TNamed, templated

__all__ = ["TDirectory", "gDirectory", "current_directory"]

#: The current directory, once one has been gone into; memory until then.
_CURRENT: list[Any] = []


def current_directory() -> Any:
    """What ``gDirectory`` is now: the directory last gone into, or ``gROOT``."""
    if _CURRENT:
        return _CURRENT[0]
    from .troot import gROOT

    return gROOT


def set_current(directory: Any) -> None:
    """Go into ``directory``: what ``cd`` does, without the checks."""
    _CURRENT[:] = [directory]


def forget(obj: Any) -> None:
    """Take ``obj`` out of the directory it is kept in, if it is kept in one."""
    home = getattr(obj, "_directory", None)
    if home is not None:
        home.Remove(obj)


class TDirectory(TNamed):
    """``TDirectory``: a named list of objects, and the directories below it."""

    def __init__(self, name: Any = "", title: Any = "", classname: str = "", mother: Any = None):
        super().__init__(name, title)
        self._list = TList()
        self._mother = mother

    # -- what it holds ----------------------------------------------------------

    def GetList(self) -> TList:
        """``GetList``: the objects in memory in this directory."""
        return self._list

    def GetListOfKeys(self) -> TList:
        """``GetListOfKeys``: the keys on file - none, for a directory in memory."""
        return TList()

    def Append(self, obj: Any, replace: bool = False) -> None:
        """``Append``: keep ``obj`` here - replacing, with ROOT's warning, one of its name."""
        if replace and obj.GetName():
            self._replace(obj)
        self._list.Add(obj)
        if "_directory" in getattr(obj, "__dict__", {}):
            obj._directory = self

    def _replace(self, obj: Any) -> None:
        old = self._list.FindObject(obj.GetName())
        while old is not None:
            if old is not obj:
                message(
                    "Warning",
                    f"{self.ClassName()}::Append",
                    "Replacing existing %s: %s (Potential memory leak).",
                    obj.ClassName(),
                    obj.GetName(),
                )
            self.Remove(old)
            old = self._list.FindObject(obj.GetName())

    def Add(self, obj: Any, replace: bool = False) -> None:
        self.Append(obj, replace)

    def Remove(self, obj: Any) -> Any:
        """``Remove``: stop keeping ``obj`` here, and hand it back."""
        found = self._list.Remove(obj)
        if found is not None and getattr(found, "_directory", None) is self:
            found._directory = None
        return found

    def Clear(self, option: str = "") -> None:
        for obj in list(self._list):
            self.Remove(obj)

    def FindObject(self, name: Any) -> Any:
        """``FindObject``: an object in memory here, by name or by being it."""
        return self._list.FindObject(name)

    def FindObjectAny(self, name: Any) -> Any:
        """``FindObjectAny``: here, then in every directory below."""
        found = self.FindObject(name)
        for sub in self._subdirectories():
            found = found if found is not None else sub.FindObjectAny(name)
        return found

    def _subdirectories(self) -> list[Any]:
        return [obj for obj in self._list if isinstance(obj, TDirectory)]

    @templated
    def Get(self, namecycle: Any) -> Any:
        """``Get``: an object by name - ``dir/name`` walks down, ``;cycle`` is ignored here."""
        name = str(namecycle).split(";")[0]
        head, _, rest = name.partition("/")
        if rest:
            below = self.GetDirectory(head)
            return None if below is None else below.Get(rest)
        return self._list.FindObject(name)

    def GetObject(self, namecycle: Any, holder: Any = None) -> Any:
        """``GetObject(name, ptr)``: what ``Get`` finds, into ``ptr`` - and handed back.

        C++ passes the pointer by reference to be set, ``nullptr`` if nothing is
        found; the translator hands over a cell (anything with a ``.value``),
        which is set the same way. Anything else given is left alone.
        """
        found = self.Get(namecycle)
        if hasattr(holder, "value"):
            holder.value = found
        return found

    def GetDirectory(self, path: Any, printError: bool = False, funcname: str = "") -> Any:
        """``GetDirectory``: the directory at ``path`` below this one, or ``None``."""
        here: Any = self
        for part in str(path).strip("/").split("/"):
            if part in ("", "."):
                continue
            here = here._mother if part == ".." else here._child(part)
            if here is None:
                return None
        return here

    def _child(self, name: str) -> Any:
        found = self.Get(name)
        return found if isinstance(found, TDirectory) else None

    # -- where it is ---------------------------------------------------------------

    def GetMother(self) -> Any:
        return self._mother

    def GetMotherDir(self) -> Any:
        return self._mother

    def GetFile(self) -> Any:
        """``GetFile``: the file this directory is in, or ``None`` in memory."""
        return None if self._mother is None else self._mother.GetFile()

    def GetPath(self) -> str:
        """``GetPath``: ``file.root:/dir/sub``, or ``PyROOT:/`` for memory."""
        if self._mother is None:
            return f"{self.GetName()}:/"
        above = self._mother.GetPath().rstrip("/")
        return f"{above}/{self.GetName()}"

    def pwd(self) -> None:
        """``pwd``: print where this directory is."""
        print(self.GetPath())

    def cd(self, path: Any = None) -> bool:
        """``cd``: make this directory - or the one ``path`` names below it - the current one."""
        target = self if not path else self.GetDirectory(path)
        if target is None:
            message("Error", f"{self.ClassName()}::cd", "Unknown directory %s", str(path))
            return False
        set_current(target)
        return True

    def mkdir(self, name: Any, title: Any = "", returnExistingDirectory: bool = False) -> Any:
        """``mkdir``: a directory below this one, in memory; ``a/b`` makes both."""
        head, _, rest = str(name).strip("/").partition("/")
        found = self._child(head)
        if found is None:
            found = self._made_directory(head, str(title) if not rest else "")
        elif not rest and not returnExistingDirectory:
            message(
                "Error", f"{self.ClassName()}::mkdir", "An object with name %s exists already", head
            )
            return None
        return found.mkdir(rest, title, returnExistingDirectory) if rest else found

    def _made_directory(self, name: str, title: str) -> Any:
        made = TDirectory(name, title, mother=self)
        self.Append(made)
        return made

    def IsWritable(self) -> bool:
        return False

    # -- writing -------------------------------------------------------------------------

    def WriteTObject(self, obj: Any, name: Any = None, option: Any = "", bufsize: int = 0) -> int:
        """``WriteTObject``: refused, with ROOT's error, for a directory not in a file."""
        called = str(name) if name else obj.GetName()
        message(
            "Error",
            f"{self.ClassName()}::WriteTObject",
            "The current directory (%s) is not associated with a file. "
            "The object (%s) has not been written.",
            self.GetName(),
            called,
        )
        return 0

    def WriteObject(self, obj: Any, name: Any, option: Any = "", bufsize: int = 0) -> int:
        return self.WriteTObject(obj, name, option)

    def Write(self, name: Any = None, option: int = 0, bufsize: int = 0) -> int:
        """``Write``: every object in memory here writes itself, each directory its own."""
        total = 0
        previous = current_directory()
        set_current(self)
        try:
            for obj in list(self._list):
                total += obj.Write()
        finally:
            set_current(previous)
        return total

    def SaveSelf(self, force: bool = False) -> None:
        """``SaveSelf``: write this directory's own record, which closing the file does."""

    def ls(self, option: str = "") -> None:
        """``ls``: what is in memory here, one level in."""
        self._ls_memory(option)

    def _ls_memory(self, option: str) -> None:
        Indent.deeper()
        try:
            for obj in _chosen(self._list, option):
                obj.ls(option)
        finally:
            Indent.deeper(-1)

    def Close(self, option: str = "") -> None:
        self.Clear()

    def DeleteAll(self, option: str = "") -> None:
        self.Clear()

    def Delete(self, namecycle: Any = "") -> None:
        """``Delete(name)``: forget the object called ``name``; ``"*"`` forgets them all."""
        text = str(namecycle).split(";")[0]
        for obj in list(self._list):
            if text in ("*", "") or obj.GetName() == text:
                self.Remove(obj)

    def ReadAll(self, option: str = "") -> None:
        """``ReadAll``: bring every object on file into memory, which a memory one has done."""

    class TContext:
        """``TDirectory::TContext``: go back to the directory it was made in, when left.

        >>> with ROOT.TDirectory.TContext():            # doctest: +SKIP
        ...     f = ROOT.TFile("out.root", "RECREATE")
        """

        def __init__(self, *directories: Any) -> None:
            self._previous = current_directory()
            if directories and directories[-1] is not None:
                directories[-1].cd()

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *exc: object) -> None:
            set_current(self._previous)


def _chosen(held: Any, option: str) -> list[Any]:
    """What an ``ls`` with ``option`` lists: all, or the names the wildcard matches."""
    import fnmatch

    text = str(option).strip()
    if text.startswith("-m") or text.startswith("-d"):
        text = text[2:]
    if not text:
        return list(held)
    return [obj for obj in held if fnmatch.fnmatchcase(obj.GetName(), text)]


class _CurrentDirectory:
    """``gDirectory``: whichever directory is current now, answering as that one."""

    def __getattr__(self, name: str) -> Any:
        return getattr(current_directory(), name)

    def __eq__(self, other: object) -> bool:
        return current_directory() is (
            other.__real__() if isinstance(other, _CurrentDirectory) else other
        )

    def __hash__(self) -> int:
        return id(self)

    def __real__(self) -> Any:
        """The directory this stands for right now."""
        return current_directory()

    def __repr__(self) -> str:
        return repr(current_directory())


#: ``gDirectory``, always the current directory.
gDirectory: Any = _CurrentDirectory()
