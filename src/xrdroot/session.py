"""``gROOT`` and ``gDirectory``: ROOT's global session, as one Python object.

    >>> from xrdroot import gROOT, gDirectory
    >>> f = gROOT.open("tests/data/dirs-6.14.00.root")        # doctest: +SKIP
    >>> gROOT.cd("dir1/dir11")                                # doctest: +SKIP
    >>> gROOT.pwd()                                           # doctest: +SKIP
    'tests/data/dirs-6.14.00.root:/dir1/dir11'
    >>> gDirectory["h1"]                                      # doctest: +SKIP

ROOT keeps a list of the files a process has open and a directory it is "in",
and a name typed at its prompt is looked up in that directory first and then
everywhere else. That is convenient at a prompt and a trap in a program - what
a name means depends on what was opened last - so here it is an object you
can ask for rather than the state every call quietly reads: nothing else in
this library looks at it, and a program that never imports it never has one.

Every file :func:`~xrdroot.open_root` opens is listed while something holds
it; :meth:`Session.open` holds the files it opens itself until
:meth:`~Session.close_all` closes them. Objects made in memory are only listed when
:meth:`~Session.add` puts them there, which is ROOT's
``TH1::AddDirectory`` made something you say rather than something that
happens to you.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .file import OPENED, open_root

__all__ = ["Session", "CurrentDirectory", "gROOT", "gDirectory", "split_location", "preloaded"]

#: What the top of the session - no file, only memory - calls itself, as ROOT
#: does at its prompt.
TOP = "Rint"

#: A location's file part: everything up to the last ``.root`` that is followed
#: by a ``:`` and a path, or ends the text. Greedy, so that a URL's own colons -
#: ``root://host:1094//f.root:dir/h`` - stay in the file.
LOCATION = re.compile(r"^(?P<file>.*\.root)(?::(?P<path>.*))?$", re.DOTALL)

#: What ``.help`` at the prompt prints.
HELP = """\
ROOT's prompt commands, and what each is here:
  .ls [dir]        print(gROOT.ls())      what the current directory holds
  .pwd             print(gROOT.pwd())     where the session is
  .cd [dir]        gROOT.cd("dir")        a directory, "file.root:/dir", ".." or the top
  .x macro.py(a)   gROOT.macro(...)       run a Python macro, then its function of the same name
  .q               exit()                 leave
  .help            this text
Everything in xrdroot is here by name, with numpy as np. gROOT.get("name")
looks in the current directory, then memory, then every open file;
gROOT.open(url) opens a file and goes into it; gROOT.files lists them."""


def split_location(text: str) -> tuple[str | None, str]:
    """``"file.root:dir/h"`` as ``("file.root", "dir/h")``; a bare path as ``(None, path)``.

        >>> split_location("root://host:1094//store/f.root:dir/h")
        ('root://host:1094//store/f.root', 'dir/h')
        >>> split_location("f.root")
        ('f.root', '')
        >>> split_location("dir/h")
        (None, 'dir/h')

    A URL has colons of its own, so the split is at the last ``.root`` that
    ends the text or is followed by one; a leading ``/`` on the path is
    ROOT's way of saying "from the top of the file", which is where a path
    in a file starts anyway.
    """
    found = LOCATION.match(text)
    if found is None:
        return None, text
    return found["file"], (found["path"] or "").strip("/")


def preloaded() -> dict[str, Any]:
    """The names a macro and the shell start with: all of :mod:`xrdroot`, and NumPy as ``np``."""
    import numpy

    import xrdroot

    names: dict[str, Any] = {name: getattr(xrdroot, name) for name in xrdroot.__all__}
    names["np"] = numpy
    names["xrdroot"] = xrdroot
    return names


def _joined(base: str, path: str) -> str:
    """``path`` walked from ``base``: ``..`` goes up, ``.`` stays, a leading ``/`` starts over."""
    parts = [] if path.startswith("/") else [part for part in base.split("/") if part]
    for part in path.split("/"):
        if part == "..":
            parts = parts[:-1]
        elif part and part != ".":
            parts.append(part)
    return "/".join(parts)


def _is_directory(value: Any) -> bool:
    """Is this something to be ``cd``'d into - a file or a directory, not a histogram?"""
    from .file import Directory

    return isinstance(value, Directory)


def _describe(value: Any) -> tuple[str, str, str]:
    """An object's class, name and title, however it keeps them."""
    classname = getattr(value, "classname", type(value).__name__)
    return str(classname), str(getattr(value, "name", "")), str(getattr(value, "title", ""))


class Session:
    """ROOT's ``gROOT``: the open files, where the session is, and what it holds.

        >>> gROOT.open("f.root")          # doctest: +SKIP
        >>> gROOT["h"]                    # the current directory, memory, then every file
        >>> gROOT.get("f.root:/dir/h")    # or exactly where it is

    There is one, :data:`gROOT`, for the shell and for macros; a program is
    better off with the file it opened than with a global, and a test with
    a ``Session()`` of its own.
    """

    def __init__(self) -> None:
        #: The files this session opened and holds open until closed.
        self._held: list[Any] = []
        #: The objects put in memory by name, in ROOT's ``gROOT`` list.
        self._objects: dict[str, Any] = {}
        #: The file the session is in, or ``None`` at the top, and the path in it.
        self._top: Any = None
        self._path = ""

    def __repr__(self) -> str:
        return f"<Session at {self.pwd()} with {len(self.files)} files open>"

    # -- files ----------------------------------------------------------------

    def open(self, target: Any, *, config: Any = None) -> Any:
        """``TFile::Open``: open a file, hold it open, and go into it."""
        opened = open_root(target, config=config)
        self._held.append(opened)
        self._top, self._path = opened, ""
        return opened

    @property
    def files(self) -> list[Any]:
        """Every file open in this process, in the order it was opened."""
        return [found for _, found in sorted(OPENED.items()) if not found.closed]

    def close_all(self) -> None:
        """``gROOT->CloseFiles()``: close the files this session opened, and go back to the top.

        Only its own: a file :func:`~xrdroot.open_root` opened is listed here
        but closed by whoever opened it, on the principle ``open_root`` keeps
        for a file object - and so that a shell's exit never pulls a file out
        from under the program that opened it.
        """
        for found in self._held:
            found.close()
        self._held.clear()
        self.cd()

    def _file_named(self, name: str) -> Any:
        """The open file ``name`` means: by what it was opened as, its base name, or its path."""
        for found in self.files:
            if name in (found.name, os.path.basename(found.name)) or _same_path(found.name, name):
                return found
        listed = ", ".join(found.name for found in self.files) or "none"
        raise KeyError(
            f"{name!r} is not an open file (the open ones are {listed}); "
            f"open it with gROOT.open({name!r}) first"
        )

    # -- where the session is --------------------------------------------------

    def _where(self, where: str) -> tuple[Any, str]:
        """The file and path a ``cd`` or ``ls`` argument names, without going there."""
        location, path = split_location(where)
        if location is not None:
            return self._file_named(location), path
        top = self._current_top()
        if top is None:
            raise KeyError(
                f"{where!r} is a path in a file, and the session is not in one: "
                f"gROOT.cd('file.root:/{where}') names the file too"
            )
        return top, _joined(self._path, where)

    def _current_top(self) -> Any:
        """The file the session is in - or ``None``, once that file has been closed."""
        if self._top is not None and getattr(self._top, "closed", False):
            self._top, self._path = None, ""
        return self._top

    def _owner(self, directory: Any) -> tuple[Any, str]:
        """The open file a directory object is in, and its path there.

        A directory read out of a file does not hold on to the file, only to
        its bytes, so the file is the open one reading the same bytes; a
        directory whose file is not listed is its own top.
        """
        for found in self.files:
            if found._source is directory._source:
                return found, directory.path
        return directory, ""

    @staticmethod
    def _enter(top: Any, path: str) -> Any:
        """The directory at ``path`` in ``top``, refused by name when it is not one."""
        found = top[path] if path else top
        if not _is_directory(found):
            raise NotADirectoryError(
                f"{path!r} is a {_describe(found)[0]}, not a directory: "
                f"cd goes into files and the directories in them"
            )
        return found

    def cd(self, where: Any = None) -> Any:
        """``TDirectory::cd``: go into a file or a directory, and hand it back.

            >>> gROOT.cd("dir1")               # doctest: +SKIP
            >>> gROOT.cd("..")                 # doctest: +SKIP
            >>> gROOT.cd("f.root:/dir1/dir11") # doctest: +SKIP
            >>> gROOT.cd(f)                    # doctest: +SKIP
            >>> gROOT.cd()                     # the top: memory, and no file

        A path is from where the session is, or from the top of the file
        with a leading ``/``; ``file.root:path`` names an open file first.
        """
        if where is None:
            self._top, self._path = None, ""
            return self
        if not isinstance(where, str):
            self._top, self._path = self._owner(self._enter(where, ""))
            return where
        top, path = self._where(where)
        found = self._enter(top, path)
        self._top, self._path = top, path
        return found

    @property
    def directory(self) -> Any:
        """``gDirectory``: the directory the session is in, or the session itself at the top."""
        top = self._current_top()
        return self if top is None else self._enter(top, self._path)

    def pwd(self) -> str:
        """``TDirectory::pwd``: ``file.root:/dir/sub``, or ``Rint:/`` at the top."""
        top = self._current_top()
        name = TOP if top is None else getattr(top, "name", type(top).__name__)
        return f"{name}:/{self._path}"

    def ls(self, where: str | None = None) -> str:
        """``TDirectory::ls``: what the current directory - or ``where`` - holds, as text."""
        if where is not None:
            top, path = self._where(where)
            return _listing(self._enter(top, path), getattr(top, "name", ""), path)
        top = self._current_top()
        if top is None:
            return self._top_listing()
        return _listing(self._enter(top, self._path), getattr(top, "name", ""), self._path)

    def _top_listing(self) -> str:
        """The top: the objects in memory, then the open files."""
        lines = [f"{TOP}*\t\t{TOP}\t\tROOT's session, in Python"]
        for name, value in self._objects.items():
            classname, _, title = _describe(value)
            lines.append(f" OBJ: {classname}\t{name}\t{title}")
        lines.extend(f" TFile*\t\t{found.name}" for found in self.files)
        return "\n".join(lines)

    # -- names ----------------------------------------------------------------

    def _places(self) -> Iterator[Any]:
        """Where a bare name is looked for, in ROOT's order: here, memory, then every file."""
        top = self._current_top()
        if top is not None:
            yield self._enter(top, self._path)
        yield self._objects
        yield from self.files

    def __getitem__(self, name: str) -> Any:
        location, path = split_location(name)
        if location is not None:
            found = self._file_named(location)
            return found[path] if path else found
        if name.startswith("/"):
            top = self._current_top()
            if top is not None:
                return top[name.strip("/")]
        for place in self._places():
            try:
                return place[name]
            except KeyError:
                continue
        raise KeyError(
            f"{name!r} is not in {self.pwd()}, in memory, or in any open file "
            f"({', '.join(found.name for found in self.files) or 'there are none'})"
        )

    def get(self, name: str, default: Any = None) -> Any:
        """``gROOT->Get`` and ``FindObject``: the object ``name`` means, or ``default``.

        ``"file.root:/dir/h"`` is exactly where it says; a bare name is looked
        for in the current directory, then among the objects in memory, then
        in every open file in the order they were opened.
        """
        try:
            return self[name]
        except KeyError:
            return default

    def __contains__(self, name: object) -> bool:
        return self.get(str(name)) is not None

    def keys(self) -> list[str]:
        """The names of the objects in memory, which is what the top of the session holds."""
        return list(self._objects)

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self._objects)

    @property
    def objects(self) -> dict[str, Any]:
        """The objects :meth:`add` has put in memory, by name."""
        return dict(self._objects)

    def add(self, value: Any, name: str | None = None) -> Any:
        """Put an object in memory under its name - or ``name`` - and hand it back.

        >>> h = gROOT.add(Histogram.book("h", (10, 0, 1)))   # doctest: +SKIP
        >>> gROOT["h"] is h                                  # doctest: +SKIP
        True
        """
        called = name if name is not None else _describe(value)[1]
        if not called:
            raise ValueError(
                f"a {type(value).__name__} has no name to be found by: give it one, "
                f"as gROOT.add(value, 'name')"
            )
        self._objects[called] = value
        return value

    def remove(self, name: str) -> Any:
        """Take an object out of memory by name, and hand it back."""
        return self._objects.pop(name)

    # -- macros ---------------------------------------------------------------

    def run(self, source: str, filename: str = "<macro>") -> dict[str, Any]:
        """Run Python source with the ROOT names preloaded, and hand back what it defined."""
        namespace = preloaded()
        namespace.update(__name__="__main__", __file__=filename)
        exec(compile(source, filename, "exec"), namespace)
        return namespace

    def macro(self, path: str | os.PathLike[str], *args: Any) -> Any:
        """``.x macro.py(args)``: run a Python macro, then call its function of the same name.

            >>> gROOT.macro("fill.py", 1000)       # doctest: +SKIP

        A macro called ``fill.py`` that defines ``fill`` has it called with
        ``args``, and what it returns comes back, as ROOT's ``.x`` does for a
        C++ macro; one that defines no such function is just run.
        """
        where = Path(path)
        namespace = self.run(where.read_text(), str(where))
        entry = namespace.get(where.stem)
        if callable(entry):
            return entry(*args)
        if args:
            raise TypeError(
                f"{where.name} defines no function {where.stem}() to hand "
                f"{', '.join(map(repr, args))} to"
            )
        return None

    @staticmethod
    def help() -> str:
        """What the prompt's dot-commands are, and what they stand for."""
        return HELP


def _same_path(one: str, other: str) -> bool:
    """Do two local paths name one file, however each was written?"""
    if "://" in one or "://" in other:
        return False
    return os.path.abspath(one) == os.path.abspath(other)


def _listing(directory: Any, filename: str, path: str) -> str:
    """ROOT's ``ls`` of a directory: what it is, then a ``KEY:`` line per key."""
    if path:
        lines = [f"TDirectoryFile*\t\t{path.rsplit('/', 1)[-1]}\t{filename}:/{path}"]
    else:
        lines = [f"TFile**\t\t{filename}", f" TFile*\t\t{filename}"]
    lines.extend(
        f"  KEY: {key.classname}\t{key.name};{key.cycle}\t{key.title}"
        for key in directory.all_keys()
    )
    return "\n".join(lines)


class CurrentDirectory:
    """ROOT's ``gDirectory``: whatever directory the session is in when it is used.

        >>> gDirectory["h"]         # doctest: +SKIP
        >>> gDirectory.keys()       # doctest: +SKIP

    Always the current one, not the one current when it was imported: every
    use asks :data:`gROOT` where it is. At the top it is the session itself.
    """

    __slots__ = ("_session",)

    def __init__(self, session: Session) -> None:
        self._session = session

    def __getattr__(self, name: str) -> Any:
        return getattr(self._session.directory, name)

    def __getitem__(self, name: str) -> Any:
        return self._session.directory[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._session.directory)

    def __len__(self) -> int:
        return len(self._session.directory)

    def __contains__(self, name: object) -> bool:
        return name in self._session.directory

    def __repr__(self) -> str:
        return f"<gDirectory {self._session.pwd()}>"

    def cd(self, where: Any = None) -> Any:
        """Go somewhere, as :meth:`Session.cd` does."""
        return self._session.cd(where)

    def pwd(self) -> str:
        """Where the session is."""
        return self._session.pwd()

    def ls(self, where: str | None = None) -> str:
        """What the current directory holds, as text."""
        return self._session.ls(where)


#: The session: ROOT's ``gROOT``.
gROOT = Session()
#: The directory the session is in: ROOT's ``gDirectory``.
gDirectory = CurrentDirectory(gROOT)
