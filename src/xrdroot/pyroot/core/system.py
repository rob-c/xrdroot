"""``gSystem``: ROOT's operating system, with ROOT's answers - some of them backwards.

``gSystem->AccessPathName(path)`` is *true when the path cannot be reached*,
and every ROOT script that asks it is written for that, so it is so here.
Loading a library is the translator's: ``Load`` finds the library a C++
build made - a dictionary, ACLiC's ``macro_C`` - and loads its translation,
and ``CompileMacro`` is ACLiC's build, both through :mod:`xrdroot.pycint`
when it is there; ROOT's own libraries are xrdroot's packages, loaded from
the start. Adding an include path is noted and nothing more; the rest - the environment,
the working directory, running a command, sleeping, finding a file - is what
Python's own modules do, under ROOT's names.
"""

from __future__ import annotations

import ctypes
import os
import pathlib
import resource
import shutil
import socket
import struct
import subprocess
import sys
import time
from collections.abc import Iterator
from typing import Any

from .objects import TNamed
from .strings import TString


def _translator(name: str) -> Any:
    """The translator's function ``name`` - ``None`` without the translator."""
    from .troot import _translator as found

    return found(name)

__all__ = [
    "TSystem",
    "gSystem",
    "ProcInfo_t",
    "kFileExists",
    "kExecutePermission",
    "kWritePermission",
    "kReadPermission",
    "FileStat_t",
    "Internal",
]

#: What ``GetMakeSharedLib`` says first: ROOT's template, which ``build.C``-style macros edit.
MAKE_SHARED_LIB = (
    "cd $BuildDir ; c++ -c $Opt -fPIC $IncludePath $SourceFiles ; "
    "c++ $ObjectFiles -shared $LinkedLibs $DepLibs -o $SharedLib"
)

#: ``EAccessMode``.
kFileExists = 0
kExecutePermission = 1
kWritePermission = 2
kReadPermission = 4

#: Each access mode, as ``os.access`` asks it.
_MODES = {
    kFileExists: os.F_OK,
    kExecutePermission: os.X_OK,
    kWritePermission: os.W_OK,
    kReadPermission: os.R_OK,
}


class FileStat_t:
    """``FileStat_t``: what ``GetPathInfo`` found of a file - its size, mode, owner, times."""

    def __init__(self) -> None:
        self.fDev = self.fIno = self.fMode = self.fUid = self.fGid = 0
        self.fSize = self.fMtime = 0
        self.fIsLink = False
        self.fUrl = ""


class _Internal:
    """``ROOT::Internal``: the one of its functions a program calls directly here."""

    @staticmethod
    def GetCryptoRandom(buffer: Any, length: Any) -> int:
        """``GetCryptoRandom(buf, len)``: ``len`` bytes from the system's secure source written
        into ``buf``; how many, as ROOT returns it."""
        made = os.urandom(int(length))
        for at, byte in enumerate(made):
            buffer[at] = byte
        return len(made)

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(f"ROOT has Internal.{name}; xrdroot.pyroot does not yet")


#: ``ROOT.Internal``.
Internal = _Internal()


class TSystem(TNamed):
    """``TSystem``: the machine a script is running on, by ROOT's names."""

    kFileExists = kFileExists
    kExecutePermission = kExecutePermission
    kWritePermission = kWritePermission
    kReadPermission = kReadPermission

    def __init__(self) -> None:
        super().__init__("Unix", "Unix System")
        self._include_path = ""
        self._dynamic_path = os.environ.get("LD_LIBRARY_PATH", "")
        self._libraries: list[str] = []
        self._build_dir = ""
        self._build_flat = False
        self._make_shared_lib = MAKE_SHARED_LIB

    # -- libraries and the compiler, which Python does not need -------------------------------

    def Load(self, module: Any, entry: Any = "", system: bool = False) -> int:
        """``Load``: ``0`` loaded, ``1`` loaded already, ``-1`` not found - as ROOT answers.

        Without the translator there is nothing to load a translated library
        with, and every library a Python script loads is already Python: ``0``.
        """
        loading = _translator("load_library")
        if loading is None:
            self._libraries.append(str(module))
            return 0
        return int(loading(str(module)))

    def Unload(self, module: Any) -> None:
        """``Unload``: forget a library ``Load`` loaded, so loading it again loads it anew."""
        if str(module) in self._libraries:
            self._libraries.remove(str(module))
        unloading = _translator("unload_library")
        if unloading is not None:
            unloading(str(module))

    def LoadAllLibraries(self, isroot: bool = True) -> int:
        """``LoadAllLibraries``: the libraries a rootmap names, all loaded - of which there are
        none here, since a rootmap is what this does not keep (``GetMapfile`` is null)."""
        return 0

    def GetLibraries(self, regexp: Any = "", option: Any = "", isRegexp: bool = True) -> str:
        """``GetLibraries``: the libraries loaded - those whose name has ``regexp`` in it."""
        listed = _translator("loaded_libraries")
        names = [*self._libraries, *(listed() if listed is not None else [])]
        return " ".join(name for name in names if str(regexp or "") in name)

    def AddIncludePath(self, path: Any) -> None:
        self._include_path = f"{self._include_path} {path}".strip()

    def SetIncludePath(self, path: Any) -> None:
        self._include_path = str(path)

    def GetIncludePath(self) -> str:
        return self._include_path

    def AddLinkedLibs(self, libs: Any) -> None:
        """``AddLinkedLibs``: nothing is linked, so nothing to add."""

    def AddDynamicPath(self, path: Any) -> None:
        self._dynamic_path = f"{self._dynamic_path}:{path}".strip(":")

    def GetDynamicPath(self) -> str:
        return self._dynamic_path

    def SetDynamicPath(self, path: Any) -> None:
        self._dynamic_path = str(path)

    def CompileMacro(self, filename: Any, option: Any = "", *rest: Any) -> int:
        """``CompileMacro``: ACLiC's build of the macro, ``1`` when it is built and loaded."""
        compiling = _translator("compile_macro")
        return 1 if compiling is None else int(compiling(filename, option, *rest))

    def CleanCompiledMacros(self) -> None:
        """``CleanCompiledMacros``: ACLiC's libraries removed; a translation leaves none behind
        but the ``.xrdlib`` it wrote, which stays as a build's cache."""

    def GetMakeSharedLib(self) -> str:
        """``GetMakeSharedLib``: the command ACLiC would build with, kept for macros that edit it;
        a translation is built with none."""
        return self._make_shared_lib

    def SetMakeSharedLib(self, command: Any) -> None:
        self._make_shared_lib = str(command)

    def SetBuildDir(self, path: Any, isflat: bool = False) -> None:
        """``SetBuildDir``: where ACLiC puts the libraries it builds - flat, or under each
        macro's own path."""
        self._build_dir = str(path)
        self._build_flat = bool(isflat)

    def GetBuildDir(self) -> str:
        return self._build_dir

    def IsBuildDirFlat(self) -> bool:
        return self._build_flat

    # -- the environment --------------------------------------------------------------------

    def Getenv(self, name: Any) -> str:
        """``Getenv``: the variable's value, or the empty string if it is not set.

        ROOT's ``TUnixSystem::GetEnv`` hands back ``getenv``'s null pointer as ``""``,
        never as a null one, so a script reading a variable it may not be given sees a
        string either way.
        """
        return os.environ.get(str(name), "")

    def Setenv(self, name: Any, value: Any) -> None:
        os.environ[str(name)] = str(value)

    def Unsetenv(self, name: Any) -> None:
        os.environ.pop(str(name), None)

    def HostName(self) -> str:
        return socket.gethostname()

    def GetPid(self) -> int:
        return os.getpid()

    def GetUid(self, user: Any = None) -> int:
        return os.getuid()

    def GetUserInfo(self, *args: Any) -> None:
        return None

    def GetProcInfo(self, info: Any) -> int:
        """``GetProcInfo``: fill ``info``, a ``ProcInfo_t``, with this process's figures; ``0``."""
        usage = resource.getrusage(resource.RUSAGE_SELF)
        info.fCpuUser = int(usage.ru_utime * 1000)
        info.fCpuSys = int(usage.ru_stime * 1000)
        info.fMemResident, info.fMemVirtual = _memory_sizes()
        return 0

    def GetBuildArch(self) -> str:
        return os.uname().machine

    def Now(self) -> int:
        """``Now``: milliseconds of the clock, as ``TTime`` counts them."""
        return int(time.time() * 1000)

    # -- commands and the event loop ------------------------------------------------------

    def Exec(self, shellcmd: Any) -> int:
        """``Exec``: run a command in the shell, and hand back ``system``'s waiting status.

        ROOT returns what ``system()`` returns - the exit code shifted up by eight bits,
        so ``false`` is 256 - and a script written for ROOT shifts it back down itself.
        """
        return int(os.system(str(shellcmd)))

    def GetFromPipe(self, command: Any) -> TString:
        """``GetFromPipe``: what a command printed, its last newline taken off."""
        done = subprocess.run(str(command), shell=True, capture_output=True, text=True, check=False)
        return TString(done.stdout.rstrip("\n"))

    def ProcessEvents(self) -> bool:
        """``ProcessEvents``: there are no windows, so no events, and nothing interrupted."""
        return False

    def Sleep(self, milliseconds: Any) -> None:
        time.sleep(float(milliseconds) / 1000.0)

    def Exit(self, code: int = 0, mode: bool = True) -> None:
        raise SystemExit(int(code))

    def Abort(self, code: int = 0) -> None:
        raise SystemExit(int(code) or 1)

    # -- paths ----------------------------------------------------------------------------

    def GetPathInfo(self, path: Any, *found: Any) -> int:
        """``GetPathInfo(path, stat)`` - a ``FileStat_t`` filled - or ``GetPathInfo(path, &id,
        &size, &flags, &modtime)``: 0 when the path is there, 1 when it is not."""
        try:
            info = os.stat(os.path.expanduser(str(path)))
        except OSError:
            return 1
        if len(found) == 1:
            _filled(found[0], info, os.path.islink(str(path)))
        else:
            from .refs import store

            flags = (1 if os.access(str(path), os.X_OK) else 0) | \
                (2 if os.path.isdir(str(path)) else 0)
            for target, value in zip(found, (info.st_ino, info.st_size, flags,
                                             int(info.st_mtime)), strict=False):
                store(target, value)
        return 0

    def AccessPathName(self, path: Any, mode: int = kFileExists) -> bool:
        """``AccessPathName``: *true* when ``path`` can **not** be reached - ROOT's way round."""
        return not os.access(os.path.expanduser(str(path)), _MODES.get(int(mode), os.F_OK))

    def pwd(self) -> str:
        return os.getcwd()

    def WorkingDirectory(self) -> str:
        return os.getcwd()

    def HomeDirectory(self, user: Any = None) -> str:
        return os.path.expanduser("~")

    def TempDirectory(self) -> str:
        import tempfile

        return tempfile.gettempdir()

    def TempFileName(self, base: Any, directory: Any = None, suffix: Any = None) -> Any:
        """``TempFileName``: a new file, open to write, whose name is ``base`` and a unique tail.

        The name is written back into a ``TString`` ``base``, in the temporary
        directory unless ``base`` already has a directory or one is given.
        """
        import tempfile

        from ...cppstd.cstdio import FILE

        prefix = str(base)
        where = str(directory) if directory else (os.path.dirname(prefix) or
                                                    self.TempDirectory())
        handle, name = tempfile.mkstemp(suffix=str(suffix or ""), dir=where,
                                        prefix=os.path.basename(prefix))
        if isinstance(base, TString):
            base._s = name
        return FILE(os.fdopen(handle, "w+b"), name)

    def ChangeDirectory(self, path: Any) -> bool:
        """``ChangeDirectory``: go there, true if it could be gone into."""
        try:
            os.chdir(os.path.expanduser(str(path)))
        except OSError:
            return False
        return True

    def cd(self, path: Any) -> bool:
        return self.ChangeDirectory(path)

    def BaseName(self, path: Any) -> str:
        """``BaseName``: what follows the last slash, which a path ending in one lacks.

        ROOT takes the last component as it stands, so ``/a/b/`` has no base name at all
        rather than ``b``; ``/`` itself, the one path that is all slash, is its own.
        """
        text = str(path)
        return "/" if text == "/" else text[text.rfind("/") + 1 :]

    def DirName(self, path: Any) -> str:
        return os.path.dirname(str(path).rstrip("/")) or ("/" if str(path).startswith("/") else ".")

    def GetDirName(self, path: Any) -> TString:
        return TString(self.DirName(path))

    def ConcatFileName(self, directory: Any, name: Any) -> str:
        return os.path.join(str(directory), str(name))

    def PrependPathName(self, directory: Any, name: Any) -> str:
        """``PrependPathName``: ``directory/name``, put back into a ``TString`` ``name``."""
        joined = os.path.join(str(directory), str(name))
        if isinstance(name, TString):
            name._s = joined
        return joined

    def IsAbsoluteFileName(self, path: Any) -> bool:
        return os.path.isabs(str(path))

    def ExpandPathName(self, path: Any) -> Any:
        """``ExpandPathName``: ``~`` and ``$VARIABLE`` filled in; a ``TString`` in place."""
        expanded = os.path.expandvars(os.path.expanduser(str(path)))
        if isinstance(path, TString):
            path._s = expanded
            return False
        return expanded

    def UnixPathName(self, path: Any) -> str:
        return str(path)

    def Which(self, search: Any, name: Any, mode: int = kFileExists) -> str:
        """``Which``: ``name`` in one of the ``search`` path's directories, or ``""``.

        ROOT hands back an empty ``TString`` rather than a null pointer when it finds
        nothing, so ``Which`` is always a string a script may test.
        """
        if os.path.isabs(str(name)):
            return str(name) if not self.AccessPathName(name, mode) else ""
        for directory in str(search).split(":"):
            found = os.path.join(directory or ".", str(name))
            if not self.AccessPathName(found, mode):
                return found
        if mode == kExecutePermission:
            return shutil.which(str(name)) or ""
        return ""

    def FindFile(self, search: Any, name: Any, mode: int = kFileExists) -> str:
        return self.Which(search, name, mode)

    def mkdir(self, name: Any, recursive: bool = False) -> int:
        """``mkdir``: 0 made, -1 not - already there, or nowhere to put it."""
        try:
            (os.makedirs if recursive else os.mkdir)(str(name))
        except OSError:
            return -1
        return 0

    def MakeDirectory(self, name: Any) -> int:
        return self.mkdir(name)

    def Unlink(self, name: Any) -> int:
        """``Unlink``: remove a file or an empty directory, 0 if it went."""
        path = str(name)
        try:
            os.rmdir(path) if os.path.isdir(path) else os.remove(path)
        except OSError:
            return -1
        return 0

    def Rename(self, old: Any, new: Any) -> int:
        try:
            os.replace(str(old), str(new))
        except OSError:
            return -1
        return 0

    def CopyFile(self, source: Any, target: Any, overwrite: bool = False) -> int:
        """``CopyFile``: 0 copied, -2 there already and ``overwrite`` unset, -1 it failed.

        ROOT keeps those two refusals apart, so a script that asks a file in beside one
        already there can tell that apart from the copy having gone wrong.
        """
        if os.path.exists(str(target)) and not overwrite:
            return -2
        try:
            shutil.copyfile(str(source), str(target))
        except OSError:
            return -1
        return 0

    def OpenDirectory(self, name: Any) -> Iterator[str] | None:
        """``OpenDirectory``: something to hand ``GetDirEntry``, or ``None`` if it will not open."""
        try:
            names = [".", "..", *sorted(os.listdir(str(name)))]
        except OSError:
            return None
        return iter(names)

    def GetDirEntry(self, dirp: Any) -> str | None:
        """``GetDirEntry``: the next name in an opened directory, ``None`` at the end."""
        return next(dirp, None) if dirp is not None else None

    def FreeDirectory(self, dirp: Any) -> None:
        """``FreeDirectory``: nothing is held open."""

    def IsFileInIncludePath(self, name: Any, fullpath: Any = None) -> bool:
        return os.path.exists(str(name))


def _filled(stat: FileStat_t, info: os.stat_result, link: bool) -> None:
    """A ``FileStat_t`` given what ``os.stat`` found."""
    stat.fDev, stat.fIno, stat.fMode = info.st_dev, info.st_ino, info.st_mode
    stat.fUid, stat.fGid, stat.fSize = info.st_uid, info.st_gid, info.st_size
    stat.fMtime, stat.fIsLink = int(info.st_mtime), link


class ProcInfo_t:
    """``ProcInfo_t``: what ``TSystem::GetProcInfo`` fills in about this process.

    ROOT's own layout, in its own units: two CPU times in milliseconds and two memory sizes in
    kilobytes - the resident set as it is now, and the address space the process has mapped.
    """

    __slots__ = ("fCpuUser", "fCpuSys", "fMemResident", "fMemVirtual")

    def __init__(self) -> None:
        self.fCpuUser, self.fCpuSys = 0, 0
        self.fMemResident, self.fMemVirtual = 0, 0

    def __repr__(self) -> str:
        return (
            f"<ProcInfo_t cpu {self.fCpuUser}ms+{self.fCpuSys}ms rss {self.fMemResident}KB "
            f"vsz {self.fMemVirtual}KB>"
        )


def _linux_memory() -> tuple[int, int]:  # pragma: no cover - Linux only
    """``/proc/self/statm``, in pages: the size of the program, then its resident pages."""
    try:
        pages = pathlib.Path("/proc/self/statm").read_text().split()
        size = os.sysconf("SC_PAGE_SIZE") // 1024
        return int(pages[1]) * size, int(pages[0]) * size
    except (OSError, IndexError, ValueError):
        return 0, 0


def _darwin_memory() -> tuple[int, int]:  # pragma: no cover - macOS only
    """``proc_pidinfo``'s task figures: the resident size, then the virtual size, in kilobytes.

    ``proc_taskinfo`` begins with the two 64-bit sizes, so the first sixteen bytes of the record
    are all this needs; the call itself is in ``libproc``, which no standard module wraps.
    """
    libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
    libproc.proc_pidinfo.restype = ctypes.c_int
    libproc.proc_pidinfo.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    record = ctypes.create_string_buffer(4096)
    filled = libproc.proc_pidinfo(os.getpid(), 4, 0, record, 4096)  # PROC_PIDTASKINFO
    if filled < 16:
        return 0, 0
    virtual, resident = struct.unpack_from("QQ", record.raw)
    return int(resident) // 1024, int(virtual) // 1024


#: The kernel reader this platform has: ``proc_pidinfo`` on macOS, ``/proc`` elsewhere.
_MEMORY_SIZES = _darwin_memory if sys.platform == "darwin" else _linux_memory


def _memory_sizes() -> tuple[int, int]:
    """This process's resident and virtual sizes in kilobytes, as the kernel reports them."""
    return _MEMORY_SIZES()


#: ``gSystem``.
gSystem = TSystem()
