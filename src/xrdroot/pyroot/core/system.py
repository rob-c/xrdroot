"""``gSystem``: ROOT's operating system, with ROOT's answers - some of them backwards.

``gSystem->AccessPathName(path)`` is *true when the path cannot be reached*,
and every ROOT script that asks it is written for that, so it is so here.
Loading a library or adding an include path succeeds and does nothing, since
everything a macro could load is already Python; the rest - the environment,
the working directory, running a command, sleeping, finding a file - is what
Python's own modules do, under ROOT's names.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
from collections.abc import Iterator
from typing import Any

from .objects import TNamed
from .strings import TString

__all__ = [
    "TSystem",
    "gSystem",
    "kFileExists",
    "kExecutePermission",
    "kWritePermission",
    "kReadPermission",
]

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

    # -- libraries and the compiler, which Python does not need -------------------------------

    def Load(self, module: Any, entry: Any = "", system: bool = False) -> int:
        """``Load``: 0, loaded - there is no C++ library a Python script needs to load."""
        self._libraries.append(str(module))
        return 0

    def Unload(self, module: Any) -> None:
        """``Unload``: forget a library ``Load`` noted."""
        if str(module) in self._libraries:
            self._libraries.remove(str(module))

    def GetLibraries(self, regexp: Any = "", option: Any = "", isRegexp: bool = True) -> str:
        return " ".join(self._libraries)

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
        """``CompileMacro``: 1, done - a macro is translated when it is run, not compiled."""
        return 1

    def SetBuildDir(self, path: Any, isflat: bool = False) -> None:
        """``SetBuildDir``: nothing is built."""

    # -- the environment --------------------------------------------------------------------

    def Getenv(self, name: Any) -> str | None:
        """``Getenv``: the variable's value, or ``None`` - a null pointer - if it is not set."""
        return os.environ.get(str(name))

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

    def GetBuildArch(self) -> str:
        return os.uname().machine

    def Now(self) -> int:
        """``Now``: milliseconds of the clock, as ``TTime`` counts them."""
        return int(time.time() * 1000)

    # -- commands and the event loop ------------------------------------------------------

    def Exec(self, shellcmd: Any) -> int:
        """``Exec``: run a command in the shell, and hand back its status."""
        return int(subprocess.call(str(shellcmd), shell=True))

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
        return os.path.basename(str(path).rstrip("/")) or "/"

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

    def Which(self, search: Any, name: Any, mode: int = kFileExists) -> str | None:
        """``Which``: ``name`` in one of the ``search`` path's directories, or ``None``."""
        if os.path.isabs(str(name)):
            return str(name) if not self.AccessPathName(name, mode) else None
        for directory in str(search).split(":"):
            found = os.path.join(directory or ".", str(name))
            if not self.AccessPathName(found, mode):
                return found
        return shutil.which(str(name)) if mode == kExecutePermission else None

    def FindFile(self, search: Any, name: Any, mode: int = kFileExists) -> str | None:
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
        if os.path.exists(str(target)) and not overwrite:
            return -1
        shutil.copyfile(str(source), str(target))
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


#: ``gSystem``.
gSystem = TSystem()
