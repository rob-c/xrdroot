"""What every backend does the same way: code sent to the workers, and results handled.

Headers are declared through ``gInterpreter`` - xrdroot translating the C++
in them - and shared libraries loaded with ``gSystem.Load``, which takes
xrdroot's own ``.xrdlib`` libraries that ``xrdroot build library`` makes.
The rest turns what a task booked into what it sends back
(:func:`get_mergeablevalue`), adds two of those up (:func:`merge_values`),
books a result again for the next task (:func:`clone_action`) and gives the
sum to the node the user holds (:func:`set_value_on_node`).
"""

from __future__ import annotations

import os
import pathlib
from collections.abc import Iterable
from functools import singledispatch
from typing import Any

from ...rdf.asnumpy import AsNumpyResult
from ...rdf.asnumpy import clone as _clone_asnumpyresult
from ...rdf.frame import RResultPtr, columns_of, presented
from ..PythonMergeables import MergeableValue, SnapshotResult

__all__ = [
    "extend_include_path",
    "distribute_headers",
    "distribute_shared_libraries",
    "get_paths_set_from_string",
    "check_pcm_in_library_path",
    "register_files",
    "register_headers",
    "register_shared_libs",
    "get_mergeablevalue",
    "merge_values",
    "set_value_on_node",
    "clone_action",
    "trigger_run",
]

#: What a shared library's name ends in: ROOT's, and xrdroot's own.
SHARED_LIBRARY_FORMATS = (".so", ".dll", ".dylib", ".xrdlib")


def _root() -> Any:
    from .... import pyroot

    return pyroot


def extend_include_path(include_path: str) -> None:
    """Let the interpreter find headers in ``include_path``."""
    _root().gInterpreter.AddIncludePath(f"-I{include_path}")


def distribute_headers(headers_to_include: Iterable[str]) -> None:
    """Declare each header, its directory added to the include path first."""
    for header in headers_to_include:
        extend_include_path(os.path.dirname(header))
        _root().gInterpreter.Declare(f'#include "{header}"\n')


def distribute_shared_libraries(libraries_to_include: Iterable[str]) -> None:
    """Load each shared library, refusing one that is not there or will not load."""
    for library in libraries_to_include:
        if _root().gSystem.Load(library) == -1:
            if not os.path.exists(library):
                raise OSError(f"Shared library {library!r} does not exist!")
            raise RuntimeError(
                f"xrdroot could not load the shared library {library!r}: it loads the .xrdlib "
                f"libraries 'xrdroot build library' makes of C++ sources, not compiled ones; "
                f"build one of the library's sources and distribute that instead"
            )


def get_paths_set_from_string(path_string: str) -> set[str]:
    """Every file under a directory, or the file itself; nothing for a path that is neither."""
    if os.path.isdir(path_string):
        return {
            os.path.join(root, filename)
            for root, _, filenames in os.walk(path_string)
            for filename in filenames
        }
    if os.path.isfile(path_string):
        return {path_string}
    return set()


def check_pcm_in_library_path(shared_library_path: str) -> tuple[set[str], set[str]]:
    """The ``.pcm`` files and the shared libraries a library's path - or a directory - names."""
    directory, prefix = _library_prefix(shared_library_path)
    found = [path for path in get_paths_set_from_string(directory) if path.startswith(prefix)]
    pcms = {path for path in found if path.endswith(".pcm")}
    libraries = {path for path in found if path.endswith(SHARED_LIBRARY_FORMATS)}
    return pcms, libraries


def _library_prefix(shared_library_path: str) -> tuple[str, str]:
    """The directory to look in, and what a file's path there begins with: the library's
    own name for a library, anything in a directory given."""
    if not shared_library_path.endswith(SHARED_LIBRARY_FORMATS):
        directory = os.path.abspath(shared_library_path)
        return directory, os.path.join(directory, "")
    directory = os.path.dirname(os.path.abspath(shared_library_path))
    return directory, os.path.join(directory, pathlib.PurePosixPath(shared_library_path).stem)


def _each_path(paths: str | Iterable[str]) -> list[str]:
    return [paths] if isinstance(paths, str) else list(paths)


def register_files(paths_to_files: str | Iterable[str]) -> set[str]:
    """The files under each path, to send to the workers."""
    return {
        found for path in _each_path(paths_to_files) for found in get_paths_set_from_string(path)
    }


def register_headers(paths_to_headers: str | Iterable[str]) -> set[str]:
    """The headers under each path, declared here first, to send to the workers."""
    headers = register_files(paths_to_headers)
    distribute_headers(headers)
    return headers


def register_shared_libs(paths: str | Iterable[str]) -> tuple[set[str], set[str]]:
    """The shared libraries and ``.pcm`` files each path names, loaded here first."""
    libraries: set[str] = set()
    pcms: set[str] = set()
    for path in _each_path(paths):
        found_pcms, found_libraries = check_pcm_in_library_path(path)
        pcms |= found_pcms
        libraries |= found_libraries
    distribute_shared_libraries(libraries)
    return libraries, pcms


def trigger_run(starting_node: Any) -> None:
    """``TriggerRun``: the loop of the task's frame, over what was booked since the last."""
    from ....rdf.distributed import trigger

    trigger(starting_node._inner)


@singledispatch
def get_mergeablevalue(resultptr: Any) -> Any:
    """What a task sends back for a booked result: its partial, to be added to others'."""
    from ....rdf.distributed import partial_of

    return MergeableValue(partial_of(resultptr._inner), presented)


@get_mergeablevalue.register
def _(resultptr: AsNumpyResult) -> Any:
    from ....rdf.distributed import partial_of

    return MergeableValue(partial_of(resultptr._read), columns_of)


@get_mergeablevalue.register
def _(resultptr: SnapshotResult) -> Any:
    return SnapshotResult(resultptr.treename, resultptr.filenames)


def merge_values(mergeable_out: Any, mergeable_in: Any) -> None:
    """Add the second task's result into the first's."""
    mergeable_out.Merge(mergeable_in)


@singledispatch
def set_value_on_node(mergeable: Any, node: Any, backend: Any) -> None:
    """Give the node the user holds the value everything added up to."""
    node.value = mergeable.GetValue()


@set_value_on_node.register
def _(mergeable: SnapshotResult, node: Any, backend: Any) -> None:
    node.value = mergeable.GetValue(backend)


@singledispatch
def clone_action(result_promise: Any, range_id: int) -> Any:
    """``CloneResultAndAction``: the result booked again on the same frame, for another task."""
    from ....rdf.distributed import rebook

    return RResultPtr(rebook(result_promise._inner))


@clone_action.register
def _(result_promise: AsNumpyResult, range_id: int) -> Any:
    return _clone_asnumpyresult(result_promise)


@clone_action.register
def _(snap: SnapshotResult, range_id: int) -> Any:
    """The ``Snapshot`` booked again, into the file of this task's range."""
    from ....rdf.distributed import rebook

    stem = snap.filenames[0][:-5] if snap.filenames[0].endswith(".root") else snap.filenames[0]
    path = f"{stem[: stem.rfind('_')]}_{range_id}.root"
    booked = RResultPtr(rebook(snap._resultptr._inner, path))
    return SnapshotResult(snap.treename, [path], booked)
