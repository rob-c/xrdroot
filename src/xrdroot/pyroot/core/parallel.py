"""``TThreadExecutor``, ``TProcessExecutor`` and the shared pool machinery ROOT runs work on.

ROOT's executors take a callable and a sequence and hand the answers back in the
order the sequence had them, whatever order the work finished in. ``Map`` gives
that list, ``MapReduce`` folds it with a reducer, ``Foreach`` discards it, and
``Submit`` returns a handle with ``.get()``. ``TThreadExecutor`` runs the work on
threads of this process; ``TProcessExecutor`` gives every worker its own
process, so what it runs sees no shared state - and its callable and arguments
must therefore be picklable, which the tutorials' free functions are.

``ExecutorUtils.ReduceObjects`` is subscripted by a type name and called, as the
translator writes it (``ROOT.ExecutorUtils.ReduceObjects['TH1F*']()``), and
answers a reducer for two of those objects: histograms add, numbers add,
collections extend.
"""

from __future__ import annotations

import concurrent.futures as _futures
import multiprocessing as _multiprocessing
import os
import pickle
from collections.abc import Callable, Sequence
from typing import Any, ClassVar

__all__ = [
    "TExecutor",
    "TThreadExecutor",
    "TProcessExecutor",
    "TThreadedObject",
    "TThread",
    "TThreadPool",
    "TTask",
    "TThreadPoolTaskImp",
    "ExecutorUtils",
]


def _items(args: Any) -> list[Any]:
    """What to work through: anything iterable, which a string is not, or one argument."""
    if isinstance(args, (str, bytes)) or not hasattr(args, "__iter__"):
        return [args]
    return list(args)


def _is_single(fn: Callable[..., Any]) -> bool:
    """Whether the callable takes the item itself rather than a tuple of arguments."""
    try:
        import inspect

        parameters = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return True
    positional = [
        p for p in parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL)
    ]
    if any(p.kind == p.VAR_POSITIONAL for p in positional):
        return False
    return len(positional) <= 1


def _call(fn: Callable[..., Any], item: Any) -> Any:
    """ROOT calls the work with the element; a tuple element is its arguments."""
    if isinstance(item, tuple) and not _is_single(fn):
        return fn(*item)
    return fn(item)


def _reduce(red: Callable[[Any, Any], Any], values: Sequence[Any]) -> Any:
    """Fold the answers in order with the reducer, as ROOT's ``MapReduce`` does."""
    if not values:
        return None
    total = values[0]
    for value in values[1:]:
        total = red(total, value)
    return total


class _Submitted:
    """A pending answer, with the ``.get()`` a C++ ``std::future`` has."""

    def __init__(self, future: Any) -> None:
        self._future = future

    def get(self, timeout: float | None = None) -> Any:
        """``get``: the answer, waiting for it if it is not there yet."""
        return self._future.result(timeout)

    def wait(self, timeout: float | None = None) -> bool:
        import concurrent.futures as futures

        done, _ = futures.wait([self._future], timeout)
        return self._future in done

    def __repr__(self) -> str:
        return f"<submitted {self._future}>"


class TExecutor:
    """The machinery both executors share: the pool, and the four ways to use it."""

    def __init__(self, nWorkers: int = 0) -> None:
        self._workers = int(nWorkers) if nWorkers else (os.cpu_count() or 1)
        self._pool: Any = None

    #: How this executor runs one piece of work: overridden by each kind of pool.
    def _make_pool(self, workers: int) -> Any:
        raise NotImplementedError

    def _ensure(self) -> Any:
        if self._pool is None:
            self._pool = self._make_pool(self._workers)
        return self._pool

    def Map(self, fn: Callable[..., Any], args: Any) -> list[Any]:
        """``Map``: every answer, in the order the arguments were given."""
        items = _items(args)
        pool = self._ensure()
        if self._workers == 1 or len(items) <= 1:
            return [_call(fn, item) for item in items]
        return list(pool.map(lambda item: _call(fn, item), items))

    def MapReduce(
        self,
        fn: Callable[..., Any],
        args: Any,
        red: Callable[[Any, Any], Any],
        red2: Callable[[Any, Any], Any] | None = None,
    ) -> Any:
        """``MapReduce``: ``Map`` then fold with ``red``, or with ``red`` and ``red2`` in turn.

        ``red2`` is ROOT's two-stage form: the first folds each worker's answers,
        the second folds those - with one worker's answers that is the same fold.
        """
        values = self.Map(fn, args)
        first = _reduce(red, values)
        if red2 is None:
            return first
        return _reduce(red2, [first])

    def Foreach(self, fn: Callable[..., Any], args: Any) -> None:
        """``Foreach``: the work, its answers thrown away."""
        self.Map(fn, args)
        return None

    def Submit(self, fn: Callable[..., Any], *args: Any) -> Any:
        """``Submit``: start the work and hand back a handle whose ``.get()`` is its answer."""
        pool = self._ensure()
        if self._workers == 1:
            done: Any = _futures.Future()
            done.set_result(fn(*args))
            return _Submitted(done)
        return _Submitted(pool.submit(fn, *args))

    def __enter__(self) -> TExecutor:
        return self

    def __exit__(self, *why: Any) -> None:
        self.Close()

    def Close(self) -> None:
        """``Close``: stop the workers; a later use starts new ones, as a re-opened pool."""
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

class TThreadExecutor(TExecutor):
    """``TThreadExecutor``: the work runs on threads of this process, sharing its memory."""

    def _make_pool(self, workers: int) -> Any:
        return _futures.ThreadPoolExecutor(max_workers=workers)


def _picklable(fn: Callable[..., Any], items: Sequence[Any]) -> bool:
    try:
        pickle.dumps((fn, items[:1]))
    except Exception:
        return False
    return True


class TProcessExecutor(TExecutor):
    """``TProcessExecutor``: every worker is a process, so nothing is shared between them.

    Work that cannot be pickled to a worker - a lambda, a bound method of an
    object holding a lock - runs on threads instead of failing: the answers the
    tutorials ask for are the same, and ROOT's own ``TProcessExecutor`` would
    refuse such a call outright.
    """

    def _make_pool(self, workers: int) -> Any:
        return _futures.ThreadPoolExecutor(max_workers=workers)

    def Map(self, fn: Callable[..., Any], args: Any) -> list[Any]:
        items = _items(args)
        if self._workers == 1 or len(items) <= 1 or not _picklable(fn, items):
            return super().Map(fn, items)
        context = _multiprocessing.get_context("fork" if hasattr(os, "fork") else "spawn")
        with _futures.ProcessPoolExecutor(max_workers=self._workers, mp_context=context) as pool:
            return list(pool.map(_spawn_call, [(fn, item) for item in items]))


def _spawn_call(pair: tuple[Callable[..., Any], Any]) -> Any:
    """One worker process's piece: the callable and its argument, sent together."""
    fn, item = pair
    return _call(fn, item)


class _Reducer:
    """A reducer for one type of object, as ``ReduceObjects['T*']()`` gives."""

    def __init__(self, type_name: str = "") -> None:
        self.type_name = type_name

    def __call__(self, one: Any, other: Any) -> Any:
        if one is None:
            return other
        if other is None:
            return one
        add = getattr(one, "Add", None)
        if callable(add):
            add(other)
            return one
        if isinstance(one, list):
            one.extend(other if isinstance(other, (list, tuple)) else [other])
            return one
        return one + other

    def __repr__(self) -> str:
        return f"<ReduceObjects {self.type_name or 'default'}>"


class _ReducerFactory:
    """One ``ReduceObjects<T>``: called, it makes the reducer for two ``T``."""

    def __init__(self, type_name: str = "") -> None:
        self.type_name = type_name

    def __call__(self) -> _Reducer:
        return _Reducer(self.type_name)


class _ReduceObjects:
    """``ExecutorUtils::ReduceObjects<T>`` as the translator writes it: subscript, then call."""

    def __getitem__(self, type_name: str) -> _ReducerFactory:
        return _ReducerFactory(str(type_name))

    def __call__(self) -> _ReducerFactory:
        return _ReducerFactory()


class ExecutorUtils:
    """``ROOT::ExecutorUtils``: the reducers and helpers ROOT's executors are used with."""

    ReduceObjects = _ReduceObjects()

    @staticmethod
    def ReduceVectors() -> _Reducer:
        """``ReduceVectors``: a reducer for two collections of the same kind."""
        return _Reducer("vector")


def _root_class(name: str) -> Any:
    """The ROOT class called ``name``, looked up when a threaded object is first used."""
    import xrdroot.pyroot as ROOT

    try:
        return getattr(ROOT, name)
    except AttributeError as why:
        raise AttributeError(
            f"ROOT has no class {name!r}, so a TThreadedObject of it cannot be made; "
            "name one of the classes this library provides."
        ) from why


class TThreadedObject:
    """``TThreadedObject<T>``: one ``T`` per thread, made when that thread first asks.

    ``Get()`` gives this thread's own instance - the tutorials copy it onto the
    stack and fill it in a tight loop, which is why ROOT makes the instance
    per thread rather than handing out a shared one. ``SnapshotMerge()`` sums
    what every thread has so far into a fresh object, leaving the copies in
    place, and ``Merge()`` sums them into the first one and returns it. What
    sum means is the class's own: histograms ``Add``, and a class without
    ``Add`` is added with ``+``.
    """

    def __init__(self, *arguments: Any) -> None:
        self._arguments = arguments
        self._per_thread: dict[int, Any] = {}
        self._merged: Any = None

    def _make(self) -> Any:
        raise AttributeError(
            "TThreadedObject needs the class it holds, written TThreadedObject<TH1F>; "
            "made without one there is no object to give each thread."
        )

    def Get(self) -> Any:
        """``Get``: this thread's instance, made on first use."""
        import threading

        ident = threading.get_ident()
        found = self._per_thread.get(ident)
        if found is None:
            found = self._make()
            self._per_thread[ident] = found
        return found

    def _sum(self) -> Any:
        """One object holding every thread's contribution, made fresh.

        With no thread having asked yet ROOT still answers with an object - an
        empty one, which is what a monitoring thread prints before any worker
        has filled anything.
        """
        total: Any = None
        for instance in list(self._per_thread.values()):
            if total is None:
                copyable = hasattr(instance, "Clone")
                total = instance.Clone(instance.GetName()) if copyable else instance
            else:
                total = _Reducer()(total, instance)
        if total is None:
            total = self._make()
            self._per_thread[0] = total
        return total

    def SnapshotMerge(self) -> Any:
        """``SnapshotMerge``: the sum so far, in a new object, the copies untouched."""
        self._merged = self._sum()
        return self._merged

    def Merge(self) -> Any:
        """``Merge``: the sum so far, returned; a second call sums again from the copies."""
        self._merged = self._sum()
        return self._merged

    def __repr__(self) -> str:
        return f"<TThreadedObject of {len(self._per_thread)} thread(s)>"

    @classmethod
    def __class_getitem__(cls, type_name: str) -> Any:
        """``TThreadedObject<TH1F>``: the same machinery for the class named there."""
        if not isinstance(type_name, str):
            return cls
        return type(
            f"TThreadedObject[{type_name}]",
            (cls,),
            {"_make": lambda self: _root_class(type_name)(*self._arguments),
             "__doc__": TThreadedObject.__doc__},
        )


class TTask:
    """``TTask``: a named unit of work ROOT can queue under another task.

    ``Add`` hangs a subtask off this one, ``ExecuteTask`` runs a task with this
    one as its parent, and ``Run`` does this task's own work - nothing, for a
    task that only groups others, which is how the ``TTask`` tutorials use it.
    """

    def __init__(self, name: str = "", title: str = "") -> None:
        self.name = name
        self.title = title
        self._subtasks: list[TTask] = []
        self._parent: TTask | None = None

    def GetName(self) -> str:
        return self.name

    def GetTitle(self) -> str:
        return self.title

    def SetName(self, name: str) -> None:
        self.name = name

    def SetTitle(self, title: str) -> None:
        self.title = title

    def Add(self, task: TTask) -> None:
        """``Add``: a subtask, which this task then owns."""
        task._parent = self
        self._subtasks.append(task)

    def GetListOfTasks(self) -> list[TTask]:
        return list(self._subtasks)

    def GetParent(self) -> TTask | None:
        return self._parent

    def Run(self, option: str = "") -> None:
        """``Run``: do this task's work, then every subtask's, in the order added."""
        self.ExecuteTask(self, option)
        for task in self._subtasks:
            task.Run(option)

    def ExecuteTask(self, task: TTask, option: str = "") -> None:
        """``ExecuteTask``: run one task with this one as its parent; nothing, by default."""

    def CleanTasks(self, option: str = "") -> None:
        """``CleanTasks``: forget the subtasks once they have run, as ROOT does."""
        self._subtasks.clear()

    def GetOption(self) -> str:
        return ""

    def __repr__(self) -> str:
        return f"<TTask {self.name!r} with {len(self._subtasks)} subtask(s)>"


class TThreadPoolTaskImp:
    """``TThreadPoolTaskImp<task, argument>``: the base a pool task is written against.

    ROOT's own such tasks override :meth:`Process` and the pool calls it; a task
    defined in a tutorial derives from this and overrides ``Process`` too, so the
    base gives the signature and the plumbing, and nothing else.
    """

    def __init__(self, *arguments: Any) -> None:
        self._arguments = arguments

    def Process(self, *arguments: Any) -> Any:
        """``Process``: the work; a subclass replaces this."""
        return None

    def GetTaskName(self) -> str:
        return type(self).__name__

    @classmethod
    def __class_getitem__(cls, names: Any) -> Any:
        """``TThreadPoolTaskImp<Task, Int_t>``: the base itself, keeping the names for ``repr``."""
        label = names if isinstance(names, str) else ", ".join(str(n) for n in names)
        return type(f"TThreadPoolTaskImp[{label}]", (cls,), {"__doc__": TThreadPoolTaskImp.__doc__})


class TThread:
    """``TThread``: one of ROOT's own threads, running a callable it was made with.

    ``Join`` waits for it, ``Kill`` asks it to stop (Python has no way to stop a
    thread, so the flag is set and the work is expected to look at it), and the
    class-level helpers answer for the thread asking: ``SelfId`` is this
    thread's id, ``Sleep`` waits - in ROOT's units, microseconds for the one
    number form - and ``Ps`` lists the threads ROOT knows of.
    """

    #: The threads made so far, as ``Ps`` lists them.
    _known: ClassVar[list[TThread]] = []

    def __init__(self, name: str = "", work: Any = None,
                 *arguments: Any) -> None:
        self._name = name
        self._stopped = False
        self._error: BaseException | None = None
        self._thread: Any = None
        self._work = work
        self._arguments = arguments
        if work is not None:
            self._start(work, arguments)
        TThread._known.append(self)

    def Run(self) -> None:
        """``Run``: start the work; a thread already running is left as it is."""
        if self._thread is None and self._work is not None:
            self._start(self._work, self._arguments)

    def _start(self, work: Any, arguments: tuple[Any, ...]) -> None:
        import threading

        def run() -> None:
            try:
                work(*arguments)
            except BaseException as why:  # kept, and raised by Join, as ROOT reports it
                self._error = why

        self._thread = threading.Thread(target=run, name=self._name or None, daemon=True)
        self._thread.start()

    def Join(self) -> int:
        """``Join``: wait for the work to finish, and re-raise what it raised."""
        if self._thread is not None:
            self._thread.join()
        if self._error is not None:
            why, self._error = self._error, None
            raise why
        return 0

    def Kill(self) -> int:
        """``Kill``: mark it stopped; a Python thread cannot be stopped from outside."""
        self._stopped = True
        return 0

    def IsRunning(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def GetName(self) -> str:
        return self._name

    def GetId(self) -> int:
        return 0 if self._thread is None else self._thread.ident or 0

    @staticmethod
    def SelfId() -> int:
        """``SelfId``: the id of the thread asking."""
        import threading

        return threading.get_ident()

    @staticmethod
    def Sleep(first: float = 0.0, second: float | None = None) -> None:
        """``Sleep``: microseconds; with two numbers, from the first to the second."""
        import time

        micro = first if second is None else second - first
        if micro > 0:
            time.sleep(micro / 1e6)

    @staticmethod
    def Printf(format_string: str, *arguments: Any) -> None:
        """``Printf``: ROOT's own printf - the C format, thread-safely, to standard output."""
        from .cformat import c_format

        print(c_format(format_string, *arguments))

    @staticmethod
    def Ps() -> None:
        """``Ps``: print the threads ROOT knows of, as ROOT prints them."""
        print("Threads")
        print(f"There are {len(TThread._known)} threads")
        for index, thread in enumerate(TThread._known):
            print(f" {index} {thread.GetName() or '(unnamed)'} running={thread.IsRunning()}")

    def __repr__(self) -> str:
        return f"<TThread {self._name!r} running={self.IsRunning()}>"


class TThreadPool:
    """``TThreadPool<T, A>``: tasks pushed onto a pool, which its threads then process.

    ``PushTask(task, argument)`` queues one; ``Drain`` runs everything queued on
    the pool's threads and waits; ``Stop`` drains and shuts the pool down.
    """

    def __init__(self, nThreads: int = 0, needDbg: bool = False) -> None:
        self._threads = int(nThreads) if nThreads else (os.cpu_count() or 1)
        self._debug = bool(needDbg)
        self._queued: list[tuple[Any, Any]] = []
        self._pool: Any = None

    def PushTask(self, task: Any, argument: Any = None) -> None:
        """``PushTask``: queue one task with the argument its ``Process`` takes."""
        self._queued.append((task, argument))

    def Drain(self) -> None:
        """``Drain``: run what is queued, on the pool's threads, and wait for all of it."""
        if not self._queued:
            return
        if self._pool is None:
            self._pool = _futures.ThreadPoolExecutor(max_workers=self._threads)
        work, self._queued = self._queued, []
        futures = [self._pool.submit(task.Process, argument) for task, argument in work]
        for future in futures:
            future.result()

    def Stop(self, wait: bool = True) -> None:
        """``Stop``: drain what is queued, then shut the threads down."""
        self.Drain()
        if self._pool is not None and wait:
            self._pool.shutdown(wait=True)
            self._pool = None

    def GetPoolSize(self) -> int:
        return self._threads

    @classmethod
    def __class_getitem__(cls, names: Any) -> Any:
        """``TThreadPool<T, A>``: the pool, keeping the names for ``repr``."""
        label = names if isinstance(names, str) else ", ".join(str(n) for n in names)
        return type(f"TThreadPool[{label}]", (cls,), {"__doc__": TThreadPool.__doc__})

    def __repr__(self) -> str:
        return f"<TThreadPool of {self._threads} thread(s), {len(self._queued)} queued>"
