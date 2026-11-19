"""``TRef``, ``TRefArray`` and ``TProcessID``: persistent pointers, and what numbers their targets.

A ``TRef`` does not hold an object; it holds the object's number and the
process that numbered it. ``TProcessID::AssignID`` gives an object the next
number of the current process - its ``fUniqueID``, the process's index in the
top byte - marks it ``kIsReferenced``, and files it in that process's table,
where ``TRef::GetObject`` looks it up. A process numbers at most 2^24 - 1
objects; the next one starts a new process, as ROOT's does.

A table here holds its objects weakly, so a reference to an object that has
gone finds nothing, as ROOT's finds nothing once the object is deleted. In a
file each process is a ``TProcessID`` record, ``ProcessID0`` and on, named by
its index in that file and titled with its UUID; an object read with
``kIsReferenced`` is filed under the process its file record names, which is
how a ``TRef`` read beside it finds it again.
"""

from __future__ import annotations

import uuid
import weakref
from collections.abc import Iterator
from typing import Any, ClassVar

from .objects import TNamed, TObject

__all__ = ["TProcessID", "TProcessUUID", "TRef", "TRefArray"]

#: ``TObject::kIsReferenced``: the object is filed in a process's table by its number.
IS_REFERENCED = 1 << 4
#: ``TObject::kMustCleanup``, which numbering an object sets beside ``kIsReferenced``.
MUST_CLEANUP = 1 << 3
#: ``TRef::kHasUUID``: the reference names its object by UUID rather than by number.
HAS_UUID = 1 << 5
#: The bits of an identifier that number an object within its process.
NUMBER_MASK = 0xFFFFFF
#: The most objects one process numbers before the next process takes over.
MOST = NUMBER_MASK
#: The index a process past the 255th is filed under in an identifier's top byte.
FAR = 0xFF
#: ``TObject::kIsOnHeap | kNotDeleted``, which every object made on the heap carries:
#: a ``TRef`` is one here, as ``TClass::New`` makes it when a file reads one back.
HEAP_BITS = 0x03000000
#: ``TRef::SetAction`` numbers the action it was given in the bits it sets - ``execid << 16``,
#: ``execid`` being the action's place among ``AddExec``'s, from one.
ACTION_SHIFT = 16
#: The bits of ``fBits`` that number it - ``TestBits(0xff0000) >> 16``, ROOT's.
EXEC_MASK = 0xFF0000


def _unique(obj: Any) -> int:
    getter = getattr(obj, "GetUniqueID", None)
    return int(getter()) if callable(getter) else 0


class TProcessID(TNamed):
    """``TProcessID``: one process's numbering of the objects referenced in it."""

    CLASS_TITLE = "Process Unique Identifier in time and space"

    #: ``fgPIDs``: every process known in this session, by index.
    _pids: ClassVar[list[TProcessID]] = []
    #: ``fgPID``: the process numbering objects now.
    _current: TProcessID | None = None
    #: ``fgNumber``: the number the current process gave last.
    _number = 0

    def __init__(self, name: Any = "", title: Any = "") -> None:
        super().__init__(name, title)
        self._objects: dict[int, Any] = {}

    def __repr__(self) -> str:
        return f"<TProcessID {self.GetName()!r} {self.GetTitle()!r}>"

    # -- the session's processes --------------------------------------------------

    @staticmethod
    def AddProcessID() -> TProcessID:
        """A new process, with a new UUID, filed next among the session's."""
        index = len(TProcessID._pids)
        made = TProcessID(f"ProcessID{index}", str(uuid.uuid4()))
        made.SetUniqueID(index)
        TProcessID._pids.append(made)
        if TProcessID._current is None:
            TProcessID._current = made
        return made

    @staticmethod
    def GetSessionProcessID() -> TProcessID:
        """The process this session started as: the first one."""
        if not TProcessID._pids:
            TProcessID.AddProcessID()
        return TProcessID._pids[0]

    @staticmethod
    def GetPID() -> TProcessID:
        """The process numbering objects now."""
        TProcessID.GetSessionProcessID()
        assert TProcessID._current is not None
        return TProcessID._current

    @staticmethod
    def GetPIDs() -> list[TProcessID]:
        return list(TProcessID._pids)

    @staticmethod
    def GetNProcessIDs() -> int:
        return len(TProcessID._pids)

    @staticmethod
    def GetProcessID(index: int) -> TProcessID | None:
        index = int(index)
        return TProcessID._pids[index] if 0 <= index < len(TProcessID._pids) else None

    @staticmethod
    def IsValid(pid: Any) -> bool:
        """Is ``pid`` one of the session's processes?"""
        return any(known is pid for known in TProcessID._pids)

    @staticmethod
    def GetObjectCount() -> int:
        return TProcessID._number

    @staticmethod
    def SetObjectCount(number: int) -> None:
        TProcessID._number = int(number)

    @staticmethod
    def _find(title: str) -> TProcessID | None:
        """The session's process of this UUID, if it has one."""
        return next((pid for pid in TProcessID._pids if pid.GetTitle() == title), None)

    @staticmethod
    def _adopt(name: str, title: str) -> TProcessID:
        """The session's process of UUID ``title``, made and filed if it has none: what reading a
        file's ``TProcessID`` comes to."""
        TProcessID.GetSessionProcessID()  # the session's own comes first, whatever is read
        found = TProcessID._find(title)
        if found is not None:
            return found
        index = len(TProcessID._pids)
        made = TProcessID(name, title)
        made.SetUniqueID(index)
        TProcessID._pids.append(made)
        return made

    # -- numbering objects -------------------------------------------------------

    @staticmethod
    def AssignID(obj: Any) -> int:
        """Give ``obj`` the next number, unless it has one: what making a ``TRef`` to it does."""
        number = _unique(obj) & NUMBER_MASK
        if number and obj.TestBit(IS_REFERENCED):
            return number
        pid = TProcessID.GetPID()
        if TProcessID._number >= MOST:
            pid = TProcessID._current = TProcessID.AddProcessID()
            TProcessID._number = 0
        TProcessID._number += 1
        number = TProcessID._number
        obj.SetBit(IS_REFERENCED | MUST_CLEANUP)
        index = min(pid.GetUniqueID(), FAR)
        obj.SetUniqueID(number + (index << 24))
        pid.PutObjectWithID(obj, number)
        return number

    @staticmethod
    def GetProcessWithUID(uid: Any, obj: Any = None) -> TProcessID | None:
        """The process an identifier's top byte names - or, past the 255th, the one that
        has ``obj``."""
        identifier = _unique(uid) if isinstance(uid, TObject) else int(uid)
        index = identifier >> 24
        if index < FAR:
            return TProcessID.GetProcessID(index)
        holder = obj if obj is not None else uid
        return next((pid for pid in TProcessID._pids[FAR:] if pid._holds(holder)), None)

    def _holds(self, obj: Any) -> bool:
        return self.GetObjectWithID(_unique(obj)) is obj

    def PutObjectWithID(self, obj: Any, uid: int = 0) -> None:
        number = (int(uid) or _unique(obj)) & NUMBER_MASK
        try:
            self._objects[number] = weakref.ref(obj)
        except TypeError:  # an object Python cannot point at weakly is held outright
            self._objects[number] = lambda: obj

    def GetObjectWithID(self, uid: Any) -> Any:
        held = self._objects.get(int(uid) & NUMBER_MASK)
        return None if held is None else held()

    def RecursiveRemove(self, obj: Any) -> None:
        number = _unique(obj) & NUMBER_MASK
        if self.GetObjectWithID(number) is obj:
            del self._objects[number]

    def GetObjects(self) -> list[Any]:
        return [found for found in (held() for held in self._objects.values()) if found]

    def Clear(self, option: str = "") -> None:
        self._objects.clear()


class TProcessUUID(TProcessID):
    """``TProcessUUID``: the process that numbers objects referred to by UUID."""

    CLASS_TITLE = "TProcessID managing UUIDs"


class TRef(TObject):
    """``TRef``: a pointer that survives being written - the object's number and its process.

    An action - ``SetAction`` given a command, or one of ``AddExec``'s - is what
    ``GetObject`` runs when the object is not in memory, as ROOT's ``TExec``
    would load it; then it looks again.
    """

    CLASS_TITLE = "Persistent Reference link to a TObject"

    #: ``TRef::GetListOfExecs``: the actions named so far, by the number ``AddExec`` gave.
    _execs: ClassVar[list[str]] = []

    def __init__(self, obj: Any = None) -> None:
        super().__init__()
        # ``TObject()`` asks ``TStorage::UpdateIsOnHeap`` where it was made; every object
        # here is made on the heap, so a fresh reference carries ROOT's two memory bits.
        self._store_bits(HEAP_BITS)
        self._pid: TProcessID | None = None
        self._action: Any = None
        if isinstance(obj, TRef):
            self._pid = obj._pid
            self.SetUniqueID(obj.GetUniqueID())
        else:
            self.Set(obj)

    def Set(self, obj: Any) -> TRef:
        """``operator=(TObject*)``: refer to ``obj`` - number it if it is not numbered yet."""
        if obj is None or obj == 0:
            self._pid = None
            self.SetUniqueID(0)
            return self
        number = TProcessID.AssignID(obj)
        self._pid = TProcessID.GetProcessWithUID(_unique(obj), obj)
        self.ResetBit(HAS_UUID)
        self.SetUniqueID(number)
        return self

    def _assign(self, value: Any) -> None:
        """``operator=``, which writes into this reference where ``tref = other`` would bind
        a new one over it: the bits - and with them the action the reference was given -
        stay, as C++'s assignment leaves them; only the number and the process move."""
        if isinstance(value, TRef):
            self.SetUniqueID(value.GetUniqueID())
            self._pid = value._pid
            self.SetBit(HAS_UUID, value.TestBit(HAS_UUID))
            return
        self.Set(value)

    def GetObject(self) -> Any:
        """The object referred to, if its process still has it - after running the action,
        if there is one and the object was not there - and ``None`` otherwise."""
        if self._pid is None or not TProcessID.IsValid(self._pid):
            return None
        found = self._pid.GetObjectWithID(self.GetUniqueID())
        if found is not None:
            return found
        action = self._action if isinstance(self._action, str) else self._numbered_action()
        if action is None:
            return None
        from .troot import gROOT

        gROOT.ProcessLine(action)
        return self._pid.GetObjectWithID(self.GetUniqueID())

    def _numbered_action(self) -> str | None:
        """The action this reference's bits number, which is how a reference read back from
        a file - one ``SetAction`` was never called on here - knows the one it was given:
        ROOT keeps only the exec's number in ``fBits``."""
        execid = (self._bitword() & EXEC_MASK) >> ACTION_SHIFT
        return TRef._execs[execid - 1] if 0 < execid <= len(TRef._execs) else None

    @staticmethod
    def AddExec(name: str) -> int:
        """``AddExec``: an action by name, and its number - the same number for the same name."""
        if name not in TRef._execs:
            TRef._execs.append(str(name))
        return TRef._execs.index(name) + 1

    @staticmethod
    def GetListOfExecs() -> list[str]:
        return TRef._execs

    def SetAction(self, action: Any) -> None:
        """``SetAction``: a command to run when the object is wanted and not here - or the
        object whose class's ``//EXEC:`` comment names it, which nothing here reads.

        ROOT looks the command up among ``AddExec``'s and, finding it, sets the bit that
        numbers it; a command it does not know it refuses. Here an unknown command is kept
        and run all the same - a macro's ``//EXEC:`` line names one nothing registered - so
        only a registered action moves the bits."""
        if not isinstance(action, str):
            self._action = action
            return
        self._action = action
        if action in TRef._execs:
            self.SetBit((TRef._execs.index(action) + 1) << ACTION_SHIFT)

    def GetAction(self) -> Any:
        return self._action

    def GetPID(self) -> TProcessID | None:
        return self._pid

    def IsValid(self) -> bool:
        return self.GetUniqueID() != 0 and self._pid is not None

    def Clear(self, option: str = "") -> None:
        self.Set(None)

    def __bool__(self) -> bool:
        return self.GetObject() is not None

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TRef):
            return self._pid is other._pid and self.GetUniqueID() == other.GetUniqueID()
        return self.GetObject() is other

    def __hash__(self) -> int:
        return hash((id(self._pid), self.GetUniqueID()))


class TRefArray(TObject):
    """``TRefArray``: references to objects of one process, by their numbers, in slots."""

    CLASS_TITLE = "An array of references to TObjects"

    def __init__(self, *args: Any) -> None:
        super().__init__()
        given = next((arg for arg in args if isinstance(arg, TProcessID)), None)
        #: The process every object here belongs to: the session's unless one is given.
        self._pid: TProcessID = given or TProcessID.GetSessionProcessID()
        self._uids: list[int] = []
        self._lower = 0
        self._name = ""

    def GetName(self) -> str:
        return self._name

    def SetName(self, name: Any) -> None:
        self._name = str(name)

    def _take(self, obj: Any) -> int:
        """The identifier to file for ``obj``, which has to be of this array's process: one
        already referenced as it is, one not yet numbered by the session's process -
        ``TRefArray::GetObjectUID``. Any other is refused with ROOT's error; -1 says so."""
        if obj is None:
            return 0
        if obj.TestBit(HAS_UUID):
            valid = False
        elif obj.TestBit(IS_REFERENCED):
            valid = TProcessID.GetProcessWithUID(obj, obj) is self._pid
        else:
            valid = self._pid is TProcessID.GetSessionProcessID()
        if valid:
            return _unique(obj) if obj.TestBit(IS_REFERENCED) else TProcessID.AssignID(obj)
        from .messages import message

        message(
            "Error",
            "TRefArray::AddAtAndExpand",
            "The object at %s is not registered in the process the TRefArray points to "
            "(pid = %s/%s)",
            f"0x{id(obj):x}",
            self._pid.GetName(),
            self._pid.GetTitle(),
        )
        return -1

    def Add(self, obj: Any) -> None:
        self.AddLast(obj)

    def AddLast(self, obj: Any) -> None:
        number = self._take(obj)
        if number >= 0:
            self._uids.append(number)

    def AddAt(self, obj: Any, index: int) -> None:
        number = self._take(obj)
        if number < 0:
            return
        index = int(index) - self._lower
        while len(self._uids) <= index:
            self._uids.append(0)
        self._uids[index] = number

    AddAtAndExpand = AddAt

    def At(self, index: int) -> Any:
        index = int(index) - self._lower
        if not 0 <= index < len(self._uids):
            return None
        number = self._uids[index]
        return self._pid.GetObjectWithID(number) if number else None

    UncheckedAt = At

    def __getitem__(self, index: int) -> Any:
        return self.At(index)

    def __iter__(self) -> Iterator[Any]:
        return (self.At(index + self._lower) for index in range(self.GetEntriesFast()))

    def __len__(self) -> int:
        return self.GetEntriesFast()

    def GetEntriesFast(self) -> int:
        """Up to the last slot filled, empty ones and all."""
        filled = [index for index, number in enumerate(self._uids) if number]
        return filled[-1] + 1 if filled else 0

    def GetLast(self) -> int:
        return self.GetEntriesFast() - 1 + self._lower

    def GetAbsLast(self) -> int:
        return self.GetEntriesFast() - 1

    def GetEntries(self) -> int:
        return sum(1 for number in self._uids if number)

    def GetUID(self, index: int) -> int:
        return self._uids[int(index) - self._lower]

    def GetPID(self) -> TProcessID:
        return self._pid

    def GetLowerBound(self) -> int:
        return self._lower

    def IsEmpty(self) -> bool:
        return self.GetEntries() == 0

    def Clear(self, option: str = "") -> None:
        self._uids = []

    Delete = Clear

    def Compress(self) -> None:
        """``Compress``: the empty slots taken out, the rest moved up."""
        self._uids = [number for number in self._uids if number]

    def RemoveAt(self, index: int) -> Any:
        found = self.At(index)
        self._uids[int(index) - self._lower] = 0
        return found

    def Remove(self, obj: Any) -> Any:
        for index, number in enumerate(self._uids):
            if number and self._pid.GetObjectWithID(number) is obj:
                self._uids[index] = 0
                return obj
        return None

    def IndexOf(self, obj: Any) -> int:
        for index in range(self.GetEntriesFast()):
            if self.At(index + self._lower) is obj:
                return index + self._lower
        return self._lower - 1


# -- in a file ------------------------------------------------------------------------


def file_process(file: Any, index: int) -> TProcessID | None:
    """The session's process that the file's ``ProcessID{index}`` record names - made and
    filed when the session has none of its UUID - or ``None`` when the file has no such record:
    ``TFile::ReadProcessID``."""
    held = file.__dict__.setdefault("_processes", {})
    if index in held:
        return held[index]  # type: ignore[no-any-return]
    reader = file._reader()
    name = f"ProcessID{index}"
    if reader is None or name not in reader.keys():
        return None
    members = reader[name]
    named = members.get("TNamed", members)
    held[index] = TProcessID._adopt(str(named.get("fName", name)), str(named.get("fTitle", "")))
    return held[index]  # type: ignore[no-any-return]


def settle(found: Any, members: Any, file: Any) -> None:
    """What reading an object read from ``file`` leaves to the session: a reference given the
    process its file names, and an object a reference may point at filed in its process.

    The references among an object's members are given their process too. ROOT reads one as
    part of the member's own ``Streamer``, which nothing here runs - a declared class is made
    by its constructor and then filled from its record - so without this a ``TRef fReference``
    read back names no process, and ``GetObject`` gives up before it can look its object up, or
    run the action its bits number (roottest/root/io/references/actions).
    """
    if not isinstance(members, dict) or file is None:
        return
    if isinstance(found, (TRef, TRefArray)):
        _give_process(found, members, file)
        return
    tobject = members if "pidf" in members else members.get("TObject")
    if isinstance(tobject, dict) and "pidf" in tobject and isinstance(found, TObject):
        _filed(found, tobject, file_process(file, int(tobject["pidf"])))
    _settle_members(found, members, file)


def _settle_members(found: Any, members: dict[str, Any], file: Any) -> None:
    """The references held by an object read, given the process their records name."""
    from ...pycxx.symbols import member_name

    for name, record in members.items():
        if isinstance(record, dict) and "fPID" in record:
            held = getattr(found, member_name(name), None)
            if isinstance(held, (TRef, TRefArray)):
                _give_process(held, record, file)


def _give_process(found: Any, members: dict[str, Any], file: Any) -> None:
    """``found`` given the session's process that the file's ``ProcessID`` record names."""
    pid = file_process(file, int(members.get("fPID", 0)))
    if pid is not None:
        found._pid = pid


def _filed(found: Any, tobject: dict[str, Any], pid: TProcessID | None) -> None:
    """An object read that a reference may point at, numbered and filed in ``pid``."""
    if pid is None:
        return
    number = int(tobject.get("fUniqueID", 0)) & NUMBER_MASK
    found.SetUniqueID(number + (min(pid.GetUniqueID(), FAR) << 24))
    found.SetBit(IS_REFERENCED)
    pid.PutObjectWithID(found, number)


def _tref_from(members: Any) -> TRef:
    made = TRef()
    tobject = members.get("TObject", {})
    made.SetUniqueID(int(tobject.get("fUniqueID", 0)))
    # ROOT's ``TObject::Streamer`` leaves the memory bits out of what it writes - a stack
    # object's ``kIsOnHeap`` is not the reader's - so an object read is given them afresh.
    made._store_bits(int(tobject.get("fBits", 0)) | HEAP_BITS)
    return made


def _trefarray_from(members: Any) -> TRefArray:
    made = TRefArray()
    made._name = str(members.get("fName", ""))
    made._lower = int(members.get("fLowerBound", 0))
    made._uids = [int(uid) for uid in members.get("fUIDs", ())]
    return made


def _process_from(members: Any) -> TProcessID:
    named = members.get("TNamed", members)
    return TProcessID._adopt(str(named.get("fName", "")), str(named.get("fTitle", "")))


def _register() -> None:
    from .wrapping import register_members

    register_members("TRef", _tref_from)
    register_members("TRefArray", _trefarray_from)
    register_members("TProcessID", _process_from)


_register()
