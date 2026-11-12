"""Process identity and proxy reference numbers outlive an individual workspace write."""

from __future__ import annotations

import uuid
from threading import RLock
from typing import Any
from weakref import WeakKeyDictionary

from ...buffer import Buffer
from ...errors import UnsupportedFeatureError
from ...file import Key

#: TObject stores reference identifiers in its low 24 bits.
MAX_UID = 0xFFFFFF


class Session:
    """Weak owner/member identities prevent both UID collisions and retained dead models."""

    def __init__(self) -> None:
        self.title = str(uuid.uuid4())
        self.next_uid = 0
        self.owners: WeakKeyDictionary[Any, dict[tuple[str, str], int]] = WeakKeyDictionary()
        self.files: WeakKeyDictionary[Any, int] = WeakKeyDictionary()
        self.lock = RLock()

    def uid(self, owner: Any, classname: str, name: str) -> int:
        with self.lock:
            members = self.owners.setdefault(owner, {})
            key = (classname, name)
            if key not in members:
                if self.next_uid == MAX_UID:
                    raise UnsupportedFeatureError(
                        "Writing a ROOT reference session after its 24-bit identifier namespace "
                        "is full is not implemented. Start another Python process to save more "
                        "models."
                    )
                self.next_uid += 1
                members[key] = self.next_uid
            return members[key]

    def file_index(self, directory: Any) -> tuple[int, bool]:
        if directory in self.files:
            return self.files[directory], False
        highest = -1
        for raw in directory._keys:
            key = Key(Buffer(raw))
            if key.classname != "TProcessID" or not key.name.startswith("ProcessID"):
                continue
            suffix = key.name[len("ProcessID"):]
            if not suffix.isdecimal():
                continue
            index = int(suffix)
            highest = max(highest, index)
        return highest + 1, True

    def remember(self, directory: Any, index: int) -> None:
        """Only this open writer has installed the session identity in its file-local list."""
        self.files[directory] = index


#: ROOT's session process object is reused across files and workspace keys.
SESSION = Session()
