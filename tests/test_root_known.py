"""Classes a file holds without describing them, read by the layout ROOT declares."""

from __future__ import annotations

from xrdroot.buffer import IS_REFERENCED, Buffer
from xrdroot.known import HAS_UUID, KNOWN
from xrdroot.writer import WBuffer


def _tref(unique: int, bits: int, process: int | str) -> Buffer:
    out = WBuffer()
    out.u16(1)
    out.u32(unique)
    out.u32(bits)
    if bits & IS_REFERENCED:
        out.u16(9)
    if isinstance(process, str):
        out.string(process)
    else:
        out.u16(process)
    out.u16(0xBEEF)  # what follows the reference, which must be left unread
    return Buffer(bytes(out.data))


def _read(buf: Buffer) -> dict:
    return KNOWN["TRef"](None)(buf)


def test_a_tref_is_the_identifier_of_what_it_points_at_and_its_process_number():
    buf = _tref(7, 0, 3)
    assert _read(buf) == {"TObject": {"fUniqueID": 7, "fBits": 0}, "fPID": 3}
    assert buf.u16() == 0xBEEF


def test_a_tref_that_is_itself_referenced_steps_over_its_own_process_first():
    buf = _tref(2, IS_REFERENCED, 1)
    assert _read(buf)["fPID"] == 1
    assert buf.u16() == 0xBEEF


def test_a_tref_that_names_its_process_by_uuid_keeps_the_uuid():
    buf = _tref(5, HAS_UUID, "594fa8e5-b927")
    assert _read(buf)["fPID"] == "594fa8e5-b927"
    assert buf.u16() == 0xBEEF
