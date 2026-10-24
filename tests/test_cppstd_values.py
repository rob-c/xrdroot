"""The generic value runtime uses explicit container provenance without loading ROOT."""

from __future__ import annotations

import numpy as np
import pytest

from xrdroot.cppstd import references, valuecopy, vectorassign, vectorvalue


def test_the_legacy_vector_paths_share_the_generic_factories_and_registries() -> None:
    from xrdroot.pyroot import vectorassign as old_assign
    from xrdroot.pyroot import vectorvalue as old_value

    assert old_assign.register is vectorassign.register
    assert old_assign.assigned is vectorassign.assigned
    assert old_assign._LAYOUTS is vectorassign._LAYOUTS
    assert old_value.register is vectorvalue.register
    assert old_value.native is vectorvalue.native
    assert old_value._KINDS is vectorvalue._KINDS


def test_interpreter_reference_exports_keep_the_generic_implementations() -> None:
    from xrdroot.cint.runtime import objects
    from xrdroot.cint.runtime import references as legacy

    assert objects.value_copy is valuecopy.value_copy
    assert legacy.__all__ == [
        "assign_call", "assign_method", "assign_into", "store_through", "assign_vector",
        "store_reference", "increment_reference",
    ]
    for name in references.__all__:
        assert getattr(legacy, name) is getattr(references, name)


def test_value_copy_preserves_raw_floating_payloads_and_independent_storage() -> None:
    bits = np.array([0x80000000, 0x7F800043, 0x7FC00042], dtype="uint32")
    original = bits.view("float32")
    copied = valuecopy.value_copy(original)
    assert copied.view("uint32").tobytes() == bits.tobytes()
    assert not np.shares_memory(original, copied)


def test_explicit_generic_registration_keeps_assignment_storage(monkeypatch: pytest.MonkeyPatch):
    class Vector:
        dtype = np.dtype("float32")

        def __init__(self, values):
            self._data = np.zeros(8, dtype=self.dtype)
            self._size = len(values)
            self._data[:self._size] = values

        def data(self):
            return self._data[:self._size]

    # Restore the registry entries after the test's private fixture class is no longer needed.
    monkeypatch.setitem(vectorvalue._KINDS, Vector, None)
    monkeypatch.setitem(vectorassign._LAYOUTS, Vector, None)
    vectorvalue.register(Vector, Vector.dtype, True)
    vectorassign.register(Vector, "std", {"data": Vector.data})
    source, target = Vector([1, 2, 3]), Vector([4])
    storage = target._data
    assert references.assign_into(target, source) is target
    assert target._data is storage
    assert target._size == 3
    assert target.data().tobytes() == source.data().tobytes()
    assert not np.shares_memory(target.data(), source.data())
