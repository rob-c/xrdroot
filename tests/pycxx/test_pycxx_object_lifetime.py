"""A translated object's lifetime: its destructor, its class's statics, and cppyy's heuristic.

roottest/python/memory/memory is what these pin: an object counts itself in and out through a
class static, ``del`` runs the destructor, ``obj.counter = 1`` writes the static C++ would write,
and a call the heuristic policy takes as handing the object to C++ leaves the destructor to the
explicit ``TClass::Destructor``.
"""

from __future__ import annotations

import pytest

from xrdroot.pycxx import run_source, translate
from xrdroot.pycxx.runtime import objects

#: The counted class of roottest/python/memory/MemTester.C, less the copy constructor.
COUNTED = r"""
class Counted {
public:
   static int live;
   Counted() { ++live; }
   ~Counted() { --live; }
   void Nothing() {}
   static void Take(Counted *) {}
   static void Look(const Counted &) {}
};

int Counted::live = 0;
"""


def _counted(source: str = COUNTED):
    """The translated class, and the module it lives in - its statics last as long as they do."""
    module = run_source(source, "counted.cxx", call=False)
    return module["Counted"]


def test_a_collected_object_runs_its_destructor():
    """``del a`` is the end of the instance's Python life, and so of its C++ one."""
    counted = _counted()
    assert counted.live == 0
    a = counted()
    b = counted()
    assert counted.live == 2
    del a
    assert counted.live == 1
    del b
    assert counted.live == 0


def test_an_instance_writing_a_static_writes_the_class():
    """C++'s ``obj.counter = 1`` is the static's, not a field the instance gains."""
    counted = _counted()
    a = counted()
    a.live = 5
    assert counted.live == 5
    assert "live" not in vars(a)


def test_an_instance_writing_a_field_keeps_it_to_itself():
    counted = _counted()
    a = counted()
    a.note = "mine"
    assert a.note == "mine"
    assert "note" not in vars(counted)
    assert not hasattr(counted, "note")


def test_a_subclass_writing_a_static_writes_the_class_that_declares_it():
    """A Python subclass of a translated class inherits ``__setattr__``; C++ still writes the
    static of the class that declared it, so ``D().live = 7`` is ``Counted::live``, not ``D``'s."""
    counted = _counted()

    class Derived(counted):
        def __init__(self):
            super().__init__()

    d = Derived()
    d.live = 7
    assert counted.live == 7
    assert "live" not in vars(Derived)


def test_a_class_without_statics_has_no_static_write_to_redirect():
    python = translate("class Plain { public: int f; };\n", "plain.cxx")
    assert "_cxx_statics_" not in python
    assert "__setattr__" not in python


def test_a_class_without_a_destructor_gets_no_finalizer():
    """Nothing is emitted to run at collection, so an instance is collected as any Python one."""
    python = translate("class Plain { public: int f; };\n", "plain.cxx")
    assert "__del__" not in python


def test_a_destructor_runs_once_however_many_paths_reach_it():
    """``delete p``, a scope's end and ``del`` may all arrive; the first one is the destructor."""
    counted = _counted()
    a = counted()
    a._destruct()
    a._destruct()
    assert counted.live == 0
    del a
    assert counted.live == 0


def test_a_released_object_survives_del_until_it_is_destroyed_explicitly(monkeypatch):
    """With the heuristic policy on, a non-const pointer call is taken as C++ keeping the
    object: ``del`` leaves it, and ``TClass::Destructor`` is what destroys it - once."""
    monkeypatch.setattr(objects, "HEURISTIC", True)
    counted = _counted()
    kept = counted()
    counted.Take(kept)
    assert counted.live == 1
    del kept
    assert counted.live == 1

    gone = counted()
    counted.Take(gone)
    gone._destruct()
    assert counted.live == 1
    del gone
    assert counted.live == 1


def test_a_const_reference_call_gives_up_nothing(monkeypatch):
    """A const reference cannot take ownership, so the heuristic says nothing about it."""
    monkeypatch.setattr(objects, "HEURISTIC", True)
    counted = _counted()
    a = counted()
    counted.Look(a)
    assert not a._cxx_released
    del a
    assert counted.live == 0


def test_a_call_releases_nothing_while_the_heuristic_is_off():
    """ROOT 6.40's default: the caller owns what it passes, so ``del`` destroys it."""
    counted = _counted()
    a = counted()
    counted.Take(a)
    assert not a._cxx_released
    del a
    assert counted.live == 0


def test_a_released_value_that_cannot_be_marked_is_left_alone(monkeypatch):
    """A call may be handed something with no ``__dict__`` - a number, a plain object - and
    the heuristic has nowhere to put its flag; that is not an error, just nothing to say."""
    monkeypatch.setattr(objects, "HEURISTIC", True)
    objects.release(3)
    objects.release(object())


def test_the_heuristic_switch_answers_with_the_policy_it_replaced(monkeypatch):
    """``ROOT.SetHeuristicMemoryPolicy`` asks this for the value in force before the call."""
    monkeypatch.setattr(objects, "HEURISTIC", False)
    assert (objects.set_heuristic(True), objects.HEURISTIC) == (False, True)
    assert (objects.set_heuristic(1), objects.HEURISTIC) == (True, True)
    assert (objects.set_heuristic(0), objects.HEURISTIC) == (True, False)


@pytest.mark.parametrize("python", ["release", "set_heuristic", "HEURISTIC"])
def test_the_heuristic_is_the_translators_to_keep(python: str):
    """What a translated destructor's lifetime depends on is spelled in the runtime's own file."""
    assert hasattr(objects, python)
