"""Placement new into a ``TClonesArray``'s slot: ``new (clones[i]) TTrack(args)``."""

from __future__ import annotations

from typing import Any

import pytest

from cintfake import fake
from xrdroot.cint import Refusal, translate
from xrdroot.cint.execute import run_source
from xrdroot.cint.runtime import construct_at


class Clones:
    """A stand-in for ``TClonesArray``: slots, filled by ``AddAt``."""

    def __init__(self, kind: str = "") -> None:
        self.slots: dict[int, Any] = {}

    def AddAt(self, obj: Any, index: int) -> None:
        self.slots[index] = obj


def test_an_object_built_in_a_slot_is_put_there_and_is_the_new_expressions_value(
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = fake()
    root.TClonesArray = Clones
    source = """
    struct Hit { int n; Hit(int k) : n(k) {} };
    void t() {
      TClonesArray *arr = new TClonesArray("Hit"); TClonesArray &ar = *arr; int used = 0;
      for (int i = 0; i < 3; i++) new (ar[i]) Hit(10 * i);
      Hit *last = new (ar[used++]) Hit(7);
      printf("%d %d %d\\n", ((Hit *)arr->slots[2])->n, last->n, used);
    }
    """
    text = translate(source, "t.C")
    assert "construct_at(ar, i, Hit(10 * i))" in text
    run_source(source, "t.C", root=root)
    assert capsys.readouterr().out == "20 7 1\n"


def test_an_object_is_put_in_a_list_by_its_index_when_there_is_no_add_at() -> None:
    items: list[Any] = [None, None]
    assert construct_at(items, 1, "made") == "made" and items == [None, "made"]


def test_placement_new_into_a_buffer_is_refused() -> None:
    with pytest.raises(Refusal, match="placement new of anything but an object into an element"):
        translate("void t() { char buf[8]; TH1F *h = new (buf) TH1F(); }", "t.C")
