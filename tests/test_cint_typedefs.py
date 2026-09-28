"""``typedef struct [Tag] {...} T;``: a class defined in a typedef, named by it."""

from __future__ import annotations

import pytest

from cintfake import fake
from xrdroot.cint import translate
from xrdroot.cint.execute import run_source


def test_an_unnamed_struct_takes_the_typedefs_name(capsys: pytest.CaptureFixture[str]) -> None:
    source = """
    typedef struct { float vect[3]; int nmec; } Gctrak_t;
    void t() { Gctrak_t g; g.nmec = 4; g.vect[1] = 2.5; printf("%d %g\\n", g.nmec, g.vect[1]); }
    """
    assert "class Gctrak_t:" in translate(source, "t.C")
    run_source(source, "t.C", root=fake())
    assert capsys.readouterr().out == "4 2.5\n"


def test_a_tagged_struct_is_its_tag_and_every_typedef_name_is_a_name_for_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = """
    void t() {
      typedef struct _Chunk { unsigned short idnum; unsigned int len; } Chunk, *PChunk;
      Chunk c; c.len = 7; PChunk p = &c;
      printf("%u\\n", p->len);
    }
    """
    text = translate(source, "t.C")
    assert "class _Chunk:" in text and "c = _Chunk()" in text
    run_source(source, "t.C", root=fake())
    assert capsys.readouterr().out == "7\n"
