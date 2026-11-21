"""``char[N][M]``: one buffer of the characters, its rows the C strings C makes them.

A ``char fChar[2][3]`` is not two C strings kept apart: it is six bytes, and
``fChar[i]`` is the row view of three of them. ``strcpy(fChar[i], ...)``,
``&fChar[0][0]`` and ``fChar[i][j] = c`` all reach that one memory, which is
what a class's constructor does to its member and what ROOT then writes out.
"""

from __future__ import annotations

import pytest

from cintfake import fake
from xrdroot.pycint import translate
from xrdroot.pycint.execute import run_source


def output(capsys: pytest.CaptureFixture[str], source: str) -> str:
    run_source(source, "t.C", root=fake())
    return capsys.readouterr().out


def test_a_declared_char_grid_is_a_buffer_of_the_shape_it_declares() -> None:
    text = translate("void t() {\nchar more[3][4]; char named[2][8] = {\"a\", \"b\"};\n}\n", "t.C")
    assert "more = char_array((3, 4))" in text
    assert "named = char_array((2, 8), ['a', 'b'])" in text


def test_a_constructor_writes_element_by_element_of_a_char_member(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = """
    class Ex {
    public:
       char fChar[2][3];
       Ex() { fChar[0][0] = 'a'; fChar[1][2] = 'z'; }
       void Dump() {
          for (int i = 0; i < 2; i++)
             for (int j = 0; j < 3; j++) printf("%d ", (int)fChar[i][j]);
          printf("\\n");
       }
    };
    void t() { Ex e; e.Dump(); }
    """
    assert output(capsys, source) == "97 0 0 0 0 122 \n"


def test_the_innermost_length_is_the_last_index_of_a_char_grid(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = """
    class Grid {
    public:
       char fGrid[2][3][4];
       Grid() {
          for (int i = 0; i < 2; i++)
             for (int j = 0; j < 3; j++)
                for (int k = 0; k < 4; k++) fGrid[i][j][k] = 48 + i + j + k;
       }
       void Dump() {
          for (int i = 0; i < 2; i++)
             for (int j = 0; j < 3; j++) printf("[%s]", fGrid[i][j]);
          printf("\\n");
       }
    };
    void t() { Grid g; g.Dump(); }
    """
    assert output(capsys, source) == "[0123][1234][2345][1234][2345][3456]\n"


def test_a_row_of_a_char_grid_is_written_into_in_place(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = """
    class Names {
    public:
       char fNames[2][8];
       Names() { strcpy(fNames[0], "ab"); strcpy(fNames[1], "cde"); }
       void Dump() {
          printf("[%s][%s] %d\\n", fNames[0], fNames[1], (int)sizeof(fNames));
       }
    };
    void t() { Names n; n.Dump(); }
    """
    assert output(capsys, source) == "[ab][cde] 16\n"


def test_a_row_of_a_char_grid_streams_as_the_string_it_holds(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = """
    class Grid {
    public:
       char fGrid[2][1][4];
       Grid() {
          for (int k = 0; k < 4; k++) fGrid[1][0][k] = 65 + k;
       }
       void Dump() { std::cout << fGrid[1][0] << std::endl; }
    };
    void t() { Grid g; g.Dump(); }
    """
    assert output(capsys, source) == "ABCD\n"


def test_each_string_of_an_initialiser_is_copied_into_a_row_of_its_own(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = """
    class Names {
    public:
       char fNames[2][4] = {"ab", "cde"};
       void Dump() { printf("[%s][%s]\\n", fNames[0], fNames[1]); }
    };
    void t() { Names n; n.Dump(); }
    """
    assert output(capsys, source) == "[ab][cde]\n"


def test_characters_given_a_grid_fill_it_in_the_order_c_lays_them_down(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = """
    void t() {
       char grid[2][3] = {'a', 'b', 'c', 'd'};
       printf("%s|%s|%d\\n", grid[0], grid[1], (int)grid[1][0]);
    }
    """
    # The braced list is a run of bytes, so ``grid[1][0]`` is the fourth of them.
    assert output(capsys, source) == "abc|d|100\n"
