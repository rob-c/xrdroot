"""Memory from ``new char[n]``: one buffer every pointer into it shares, written in place.

Each expected output is what the same program prints compiled by ROOT 6.40.04's cling.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from xrdroot.cppstd.buffers import CharBuffer, put_char, text_into
from xrdroot.cppstd.objects import array
from xrdroot.cppstd.strings import char_at, cstr, with_char
from xrdroot.pycxx import run_program
from xrdroot.pycxx.errors import MacroError, Refusal
from xrdroot.pycxx.translation import translation


def run(tmp_path: Path, capsys: pytest.CaptureFixture[str], body: str) -> str:
    source = "#include <cstdio>\n#include <cstring>\n" + body
    path = tmp_path / "buffers.cxx"
    path.write_text(source)
    assert run_program([path], use_cache=False) == 0
    return capsys.readouterr().out


def test_a_buffer_reads_as_the_c_string_it_holds() -> None:
    buffer = array("char", 6)
    assert isinstance(buffer, CharBuffer) and str(buffer) == "" and len(buffer) == 6
    assert text_into(buffer, "abc") is buffer
    assert str(buffer) == "abc" and cstr(buffer) == "abc" and f"{buffer:>4}" == " abc"
    assert repr(buffer[1:]) == "CharBuffer('bc')"
    assert char_at(buffer, 3) == 0 and char_at(buffer, 0) == ord("a")
    assert put_char(buffer, 1, 300) == 300 and char_at(buffer, 1) == 44
    assert put_char(buffer, 1, 200) == 200 and char_at(buffer, 1) == -56
    unsigned = array("unsigned char", 2)
    put_char(unsigned, 0, -56)
    assert char_at(unsigned, 0) == 200


def test_a_write_a_buffer_cannot_take_is_refused_as_it_runs() -> None:
    with pytest.raises(TypeError, match="only memory from new char"):
        put_char("text", 0, 65)
    with pytest.raises(TypeError, match="buffer the caller passed"):
        text_into("text", "more", True)
    assert text_into("old", "new") == "new"
    with pytest.raises(IndexError, match="past its end"):
        text_into(CharBuffer(3), "abc")
    assert np.asarray(CharBuffer(2)).dtype == np.int8


def test_pointers_into_one_buffer_see_each_others_writes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    body = """
struct B {
   char *fName;
   B(const char *name) { fName = (char *)new char[strlen(name) + 1]; strcpy(fName, name); }
   void Up() { fName[0] = 'A'; }
};
void fill(char *out) { strcpy(out, "filled"); out[6] = '!'; out[7] = 0; }
int main() {
   B b("hello");
   char *alias = b.fName;
   b.Up();
   printf("%s %s\\n", b.fName, alias);
   char *c = new char[4];
   c[0] = 'i'; c[1] = '0'; c[2] = '0'; c[3] = 0;
   char *q = c + 1;
   q[0] = '1';
   (c[2])++;
   c[2] += 1;
   printf("%s %c %d\\n", c, *q, (int)strlen(c));
   char *big = new char[20];
   fill(big);
   strcat(big, "?");
   printf("%s\\n", big);
   sprintf(big + 2, "%d", 42);
   snprintf(big, 4, "%s", "zzzzz");
   printf("%s\\n", big);
   char buf[20];
   sprintf(buf, "%d", 3);
   strcat(buf, "x");
   printf("%s\\n", buf);
   delete [] c;
   return 0;
}
"""
    assert run(tmp_path, capsys, body) == "Aello Aello\ni12 1 3\nfilled!?\nzzz\n3x\n"


def test_a_pointer_steps_through_its_buffer_without_touching_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    body = """
int main() {
   char *a = new char[4];
   a[0] = 'A'; a[1] = 'B'; a[2] = 'C'; a[3] = 0;
   char *p = a;
   char c = *p++;
   char d = *p;
   char e = *++p;
   int n = 0;
   for (char *r = a; *r; ++r) n++;
   char *w = a;
   while (*w) { w++; }
   double *x = new double[4];
   x[1] = 5;
   double *y = x;
   y++;
   y += 1;
   *y = 2;
   printf("%c %c %c %d %s %g %g\\n", c, d, e, n, a, x[1], x[2]);
   return 0;
}
"""
    assert run(tmp_path, capsys, body) == "A B C 3 ABC 5 2\n"


def test_a_char_written_through_a_pointer_to_text_fails_as_it_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    body = 'void poke(char *p) { p[0] = 1; }\nint main() { char s[] = "abc"; poke(s); }\n'
    with pytest.raises(MacroError, match="only memory from new char"):
        run(tmp_path, capsys, body)


def test_a_pointer_step_that_is_not_a_variables_is_refused() -> None:
    with pytest.raises(Refusal, match=r"char\*\+\+ of a pointer that is not a variable"):
        translation("struct S { char *p; };\nvoid f(S s) { char c = *s.p++; }", "t.cxx")


def test_a_byte_array_and_a_new_array_of_rows_are_numbers_in_place(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    body = """
#include <cstdint>
int main() {
   uint8_t buf[4];
   for (int n = 0; n < 4; n++) buf[n] = n * 100;
   unsigned char c[3] = {1, 2, 3};
   c[1] += 250;
   int s = 0;
   for (int n = 0; n < 4; n++) s += buf[n];
   int n = 3;
   auto a2 = new float[n][2];
   a2[1][1] = 5;
   float (*b)[2] = new float[n][2];
   b[2][0] = 1;
   printf("%d %d %d %g %g\\n", s, c[1], (int)sizeof(buf), a2[1][1], b[2][0]);
   return 0;
}
"""
    assert run(tmp_path, capsys, body) == "344 252 4 5 1\n"
    with pytest.raises(Refusal, match="new of an array of arrays of S"):
        translation("struct S {};\nvoid f(int n) { auto p = new S[n][2]; }", "t.cxx")


def test_a_char_array_a_library_writes_into_is_a_buffer() -> None:
    from dataclasses import replace

    from xrdroot.pycxx.dialect import CXX, using

    lab = replace(CXX, name="lab-buffers", buffer_methods=frozenset({"Branch"}))
    source = 'void f(T &t) { char s[10] = "onetwo"; char u[4]; t.Branch(s); s[3] = 0; }'
    with using(lab):
        python = translation(source, "t.cxx").python
    assert "s = text_into(array('char', 10), 'onetwo')" in python
    assert "u = ''" in python
    buffer = text_into(array("char", 10), "onetwo")
    assert with_char(buffer, 3, 0, True) is buffer and str(buffer) == "one"


def test_a_line_read_and_scanned_gives_its_text_to_the_variables_they_stand_for(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``while (fgets(line, 160, fp)) sscanf(line, "%s %d %d", name, &a, &b)`` - ROOT's
    ``test/stressGraphics.cxx`` and ``tree500_cernbuild.C``. Neither ``line`` nor ``name``
    is memory here, so each callee is handed a cell in the variable's place and the variable
    takes the cell's text once the call has run. ROOT 6.40.04 prints the same three lines.
    """
    rows = tmp_path / "rows.txt"
    rows.write_text("staff 12 -3\nother 7 8\nnot-enough\n")
    body = f'''
int main() {{
   FILE *fp = fopen("{rows}", "r");
   char line[160];
   char name[100];
   int a = 0, b = 0;
   while (fgets(line, 160, fp)) {{
      if (sscanf(line, "%s %d %d", name, &a, &b) != 3) continue;
      printf("[%s] %d %d\\n", name, a, b);
   }}
   fclose(fp);
   return 0;
}}
'''
    assert run(tmp_path, capsys, body) == "[staff] 12 -3\n[other] 7 8\n"


def test_a_scan_into_an_array_of_numbers_writes_its_first_element(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Only a character array is text to give back to a variable: an array of numbers a
    pointer names is written where it already lies."""
    body = """
int main() {
   int numbers[2] = {0, 0};
   sscanf("5", "%d", numbers);
   printf("%d %d\\n", numbers[0], numbers[1]);
   return 0;
}
"""
    assert run(tmp_path, capsys, body) == "5 0\n"


def test_a_scan_into_a_parameter_s_array_is_the_caller_s_to_write(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``char buf[10]`` a function was handed belongs to its caller: the callee cannot give
    it the text, so the scan refuses rather than read into a ``str`` the caller keeps."""
    body = ('void fill(char buf[10]) { sscanf("text", "%s", buf); }\n'
            'int main() { char s[10]; fill(s); }\n')
    with pytest.raises(MacroError, match=r"new char\[n\]"):
        run(tmp_path, capsys, body)


def test_a_scan_writes_only_into_the_text_a_variable_stands_for() -> None:
    """A target that is not the macro's own variable - ``p + 1`` into memory a pointer
    already names - has no variable to give the text back to, and is written in place.
    """
    python = translation('void f(char *p) { sscanf("x", "%s", p + 1); }', "t.cxx").python
    assert "written_after" not in python and "sscanf('x', '%s', p[1:])" in python


def test_a_scan_into_a_char_array_a_library_writes_into_writes_it_in_place() -> None:
    from dataclasses import replace

    from xrdroot.pycxx.dialect import CXX, using

    lab = replace(CXX, name="lab-buffers", buffer_methods=frozenset({"Branch"}))
    source = 'void f(T &t) { char s[10]; t.Branch(s); sscanf("x", "%s", s); }'
    with using(lab):
        python = translation(source, "t.cxx").python
    assert "written_after" not in python
    assert "sscanf('x', '%s', s)" in python


def test_an_object_moved_by_its_base_offset_is_itself_at_offset_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    body = """
struct B { int v = 7; };
struct D : B { int w = 1; };
template <typename T> int peek(T *obj, unsigned off) {
   obj = (T *)(((char *)obj) - off);
   return obj->v;
}
int main() { D d; printf("%d\\n", peek<D>(&d, 0)); return peek<D>(&d, 8); }
"""
    with pytest.raises(MacroError, match="moved by -8 bytes, into the middle of it"):
        run(tmp_path, capsys, body)
    assert capsys.readouterr().out == "7\n"


def test_a_buffer_written_into_an_element_is_worked_out_once_and_is_never_null(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    body = """
int main() {
   char *arr[4] = {0};
   int num = 0;
   const char *words[] = {"one", "two"};
   for (int w = 0; w < 2; w++) {
      arr[num] = new char[8];
      strcpy(arr[num++], words[w]);
   }
   strcat(arr[--num], "!");
   int n = 0;
   for (int i = 0; arr[i]; ++i) n++;
   printf("%d %d %s %s\\n", num, n, arr[0], arr[1]);
   return 0;
}
"""
    assert run(tmp_path, capsys, body) == "1 2 one two!\n"
