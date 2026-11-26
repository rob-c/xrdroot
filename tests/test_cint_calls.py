"""Calls into the standard library, ROOT and the macro itself, as the Python they become."""

from __future__ import annotations

import pytest

from cintfake import fake
from xrdroot.pycint import Refusal, translate
from xrdroot.pycint.execute import run_source


def printed(capsys: pytest.CaptureFixture[str], source: str) -> str:
    run_source(source, "t.C", root=fake())
    return capsys.readouterr().out


def body(source: str) -> str:
    text = translate("void t() {\n" + source + "\n}\n", "t.C")
    compile(text, "t.C", "exec")
    return text


def test_std_size_and_equal_are_written_as_the_helpers_that_do_them() -> None:
    """``std::size`` and ``std::equal`` have no Python of their own, so they are rewritten."""
    assert "size_of(a)" in body("double a[3] = {1, 2, 3}; int n = (int)std::size(a);")
    written = body(
        "double a[3] = {1, 2, 3}; double b[3] = {1, 2, 4};"
        " bool e = std::equal(a, a + 3, b);"
    )
    assert "equal_ranges(a, 0, 3, b, 0)" in written


def test_the_standard_headers_limits_are_their_numbers(capsys: pytest.CaptureFixture[str]) -> None:
    """``<climits>`` and ``<cstdint>``'s macros, as LP64 has them."""
    written = (
        'printf("%lu %ld %d %d %d %d\\n", UINT64_MAX, LONG_MAX, INT32_MIN, INT_MAX, CHAR_BIT,'
        " UCHAR_MAX);"
    )
    assert printed(capsys, "void t() {\n" + written + "\n}\n") == (
        "18446744073709551615 9223372036854775807 -2147483648 2147483647 8 255\n"
    )


@pytest.mark.parametrize(
    ("source", "text"),
    [
        (
            'auto h = std::make_unique<TH1F>("h", "h", 2, 0, 1); printf("%s\\n", h->GetName());',
            "h\n",
        ),
        (
            'std::shared_ptr<TH1F> s = std::make_shared<TH1F>("s", "s", 1, 0, 1);'
            'TH1F *raw = s.get(); printf("%s\\n", raw->GetName());',
            "s\n",
        ),
        (
            'std::unique_ptr<TH1F> e = std::unique_ptr<TH1F>(); printf("%d\\n", e == nullptr);',
            "1\n",
        ),
        (
            'std::unique_ptr<TH1F> r(new TH1F("r", "r", 1, 0, 1)); TH1F *x = r.release();'
            'printf("%s\\n", x->GetName());',
            "r\n",
        ),
        (
            'printf("%g %d %g\\n", std::numeric_limits<double>::max(),'
            " std::numeric_limits<int>::min(), std::numeric_limits<float>::epsilon());",
            "1.79769e+308 -2147483648 1.19209e-07\n",
        ),
        (
            'double a[4] = {3, 1, 2, 0}; std::sort(a, a + 3); printf("%g %g\\n", a[0], a[3]);',
            "1 0\n",
        ),
        (
            "std::vector<int> v; v.push_back(3); v.push_back(1); v.push_back(2);"
            'std::sort(v.begin() + 1, v.end()); printf("%d %d %d\\n", v[0], v[1], v[2]);',
            "3 1 2\n",
        ),
        (
            "std::vector<int> v; v.push_back(1); v.push_back(2); std::reverse(v.begin(), v.end());"
            'printf("%d\\n", v[0]);',
            "2\n",
        ),
        (
            "std::vector<double> v; v.push_back(1.5); v.push_back(2);"
            'printf("%g\\n", std::accumulate(v.begin(), v.end(), 0.0));',
            "3.5\n",
        ),
        (
            "std::vector<int> v{1, 2, 3};"
            'printf("%d\\n", std::accumulate(v.begin(), v.end(), 0,'
            " [](int a, int b) { return a + 2 * b; }));",
            "12\n",
        ),
        (
            'std::string s = "a b"; std::replace(s.begin(), s.end(), \' \', \'_\');'
            'printf("%s\\n", s.c_str());',
            "a_b\n",
        ),
        (
            "std::vector<int> v{1, 2, 1}; std::replace(v.begin(), v.end(), 1, 9);"
            'printf("%d %d %d\\n", v[0], v[1], v[2]);',
            "9 2 9\n",
        ),
        (
            "double a[3] = {1, 2, 1}; std::replace(a, a + 3, 1.0, 2.0);"
            'printf("%g %g %g\\n", a[0], a[1], a[2]);',
            "2 2 2\n",
        ),
        (
            "std::unique_ptr<char[]> buf(new char[4]); buf[2] = 7;"
            'char *p = buf.get() + 2; printf("%d\\n", p[0]);',
            "7\n",
        ),
        (
            "std::vector<int> v{0, 0, 0}; std::iota(v.begin(), v.end(), 5);"
            'printf("%d %d %d\\n", v[0], v[1], v[2]);',
            "5 6 7\n",
        ),
        (
            "std::vector<int> v{0, 0, 0};"
            "std::generate(v.begin(), v.end(), [n = 0]() mutable { n++; return n * 10; });"
            'printf("%d %d %d\\n", v[0], v[1], v[2]);',
            "10 20 30\n",
        ),
        (
            'std::string s = "abc:"; s.pop_back(); printf("%s\\n", s.c_str());',
            "abc\n",
        ),
        (
            'std::string s = "abcdef"; s.erase(1, 2); s.erase(2);'
            'printf("%s\\n", s.c_str());',
            "ad\n",
        ),
        (
            'std::string s = "ac"; s.insert(1, "b"); s.replace(1, 1, "xy");'
            'printf("%s\\n", s.c_str());',
            "axyc\n",
        ),
        (
            'std::string s = "abc"; s.erase(); printf("[%s]\\n", s.c_str());',
            "[]\n",
        ),
        (
            'std::string s = "abcdef"; int i = 2; s.erase(i + 1, 2);'
            'printf("%s\\n", s.c_str());',
            "abcf\n",
        ),
        (
            'std::string s; s.assign(3, \'x\'); s.assign("yz");'
            'printf("%s\\n", s.c_str());',
            "yz\n",
        ),
        ('double w[3] = {1, 2, 3}; printf("%g\\n", std::accumulate(w, w + 2, 10.0));', "13\n"),
        (
            'double a[4] = {3, 1, 2, 0}; std::sort(&a[0], &a[0] + 3);'
            'printf("%g %g\\n", a[0], a[3]);',
            "1 0\n",
        ),
        (
            "std::vector<int> v; v.push_back(3); v.push_back(1); v.push_back(2);"
            'std::sort(&v[0], &v[0] + 2); printf("%d %d %d\\n", v[0], v[1], v[2]);',
            "1 3 2\n",
        ),
        (
            "std::vector<int> v; v.push_back(3); v.push_back(1); v.push_back(2);"
            'std::sort(&v.front(), &v.front() + 2); printf("%d %d %d\\n", v[0], v[1], v[2]);',
            "1 3 2\n",
        ),
        (
            "std::vector<int> v; v.push_back(3); v.push_back(1); v.push_back(2);"
            'int *p = &v[0]; std::reverse(p, p + 3); printf("%d %d %d\\n", v[0], v[1], v[2]);',
            "2 1 3\n",
        ),
        (
            'double a[3] = {1, 2, 3}; double b[3] = {0, 0, 0};'
            'std::copy(&a[0], &a[3], &b[0]); printf("%g %g %g\\n", b[0], b[1], b[2]);',
            "1 2 3\n",
        ),
        ('printf("%d %d %d\\n", abs(-3), std::max(2, 5), std::min(1.5, 0.5) < 1);', "3 5 1\n"),
        ('long big = 5; unsigned char c = 260; printf("%ld %d\\n", big, c);', "5 4\n"),
        ('int x; x = {3}; printf("%d\\n", x);', "3\n"),
        ('std::string s = "a,b"; printf("%d\\n", s.find(",") != std::string::npos);', "1\n"),
        ('printf("%d %d\\n", kTRUE, kFALSE);', "1 0\n"),
        ('assert(1 + 1 == 2); printf("ok\\n");', "ok\n"),
    ],
)
def test_library_calls_do_what_the_library_does(
    capsys: pytest.CaptureFixture[str], source: str, text: str
) -> None:
    assert printed(capsys, "void t() {\n" + source + "\n}\n") == text


@pytest.mark.parametrize(
    ("source", "fragment"),
    [
        ('auto f = Format("x");', "f = ROOT.Format('x')"),
        ("auto g = std::make_unique(3);", "g = ROOT.std.make_unique(3)"),
        ("auto a = (*fp)(2); auto b = f()();", "b = ROOT.f()()"),
        ("std::vector<std::array<int, 3>> v;", "ROOT.std.vector['std::array<int,3>']()"),
        ('fprintf(stderr, "x");', "fprintf('stderr', 'x')"),
    ],
)
def test_calls_are_written_as_their_python(source: str, fragment: str) -> None:
    assert fragment in body(source)


@pytest.mark.parametrize(
    ("source", "why"),
    [
        ("auto x = std::numeric_limits<std::string>::max();", r"numeric_limits<std::string>::max"),
        ("std::sort(v.begin());", "std::sort without a range"),
        ("auto s = std::accumulate(v.begin());", "std::accumulate with 1 arguments"),
        ("std::sort(v.begin(), w.end());", "a range whose two ends are in different containers"),
        ("std::sort(it, other);", "an iterator this translator cannot follow back"),
        ("std::unique_ptr<TH1F> p; long n = p.use_count();", r"the smart pointer's use_count\(\)"),
    ],
)
def test_calls_with_no_python_that_does_the_same_are_refused(source: str, why: str) -> None:
    with pytest.raises(Refusal, match=why):
        body(source)


def test_a_method_calls_its_classs_other_members_through_self(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = """
    class Pair { public: static int count; static int Count() { return count; }
      int Twice() { return 2 * One(); } int One() { return 1; } };
    int Pair::count = 7;
    void t() { Pair p; printf("%d %d %d\\n", p.Twice(), Pair::Count(), Pair::count); }
    """
    assert printed(capsys, source) == "2 7 7\n"


def test_a_name_written_from_the_global_scope_is_the_files_own(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``::Report(3)`` in a class's method is the file's free function, never the member of that
    name: a leading ``::`` is C++'s global namespace. Without that, a method calling the free
    function of its own name calls itself - roottest's ``Holder::Print`` did."""
    source = """
    void Report(int n) { printf("free %d\\n", n); }
    class Holder { public: void Report() { printf("member\\n"); }
      void Go() { ::Report(3); Report(); } };
    void t() { Holder h; h.Go(); }
    """
    assert printed(capsys, source) == "free 3\nmember\n"


def test_a_base_class_offset_is_bytes_a_char_pointer_moves_by() -> None:
    """A dialect's table entry is a declaration, typedefs and all: ``GetBaseClassOffset`` is an
    ``Int_t``, which is the ``int`` pointer arithmetic is written against - read as a bare name it
    is no arithmetic type, and ``((char *)obj) + off`` became a string concatenation."""
    text = translate(
        "void t() { TClass *cl = 0; TBaseClass *b = 0; void *obj = 0;"
        " void *p = ((char *)obj) + cl->GetBaseClassOffset(b); }",
        "t.C",
    )
    assert "byte_offset(obj, (cl.GetBaseClassOffset(b)))" in text


def test_a_class_deriving_from_roots_reaches_roots_members_through_self() -> None:
    source = """
    class MySelector : public TSelector { public:
      Bool_t Process(Long64_t entry) { fChain->GetEntry(entry); return kTRUE; } };
    class Deeper : public MySelector { public: void Go() { fInput->Print(); } };
    class Plain { public: int fN; void Go() { fN = 1; fOther = 2; } };
    """
    text = translate(source, "t.C")
    assert "self.fChain.GetEntry(entry)" in text
    assert "self.fInput.Print()" in text
    assert "ROOT.fOther = 2" in text


def test_an_element_taken_by_at_is_a_pointer_into_the_container(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``&v.at(i)`` handed to something other than a range - an initialiser here - is a pointer
    into the vector's own buffer: what it points at is that element."""
    source = (
        "std::vector<double> v; v.push_back(1.5); v.push_back(2.5);"
        "double *q = &v.at(1);"
        'printf("%g\\n", *q);'
    )
    assert printed(capsys, "void t() {" + source + "}") == "2.5\n"


def test_an_element_taken_by_front_is_a_pointer_into_the_container(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``&v.front()`` likewise: element zero of the vector, reached through a pointer."""
    source = (
        "std::vector<double> v; v.push_back(2.5); v.push_back(3.5);"
        "double *r = &v.front();"
        'printf("%g\\n", *r);'
    )
    assert printed(capsys, "void t() {" + source + "}") == "2.5\n"


def test_the_address_of_a_call_that_names_no_element_is_not_a_pointer_into_it() -> None:
    """``&v.size()`` is a call whose name is neither ``front`` nor ``at``, so it names no
    element: the address taken is the member's own, not a pointer into the vector."""
    assert "ItemRef" not in body("std::vector<int> v; auto *p = &v.size(); (void)p;")


def test_a_range_whose_ends_are_elements_taken_by_at_sorts_those_elements(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``std::sort(&v.at(0), &v.at(2))``: a range end taken by ``at`` is the same vector and
    index a range end taken by ``[i]`` is, so the algorithm sorts inside the vector."""
    source = (
        "std::vector<int> v; v.push_back(3); v.push_back(1); v.push_back(2);"
        "std::sort(&v.at(0), &v.at(2));"
        'printf("%d %d %d\\n", v[0], v[1], v[2]);'
    )
    assert printed(capsys, "void t() {" + source + "}") == "1 3 2\n"


def test_the_address_of_a_member_call_or_a_plain_name_is_not_an_element_of_a_vector() -> None:
    """``&v.size()`` names no element and ``&d`` of a plain ``double`` is no vector at all:
    neither becomes the vector-and-offset a range end is."""
    source = (
        "std::vector<int> v; int *p = &v.size(); (void)p;"
        "double d = 1.; double *q = &d; (void)q;"
    )
    text = body(source)
    assert "ItemRef(v" not in text


def test_a_range_end_that_names_no_element_is_refused_by_name() -> None:
    """``std::sort(&v.size(), …)``: a call whose name is neither ``front`` nor ``at``, and a
    plain name's address, are no vector element, so the iterator form is refused - not guessed."""
    with pytest.raises(Refusal, match="an iterator this translator cannot follow back"):
        body("std::vector<int> v; std::sort(&v.size(), &v.size() + 1);")
    # A plain name's address is no vector element either, but it *is* a pointer this
    # translator can follow: the array form takes it, which is what the check is for.
    assert "sort_range(d, 0, 1)" in body("double d = 1.; std::sort(&d, &d + 1);")


def test_a_qualified_static_call_fills_the_local_handed_to_it_by_reference() -> None:
    """``TBufferJSON::FromJSON(obj, json)`` takes ``T*&``: the local is a cell the callee writes
    through, exactly as ``obj.FromJSON(...)`` does - the table is keyed by the name either way."""
    text = body("int x; TBufferJSON::FromJSON(x, s);")
    assert "x = Cell(" in text
    assert "ROOT.TBufferJSON.FromJSON(x," in text


def test_a_cell_holding_an_object_pointer_carries_the_class_it_was_declared_with() -> None:
    """``TArrayI *a2 = nullptr; FromJSON(a2, json)`` makes an object of the class the
    *declaration* names - ROOT's template argument - and a cell holding ``None`` has no
    value that could say which class that is, so the cell carries the type's name."""
    text = body("TArrayI *a2 = nullptr; TBufferJSON::FromJSON(a2, s);")
    assert "a2 = Cell(None, 'TArrayI')" in text
    assert "ROOT.TBufferJSON.FromJSON(a2," in text


def test_strftime_gives_its_character_array_the_time_it_formatted(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``strftime(buf, size, fmt, tt)``: a ``char[]`` is text here, which no write reaches in
    place, so the call returns what it formatted and the variable keeps it - as ``sprintf``'s.
    A ``struct tm``'s fields are C's: the year counts from 1900."""
    text = body('time_t t = 0; struct tm* tt = gmtime(&t); char buf[64];'
                "strftime(buf, sizeof(buf), \"%Y-%m-%d\", tt);")
    assert "buf = text_into(buf, formatted_time(" in text
    source = r"""time_t t = 0; struct tm* tt = gmtime(&t); char buf[64];
    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", tt);
    printf("%s %d\n", buf, tt->tm_year + 1900);"""
    assert printed(capsys, "void t() {" + source + "}") == "1970-01-01 00:00:00 1970\n"
