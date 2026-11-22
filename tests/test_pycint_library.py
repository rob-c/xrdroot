"""ROOT's build model, translated: dictionaries, libraries, ACLiC, loading and autoloading."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from pyrootsupport import fresh
from xrdroot import pyroot as ROOT
from xrdroot.pycint.execute import load as dot_l
from xrdroot.pycint.execute import preload, process_line, run, run_program
from xrdroot.pycint.library import (
    BuildError,
    aclic,
    build_aclic,
    build_dictionary,
    build_library,
    compile_macro,
    compiled,
    find,
    load,
    load_library,
    loaded_names,
    loading,
    manifest,
    read,
    selection,
    stem_of,
    translate,
)
from xrdroot.pycint.library.build import translator_version
from xrdroot.pycxx import cache
from xrdroot.pycxx.errors import CxxError, Refusal
from xrdroot.pycxx.runtime.namespace import DECLARED
from xrdroot.pyroot.core import troot

#: A header with a class in a namespace, a static member and a function only declared.
SHAPES_H = """\
#ifndef SHAPES_H
#define SHAPES_H
#include "TObject.h"
namespace geo {
class Shape : public TObject {
public:
   static int count;
   double fSize;
   Shape(double s = 1) : fSize(s) { ++count; }
   double Area() const;
   ClassDef(Shape, 1)
};
namespace inner { int Depth(); }
int Undefined();
}
int CountShapes();
#endif
"""

#: Its implementation, which the header only declares.
SHAPES_CXX = """\
#include "Shapes.h"
int geo::Shape::count = 0;
double geo::Shape::Area() const { return fSize * fSize; }
int geo::inner::Depth() { return 2; }
int CountShapes() { return geo::Shape::count; }
"""

#: What rootcling is asked to select: one class there, one not.
LINKDEF = """\
#ifdef __CLING__
#pragma link off all classes;
#pragma link C++ class geo::Shape+;
#pragma link C++ class Missing+;
#pragma link C++ class std::vector<geo::Shape>+;
#pragma link C++ class TH1F;
#pragma link C++ class geo::*;
#pragma link C++ function CountShapes;
#endif
"""


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Each test in its own directory, with no library loaded and nothing declared."""
    monkeypatch.setenv(cache.ENVIRONMENT, str(tmp_path / "cache"))
    for variable in loading.ENVIRONMENT:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(ROOT.gSystem, "_dynamic_path", "")
    monkeypatch.setattr(ROOT.gSystem, "_build_dir", "")
    saved, names = dict(DECLARED), set(vars(ROOT))
    loading.forget_all()
    try:
        yield from fresh(tmp_path)
    finally:
        loading.forget_all()
        DECLARED.clear()
        DECLARED.update(saved)
        for name in set(vars(ROOT)) - names:  # what gInterpreter.Declare made ROOT's
            if getattr(getattr(ROOT, name), "__module__", None) == "__cint__":
                delattr(ROOT, name)


def ours() -> list[str]:
    """The libraries loaded past those ROOT starts with, by name."""
    return [stem_of(name) for name in loaded_names()[len(loading.STARTED) :]]


def write(name: str, text: str) -> Path:
    path = Path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def shapes(out: Path | None = None) -> Path:
    """``libShapes.xrdlib``, rootcling's dictionary of the shapes, with their sources."""
    write("Shapes.h", SHAPES_H)
    write("Shapes.cxx", SHAPES_CXX)
    write("LinkDef.h", LINKDEF)
    return build_dictionary("Shapes", [Path("Shapes.h"), Path("Shapes.cxx")],
                            linkdef=Path("LinkDef.h"), out=out)  # fmt: skip


# -- building -------------------------------------------------------------------------------


def test_a_dictionary_is_the_translation_of_its_files_and_what_they_declare(
    capsys: pytest.CaptureFixture[str],
) -> None:
    where = shapes()
    assert where == Path.cwd() / "libShapes.xrdlib"
    assert capsys.readouterr().err == "Warning: Unused class rule: Missing\n"
    library = read(where)
    assert library is not None
    assert (library.kind, library.rootmap, library.name) == ("dictionary", True, "libShapes")
    assert (library.names, library.classes, library.namespaces) == (
        ["geo", "CountShapes"], ["geo::Shape"],
        {"geo": {"Shape": None, "inner": {"Depth": None}, "Undefined": None}},
    )  # fmt: skip
    sources = [str(Path(name).resolve()) for name in ("Shapes.h", "Shapes.cxx")]
    assert (library.resolved(library.sources), library.translator) == (
        sources, translator_version())  # fmt: skip
    stored = json.loads(where.read_text())
    assert stored["sources"] == ["Shapes.h", "Shapes.cxx"]  # relative: movable as a whole


def test_a_selection_file_is_genreflex_and_a_library_has_no_rootmap(
    capsys: pytest.CaptureFixture[str],
) -> None:
    write("A.h", "class A {}; namespace ns { struct B {}; }")
    write("sel.xml", '<lcgdict><class name="A"/><class name="Gone"/></lcgdict>')
    build_dictionary("A", [Path("A.h")], selection=Path("sel.xml"), rootmap=False,
                     libraries=["libOther"])  # fmt: skip
    assert capsys.readouterr().err == "Warning: Unused class rule: Gone\n"
    library = read(Path("libA.xrdlib"))
    assert library is not None and not library.rootmap and library.libraries == ["libOther"]
    build_library("libB", [Path("A.h")], out=Path("out"))
    made = read(Path("out/libB.xrdlib"))
    assert made is not None and (made.kind, made.rootmap) == ("library", False)


def test_a_build_refuses_missing_files_and_cpp_it_cannot_translate_as_rootcling_does(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(BuildError, match=r"rootcling: compilation failure \(gone.h\)"):
        build_dictionary("X", [Path("gone.h")])
    assert capsys.readouterr().err == "fatal error: 'gone.h' file not found\n"
    with pytest.raises(BuildError, match=r"genreflex: compilation failure \(gone.xml\)"):
        build_dictionary("X", [write("ok.h", "")], selection=Path("gone.xml"))
    capsys.readouterr()
    write("bad.h", "class C { int x = ; };")
    with pytest.raises(Refusal) as refused:  # a build that cannot be done, as refused
        build_dictionary("X", [Path("bad.h")], selection=write("s.xml", "<lcgdict/>"))
    assert refused.value.where is not None and refused.value.where.line == 1
    assert capsys.readouterr().err == ""


def test_aclic_s_library_is_the_macro_s_translation_with_its_entry(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    write("twice.C", "int twice(int n = 2) { return 2 * n; }")
    library = build_aclic(Path("twice.C"), "twice_C", Path("."))
    assert (library.kind, library.entry, library.returns) == ("aclic", "twice", "int")
    write("broken.C", "int broken() { return ; ; }}")
    with pytest.raises(Refusal):
        build_aclic(Path("broken.C"), "broken_C", Path("."))
    with pytest.raises(Refusal):
        compiled(Path("broken.C"), explain=False)
    with pytest.raises(BuildError, match="ACLiC: compilation failure"):
        compiled(Path("gone.C"))
    capsys.readouterr()
    with pytest.raises(BuildError, match="Compilation failed!"):
        compiled(Path("broken.C"))
    assert "broken.C:1:1: error: " in capsys.readouterr().err

    def unplaced(*args: object) -> None:
        raise CxxError("no place")

    monkeypatch.setattr("xrdroot.pycint.library.build.translated_unit", unplaced)
    with pytest.raises(BuildError):
        compiled(Path("twice.C"), force=True)
    assert capsys.readouterr().err.endswith("error: no place\n")


# -- what a library file is ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("given", "stem"),
    [("libFoo.so", "libFoo"), ("./dir/libFoo.dylib", "libFoo"), ("libFoo", "libFoo"),
     ("macro_C.xrdlib", "macro_C"), (".so", ".so")],
)  # fmt: skip
def test_a_library_is_named_without_its_extension(given: str, stem: str) -> None:
    assert stem_of(given) == stem


def test_what_is_not_a_library_file_this_version_wrote_is_not_read() -> None:
    assert read(Path("none.xrdlib")) is None
    assert read(write("a.xrdlib", "not json")) is None
    assert read(write("b.xrdlib", "[1]")) is None
    assert read(write("c.xrdlib", '{"format": "other"}')) is None
    assert read(write("d.xrdlib", f'{{"format": "{manifest.FORMAT}", "version": 0}}')) is None
    assert read(write("e.xrdlib", f'{{"format": "{manifest.FORMAT}", "version": 1}}')) is None
    this = f'{{"format": "{manifest.FORMAT}", "version": {manifest.VERSION}}}'
    assert read(write("f.xrdlib", this)) is None  # this version's, with nothing in it


def test_a_library_s_files_are_kept_relative_to_it_and_found_absolute_when_read() -> None:
    write("sub/h.h", "")
    library = manifest.Library("libL", "library", "", "unit", sources=["sub/h.h", "/abs/x.h"])
    assert library.resolved(["sub/h.h"]) == [str(Path("sub/h.h").resolve())]
    where = manifest.write(library, Path("lib"))
    assert json.loads(where.read_text())["sources"] == [str(Path("sub/h.h").resolve()), "/abs/x.h"]
    found = read(where)
    assert found is not None and found.translation().file == "unit"


def test_what_a_unit_declares_is_read_off_its_namespaces_and_classes() -> None:
    source = """
    namespace a { namespace { int hidden; } class C { class In { class Deep {}; }; };
      enum E { e1 }; enum class S { s1 }; enum { anon }; typedef int T; void f(); }
    void a::f() {}
    int top;
    using namespace std;
    struct D { static int n; };
    int D::n = 1;
    """
    found = translate.translated_unit(source, "u.C", [])[1]
    assert found.top == ["a", "top", "D"]
    assert found.classes == ["a::C", "a::C::In", "a::C::In::Deep", "a::E", "a::S", "a::T", "D"]
    assert found.others == ["a::hidden", "a::e1", "a::anon", "a::f", "top"]
    for name in ("a::C", "a::C<int>", "::a", "a", "top", "a::E"):
        assert found.knows(name), name
    for name in ("b", "a::Z", "a::C::Q"):
        assert not found.knows(name), name


def test_the_digest_of_files_changes_with_their_text_and_their_absence() -> None:
    first = translate.digest([str(write("x.h", "1"))])
    assert translate.digest([str(write("x.h", "2"))]) != first
    assert translate.digest(["gone.h"]) != translate.digest(["gone.h", "gone.h"])


def test_a_selection_is_read_from_a_linkdef_and_a_genreflex_file() -> None:
    assert selection.linkdef_rules(LINKDEF) == [
        "geo::Shape", "Missing", "std::vector<geo::Shape>", "TH1F", "geo::*",
    ]  # fmt: skip
    xml = """<lcgdict><selection><class name="A"/><class pattern="B*"/><enum name="E"/>
             </selection><exclusion><class name="X"/></exclusion><struct name="S"/></lcgdict>"""
    assert selection.xml_rules(xml) == ["A", "E", "S"]
    lenient = """<rootdict> <!-- <class name="Commented"/> -->
       <class name="std::vector<vector<TH1F>>" /> <class name='single' id="7">
       <field name="f" transient="true"/></class> <enum pattern="E*"/>"""  # no closing tag
    assert selection.xml_rules(lenient) == ["std::vector<vector<TH1F>>", "single"]
    assert selection.requested(None, None) == []


# -- loading ---------------------------------------------------------------------------------


def test_a_library_is_loaded_by_name_once_and_what_it_defines_is_found_everywhere(
    capsys: pytest.CaptureFixture[str],
) -> None:
    shapes()
    assert load("libShapes") == 0
    assert load("./libShapes.so") == 1
    assert ours() == ["libShapes"]
    shape = ROOT.geo.Shape(3)
    assert (shape.Area(), ROOT.CountShapes(), ROOT.geo.inner.Depth()) == (9.0, 1, 2)
    assert repr(DECLARED["geo"]) == "<namespace geo>"
    assert ROOT.gSystem.GetLibraries("libShapes") == str(Path("libShapes.xrdlib").resolve())
    assert ROOT.gSystem.GetLibraries().startswith("libRint libCore ")


def test_a_library_is_looked_for_where_root_looks_and_as_lib_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shapes(out=Path("libs"))
    assert find("libShapes") is None
    monkeypatch.setenv("LD_LIBRARY_PATH", "libs")
    assert find("Shapes") == Path("libs/libShapes.xrdlib")
    monkeypatch.delenv("LD_LIBRARY_PATH")
    ROOT.gSystem.AddDynamicPath("libs")
    assert find("libShapes.so") == Path("libs/libShapes.xrdlib")
    assert find("libs/libShapes") == Path("libs/libShapes.xrdlib")
    monkeypatch.delitem(sys.modules, "xrdroot.pyroot.core.system")
    assert loading.search_directories() == [Path(".")]


def test_root_s_own_libraries_load_and_others_are_not_found(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (load("libCore"), load("libTree"), load("libTree.so")) == (1, 0, 1)
    assert load("libNope") == -1
    assert capsys.readouterr().err.startswith(
        "Error in <" + loading._system() + "::FindDynamicLibrary>: "
        "libNope[.so | .dll | .dylib | .sl | .dl | .a] does not exist in ."
    )  # fmt: skip
    write("libJunk.xrdlib", "junk")
    assert load("libJunk") == -1
    assert "is not a library xrdroot built" in capsys.readouterr().err


def test_a_library_loads_the_libraries_it_needs_first() -> None:
    write("A.h", "int fromA() { return 1; }\nint main() { return 9; }")
    build_library("A", [Path("A.h")])
    write("B.h", "int fromB() { return 2; }")
    build_library("B", [Path("B.h")], libraries=["libA"])
    assert load("libB") == 0
    assert ours() == ["libA", "libB"]
    write("t.C", "#include <gtest/gtest.h>\nTEST(T, t) { EXPECT_EQ(fromA(), 1); }")
    assert run_program(["t.C"]) == 0  # its tests, not the library's main()
    library = read(Path("libB.xrdlib"))
    assert library is not None
    library.path = None
    assert load_library(library) is load_library(library)  # kept by name without a file


def test_a_namespace_two_libraries_declare_holds_both_libraries_names() -> None:
    write("A.h", "namespace ns { int a() { return 1; } }")
    write("B.h", "namespace ns { int b() { return 2; } }")
    build_library("A", [Path("A.h")])
    build_library("B", [Path("B.h")])
    assert (load("libA"), load("libB")) == (0, 0)
    assert (ROOT.ns.a(), ROOT.ns.b()) == (1, 2)


def test_a_macro_including_a_loaded_library_s_header_uses_the_library_s_definitions(
    capsys: pytest.CaptureFixture[str],
) -> None:
    shapes()
    write("use.C", """#include "Shapes.h"
    int use() { geo::Shape a(2); return CountShapes() * 10 + (int)a.Area(); }""")
    assert load("libShapes") == 0
    assert run("use.C") == 14


def test_a_macro_compiled_against_a_loaded_library_calls_the_library_s_bodies() -> None:
    # `macro.C+` is a library of its own, and cling's link makes its calls into the loaded
    # library's: a class its header only declares is constructed by that library's code.
    shapes()
    assert load("libShapes") == 0
    write("built.C", """#include "Shapes.h"
    int built() { geo::Shape a(4); return (int)a.Area(); }""")
    assert run("built.C+") == 16


def test_a_loaded_library_s_classes_are_the_ones_with_a_dictionary() -> None:
    # `if (!TClassTable::GetDict("geo::Shape")) gSystem->Load("libShapes")` is how a program
    # decides, and the dictionary it asks about is the library's own.
    shapes()
    assert ROOT.TClassTable.GetDict("geo::Shape") is None
    assert load("libShapes") == 0
    found = ROOT.TClassTable.GetDict("geo::Shape")
    assert found is not None and found.GetName() == "Shape"
    assert ROOT.TClassTable.GetDict("CountShapes") is None  # a function, not a class


def test_a_header_a_library_provides_declared_in_python_or_a_line_is_the_library_s(
) -> None:
    shapes()
    assert ROOT.gSystem.Load("libShapes") == 0
    ROOT.gInterpreter.Declare('#include "Shapes.h"')
    assert process_line('#include "Shapes.h"') == 0
    shape = ROOT.geo.Shape(2)
    assert type(shape) is ROOT.Shape and (shape.Area(), ROOT.CountShapes()) == (4.0, 1)


def test_a_library_loaded_after_a_macro_including_its_header_binds_its_names_there() -> None:
    shapes()
    write("late.C", """#include "Shapes.h"
    int late() { gSystem->Load("libShapes"); geo::Shape a(3); return (int)a.Area(); }""")
    write("plain.C", '#include "Other.h"\nint plain() { return 1; }')
    write("Other.h", "int other();")
    assert run("plain.C") == 1
    assert run("late.C") == 9


def test_a_name_a_library_provides_is_autoloaded_as_its_rootmap_allows() -> None:
    shapes()
    write("A.h", "int fromA() { return 5; }")
    build_library("A", [Path("A.h")])  # no rootmap: never autoloaded
    write("auto.C", "int auto_() { return geo::Shape(2).Area() + CountShapes(); }")
    assert process_line("CountShapes()") == 0  # loading libShapes, whose count is 0 so far
    assert ours() == ["libShapes"]
    assert not loading.autoload("_private") and not loading.autoload("fromA")
    assert not loading.autoload("geo")  # its library is loaded already
    with pytest.raises(AttributeError):
        _ = ROOT.fromA


def test_autoloading_reads_a_library_once_while_it_is_unchanged() -> None:
    shapes()
    path = Path("libShapes.xrdlib")
    first = loading._indexed(path)
    assert first is not None and loading._indexed(path) is first
    assert loading._indexed(Path("gone.xrdlib")) is None
    loading.AUTOLOADING.add(str(path.resolve()))
    assert not loading.autoload("CountShapes")


def test_a_name_no_library_provides_is_still_not_found() -> None:
    with pytest.raises(AttributeError, match="ROOT has Nowhere"):
        _ = ROOT.Nowhere


def test_a_translation_s_names_are_mapped_to_the_files_that_defined_them() -> None:
    from xrdroot.pycxx.translation import Translation

    python = "import os\nx = 1\na, b = 2, 3\nclass K:\n    pass\nos.sep\n"
    made = Translation(python, "m.C", {2: ("m.C", 1), 3: ("m.C", 2), 4: ("h.h", 1), 6: ("m.C", 3)})
    file = str(Path("h.h").resolve())
    assert loading._defined_in(made) == [("x", str(Path("m.C").resolve())), ("K", file)]


# -- ACLiC -----------------------------------------------------------------------------------


def test_aclic_s_options_and_library_name() -> None:
    assert aclic.split("m.C") == ("m.C", "")
    assert aclic.split("m.C+") == ("m.C", "+")
    assert aclic.split(" dir/m.C++g ") == ("dir/m.C", "++g")
    assert aclic.library_name(Path("dir/hsimple.C")) == "hsimple_C"


def test_aclic_builds_once_and_again_only_when_the_macro_changes_or_it_is_forced(
    capsys: pytest.CaptureFixture[str],
) -> None:
    macro = write("m.C", '#include "m.h"\nint m(int n = 1) { return n * K; }')
    write("m.h", "const int K = 3;")
    first = compiled(macro)
    assert "creating shared library" in capsys.readouterr().err
    assert compiled(macro).digest == first.digest
    assert capsys.readouterr().err == ""
    compiled(macro, force=True)
    assert "creating shared library" in capsys.readouterr().err
    write("m.h", "const int K = 4;")
    assert compiled(macro).digest != first.digest
    library = read(Path("m_C.xrdlib"))
    assert library is not None
    library.translator = "old"
    manifest.write(library, Path("."))
    capsys.readouterr()
    compiled(macro)
    assert "creating shared library" in capsys.readouterr().err


def test_aclic_compiles_with_what_add_include_path_defines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write("inc/k.h", "const int K = 10;")
    write("d.C", '#include "k.h"\nint d() { return VERSION + K; }\n#ifdef CASE_2\n'
          "int twice() { return 2; }\n#endif")  # fmt: skip
    monkeypatch.setattr(ROOT.gSystem, "_include_path", "")
    ROOT.gSystem.AddIncludePath(' -DCASE_2 "-DVERSION=3" -Iinc -I ')
    assert run("d.C+") == 13 and DECLARED["twice"]() == 2
    assert compiled(Path("d.C")).digest == read(Path("d_C.xrdlib")).digest  # up to date
    ROOT.gSystem.SetIncludePath("-DVERSION=5 -Iinc")
    assert run("d.C+") == 15
    ROOT.gSystem.SetIncludePath("")
    compiled(Path("d.C"))  # built again: without the definitions it was built with
    assert not Path("d_C_ACLiC_defines.h").exists()


def test_aclic_puts_its_library_in_the_build_directory_flat_or_under_the_macro_s_path() -> None:
    macro = write("m.C", "int m() { return 1; }")
    assert aclic.library_place(macro) == Path.cwd()
    ROOT.gSystem.SetBuildDir("build", True)
    assert ROOT.gSystem.IsBuildDirFlat() and ROOT.gSystem.GetBuildDir() == "build"
    assert aclic.library_place(macro) == Path("build").resolve()
    ROOT.gSystem.SetBuildDir("build")
    assert aclic.library_place(macro) == Path("build").resolve() / str(Path.cwd()).lstrip("/")
    assert aclic.library_place(macro, Path("o")) == Path("o").resolve()


def test_aclic_without_pyroot_s_system_puts_the_library_beside_the_macro(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "xrdroot.pyroot.core.system")
    assert aclic._build_dir() == ("", False)


def test_dot_l_plus_builds_loads_and_reloads_a_changed_macro(
    capsys: pytest.CaptureFixture[str],
) -> None:
    write("lib.C", "int twice(int x) { return 2 * x; }")
    assert dot_l("lib.C+")["twice"](4) == 8
    assert load("lib_C") == 1
    write("lib.C", "int twice(int x) { return 3 * x; }")
    assert dot_l("lib.C+")["twice"](4) == 12
    assert process_line("twice(2)") == 6


def test_a_compiled_class_s_static_reads_its_own_class_as_its_body_writes_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``int A::s = A::k;``: a static's definition is written in the class's Python body, which
    runs before the class's name is bound there - so what the body reads is its own ``k``."""
    write("own.C", """\
struct Own {
   enum E { k = 3, next };
   static E s;
   static int n;
   static int doubled;
};
Own::E Own::s = Own::k;
int Own::n = (int)Own::s + Own::next;
int Own::doubled = 2 * Own::n;
""")  # fmt: skip
    loaded = dot_l("own.C+")
    klass = loaded["Own"]
    assert (klass.s, klass.n, klass.doubled) == (3, 7, 14)
    assert "creating shared library" in capsys.readouterr().err


def test_dot_x_plus_runs_the_compiled_function_and_prints_its_value(
    capsys: pytest.CaptureFixture[str],
) -> None:
    write("val.C", 'int val(int n) { printf("n=%d\\n", n); return n + 1; }')
    write("nothing.C", "int other() { return 1; }")
    write("block.C", '{ printf("block\\n"); }')
    assert process_line(".x val.C+(4)") == 5
    assert capsys.readouterr().out == "n=4\n(int) 5\n"
    assert run("val.C+", (2,)) == 3
    assert run_program(["val.C+"], (6,)) == 7
    assert run_program(["block.C+"]) == 0
    assert process_line(".x nothing.C+") is None
    assert capsys.readouterr().out == "n=2\nn=6\n(int) 7\nblock\n"


def test_compile_macro_is_one_when_built_and_zero_when_it_cannot_be(
    capsys: pytest.CaptureFixture[str],
) -> None:
    write("cm.C", "int cm() { return 1; }")
    assert ROOT.gSystem.CompileMacro("cm.C", "kc", "libcm", "out") == 1
    assert load("libcm") == -1 and load("out/libcm") == 0  # built, and not loaded: c
    assert ROOT.gSystem.CompileMacro("cm.C") == 1
    assert load("cm_C") == 1
    write("bad.C", "int bad() { return ; ; }}")
    assert compile_macro("bad.C+") == 0
    assert capsys.readouterr().err.endswith("Error in <ACLiC>: Compilation failed!\n")


# -- what a macro loads as it is read ---------------------------------------------------------


def test_a_macro_s_library_pragmas_load_before_it_is_translated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shapes(out=Path("libs"))
    write("A.h", "int fromA() { return 7; }")
    build_library("A", [Path("A.h")], out=Path("libs"))
    monkeypatch.setenv("LIBS", "libs")
    preload('R__ADD_LIBRARY_PATH($LIBS)\nR__LOAD_LIBRARY(libShapes)\n'
            '#pragma cling add_library_path("libs")\n  # pragma cling load("libA")\n')  # fmt: skip
    assert ours() == ["libShapes", "libA"]
    write("loads.C", 'R__LOAD_LIBRARY("libs/libA.so")\nint loads() { return fromA(); }')
    assert run("loads.C") == 7


def test_dot_l_of_a_library_loads_it(capsys: pytest.CaptureFixture[str]) -> None:
    shapes()
    assert "CountShapes" in dot_l("libShapes.so")
    assert dot_l("libTree.so") == {}
    assert dot_l("libNope.so") == {}
    assert process_line(".L libShapes.so") == 0


def test_a_library_is_run_as_its_macro_and_listed_by_dot_libraries(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from xrdroot.cli import main
    from xrdroot.pycint.execute import exists

    write("hello.C", 'int hello(int n = 1) { printf("hello %d\\n", n); return n; }')
    assert compile_macro("hello.C", "kc") == 1
    capsys.readouterr()
    assert (exists("hello_C.so"), exists("gone_C.so"), exists("hello.C+"), exists("x.C")) == (
        True, False, True, False)  # fmt: skip
    assert main(["root", "-l", "-b", "-q", "hello_C.so(3)", "-e", ".libraries"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Processing hello_C.so(3)...\nhello 3\n(int) 3\nlibRint\nlibCore\n")
    assert out.endswith(f"{Path('hello_C.xrdlib').resolve()}\n")
    assert process_line(".x hello_C.so(2)") == 2
    assert run("hello_C.dylib", (4,)) == 4


def test_root_refuses_a_library_that_is_not_there_and_goes_into_a_directory(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from xrdroot.cli import main

    assert main(["root", "-l", "-b", "-q", "gone_C.so"]) == 2
    Path("sub").mkdir()
    capsys.readouterr()
    assert main(["root", "-l", "-b", "-q", "sub", "-e", "gSystem->pwd()"]) == 255
    assert capsys.readouterr().out == f'(const char *) "{Path.cwd()}"\n'  # root went into it
    assert Path.cwd().name == "sub"
    with pytest.raises(FileNotFoundError, match="no library xrdroot built"):
        aclic.call("gone_C.so")


#: A header that gives a class without a ``ClassDef`` a version of its own, as roottest's
#: ``io/evolution/versions/myclass.h`` does: the version is compiled in, rootcling is not told.
VERSIONED_H = """\
class MyClass {
public:
   double a;
};

#ifndef __CLING__
RootClassVersion(MyClass,3)
#endif
"""


def test_a_class_written_without_a_classdef_takes_the_version_the_compiled_header_names(
) -> None:
    """``RootClassVersion(MyClass,3)`` is compiled into an ACLiC library - ROOT's macro is
    inside ``#ifndef __CLING__``, and only rootcling defines that - and sets the version the
    class is described by, so a class without a ``ClassDef`` is written at version 3 rather
    than as a foreign class, which is what roottest's ``AddVersion-1-3`` reads back."""
    from xrdroot.pycint.dialect import LIBRARY
    from xrdroot.pycint.dialect import ROOT as ROOT_DIALECT
    from xrdroot.pycxx import translate
    from xrdroot.pyroot.core.declared import declaration_of

    compiled: dict[str, object] = {}
    exec(translate(VERSIONED_H, "myclass.h", dialect=LIBRARY), compiled)
    declared = declaration_of(compiled["MyClass"])  # type: ignore[arg-type]
    assert (declared.version, declared.foreign) == (3, True)
    cling = translate(VERSIONED_H, "myclass.h", dialect=ROOT_DIALECT)
    assert "Class_Version" not in cling  # what Cling, which has __CLING__, leaves out

def test_a_load_of_a_name_no_library_or_macro_answers_to_is_not_found() -> None:
    """``gSystem->Load("nothing_C")``: no library, no ``nothing.C`` - ROOT's -1 and its message."""
    from xrdroot.pycint.library.loading import load

    assert load("nothing_at_all_C") == -1


def test_a_load_of_a_name_aclic_named_builds_the_macro_it_came_from() -> None:
    """``gSystem->Load("tiny_C")`` is a request to compile ``tiny.C``: ROOT names the library
    ACLiC makes of ``tiny.C`` ``tiny_C``."""
    from xrdroot.pycint.library.loading import load

    Path("tiny.C").write_text("int tiny_f(int x) { return x + 1; }\n")
    assert load("tiny_C") == 0


ENUMS_H = """\
#ifndef ENUMS_H
#define ENUMS_H
namespace myns {
enum enpclass {kNoneNonConflicting};
enum class Kind : short {kOne, kTwo};
}
class Holder {
public:
   enum Mode {kOff, kOn = 5};
};
enum Global {kFirst = 2, kNext};
#endif
"""


# -- the enumerations a dictionary carries --------------------------------------------------


def test_a_dictionary_lists_the_enumerations_of_the_headers_it_was_built_from() -> None:
    """A rootmap lists a dictionary's enumerations beside its classes, and ROOT's interpreter
    knows them as soon as the library is loaded - before anything reads the header."""
    write("Enums.h", ENUMS_H)
    build_dictionary("Enums", [Path("Enums.h")], rootmap=False)
    library = read(Path("libEnums.xrdlib"))
    assert library is not None
    assert [each["name"] for each in library.enums] == [
        "myns::enpclass", "myns::Kind", "Holder::Mode", "Global",
    ]  # fmt: skip
    assert library.enums[1]["scoped"] and library.enums[1]["underlying"] == "short"


def test_loading_a_dictionary_makes_its_enumerations_the_interpreter_s() -> None:
    """The title of one a namespace declares is the namespace's name, which is what the oracle
    answers for a rootcling-built enumeration (roottest's ``qualifiedNames`` prints
    ``enpclass/myns``); one a class declares has none."""
    write("Enums.h", ENUMS_H)
    build_dictionary("Enums", [Path("Enums.h")], rootmap=False)
    assert troot.find_enum("myns::Kind") is None
    load("libEnums")
    kinds = troot.find_enum("myns::Kind")
    assert kinds is not None
    assert (kinds.GetTitle(), kinds.GetUnderlyingType()) == ("myns", 2)
    assert [(one.GetName(), one.GetValue()) for one in kinds.GetConstants()] == [
        ("kOne", 0), ("kTwo", 1),
    ]  # fmt: skip
    mode = troot.find_enum("Holder::Mode")
    assert mode is not None and mode.GetTitle() == ""
    assert [(one.GetName(), one.GetValue()) for one in mode.GetConstants()] == [
        ("kOff", 0), ("kOn", 5),
    ]  # fmt: skip
    assert [one.GetName() for one in troot.class_enums("myns")] == ["enpclass", "Kind"]


def test_a_library_is_autoloaded_by_the_name_of_an_enumeration_it_declares() -> None:
    """``TEnum::GetEnum`` loads the library a rootmap lists an enumeration in, and looks
    again: roottest's ``qualifiedNames`` never loads its dictionary and still finds it."""
    write("Enums.h", ENUMS_H)
    build_dictionary("Enums", [Path("Enums.h")])
    assert loading.provides(read(Path("libEnums.xrdlib")), "myns::Kind")
    assert not loading.autoload_enum("myns::Nowhere")
    assert loading.autoload_enum("myns::Kind")
    assert ours() == ["libEnums"]
    assert troot.find_enum("myns::Kind") is not None


def test_a_dictionary_describes_only_the_classes_it_selects() -> None:
    """A header's classes are translated with the dictionary it is built from, but only the
    ones a LinkDef selects have a dictionary - which is what ``GetMissingDictionaries``
    reports of a class of such a header."""
    write("Two.h", "class First { public: int a; };\nclass Second { public: int b; };\n")
    write("TwoLinkDef.h", "#pragma link C++ class First+;\n")
    build_dictionary("Two", [Path("Two.h")], linkdef=Path("TwoLinkDef.h"))
    assert load("libTwo") == 0
    assert ROOT.TClassTable.GetDict("First") is not None
    assert ROOT.TClassTable.GetDict("Second") is None


def test_a_selection_of_patterns_describes_every_class_the_header_declares() -> None:
    """``pattern="Fir*"`` selects whatever of the header matches, so a dictionary built from
    a selection of patterns describes every class its headers declare, not a list of names."""
    write("Pat.h", "class First { public: int a; };\nclass Second { public: int b; };\n")
    write("pat.xml", '<lcgdict>\n  <class pattern="Fir*"/>\n</lcgdict>\n')
    build_dictionary("Pat", [Path("Pat.h")], selection=Path("pat.xml"))
    assert load("libPat") == 0
    assert ROOT.TClassTable.GetDict("First") is not None
    assert ROOT.TClassTable.GetDict("Second") is not None
