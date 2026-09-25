"""Running macros as ROOT runs them: files, ``.x`` arguments, ProcessLine, the cache, errors."""

from __future__ import annotations

from pathlib import Path

import pytest

from cintfake import fake
from xrdroot.cint import MacroError, Refusal, cache, translate_file
from xrdroot.cint.execute import arguments, load, process_line, run, run_source, split_call
from xrdroot.cint.runtime import ROOT


@pytest.fixture(autouse=True)
def private_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    where = tmp_path / "cache"
    monkeypatch.setenv(cache.ENVIRONMENT, str(where))
    return where


def write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def test_a_macro_file_runs_its_function_of_the_same_name_with_the_arguments_given(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write(
        tmp_path, "twice.C", 'int twice(int n = 1) { printf("%d\\n", 2 * n); return 2 * n; }'
    )
    assert run(path, root=fake()) == 2
    assert run(f"{path}+", (21,), root=fake()) == 42
    assert capsys.readouterr().out == "2\n42\n"


def test_an_unnamed_macro_runs_its_block(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write(tmp_path, "block.C", '{\n  int a = 5;\n  printf("%d\\n", a / 2);\n}\n')
    run(path, root=fake())
    assert capsys.readouterr().out == "2\n"
    with pytest.raises(TypeError, match=r"block\.C is an unnamed macro, which takes no arguments"):
        run(path, (1,), root=fake())


def test_a_macro_without_its_function_is_only_defined(tmp_path: Path) -> None:
    path = write(tmp_path, "library.C", "int helper() { return 3; }")
    assert run(path, root=fake()) is None
    assert load(path, root=fake())["helper"]() == 3


def test_a_translation_is_kept_and_found_again_until_the_macro_or_its_header_changes(
    tmp_path: Path, private_cache: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write(tmp_path, "values.h", "const int kValue = 1;")
    path = write(
        tmp_path, "cached.C", '#include "values.h"\nvoid cached() { printf("%d\\n", kValue); }'
    )
    run(path, root=fake())
    assert len(list(private_cache.glob("*.json"))) == 1
    run(path, root=fake())
    write(tmp_path, "values.h", "const int kValue = 2;")
    run(path, root=fake())
    assert capsys.readouterr().out == "1\n1\n2\n"
    assert len(list(private_cache.glob("*.json"))) == 1


def test_a_cache_that_cannot_be_written_or_read_is_simply_not_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    blocker = write(tmp_path, "not-a-directory", "")
    monkeypatch.setenv(cache.ENVIRONMENT, str(blocker))
    path = write(tmp_path, "plain.C", 'void plain() { printf("ok\\n"); }')
    run(path, root=fake())
    corrupt = tmp_path / "corrupt"
    corrupt.mkdir()
    monkeypatch.setenv(cache.ENVIRONMENT, str(corrupt))
    (corrupt / f"{cache.key(path.read_text(), str(path))}.json").write_text("{not json")
    run(path, root=fake())
    assert capsys.readouterr().out == "ok\nok\n"


def test_the_cache_is_under_the_users_cache_directory_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(cache.ENVIRONMENT)
    assert cache.directory() == Path.home() / ".cache" / "xrdroot" / "cint"


def test_a_header_that_is_gone_makes_its_translation_stale(tmp_path: Path) -> None:
    header = write(tmp_path, "gone.h", "int kGone = 1;")
    path = write(tmp_path, "user.C", '#include "gone.h"\nint user() { return kGone; }')
    assert run(path, root=fake()) == 1
    header.unlink()
    with pytest.raises(MacroError):
        run(path, root=fake())


def test_an_error_as_the_macro_runs_is_placed_at_its_cpp_line(tmp_path: Path) -> None:
    path = write(tmp_path, "broken.C", "void broken() {\n  int a = 0;\n  int b = 1 / a;\n}\n")
    with pytest.raises(MacroError, match=r"broken.C:3: ZeroDivisionError: integer division") as why:
        run(path, root=fake())
    assert why.value.where is not None and why.value.where.line == 3


def test_an_error_in_rooot_is_placed_at_the_line_that_called_it() -> None:
    with pytest.raises(MacroError, match=r"^t.C:1: AttributeError: .* no attribute 'NoSuchThing'"):
        run_source("void t() { NoSuchThing(); }", "t.C", root=fake())


def test_an_error_outside_the_translation_has_no_line_to_be_placed_at() -> None:
    with pytest.raises(MacroError, match=r"^TypeError: t\(\) takes 1 positional"):
        run_source("void t(int a) {}", "t.C", (1, 2), root=fake())


def test_refusals_and_exits_pass_through_unchanged(tmp_path: Path) -> None:
    with pytest.raises(Refusal, match="goto"):
        run_source("void t() { goto end; end: ; }", "t.C", root=fake())
    with pytest.raises(SystemExit):
        run_source("void t() { exit(2); }", "t.C", root=fake())


@pytest.mark.parametrize(
    ("text", "path", "given"),
    [
        ("hsimple.C", "hsimple.C", ""),
        ("hsimple.C+", "hsimple.C", ""),
        ('fit.C++(1000, "gaus")', "fit.C", '1000, "gaus"'),
        (" dir/m.cxx ( 2 ) ", "dir/m.cxx", "2"),
    ],
)
def test_a_macro_call_is_split_into_its_path_and_its_arguments(
    text: str, path: str, given: str
) -> None:
    assert split_call(text) == (path, given)


def test_what_is_no_macro_call_is_refused() -> None:
    with pytest.raises(ValueError, match="does not name a macro"):
        split_call("")


def test_the_arguments_of_a_call_are_read_as_cpp() -> None:
    assert arguments("") == ()
    assert arguments('1000, "gaus", 2.5, 7 / 2, true') == (1000, "gaus", 2.5, 3, True)
    with ROOT.bind(fake()):
        assert arguments("kRed + 1") == (633,)


def test_process_line_runs_statements_and_dot_commands(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write(tmp_path, "hello.C", 'void hello(int n) { printf("hello %d\\n", n); }')
    process_line('int a = 7; printf("%d %g\\n", a / 2, a / 2.)', root=fake())
    process_line(f".x {path}(3)", root=fake())
    assert process_line(f".L {path}", root=fake())["hello"] is not None
    process_line('{ printf("block\\n"); }', root=fake())
    assert capsys.readouterr().out == "3 3.5\nhello 3\nblock\n"
    with pytest.raises(ValueError, match=r"\.q is not a command"):
        process_line(".q")


def test_a_kept_translation_reads_like_the_macro_and_runs_as_a_script() -> None:
    python = translate_file(Path(__file__).parent / "data" / "cint" / "math" / "Legendre.C")
    assert python.startswith("# Translated from Legendre.C by xrdroot.cint.")
    assert "from xrdroot.cint.runtime import *" in python
    assert "    L = array('TF1*', 5)\n" in python
    assert python.endswith('if __name__ == "__main__":\n    Legendre()\n')


def test_arguments_for_a_macro_with_no_function_to_take_them_are_refused(tmp_path: Path) -> None:
    path = write(tmp_path, "other.C", "int helper() { return 3; }")
    with pytest.raises(TypeError, match=r"other.C defines no function other\(\) to hand 1, 'x'"):
        run(path, (1, "x"), root=fake())
