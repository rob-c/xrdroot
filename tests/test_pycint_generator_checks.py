"""What rootcling and genreflex refuse in a selection, refused with their words and status.

Every message and exit status here is what ROOT 6.40.04's genreflex or rootcling gives for
the same files.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from pyrootsupport import fresh
from xrdroot.cli import main
from xrdroot.pycint.library import build_dictionary, checks
from xrdroot.pycint.library.build import Strictness
from xrdroot.pycxx import cache


@pytest.fixture(autouse=True)
def _fresh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv(cache.ENVIRONMENT, str(tmp_path / "cache"))
    yield from fresh(tmp_path)


def write(name: str, text: str) -> Path:
    Path(name).write_text(text)
    return Path(name)


def dictionary(*words: str) -> int:
    return main(["build", "dictionary", "--name", "D", *words])


def test_a_switch_that_is_neither_true_nor_false_is_an_xml_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    write("foo.h", "class Foo { int fData; };\n")
    write("sel.xml", '<lcgdict>\n<selection>\n<class name="Foo" noStreamer="invalid" />\n'
          "</selection>\n</lcgdict>\n")  # fmt: skip
    assert dictionary("--selection", "sel.xml", "foo.h") == 1
    assert capsys.readouterr().err == (
        "Error: XML at line 3: class attribute 'noStreamer' must be 'true' or 'false' "
        "(it was invalid)\n"
    )


CONFLICTING = """<rootdict>
   <class name="A" />
   <class name="A">
      <field name="fMember" transient = "true"/>
   </class>
<!-- a comment
   over lines -->
   <class pattern="Long*" ClassVersion="3" />
   <class pattern="Long*" />
</rootdict>
"""


def test_two_rules_for_one_name_that_differ_stop_the_build(
    capsys: pytest.CaptureFixture[str],
) -> None:
    write("a.h", "class A { int fMember; };\n")
    write("sel.xml", CONFLICTING)
    assert dictionary("--selection", "sel.xml", "a.h") == 1
    err = capsys.readouterr().err
    assert err.endswith(
        "Warning in <SelectionRules::CheckDuplicates>: Duplicated rule found.\nRule:\n"
        "\t\tSelected (line 2): Yes\n\t\tAttributes: \n\t\tname = A\n\t\tNo field sel rules\n"
        "\t\tNo method sel rules\nConflicting rule already stored:\n"
        "\t\tSelected (line 3): Yes\n\t\tAttributes: \n\t\tname = A\n\t\tField 0:\n"
        "\t\tSelected: Yes\n\t\t\tname = fMember\n\t\t\ttransient = true\n"
        "\t\tNo method sel rules\n"
        "Error in <SelectionRules::CheckDuplicates>: Duplicates in rules were found.\n"
    )
    assert err.index("line 8") < err.index("line 2")  # the last pair first
    assert "Identical" in err[: err.index("line 2")]  # a version alone makes no difference


def test_two_identical_rules_are_only_warned_about(capsys: pytest.CaptureFixture[str]) -> None:
    write("a.h", "class A {};\n")
    write("sel.xml", '<rootdict>\n<class name="A" />\n<class name="A" />\n</rootdict>\n')
    assert dictionary("--selection", "sel.xml", "a.h") == 0
    assert "Identical rule already stored:\n\t\tSelected (line 3)" in capsys.readouterr().err


UNSUPPORTED = """<rootdict>
  <class name="vector<int>" />
  <class name="std::array<int,3>" />
  <class name="std::tuple<int,double,float>" />
  <class name="std::chrono::duration<int>" />
  <class name="std::ratio<1,1>" />
  <class name="std::thread" />
</rootdict>
"""


def test_a_standard_class_without_io_is_an_error_each_and_the_status_their_count(
    capsys: pytest.CaptureFixture[str],
) -> None:
    write("u.h", "int x;\n")
    write("sel.xml", UNSUPPORTED)
    assert dictionary("--selection", "sel.xml", "u.h") == 4

    def missing(name: str) -> str:
        return (f"Error: Class {name} has been selected but currently the support for its I/O "
                f"is not yet available. Note that {name}, even if not selected, will be "
                "available for interpreted code.\n")  # fmt: skip

    assert capsys.readouterr().err == (
        "Error: It is not necessary to explicitly select class array<int,3>. I/O is supported "
        "for it transparently.\n" + missing("ratio<1,1>")
        + missing("chrono::duration<int,ratio<1,1> >") + missing("thread")
    )


def test_genreflex_refuses_to_inline_a_header_with_pragma_once(
    capsys: pytest.CaptureFixture[str],
) -> None:
    write("once.h", "// a comment\n   #pragma     once\nclass A {};\n")
    write("sel.xml", '<rootdict><class name="A" /></rootdict>')
    assert dictionary("--selection", "sel.xml", "once.h") == 1
    assert capsys.readouterr().err == (
        "Error: #pragma once directive detected in header file once.h which was requested to "
        "be inlined.\n"
    )
    write("LinkDef.h", "#pragma link C++ class A+;\n")
    assert dictionary("--linkdef", "LinkDef.h", "once.h") == 0  # rootcling does not inline


def test_a_union_a_linkdef_selects_is_refused(capsys: pytest.CaptureFixture[str]) -> None:
    write("u.h", "namespace N { union U { int i; float f; }; typedef U V; }\n")
    write("LinkDef.h", "#pragma link C++ union N::U-;\n#pragma link C++ typedef N::V;\n")
    assert dictionary("--linkdef", "LinkDef.h", "u.h") == 1
    assert capsys.readouterr().err == (
        "Error: Union N::U has been selected for I/O. This is not supported. Interactive usage "
        "of unions is supported, as all C++ entities, without the need of dictionaries.\n"
    )


def test_rootcling_by_hand_with_nothing_to_select_by_is_refused(
    capsys: pytest.CaptureFixture[str],
) -> None:
    write("a.h", "class A {};\n")
    assert dictionary("--selection-required", "a.h") == 1
    assert capsys.readouterr().err.startswith("Error: No selection rules specified")
    write("b.h", "class B {};\n#pragma link C++ class B+;\n")
    assert dictionary("--selection-required", "b.h") == 0
    write("LinkDef.h", "#pragma link C++ class A+;\n")
    assert dictionary("--selection-required", "--linkdef", "LinkDef.h", "a.h") == 0
    assert dictionary("a.h") == 0


IO_RULES = """<lcgdict>
<class name="UserData"/>
  <ioread sourceClass="UserData" source="float posx; float posy;" targetClass="UserData"
    target="posx; posy;" version="[12-25]">
    <![CDATA[{ posx.resize(1, onfile.posx); }]]>
  </ioread>
  <ioread sourceClass="UserData" targetClass="UserData" target="posx" version="[2]"/>
  <read sourceClass="Old" version="[2]" targetClass="Elsewhere" />
  <read sourceClass="Old" version="[3]" targetClass="Elsewhere" />
  <read sourceClass="Old" version="[3]" targetClass="Other" />
</lcgdict>
"""


def test_an_io_rule_for_no_member_warns_and_fails_on_warnings(
    capsys: pytest.CaptureFixture[str],
) -> None:
    write("u.h", "#include <vector>\nclass UserData { std::vector<double> posx, posy; };\n")
    write("sel.xml", IO_RULES)
    assert dictionary("--selection", "sel.xml", "u.h") == 0
    expected = (
        "Warning: IO rule for class UserData data member: posx; posy; was specified as a target "
        "in the rule but doesn't seem to appear in target class\n"
        "Warning: 2 rules for target class Elsewhere were not used!\n"
        "Warning: 1 rule for target class Other was not used!\n"
    )
    assert capsys.readouterr().err == expected
    assert dictionary("--selection", "sel.xml", "--fail-on-warnings", "u.h") == 1
    assert capsys.readouterr().err == expected


def test_a_pattern_selects_the_target_class_of_an_io_rule() -> None:
    parsed = checks.rules('<lcgdict><class/><class pattern="User*"/></lcgdict>')
    assert checks.io_targets(parsed, {}) == []
    parsed.io.append({"targetClass": "UserData", "target": "x"})
    assert checks.io_targets(parsed, {"UserData": ["x"]}) == []


def test_a_nested_typedef_is_a_name_a_linkdef_may_select(
    capsys: pytest.CaptureFixture[str],
) -> None:
    write("n.h", "namespace N { class Fd { public: typedef int Number; enum E { a }; }; }\n")
    write("LinkDef.h", "#pragma link C++ typedef N::Fd::Number;\n#pragma link C++ enum N::Fd::E;"
          "\n#pragma link C++ class vector<int>;\n")  # fmt: skip
    build_dictionary("N", [Path("n.h")], linkdef=Path("LinkDef.h"), strict=Strictness())
    assert capsys.readouterr().err == ""


def test_a_selection_read_for_its_syntax_alone_refuses_no_duplicate_rule(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """genreflex's ``--selSyntaxOnly``: ROOT 6.40.04 reads a selection whose rules conflict,
    warns of each pair as it parses them, writes nothing and exits 0 - the duplicates stop
    the build that follows, not the check."""
    write("a.h", "class A { int fMember; };\n")
    write("sel.xml", CONFLICTING)
    assert main(["build", "dictionary", "--name", "D", "--selection", "sel.xml",
                 "--selection-syntax-only", "a.h"]) == 0
    assert "Conflicting rule already stored" in capsys.readouterr().err
    assert not Path("libD.xrdlib").exists()


def test_a_selection_read_for_its_syntax_alone_still_refuses_an_xml_error() -> None:
    write("a.h", "class A {};\n")
    write("sel.xml", '<rootdict>\n<class name="A" noStreamer="perhaps" />\n</rootdict>\n')
    assert main(["build", "dictionary", "--name", "D", "--selection", "sel.xml",
                 "--selection-syntax-only", "a.h"]) == 1


def test_a_syntax_check_of_a_selection_that_is_not_there_reads_none() -> None:
    """A generator run for its syntax with no selection is rootcling's, and refuses nothing."""
    write("a.h", "class A {};\n")
    assert main(["build", "dictionary", "--name", "D", "--selection-syntax-only", "a.h"]) == 0
