"""What the preprocessor says of a macro's build: MathMore present, Qt and SYCL programs out."""

from __future__ import annotations

import pytest

from xrdroot.cint import Refusal, translate


def test_mathmore_is_there_as_it_is_in_a_released_root() -> None:
    source = "#ifdef R__HAS_MATHMORE\nint n = 1;\n#else\n#error no MathMore\n#endif\n"
    assert "n = 1" in translate(source, "t.C")


@pytest.mark.parametrize(
    ("header", "what"),
    [
        ("<QWidget>", "a Qt widget program"),
        ("<QtWidgets/QApplication>", "a Qt widget program"),
        ("<sycl/sycl.hpp>", "a SYCL program"),
        ("<CL/sycl.hpp>", "a SYCL program"),
    ],
)
def test_a_qt_or_sycl_program_is_refused_by_what_it_includes(header: str, what: str) -> None:
    with pytest.raises(Refusal, match=f"makes this {what}.*not a macro, out of scope") as refused:
        translate(f"#include <vector>\n#include {header}\nvoid t() {{}}\n", "t.C")
    assert refused.value.where is not None and refused.value.where.line == 2
