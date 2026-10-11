"""C's floating division, ``<cmath>`` of an ``RVec``, ``ofstream``, ``R__ASSERT`` and ``*list``.

``b1 / b2`` with ``b2`` zero is an infinity in C++, not an exception (df004);
``exp(v)`` of an ``RVec`` is an ``RVec`` of each element's (vo002);
``std::ofstream`` writes a file as ``cout`` writes the screen (rf102);
``R__ASSERT`` is ROOT's ``assert``; ``*tree->GetListOfBranches()`` is the
list itself, which a range-for walks (df007).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from cintfake import fake
from xrdroot.cint import runtime as rt
from xrdroot.cint.errors import MacroError
from xrdroot.cint.execute import run_source


def block(capsys: pytest.CaptureFixture[str], body: str) -> str:
    run_source("{\n" + body + "\n}\n", "t.C", (), root=fake())
    return capsys.readouterr().out


def test_a_floating_division_by_zero_is_ieees_infinity_or_nan(
    capsys: pytest.CaptureFixture[str],
) -> None:
    text = block(capsys, 'double a = 1.5; int z = 0; double n = 0; printf("%g %g %g %g\\n", '
                 "a / z, -a / z, n / z, a / 2);")  # fmt: skip
    assert text == "inf -inf nan 0.75\n"
    assert rt.fdiv(1.0, -0.0) == -math.inf
    assert math.isnan(rt.fdiv(math.nan, 0.0))
    assert rt.div(3.0, 0) == math.inf
    assert rt.fdiv(np.array([1.0, 2.0]), 2.0).tolist() == [0.5, 1.0]


def test_cmath_of_an_rvec_is_each_elements_in_an_rvec_of_the_promoted_type() -> None:
    from xrdroot.pyroot.rdf.rvec import RVec

    floats = RVec["float"]([1.0, 4.0])
    root = rt.sqrt(floats)
    assert root.__cpp_name__ == "ROOT::VecOps::RVec<float>" and root.data().tolist() == [1, 2]
    ints = RVec["int"]([1, 2])
    assert rt.exp(ints).__cpp_name__ == "ROOT::VecOps::RVec<double>"
    assert rt.exp(ints).data().tolist() == [math.exp(1), math.exp(2)]
    assert rt.log(RVec["double"]([1.0, 0.0])).data().tolist() == [0.0, -math.inf]
    assert rt.pow(np.array([2.0, 3.0]), 2).tolist() == [4.0, 9.0]
    with pytest.raises(TypeError):
        rt.sqrt(object())


def test_an_ofstream_writes_a_file_as_cout_writes_and_adds_to_one_asked_to(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = (tmp_path / "out.txt").as_posix()
    block(capsys, f'std::ofstream os("{path}"); os << 1.5 << " " << 2 << std::endl; os.close();')
    stream = rt.ofstream(path, "std::ios::app")
    assert stream.is_open()
    stream << "more" << rt.endl
    stream.flush()
    stream.close()
    assert Path(path).read_text() == "1.5 2\nmore\n"
    unopened = rt.ofstream()
    assert not unopened.is_open() and unopened.flush() is unopened
    unopened.close()
    with pytest.raises(ValueError, match="before a file"):
        unopened << 1


def test_r_assert_is_an_assert(capsys: pytest.CaptureFixture[str]) -> None:
    assert block(capsys, 'R__ASSERT(1 == 1); printf("ok\\n");') == "ok\n"
    with pytest.raises(MacroError, match="AssertionError"):
        block(capsys, "R__ASSERT(1 == 2);")


def test_dereferencing_a_root_list_gives_the_list_and_a_c_array_its_first() -> None:
    from xrdroot.pyroot.trees._base import ListOf

    listed = ListOf([1, 2])
    assert rt.deref(listed) is listed
    assert rt.deref([7, 8]) == 7
