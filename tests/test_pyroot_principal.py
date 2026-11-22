"""``TPrincipal``: ROOT's principal component analysis, and the code it writes."""

from __future__ import annotations

import numpy as np
import pytest

import xrdroot.pyroot as ROOT


def correlated() -> list[list[float]]:
    """Three variables, the third being the first two added: one exact dependency."""
    rows = []
    for at in range(200):
        first = 5.0 + 0.5 * np.sin(at)
        second = 8.0 + 0.25 * np.cos(3 * at)
        rows.append([first, second, 10.0 + 0.5 * np.sin(at) - 0.25 * np.cos(3 * at)])
    return rows


def filled() -> ROOT.TPrincipal:
    principal = ROOT.TPrincipal(3, "ND")
    for row in correlated():
        principal.AddRow(row)
    principal.MakePrincipals()
    return principal


def test_the_means_sigmas_and_eigenvalues_are_the_data_s_own():
    """ROOT's table: the mean of each variable, the population sigma, and eigenvalues that add
    up to one because the N option normalises, largest first."""
    principal = filled()
    assert np.allclose(principal._means, np.mean(correlated(), axis=0), rtol=0, atol=1e-12)
    expected_sigma = np.std(correlated(), axis=0)  # population, as ROOT's
    assert np.allclose(principal._sigmas, expected_sigma, rtol=0, atol=1e-12)
    assert principal._eigenvalues.sum() == pytest.approx(1.0, abs=1e-12)
    assert list(principal._eigenvalues) == sorted(principal._eigenvalues, reverse=True)
    assert principal._eigenvalues[-1] == pytest.approx(0.0, abs=1e-12)  # the dependency


def test_printing_is_roots_table_to_the_column(capsys: pytest.CaptureFixture[str]):
    """``Print`` writes ROOT's header and one line per variable, in ROOT's own column widths."""
    principal = ROOT.TPrincipal(2, "ND")
    for row in ([1.0, 2.0], [3.0, 5.0], [5.0, 8.0]):
        principal.AddRow(row)
    principal.MakePrincipals()
    capsys.readouterr()
    principal.Print()
    said = capsys.readouterr().out.splitlines()
    assert said[0] == " Variable #  | Mean Value |   Sigma    | Eigenvalue"
    assert said[1] == "-------------+------------+------------+------------"
    assert said[2].startswith("           0 |          3 |")
    assert said[2].endswith("|          1 ")  # two variables, all the variance in one
    assert said[3].startswith("           1 |          5 |")


def test_a_row_goes_to_the_components_and_back():
    """``X2P`` and ``P2X`` are ROOT's two transformations, and ``Test`` says so with ``0``."""
    principal = filled()
    row = correlated()[7]
    again = principal.P2X(principal.X2P(row))
    assert np.allclose(again, row, atol=1e-9)
    assert principal.Test() == 0
    assert ROOT.TPrincipal(3).Test() == 0  # nothing added: nothing to test
    assert ROOT.TPrincipal(3).MakePrincipals() == 0


def test_a_row_of_the_wrong_length_is_refused():
    principal = ROOT.TPrincipal(3)
    with pytest.raises(ValueError, match=r"has 3 variables, and AddRow was given 2"):
        principal.AddRow([1.0, 2.0])


def test_the_histograms_are_kept_under_roots_own_names():
    """``MakeHistograms`` leaves one histogram per variable, named ``h_<at>``."""
    principal = filled()
    principal.MakeHistograms()
    assert ROOT.gDirectory.Get("h_0").GetName() == "h_0"
    assert ROOT.gDirectory.Get("h_2").GetNbinsX() == 100


def test_the_code_it_writes_is_roots_own_file(tmp_path, capsys: pytest.CaptureFixture[str]):
    """``MakeCode`` writes ``pca.C`` with ROOT's prologue, the four arrays and both functions, and
    says so on stdout as ROOT does."""
    principal = filled()
    target = tmp_path / "pca.C"
    capsys.readouterr()
    principal.MakeCode(target)
    assert capsys.readouterr().out == f'\nWriting on file "{target}" ... done\n'
    text = target.read_text()
    for said in (
        "// -*- mode: c++ -*-",
        "static Int_t    gNVariables = 3;",
        "static Double_t gEigenVectors[] = {",
        "static Double_t gEigenValues[] = {",
        "static Double_t gMeanValues[] = {",
        "static Double_t gSigmaValues[] = {",
        "void X2P(Double_t *x, Double_t *p) {",
        "void P2X(Double_t *p, Double_t *x, Int_t nTest) {",
        "// EOF for ",
    ):
        assert said in text
    assert "Writing on file" not in text


def test_the_relationship_of_the_object_says_what_it_holds():
    assert repr(ROOT.TPrincipal(4)) == "<TPrincipal 4 variables, 0 rows>"


def test_without_the_n_option_the_covariance_is_left_un_normalised():
    """No ``N``: ROOT decomposes the covariance itself, so the eigenvalues are variances in the
    variables' own units rather than shares of one."""
    plain = ROOT.TPrincipal(2)
    for row in ([1.0, 20.0], [3.0, 60.0], [5.0, 100.0]):
        plain.AddRow(row)
    plain.MakePrincipals()
    assert plain._eigenvalues.sum() == pytest.approx(
        np.trace(np.cov(np.array([[1.0, 20.0], [3.0, 60.0], [5.0, 100.0]]), rowvar=False,
                        bias=True)) / 2
    )  # fmt: skip
