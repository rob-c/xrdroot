"""ROOT's predefined shapes: the arguments a shape takes, and the four ways one is refused.

``TF1("f", "gaus")`` is a Gaussian because ``TFormula`` rewrites the word before it parses
anything, and this kit does the same rewriting with ROOT's own bodies. What a shape may say
besides its name is the subject here: where its parameters start (``gaus(3)``), which variable it
is in (``pol1(y,0)``), what its parameters are called (``pol1(x,[A],[B])``), and - for
``chebN`` - how far the series goes. Every refusal is a full sentence saying what a shape does
take, because a shape's arguments are the one part of a formula ROOT's user writes by hand.

A formula that is nothing but one shape gets ROOT's names for its parameters (``Constant``,
``Mean``, ``Sigma``), and a formula read back from a file keeps the names it came with. Both are
measured here, because a fit's parameter order is what a caller indexes.

Nothing opens a file: a formula is a string, which keeps ``xrdroot.function.shapes`` inside the
``hist`` component's own isolation scope.
"""

from __future__ import annotations

import pytest

from xrdroot.errors import UnsupportedFeatureError
from xrdroot.formula.errors import FormulaError
from xrdroot.function.shapes import expand, number, recognised

#: The words a body of ROOT's writes a parameter with, which is what a shape is refused by.
_NOT_A_SHAPE = "is not a shape ROOT reads: a shape takes where its parameters start"


def test_a_shape_may_say_where_its_parameters_start():
    """``gaus(3)`` is a Gaussian whose parameters are ``[3]``, ``[4]`` and ``[5]`` - which is how
    ROOT's own tests put a second Gaussian beside a first - so the formula has six parameters and
    the first three belong to whatever came before."""
    found = expand("gaus(3)")
    assert found.text == "[p3]*exp(-0.5*((x-[p4])/[p5])*((x-[p4])/[p5]))"
    assert found.names == ("p0", "p1", "p2", "p3", "p4", "p5")
    assert not found.predefined  # a shape that starts at 3 is not "one shape alone"


def test_a_shape_may_name_the_variable_it_is_in():
    """``pol1(y,0)`` is a straight line in y, not in x: ROOT's own way of fitting a projection
    against its other axis."""
    found = expand("pol1(y,0)")
    assert found.text == "([p0]+[p1]*y)"
    assert found.names == ("p0", "p1")


def test_a_shape_may_name_its_own_parameters():
    """``pol1(x,[A],[B])`` names the two parameters where it writes them, and the formula keeps
    those names - which is what a fit reports and what ``GetParameter("A")`` asks for."""
    found = expand("pol1(x,[A],[B])")
    assert found.text == "([A]+[B]*x)"
    assert found.names == ("A", "B")


def test_a_shape_argument_that_is_none_of_the_three_is_refused_in_a_sentence():
    """A shape's brackets hold one of three things and nothing else, so a fourth kind of argument
    is refused with all three spelled out - and with the note that a real C++ function is written
    with its scope."""
    with pytest.raises(FormulaError, match=_NOT_A_SHAPE):
        expand("gaus(1,[A])")  # a start and a name together are not one of the three
    with pytest.raises(FormulaError, match=_NOT_A_SHAPE):
        expand("pol2(1.5)")  # a fractional start is no place for a parameter


def test_a_chebyshev_series_is_written_as_the_library_s_own_and_stops_at_cheb10():
    """``chebN`` is not arithmetic but a call into ROOT::Math, and ROOT defines the series only up
    to ``cheb10`` - so ``cheb11`` says so rather than writing a name nothing has."""
    assert expand("cheb2").text == "ROOT::Math::Chebyshev2(x,[p0],[p1],[p2])"
    with pytest.raises(UnsupportedFeatureError, match="past cheb10, the highest Chebyshev"):
        expand("cheb11")


def test_too_few_names_for_a_series_is_refused_with_both_counts():
    """A polynomial of degree three takes four parameters, so one name for them is refused with
    how many it wanted and how many it was given."""
    with pytest.raises(FormulaError, match="pol3 has 4 parameters, and 1 names were given"):
        expand("pol3(x,[A])")


def test_a_coordinate_written_by_number_is_the_variable_it_stands_for():
    """``x[1]`` is ``y`` and ``x[2]`` is ``z``, ROOT's other spelling of a variable - and ``x[3]``
    is no variable at all, which the refusal spells out as the three there are."""
    assert expand("x[1]*x[2]").text == "y*z"
    assert expand("x[0]+y").text == "x+y"  # x[0] is x itself
    with pytest.raises(FormulaError, match=r"x\[3\] in 'x\[3\]' is not a variable"):
        expand("x[3]")


def test_a_parameter_left_open_is_refused_where_it_opens():
    """A ``[`` with no ``]`` names no parameter, and the sentence gives the character it opened
    at, which is what a caller needs to find it in a long formula."""
    with pytest.raises(FormulaError, match="'\\[A\\*x' opens a parameter at character 0"):
        expand("[A*x")


def test_a_shape_s_bracket_left_open_is_refused_where_it_opens_too():
    """The same for a shape's ``(``: the argument list is scanned for its own closing bracket, and
    one that never comes is refused at the bracket's own character."""
    with pytest.raises(FormulaError, match=r"'gaus\(0' opens a bracket at character 4"):
        expand("gaus(0")


def test_a_numbered_parameter_and_a_parameter_named_for_the_same_index_are_one_parameter():
    """``[0]`` and ``[p0]`` name the same slot: ROOT calls the first parameter ``p0`` when it is
    written by number, so a formula writing both has one parameter and not two."""
    found = expand("[0]*[p0]")
    assert found.text == "[p0]*[p0]"
    assert found.names == ("p0",)


def test_a_formula_that_is_one_shape_alone_gets_root_s_own_names():
    """ROOT names the parameters of a formula that is nothing but a shape - ``Constant``, ``Mean``,
    ``Sigma`` for a Gaussian - and marks it predefined, which is what a fit's parameter order and
    the closed-form integral both hang off."""
    found = expand("gaus")
    assert found.names == ("Constant", "Mean", "Sigma")
    assert found.predefined
    assert "[Constant]*exp(-0.5*((x-[Mean])/[Sigma])*((x-[Mean])/[Sigma]))" == found.text


def test_a_formula_read_from_a_file_keeps_the_names_it_came_with():
    """A file's text already carries its names, so the words are left as they are and the names
    are the file's own - ROOT wrote them, and reading them back must not rename anything."""
    written = "[p0]*exp(-0.5*((x-[p1])/[p2])*((x-[p1])/[p2]))"
    found = expand(written, ("Constant", "Mean", "Sigma"))
    assert found.text == written  # the words are the file's, not expanded again
    assert found.names == ("Constant", "Mean", "Sigma")
    assert not found.predefined  # nothing was recognised: the shape's word is gone


def test_a_linear_formula_multiplies_each_term_by_a_parameter_of_its_own():
    """``a++b`` is ROOT's linear formula - what ``TH1::Fit`` fits by linear least squares - and it
    is written out as each term times its own parameter, which is how it is recognised later."""
    found = expand("sin(x)++x")
    assert found.text == "([p0]*(sin(x)))+([p1]*(x))"
    assert found.names == ("p0", "p1")
    assert found.linear


def test_the_number_a_shape_carries_is_root_s_own_and_a_series_is_three_hundred_and_its_degree():
    """``TFormula::GetNumber`` is 100 for a Gaussian, 200 for an exponential, 400 for a Landau and
    500 for a crystal ball; a ``polN`` is 300 plus its degree, and no shape at all is 0."""
    assert number("gaus") == 100
    assert number("xyexpo") == 210  # the two-variable spelling is ten past the one-variable
    assert number("landau") == 400
    assert number("crystalball") == 500
    assert number("pol3") == 303
    assert number(None) == 0


def test_a_formula_is_recognised_as_the_shape_its_text_is_written_out_from():
    """A shape's word is lost when ROOT writes a formula back, so what it was is worked out by
    writing each shape out under the same parameter names and comparing - which is the only way a
    formula read from a file can still be integrated in closed form."""
    gaussian = expand("gaus")
    assert recognised(gaussian.text, gaussian.names) == "gaus"
    assert recognised("x+1", ("a",)) is None  # no shape's text is this one
    polynomial = expand("pol2")
    assert recognised(polynomial.text, polynomial.names) == "pol2"
