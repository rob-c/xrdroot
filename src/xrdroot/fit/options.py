"""ROOT's fit options: the letters ``TH1::Fit`` and ``TGraph::Fit`` read, read the same way.

``ROOT::Fit::FitOptionsMake`` upper-cases the string and asks whether it
*contains* each letter, so the order does not matter, spaces are nothing,
and ``"WL"`` is both ``W`` and ``L`` before the likelihood options turn it
into a weighted likelihood. This is that function: the same questions in
the same order, the same words - ``WIDTH``, ``NORMWIDTH``, ``MULTI``,
``SERIAL``, ``MULTITHREAD`` - taken out first so their letters do not count,
and the graph options - ``EX0``, ``ROB`` - read only for a graph. ``ROB``,
the robust linear fit, is refused by name. ``U`` - the FCN set on
``TVirtualFitter`` - does what ROOT does when no FCN was set there, which
is nothing: that is also how ``"DEBUG"``, which contains a U, still fits.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

from ..errors import UnsupportedFeatureError

__all__ = ["FitOptions", "parse"]


@dataclass
class FitOptions:
    """``Foption_t``: every option as the number ROOT gives it."""

    #: ``I``: the integral of the function over each bin, not its value at the centre.
    integral: int = 0
    #: ``W``: every error set to one - 2 for ``WW``, which takes the empty bins in too.
    w1: int = 0
    #: ``L``: 1 a Poisson likelihood, 2 weighted (``WL``), 4 multinomial, 6 both.
    like: int = 0
    #: ``P``: 1 Pearson's chi-square, 2 weighted (``PW``).
    pchi2: int = 0
    #: ``WIDTH`` 1, ``NORMWIDTH`` 2: the function scaled by the bin's width.
    bin_volume: int = 0
    #: ``E``: Hesse and Minos errors after the fit.
    errors: bool = False
    #: ``R``: the function's own range.
    range: bool = False
    #: ``G``: the function's gradient, rather than Minuit's numerical one.
    gradient: bool = False
    #: ``M``: another look for a better minimum.
    more: bool = False
    #: ``N``: the function not stored on what was fitted.
    nostore: bool = False
    #: ``0``: stored, but marked not to be drawn.
    nograph: bool = False
    #: ``+``: added to the functions already there, rather than replacing them.
    plus: bool = False
    #: ``B``: the parameters and limits given, not ROOT's guesses for a built-in shape.
    bound: bool = False
    #: ``C``: in ROOT, no chi-square for a linear fit; the chi-square is always had here.
    nochisq: bool = False
    #: ``F``: Minuit, even for a polynomial ROOT would fit by linear least squares.
    minuit: bool = False
    #: ``S``: the result as an object - which it always is here.
    store_result: bool = False
    #: ``Q`` quiet; ``V``, ``VV``, ``VVV`` more verbose.
    quiet: bool = False
    verbose: int = 0
    #: ``EX0`` (a graph): the x errors left out of the chi-square.
    no_x_errors: bool = False


def _width(opt: str, found: FitOptions) -> str:
    """``WIDTH`` and ``NORMWIDTH``, taken out of the string before its letters are read."""
    if "NORMWIDTH" in opt:
        found.bin_volume = 2
        return opt.replace("NORMWIDTH", "")
    if "WIDTH" in opt:
        found.bin_volume = 1
        return opt.replace("WIDTH", "")
    return opt


def _likelihood(opt: str, found: FitOptions, chi2: bool) -> str:
    """``L``'s variants: ``WL`` weighted, ``MULTI`` multinomial; ``P`` and ``X`` give way."""
    if "W" in opt:
        found.like, found.w1 = 2, 0
    if "MULTI" in opt:
        found.like = 6 if found.like == 2 else 4
        opt = opt.replace("MULTI", "")
    if chi2 or found.pchi2:
        warnings.warn(
            "Cannot use P or X option in combination of L. Ignore the chi2 option and "
            "perform a likelihood fit",
            RuntimeWarning,
            stacklevel=6,
        )
        found.pchi2 = 0
    return opt


def _histogram_letters(opt: str, found: FitOptions) -> str:
    """The options only a histogram has, in ``FitOptionsMake``'s order."""
    opt = _width(opt, found)
    found.integral = int("I" in opt)
    found.w1 = 2 if "WW" in opt else int("W" in opt)
    found.like = int("L" in opt)
    if "P" in opt:
        found.pchi2 = 2 if found.w1 else 1
        found.w1 = 0
    if found.like:
        opt = _likelihood(opt, found, "X" in opt)
    return opt


def _graph_letters(opt: str, found: FitOptions) -> str:
    """The options only a graph has: ``EX0``, and ``ROB``, which is refused."""
    opt = opt.replace("ROB", "H").replace("EX0", "T")
    if "H" in opt:
        raise UnsupportedFeatureError(
            "option ROB is TLinearFitter's robust fit, which drops the worst points by a "
            "least-trimmed-squares search; it is not here"
        )
    found.no_x_errors = "T" in opt
    found.w1 = int("W" in opt)
    return opt


#: The single letters every fit reads, and the member each sets.
FLAGS = (
    ("E", "errors"),
    ("R", "range"),
    ("G", "gradient"),
    ("M", "more"),
    ("N", "nostore"),
    ("0", "nograph"),
    ("+", "plus"),
    ("B", "bound"),
    ("C", "nochisq"),
    ("F", "minuit"),
    ("S", "store_result"),
)


def _verbosity(opt: str, found: FitOptions) -> None:
    """``V`` wins over ``Q``, as ROOT has it."""
    if "VVV" in opt or "DEBUG" in opt:
        found.verbose = 3
    elif "VV" in opt:
        found.verbose = 2
    elif "V" in opt:
        found.verbose = 1
    else:
        found.quiet = "Q" in opt


def parse(option: str, graph: bool = False) -> FitOptions:
    """``FitOptionsMake``: the options a string asks for, for a histogram or for a graph."""
    found = FitOptions()
    opt = (option or "").upper().replace("MULTITHREAD", "").replace("SERIAL", "")
    opt = _graph_letters(opt, found) if graph else _histogram_letters(opt, found)
    _verbosity(opt, found)
    for letter, name in FLAGS:
        if letter in opt:
            setattr(found, name, True)
    return found
