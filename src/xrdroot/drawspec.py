"""What a ``TTree::Draw`` call asks for, read out of its three strings.

``Draw("y:x>>h(40,0,4,20,-1,1)", "n > 2", "prof")`` is a small language of
its own. The expression is split at its colons into the axes - written from
the top down, so ``y:x`` is ``y`` against ``x`` - and whatever follows ``>>``
names the histogram to fill: ``>>h`` makes one called ``h``, ``>>+h`` adds to
one that is already there, and numbers in brackets bin it, three to an axis,
``x`` first. The option says what kind of thing to fill: ``prof`` makes a
profile of a two- or three-part expression, ``profs``, ``profi`` and
``profg`` a profile whose error is the spread, the spread of integers or
Gaussian, ``e`` keeps the squares of the weights from the start, ``norm``
scales the result to a sum of one, and a two-part expression drawn with an
option of points or lines - ``p``, ``l``, ``*`` - is the scatter of points
ROOT draws as a graph.

This reads all of that the way ``TSelectorDraw::Begin`` does, and refuses by
name what it does not mean here.
"""

from __future__ import annotations

from typing import NamedTuple

__all__ = ["Options", "Target", "options", "split_names", "target"]

#: The most numbers the brackets after ``>>name`` hold: three per axis.
MOST_BINNING = 9

#: Options that make ROOT draw something this does not make, and why not.
UNSUPPORTED = {
    "para": "parallel coordinates are a picture of the entries, not a histogram",
    "candle": "a candle plot is a picture of the entries, not a histogram",
    "gl5d": "a five-dimensional OpenGL picture needs ROOT's canvas",
    "entrylist": "filling an entry list is tree.arrays(..., cut=...) or an EntryList",
}

#: Drawing options that turn ROOT's scatter of a ``y:x`` draw back into a histogram.
BINNED = ("surf", "lego", "cont", "col", "hist", "scat", "box")


def split_names(varexp: str) -> list[str]:
    """``TSelectorDraw::SplitNames``: the parts of an expression, cut at its colons.

        >>> split_names("TMath::Abs(y):x > 0 ? x : -x")
        ['TMath::Abs(y)', 'x > 0 ? x : -x']

    A colon of C++'s ``::`` separates nothing, and nor does the one a ternary
    ``?`` is waiting for.
    """
    names: list[str] = []
    ternary, previous = False, 0
    padded = varexp + "\0"
    for at, char in enumerate(varexp):
        if char == "?":
            ternary = True
            continue
        if char != ":" or (at > 0 and varexp[at - 1] == ":") or padded[at + 1] == ":":
            continue
        if ternary:
            ternary = False
            continue
        names.append(varexp[previous:at])
        previous = at + 1
    names.append(varexp[previous:])
    return names


class Target(NamedTuple):
    """What the expression names to fill, after its ``>>``."""

    #: The expression before any ``>>``, as written.
    varexp: str
    #: The histogram's name, or ``None`` for ROOT's own ``htemp``.
    name: str | None
    #: ``>>+name``: add to the histogram of that name rather than start again.
    add: bool
    #: The numbers in brackets after the name, nine of them, ``None`` where not given.
    binning: tuple[float | None, ...] | None


def _number(text: str, whole: str) -> float | None:
    """One number of the brackets after ``>>name``; nothing written is none given."""
    text = text.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        raise ValueError(
            f"{text!r} in {whole!r} is not a number: the brackets after '>>name' hold the "
            f"number of bins and the two ends of each axis, as '>>h(100, 0, 10)'"
        ) from None


def _binning(name: str, parts: int) -> tuple[str, tuple[float | None, ...] | None]:
    """The name before the brackets, and the numbers in them."""
    if "(" not in name and ")" not in name:
        return name, None
    opened, closed = name.find("("), name.find(")")
    if name.count("(") != 1 or name.count(")") != 1 or closed < opened:
        raise ValueError(
            f"'>>{name}' should be a name and one pair of brackets, as '>>h(100, 0, 10)'"
        )
    numbers = [_number(text, name) for text in name[opened + 1 : closed].split(",")]
    if len(numbers) > min(MOST_BINNING, 3 * parts):
        raise ValueError(
            f"'>>{name}' gives {len(numbers)} numbers for {parts} "
            f"ax{'is' if parts == 1 else 'es'}; each axis takes three at most - the "
            f"number of bins and its two ends"
        )
    numbers += [None] * (MOST_BINNING - len(numbers))
    return name[:opened].strip(), tuple(numbers)


def target(varexp: str) -> Target:
    """Split ``"y:x>>+h(10,0,1)"`` into what to draw and what to fill."""
    at = varexp.rfind(">>")
    if at < 0:
        return Target(varexp, None, False, None)
    if at == 0:
        raise ValueError(
            f"{varexp!r} has nothing before its '>>': ROOT fills an entry list that way, "
            f"which here is tree.arrays(..., cut=...) or an EntryList"
        )
    name, drawn = varexp[at + 2 :].strip(), varexp[:at]
    add = name.startswith("+")
    name, binning = _binning(name.removeprefix("+").strip(), len(split_names(drawn)))
    if not name:
        raise ValueError(f"{varexp!r} names no histogram after its '>>'")
    return Target(drawn, name, add, binning)


class Options(NamedTuple):
    """What the option string asks for, once the drawing styles are set aside."""

    #: ``prof``, ``profs``, ``profi`` or ``profg``: a profile, and its error option.
    profile: str | None
    #: ``e``: keep the squares of the weights from the first fill.
    errors: bool
    #: ``norm``: scale what was filled to a sum of weights of one.
    norm: bool
    #: A two-part draw that ROOT shows as a scatter of points: a graph.
    graph: bool


def _profile(text: str, dimensions: int) -> str | None:
    if "prof" not in text or dimensions < 2:
        return None  # ROOT ignores "prof" for one expression
    for letter in "sig":
        if f"prof{letter}" in text:
            return letter
    return ""


def _scatter(text: str, dimensions: int, profile: str | None) -> bool:
    """``TSelectorDraw``'s test for a ``y:x`` shown as points, the empty option aside."""
    if dimensions != 2 or profile is not None:
        return False
    marked = any(mark in text for mark in "pl*")
    return marked and not any(style in text for style in BINNED)


def options(option: str, dimensions: int) -> Options:
    """The option string read as ``TSelectorDraw::Begin`` reads it.

    ``same`` and ``goff`` make no difference: nothing is drawn unless an
    ``ax`` is given, so every draw is ``goff``, and there is no pad for
    ``same`` to add to. The empty option is ``goff``'s too, so a ``y:x``
    draw with no option fills a histogram rather than make ROOT's scatter.
    """
    text = str(option).lower().replace("same", "")
    for word, why in UNSUPPORTED.items():
        if word in text:
            raise ValueError(f"option {word!r} is not drawn here: {why}")
    profile = _profile(text, dimensions)
    return Options(
        profile=profile,
        errors=dimensions == 1 and "e" in text,
        norm="norm" in text,
        graph=_scatter(text, dimensions, profile),
    )
