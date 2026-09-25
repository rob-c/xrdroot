"""ROOT's predefined shapes, written out as the arithmetic they stand for.

``TF1("f", "gaus")`` is a Gaussian because ``TFormula`` rewrites the word
before it parses anything: ``gaus`` becomes
``[0]*exp(-0.5*((x-[1])/[2])*((x-[1])/[2]))``, ``pol2`` becomes
``([0]+[1]*x+[2]*TMath::Sq(x))``, and so on through the table ROOT fills in
``TFormula::FillParametrizedFunctions``. This does the same rewriting, with
ROOT's bodies, so the formula that is kept - and written back to a file -
is the one ROOT would keep.

A shape may say where its parameters start, ``gaus(3)``, which variable it
is in, ``pol1(y, 0)``, or what its parameters are called,
``pol1(x, [A], [B])``. ``xygaus``, ``xyexpo`` and the other old spellings
name their variables in the shape's own name. A formula that is nothing but
one shape - ``gaus``, ``expo``, ``landau`` - gets ROOT's names for its
parameters, ``Constant``, ``Mean``, ``Sigma`` and the rest; every other
parameter written by number is called ``p`` and its number, as ROOT calls
it.
"""

from __future__ import annotations

import re

from ..errors import UnsupportedFeatureError
from ..formula.errors import FormulaError

__all__ = ["Expanded", "expand", "param_order"]

#: The body of each shape, by name and dimension: ``{V0}`` and the rest are
#: where the variables go, ``[k]`` its k-th parameter.
BODIES: dict[tuple[str, int], str] = {
    ("gaus", 1): "[0]*exp(-0.5*(({V0}-[1])/[2])*(({V0}-[1])/[2]))",
    ("gausn", 1): "[0]*exp(-0.5*(({V0}-[1])/[2])*(({V0}-[1])/[2]))/(sqrt(2*pi)*[2])",
    ("gaus", 2): "[0]*exp(-0.5*(({V0}-[1])/[2])^2-0.5*(({V1}-[3])/[4])^2)",
    ("gaus", 3): "[0]*exp(-0.5*(({V0}-[1])/[2])^2-0.5*(({V1}-[3])/[4])^2-0.5*(({V2}-[5])/[6])^2)",
    ("landau", 1): "[0]*TMath::Landau({V0},[1],[2],false)",
    ("landaun", 1): "[0]*TMath::Landau({V0},[1],[2],true)",
    ("landau", 2): "[0]*TMath::Landau({V0},[1],[2],false)*TMath::Landau({V1},[3],[4],false)",
    ("landaun", 2): "TMath::Landau({V0},[0],[1],true)*TMath::Landau({V1},[2],[3],true)",
    ("expo", 1): "exp([0]+[1]*{V0})",
    ("expo", 2): "exp([0]+[1]*{V0}+[2]*{V1})",
    ("crystalball", 1): "[0]*ROOT::Math::crystalball_function({V0},[3],[4],[2],[1])",
    ("crystalballn", 1): "[0]*ROOT::Math::crystalball_pdf({V0},[3],[4],[2],[1])",
    ("breitwigner", 1): "[0]*ROOT::Math::breitwigner_pdf({V0},[2],[1])",
    ("bigaus", 2): "[0]*ROOT::Math::bigaussian_pdf({V0},{V1},[2],[4],[5],[1],[3])",
}

#: The names ROOT gives the parameters of a formula that is one shape alone.
NAMES: dict[tuple[str, int], tuple[str, ...]] = {
    ("gaus", 1): ("Constant", "Mean", "Sigma"),
    ("gausn", 1): ("Constant", "Mean", "Sigma"),
    ("gaus", 2): ("Constant", "MeanX", "SigmaX", "MeanY", "SigmaY"),
    ("bigaus", 2): ("Constant", "MeanX", "SigmaX", "MeanY", "SigmaY", "Rho"),
    ("expo", 1): ("Constant", "Slope"),
    ("landau", 1): ("Constant", "MPV", "Sigma"),
    ("landaun", 1): ("Constant", "MPV", "Sigma"),
    ("crystalball", 1): ("Constant", "Mean", "Sigma", "Alpha", "N"),
    ("crystalballn", 1): ("Constant", "Mean", "Sigma", "Alpha", "N"),
    ("breitwigner", 1): ("Constant", "Mean", "Gamma"),
}

#: The older spellings that name their variables, and what they are.
ALIASES: dict[str, tuple[str, tuple[str, ...]]] = {
    "xgaus": ("gaus", ("x",)),
    "ygaus": ("gaus", ("y",)),
    "zgaus": ("gaus", ("z",)),
    "xygaus": ("gaus", ("x", "y")),
    "xyzgaus": ("gaus", ("x", "y", "z")),
    "xexpo": ("expo", ("x",)),
    "yexpo": ("expo", ("y",)),
    "zexpo": ("expo", ("z",)),
    "xyexpo": ("expo", ("x", "y")),
    "xylandau": ("landau", ("x", "y")),
    "xylandaun": ("landaun", ("x", "y")),
    "bigaus": ("bigaus", ("x", "y")),
}

#: ``polN`` and ``chebN``, with the degree.
SERIES = re.compile(r"(pol|cheb)(\d+)$")
#: The highest degree ROOT defines a Chebyshev series for.
CHEB_MAX = 10
#: The words of a formula the scan steps through whole.
WORD = re.compile(r"(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?|[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*")
#: A parameter, numbered or named.
LABEL = re.compile(r"\[([^\[\]]*)\]")
#: A coordinate by number, as ``x[1]`` is ``y``.
INDEXED = re.compile(r"\[(\d+)\]")
#: What a variable is called in a shape's arguments.
VARIABLE_NAMES = ("x", "y", "z")


class Expanded:
    """A formula with its shapes written out, and its parameters named in order."""

    __slots__ = ("text", "names", "predefined")

    def __init__(self, text: str, names: tuple[str, ...], predefined: bool) -> None:
        #: The formula as ROOT keeps it: shapes expanded, parameters by name.
        self.text = text
        #: Each parameter's name, in the order of its index.
        self.names = names
        #: Whether this is one predefined shape alone, named as ROOT names those.
        self.predefined = predefined


def _shape(word: str) -> tuple[str, tuple[str, ...]] | None:
    """The shape a word names, and the variables its name gives it, if it is one."""
    if word in ALIASES:
        return ALIASES[word]
    series = SERIES.match(word)
    if series is not None or (word, 1) in BODIES:
        return word, ()
    return None


def _closing(text: str, at: int) -> int:
    """Where the bracket opened at ``at`` closes."""
    depth = 0
    for index in range(at, len(text)):
        depth += {"(": 1, ")": -1}.get(text[index], 0)
        if depth == 0:
            return index
    raise FormulaError(f"{text!r} opens a bracket at character {at} and never closes it")


def _split(inside: str) -> list[str]:
    """The arguments between a shape's brackets, split at the commas that separate them."""
    args, depth, start = [], 0, 0
    for index, char in enumerate(inside):
        depth += {"(": 1, ")": -1}.get(char, 0)
        if char == "," and depth == 0:
            args.append(inside[start:index])
            start = index + 1
    args.append(inside[start:])
    return [arg for arg in args if arg] if inside else []


def _parameters(shape: str, args: list[str]) -> tuple[int, list[str]]:
    """Where a shape's parameters start, or the names it gives them."""
    if not args:
        return 0, []
    if len(args) == 1 and args[0].isdigit():
        return int(args[0]), []
    if all(LABEL.fullmatch(arg) for arg in args):
        return 0, args
    raise FormulaError(
        f"{shape}({','.join(args)}) is not a shape ROOT reads: a shape takes where its "
        f"parameters start, gaus(3), its variable, pol1(y,0), or its parameters, "
        f"pol1(x,[A],[B]); a function of its own is TMath::Gaus(x,mean,sigma)"
    )


def _variables(
    shape: str, named: tuple[str, ...], args: list[str]
) -> tuple[tuple[str, ...], list[str]]:
    if args and args[0] in VARIABLE_NAMES and not named:
        return (args[0],), args[1:]
    return named or ("x",), args


def _numbered(offset: int, names: list[str], count: int) -> list[str]:
    """How each of a shape's parameters is written, by position in the shape."""
    return names or [f"[{offset + k}]" for k in range(count)]


def _series(kind: str, degree: int, variable: str, labels: list[str]) -> str:
    if kind == "pol":
        terms = [labels[0]]
        for power in range(1, degree + 1):
            spelled = {1: variable, 2: f"TMath::Sq({variable})"}.get(
                power, f"TMath::Power({variable},{power})"
            )
            terms.append(f"{labels[power]}*{spelled}")
        return "(" + "+".join(terms) + ")"
    if degree > CHEB_MAX:
        raise UnsupportedFeatureError(
            f"cheb{degree} is past cheb{CHEB_MAX}, the highest Chebyshev series ROOT defines"
        )
    return f"ROOT::Math::Chebyshev{degree}({variable},{','.join(labels[: degree + 1])})"


def _labelled(shape: str, labels: list[str], count: int) -> list[str]:
    """The labels for a shape of ``count`` parameters, refusing too few names."""
    if len(labels) < count:
        raise FormulaError(
            f"{shape} has {count} parameters, and {len(labels)} names were given for them"
        )
    return labels


def _body(shape: str, variables: tuple[str, ...], offset: int, names: list[str]) -> str:
    series = SERIES.match(shape)
    if series is not None:
        degree = int(series.group(2))
        labels = _labelled(shape, _numbered(offset, names, degree + 1), degree + 1)
        return _series(series.group(1), degree, variables[0], labels)
    body = BODIES[shape, len(variables)]
    for index, variable in enumerate(variables):
        body = body.replace(f"{{V{index}}}", variable)
    count = max(int(label) for label in LABEL.findall(body)) + 1
    labels = _labelled(shape, _numbered(offset, names, count), count)
    return "(" + LABEL.sub(lambda found: labels[int(found.group(1))], body) + ")"


def _expand_one(
    text: str, at: int, word: str, shape: tuple[str, tuple[str, ...]]
) -> tuple[str, int]:
    """One shape at ``at``: what it stands for, and where the text carries on.

    A shape inside a longer formula is bracketed, so ``x/gaus`` divides by
    the whole Gaussian; one that is the whole formula is written as ROOT
    writes it, without.
    """
    end = at + len(word)
    args: list[str] = []
    if end < len(text) and text[end] == "(":
        close = _closing(text, end)
        args, end = _split(text[end + 1 : close]), close + 1
    variables, rest = _variables(shape[0], shape[1], args)
    offset, names = _parameters(shape[0], rest)
    body = _body(shape[0], variables, offset, names)
    whole = at == 0 and end == len(text) and not SERIES.match(shape[0])
    return (body[1:-1] if whole else body), end


def _coordinate(text: str, found: re.Match[str]) -> tuple[str, int]:
    """A word as it stands - except ``x[0]``, ``x[1]`` and ``x[2]``, which are
    ``x``, ``y`` and ``z`` and not parameters, and are written so."""
    index = INDEXED.match(text, found.end())
    if found.group() != "x" or index is None:
        return found.group(), found.end()
    number = int(index.group(1))
    if number >= len(VARIABLE_NAMES):
        raise FormulaError(
            f"x[{number}] in {text!r} is not a variable: x[0], x[1] and x[2] are x, y and z"
        )
    return VARIABLE_NAMES[number], index.end()


def _scan(text: str) -> str:
    """The text with every shape written out."""
    out: list[str] = []
    at = 0
    while at < len(text):
        if text[at] == "[":
            close = text.find("]", at)
            if close < 0:
                raise FormulaError(
                    f"{text!r} opens a parameter at character {at} and never closes it"
                )
            out.append(text[at : close + 1])
            at = close + 1
            continue
        found = WORD.match(text, at)
        if found is None:
            out.append(text[at])
            at += 1
            continue
        word = found.group()
        shape = _shape(word)
        if shape is None:
            spelled, at = _coordinate(text, found)
            out.append(spelled)
            continue
        body, at = _expand_one(text, at, word, shape)
        out.append(body)
    return "".join(out)


def param_order(name: str) -> tuple[int, int, str]:
    """``TFormulaParamOrder``: ``p2`` before ``p10``, numbered before named, then by name."""
    numeric = re.fullmatch(r"p?(\d+)", name)
    if numeric is not None:
        return 0, int(numeric.group(1)), ""
    return 1, 0, name


def _numbers(labels: list[str]) -> list[int]:
    """The parameters written by number, each once, in order."""
    return sorted({int(label) for label in labels if label.isdigit()})


def _words(labels: list[str]) -> list[str]:
    """The parameters written by name, each once, in the order they are first met."""
    return list(dict.fromkeys(label for label in labels if not label.isdigit()))


def _indices(text: str) -> dict[str, int]:
    """Each parameter's index: its number, or the next free one, in order of first use."""
    labels = [label.strip() for label in LABEL.findall(text)]
    numbered = _numbers(labels)
    found: dict[str, int] = {f"p{number}": number for number in numbered}
    free = iter(index for index in range(len(labels) + 1) if index not in numbered)
    for label in _words(labels):
        if label not in found:
            found[label] = next(free)
    return found


def _named(indices: dict[str, int]) -> list[str]:
    """Every parameter's name by index, a gap in the numbering filled as ``p`` and its number."""
    count = max(indices.values(), default=-1) + 1
    names = [f"p{index}" for index in range(count)]
    for label, index in indices.items():
        names[index] = label
    return names


def _predefined(text: str, count: int) -> tuple[str, ...] | None:
    """ROOT's names for the parameters of a formula that is one shape and nothing else."""
    single = _shape(text)
    if single is None:
        return None
    special = NAMES.get((single[0], max(1, len(single[1]))))
    return special if special is not None and len(special) == count else None


def expand(formula: str, known: tuple[str, ...] | None = None) -> Expanded:
    """``formula`` with its shapes written out and its parameters named as ROOT names them.

    ``known`` is the names a formula read from a file came with, which it
    keeps: ROOT wrote the text with those names already in it.
    """
    text = re.sub(r"\s+", "", formula)
    written = _scan(text)
    names = _named(_indices(written))
    special = _predefined(text, len(names)) if known is None else None
    if special is not None:
        names = list(special)
    if known is not None:
        names = list(known) + names[len(known) :]

    def spelled(found: re.Match[str]) -> str:
        label = found.group(1).strip()
        return f"[{names[int(label)] if label.isdigit() else label}]"

    return Expanded(LABEL.sub(spelled, written), tuple(names), special is not None)
