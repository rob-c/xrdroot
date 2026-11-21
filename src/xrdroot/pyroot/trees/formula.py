"""``TTreeFormula``: ROOT's expression of a tree's branches, read an entry at a time.

    >>> tf = TTreeFormula("tf", "px*px + py*py", tree)   # doctest: +SKIP
    >>> tree.GetEntry(0)                                # doctest: +SKIP
    >>> tf.EvalInstance(0)                              # doctest: +SKIP
    6.25

ROOT writes a ``TTreeFormula`` wherever a tree is asked a question - ``Draw``
and ``Scan`` make one per expression - and a macro may make its own to ask one
directly. It is compiled once, against the branches of one tree, and then
evaluated for the entry the tree last read: ``EvalInstance(instance)`` is one
value of that entry, ``instance`` choosing among the elements of a branch that
holds many, and ``GetNdata()`` says how many there are.

The expression is xrdroot's own - :mod:`xrdroot.formula`, the language
``Draw`` is written in - evaluated over the columns that entry holds, so what
this class adds is ROOT's way of asking: one entry at a time, with an instance
number, and ROOT's answer when there is none - ``nan`` past the last element,
as ROOT's dynamic arrays run out.

A formula that does not compile is ROOT's, too: ``TTreeFormula::Compile``
reports the expression and the formula stays invalid, which is why
``GetNdim()`` is 0 for it and every value 0. A macro checks exactly that.
:func:`registered_function` is the other way round - a formula a macro named
``myfunc()`` and added to ``gROOT``'s list of functions, which any other
formula may then call.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ...formula import Formula, FormulaError, compile_formula
from ...formula.functions import Function
from ...formula.nodes import Reduce, Ref, Size, Special
from ...formula.select import TreeNames, one_entry
from ...jagged import Jagged
from ..core.messages import Error
from ..core.objects import TNamed

__all__ = ["TTreeFormula", "refused_expression", "registered_function"]

#: What this class does not answer, and what to do instead, in ROOT's terms: the
#: names ROOT's ``TTreeFormula`` has and xrdroot's trees cannot give an answer for.
REFUSED = {
    "EvalObject": (
        "TTreeFormula.EvalObject hands back the C++ object an expression names, and "
        "xrdroot's trees read values out of the objects they hold rather than the objects "
        "themselves; read the branch as a value with EvalInstance, or name the member of "
        "the object in the expression"
    ),
    "EvalClass": (
        "TTreeFormula.EvalClass names the class of the object an expression hands back, "
        "which is EvalObject's question and not answered here; read the branch as a value "
        "with EvalInstance instead"
    ),
    "EvalStringInstance": (
        "TTreeFormula.EvalStringInstance gives a string branch's value as a C string, and "
        "xrdroot hands strings back as Python strings; read the column with Draw or Scan, "
        "or compare it in a selection"
    ),
    "Print": (
        "TTreeFormula.Print prints ROOT's compiled expression - every fExpr entry with its "
        "action code - and xrdroot evaluates a tree expression from its own parse tree, "
        "which has no such table to print; Draw or Scan the expression to see what it does"
    ),
}


class TTreeFormula(TNamed):
    """``TTreeFormula``: see the module's docstring."""

    CLASS_TITLE = "The Tree formula"

    #: What a formula adds to a branch name to be called by it, as ``myfunc()`` is.
    CALLED_AS = "()"

    def __init__(self, name: Any = "", formula: Any = "", tree: Any = None) -> None:
        super().__init__(str(name), str(formula))
        if tree is None:
            raise TypeError(
                "a TTreeFormula is an expression over one tree's branches and was given no "
                "tree; give it as the third argument, TTreeFormula(name, formula, tree)"
            )
        self._tree = tree
        #: The names the tree offers, members of its whole objects included, which is
        #: what an expression is resolved against and what one entry is read through.
        self._names = TreeNames(tree._backing())
        self._expression: Formula | None = None
        self._cache: tuple[int, Any, Any] | None = None
        self._axis: Any = None
        try:
            self._expression = compile_formula(
                str(formula), self._names, aliases=getattr(tree, "_aliases", None) or None
            )
        except FormulaError as refused:
            # ROOT's own: Compile reports the expression and leaves the formula invalid,
            # which the macro goes on to ask about rather than catching an exception for.
            refused_expression(refused, str(formula))

    def __getattr__(self, name: str) -> Any:
        """A method ROOT has and this class has not: refused in a sentence, for those
        that are ROOT's own questions about the C++ objects a formula names."""
        if name in REFUSED:
            _refuse(name)
        raise AttributeError(f"ROOT's TTreeFormula has {name}, or does not; xrdroot's does not")

    def Print(self, option: Any = "") -> None:
        """``Print``: ROOT's dump of its compiled expression, which is refused here."""
        _refuse("Print")

    # -- the expression -------------------------------------------------------------

    def GetNdim(self) -> int:
        """How many dimensions the formula has: ROOT's 1 for an expression, 0 if it did not
        compile."""
        return 0 if self._expression is None else 1

    def GetNpar(self) -> int:
        """How many parameters it has, which a tree's expression never has."""
        return 0

    def GetNcodes(self) -> int:
        """How many branches it reads; a formula that did not compile reads none, where
        ROOT counts the leaves it found before it gave up."""
        return 0 if self._expression is None else len(self._expression.branches)

    def GetTree(self) -> Any:
        return self._tree

    def SetTree(self, tree: Any) -> None:
        """Read another tree of the same branches: the expression is kept, the values not."""
        self._tree = tree
        self._cache = None

    def SetAxis(self, axis: Any = None) -> None:
        """The axis a string formula bins its categories in, which xrdroot has no use for."""
        self._axis = axis

    def UpdateFormulaLeaves(self) -> None:
        """Reorder the leaves after a tree changed under the formula: nothing is cached here
        but the entry's values, which the next entry taken throws away."""

    Notify = UpdateFormulaLeaves

    def ResetLoading(self) -> None:
        """Forget the entry read, which the next question reads again."""

    def IsQuickLoad(self) -> bool:
        """ROOT may reload a branch only when the entry changed; here the entry is always
        read when it is asked for, so a formula is never quicker than that."""
        return False

    def SetQuickLoad(self, quick: bool) -> None:
        """``SetQuickLoad``: honoured in that the answer above stays false."""

    # -- the values ---------------------------------------------------------------

    def GetNdata(self) -> int:
        """How many values the formula has for the entry the tree last read: the elements of
        a branch that holds many, and one for anything else."""
        if self._expression is None:
            return 1
        values, _ = self._current()
        return _row(values) if len(values) else 1

    def GetMultiplicity(self) -> int:
        """ROOT's indicator of variability: 1 when the formula gives a value per element of a
        collection, 0 when it gives one per entry."""
        if self._expression is None:
            return 0
        values, _ = self._current()
        return int(isinstance(values, Jagged))

    def IsInteger(self, fast: bool = True) -> bool:
        """Is the formula one integer the tree holds - a leaf, an indexed leaf, ``Entry$``?

        ROOT keeps a bit for that, set when the formula is a value of the tree's own and
        not an arithmetic expression over it, which is what this asks: ``values[0]`` is
        an integer, and so is ``Entry$``, while ``values[0]+1`` is a double in ROOT
        whatever it is made of.
        """
        return self._whole_value() and _kind(self._current()[0]) in "iub"

    def IsString(self) -> bool:
        """Is the formula a string the tree holds, rather than a number?"""
        return self._whole_value() and _kind(self._current()[0]) in "US"

    def _whole_value(self) -> bool:
        """Is the whole expression one value the tree holds: no operation on it at all?"""
        if self._expression is None:
            return False
        root = self._expression.root
        if isinstance(root, (Ref, Size, Special)):
            return True
        return isinstance(root, Reduce) and root.kind == "Length$"

    def _current(self) -> tuple[Any, Any]:
        """The formula's values and mask for the entry the tree last read, read once.

        ROOT reads entry 0 for a formula asked before any entry has been, and so does this.
        """
        assert self._expression is not None
        entry = max(int(self._tree.GetReadEntry()), 0)
        if self._cache is None or self._cache[0] != entry:
            found = one_entry(self._tree._backing(), self._expression, entry, self._names.whole)
            self._cache = (entry, found[0], found[1])
        return self._cache[1], self._cache[2]

    def EvalInstance(self, instance: Any = 0, stringStack: Any = None) -> float:
        """The value of instance ``instance`` for the entry the tree last read.

        Past the last element of the entry - which is what a dynamic array indexed by a
        count that ran out asks for - there is no value, and ROOT's answer is ``nan``.
        An invalid formula has no value at all and gives 0, as ROOT's does.
        """
        if self._expression is None:
            return 0.0
        values, valid = self._current()
        if not len(values):
            return 0.0  # a tree with no entry read nothing, and ROOT's answer is its buffer's
        at = int(instance)
        if not isinstance(values, Jagged):
            return float(values[0]) if at == 0 and bool(valid[0]) else math.nan
        start, count = int(values.offsets[0]), _row(values)
        return (
            float(values.content[start + at]) if at < count and bool(valid.content[start + at])
            else math.nan
        )

    def EvalInstance64(self, instance: Any = 0, stringStack: Any = None) -> int:
        """``EvalInstance`` as a ``Long64_t``, as ROOT's integer instantiation gives it."""
        return int(self.EvalInstance(instance, stringStack))


def _refuse(what: str) -> None:
    """Refuse one of ROOT's own questions, in the sentence :data:`REFUSED` words it in."""
    from ...errors import UnsupportedFeatureError

    raise UnsupportedFeatureError(REFUSED[what])


def refused_expression(refused: FormulaError, text: str) -> None:
    """An expression ``TTreeFormula::Compile`` could not compile, reported as ROOT reports it.

    ROOT names the piece that failed to resolve - the leaf no branch has - and then the
    whole expression, when that piece was an index expression it compiled on its own:
    ``Bad numerical expression : "do_not_exist"``, then ``... : "values[do_not_exist]"``.
    A mistake anywhere else is one report, and it names the same piece.
    """
    if refused.name is None:
        Error("TTreeFormula::Compile", ' Bad numerical expression : "%s"', text)
        return
    Error("TTreeFormula::Compile", ' Bad numerical expression : "%s"', refused.name)
    if _in_index(text, refused.name):
        Error("TTreeFormula::Compile", ' Bad numerical expression : "%s"', text)


def _in_index(text: str, name: str) -> bool:
    """Is ``name`` written inside a bracket of ``text`` - an index ROOT compiles on its own?"""
    depth = 0
    for letter in text[: text.find(name)]:
        depth += 1 if letter == "[" else 0
        depth -= 1 if letter == "]" else 0
    return depth > 0


def _row(values: Any) -> int:
    """How many elements the one entry of a column of rows holds, or one for a single value."""
    if isinstance(values, Jagged):
        return int(values.offsets[1] - values.offsets[0])
    return 1


def _kind(values: Any) -> str:
    """The NumPy kind of a column's values - ``i``, ``f``, ``U`` - rows of them or not."""
    content = values.content if isinstance(values, Jagged) else values
    return str(np.asarray(content).dtype.kind)


def registered_function(name: str) -> Function | None:
    """The function a formula may call by ``name``, if ROOT's session registered one.

    A macro adds a ``TTreeFormula`` called ``myfunc()`` to ``gROOT``'s list of functions
    and then writes ``myfunc()`` inside another: ROOT finds it there by name and takes
    its value for the entry it is asked about. So this asks that list, and answers with
    a function of no arguments whose value is the registered formula's - the expression
    it is written in reads the tree's current entry, which is the one being evaluated.
    """
    from ..core.troot import gROOT

    listed = gROOT.GetListOfFunctions().FindObject(name + TTreeFormula.CALLED_AS)
    if not isinstance(listed, TTreeFormula):
        return None
    return Function(0, 0, lambda: np.asarray(listed.EvalInstance(0)))
