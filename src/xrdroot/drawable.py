"""``draw`` and ``scan``: what a tree and a chain share with ROOT's ``TTree::Draw`` and ``Scan``.

Both are written once, here, for anything that reads like a tree - its
``keys``, its branches by name and its length - so a :class:`~.tree.TTree`,
one with friends, and a :class:`~.chain.Chain` of many files all draw and
scan the same way. The work is in :mod:`~xrdroot.treedraw` and
:mod:`~xrdroot.scan`, imported when first asked for.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import IO, Any

__all__ = ["Drawable"]


class Drawable:
    """ROOT's ``Draw`` and ``Scan``, for a tree or a chain."""

    __slots__ = ()

    def draw(
        self,
        varexp: str,
        selection: str = "",
        option: str = "",
        *,
        entries: int | None = None,
        first_entry: int = 0,
        bins: Any = None,
        name: str | None = None,
        title: str | None = None,
        weight: Any = None,
        ax: Any = None,
        step: int | None = None,
        estimate: int | None = None,
        histograms: MutableMapping[str, Any] | None = None,
        aliases: Mapping[str, str] | None = None,
    ) -> Any:
        """``TTree::Draw``: a histogram, profile or graph filled from expressions.

            >>> h = tree.draw("jet_pt", "Sum$(jet_pt > 30) > 1")      # doctest: +SKIP
            >>> h2 = tree.draw("y:x>>h2(40, 0, 4, 40, -2, 2)", "w")    # doctest: +SKIP
            >>> p = tree.draw("pz:px", "", "prof")                      # doctest: +SKIP

        ``varexp`` is ``x``, ``y:x`` or ``z:y:x`` - the vertical axis first, as
        ROOT writes it - each part an expression over the branches, looped
        over the elements of a collection. The ``selection``'s value
        multiplies each fill's weight, so a cut keeps what it is true for
        and a number weighs it; ``weight`` multiplies it too, a number as
        ``TTree::SetWeight`` sets one or an expression per fill. A fill of
        weight zero is not made. ``entries`` and ``first_entry`` are the
        range ROOT's last two arguments give.

        What comes back is ROOT's: a ``TH1F``, ``TH2F`` or ``TH3F`` named
        ``htemp`` and titled with the expression and ``{selection}``; a
        ``TProfile`` or ``TProfile2D`` for option ``prof`` (``profs``,
        ``profi``, ``profg`` for its other errors); or, for a ``y:x`` whose
        option asks for points or lines (``p``, ``l``, ``*``), the scatter of
        points as a graph. Its ``selected`` is the number of fills made,
        which is what ``TTree::Draw`` returns.

        ``varexp`` may end in ``>>name`` to name what is filled, ``>>name(n,
        lo, hi, ...)`` to bin it, three numbers an axis from x, and ``>>+name``
        to add to the histogram of that name in ``histograms`` - a dict that
        stands for ROOT's ``gDirectory``, which the result is put in under
        its name. ``bins`` bins it instead: ``(n, lo, hi)``, a count, or
        edges, for each axis binned. Without either, the axes are ROOT's -
        100 bins, 40 a side in two dimensions, 20 in three - with their ends
        found from the first ``estimate`` fills (a million), after which an
        axis doubles to take in a value off its end, as ROOT's does.

        Nothing is drawn unless ``ax`` brings matplotlib axes to draw on, so
        ``goff`` is always in force; ``same`` does nothing. ``step`` entries
        are read at a time, which is what memory holds. ``aliases`` are
        names standing for expressions.
        """
        from .treedraw import ESTIMATE, draw

        return draw(
            self,
            varexp,
            selection,
            option,
            entries=entries,
            first_entry=first_entry,
            bins=bins,
            name=name,
            title=title,
            weight=weight,
            ax=ax,
            step=step,
            estimate=ESTIMATE if estimate is None else estimate,
            histograms=histograms,
            aliases=aliases,
        )

    def scan(
        self,
        varexp: str = "*",
        selection: str = "",
        *,
        entries: int | None = None,
        first_entry: int = 0,
        width: int | None = None,
        precision: int | None = None,
        file: IO[str] | None = None,
        aliases: Mapping[str, str] | None = None,
    ) -> str:
        """``TTree::Scan``: the values of expressions as ROOT's table of text.

            >>> print(tree.scan("Int32:Float64", "Int32 < 3"))       # doctest: +SKIP

        ``varexp`` is expressions separated by colons, ``"*"`` for every
        column this reads and ``""`` for the first eight. The ``selection``
        keeps the rows it is true for, and a line after the table says how
        many there were. ``width`` is ROOT's ``colsize=``, which fixes every
        column at that many characters; without it a column is nine, or as
        wide as its name up to twenty. ``precision`` is the digits a number
        is printed with, nine unless told.

        The table comes back as a string; ``file`` is also written to as the
        rows are made, which for a long scan is the way to see them come.
        ROOT's pause every fifty rows is not made.
        """
        from .scan import scan

        return scan(
            self,
            varexp,
            selection,
            entries=entries,
            first_entry=first_entry,
            width=width,
            precision=precision,
            file=file,
            aliases=aliases,
        )
