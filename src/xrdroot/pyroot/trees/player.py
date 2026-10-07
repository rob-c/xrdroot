"""``Draw``, ``Scan``, ``Print`` and ``Show``: ROOT's ``TTreePlayer``, over xrdroot's.

``Draw`` and ``Scan`` are :meth:`xrdroot.TTree.draw` and ``scan`` with
ROOT's arguments - ``varexp, selection, option, nentries, firstentry`` - and
ROOT's return value, the number of entries (or fills) selected. What ``Draw``
fills is put in ``gDirectory`` under its name (``htemp`` unless ``>>name``
says otherwise), handed to the current pad unless the option says ``goff``,
and a ``varexp`` of ``>>name`` alone fills an entry list with the entries
the selection keeps, as ``TTree::Draw(">>elist", cut, "entrylist")`` does.
"""

from __future__ import annotations

import fnmatch
import re
import sys
from typing import Any

import numpy as np

from ...drawspec import split_names
from ._base import hooks
from .friends import _Friends
from .printing import element_lines, show_lines, tree_lines

__all__ = ["_Player", "MAX_ENTRIES"]

#: ``TTree::kMaxEntries``: the ``nentries`` that means all of them.
MAX_ENTRIES = 1_000_000_000_000_000_000
#: The settings ``Scan``'s option can carry.
SCAN_OPTIONS = re.compile(r"(colsize|precision)=(\d+)")


def _count(nentries: int) -> int | None:
    return None if nentries >= MAX_ENTRIES or nentries < 0 else int(nentries)


def _entries_before(varexp: str, registry: Any) -> float:
    """The entries of the histogram ``>>name`` fills, if it is there already - to count fills by."""
    from ...drawspec import target

    name = target(varexp).name
    held = registry.get(name) if name is not None else None
    return float(getattr(held, "entries", 0.0))


class _Player(_Friends):
    """What a tree draws, scans and prints."""

    def _view(self) -> Any:
        """What is drawn and scanned: every entry, or the ones the entry list names."""
        backing = self._backing()
        if self._entry_list is None:
            return backing
        from .copying import subset

        return subset(backing, self._entry_list._entries(), self._name or "tree")

    def Draw(  # type: ignore[override]  # TTree::Draw takes a tree's arguments, not TObject's
        self,
        varexp: str,
        selection: str = "",
        option: str = "",
        nentries: int = MAX_ENTRIES,
        firstentry: int = 0,
    ) -> int:
        """``TTree::Draw``: fill a histogram, profile or graph; how many were selected."""
        text = str(varexp).strip()
        selection, option = str(selection or ""), str(option or "")
        special = self._special_draw(text, selection, option, nentries, int(firstentry))
        if special is not None:
            return special
        registry = hooks.registry()
        before = _entries_before(text, registry)
        made = self._view().draw(
            text,
            selection,
            option,
            entries=_count(nentries),
            first_entry=int(firstentry),
            histograms=registry,
            aliases=self._aliases or None,
            weight=None if self._weight == 1 else self._weight,
            estimate=self._estimate,
        )
        if "goff" not in option.lower():
            hooks.draw(hooks.wrap(made), option)
        self._drawn = (text, selection, _count(nentries), int(firstentry))
        return int(getattr(made, "selected", made.entries - before))

    def _special_draw(
        self, text: str, selection: str, option: str, nentries: int, firstentry: int
    ) -> int | None:
        """``Draw(">>list")``, and ``Draw("hpx.GetRMS()")`` of methods of whole objects;
        ``None`` for any other expression."""
        if text.startswith(">>"):
            return self._draw_list(text[2:].strip(), selection, nentries, firstentry)
        from .methods import method_draw

        return method_draw(self, text, selection, option, _count(nentries), firstentry)

    def _drawn_values(self, axis: int) -> Any:
        """The values the last ``Draw`` computed for its ``axis``-th expression, as ``GetV1``
        and its kin hand them back: one per entry the selection kept."""
        drawn = getattr(self, "_drawn", None)
        if drawn is None:
            return None
        text, selection, count, first = drawn
        parts = split_names(text.split(">>")[0])
        if axis >= len(parts):
            return None
        stop = None if count is None else first + count
        found = [
            np.asarray(batch[parts[axis]], dtype=np.float64)
            for batch in self._view().iterate(
                [parts[axis]], entry_start=first, entry_stop=stop, cut=selection or None,
                aliases=self._aliases or None,
            )
        ]  # fmt: skip
        return np.concatenate(found) if found else np.zeros(0)

    def GetV1(self) -> Any:
        return self._drawn_values(0)

    def GetV2(self) -> Any:
        return self._drawn_values(1)

    def GetV3(self) -> Any:
        return self._drawn_values(2)

    def GetV4(self) -> Any:
        return self._drawn_values(3)

    def GetSelectedRows(self) -> int:
        values = self._drawn_values(0)
        return 0 if values is None else len(values)

    def _draw_list(self, name: str, selection: str, nentries: int, firstentry: int) -> int:
        """``Draw(">>elist", cut)``: the entries the cut keeps, as a ``TEntryList``."""
        from .entrylist import TEntryList

        adding = name.startswith("+")
        name = name.lstrip("+").strip()
        stop = None if _count(nentries) is None else firstentry + int(nentries)
        kept = [
            int(entry)
            for batch in self._view().iterate(
                ["Entry$"], entry_start=firstentry, entry_stop=stop, cut=selection or None
            )
            for entry in batch["Entry$"]
        ]
        if self._entry_list is not None:
            kept = [int(self._entry_list.GetEntry(entry)) for entry in kept]
        registry = hooks.registry()
        found = registry.get(name) if adding else None
        made = found if isinstance(found, TEntryList) else TEntryList(name, selection)
        made._enter_many(kept, self)
        registry[name] = made
        return len(kept)

    def Scan(
        self,
        varexp: str = "",
        selection: str = "",
        option: str = "",
        nentries: int = MAX_ENTRIES,
        firstentry: int = 0,
    ) -> int:
        """``TTree::Scan``: print ROOT's table of values; how many entries it selected."""
        settings = {key: int(value) for key, value in SCAN_OPTIONS.findall(str(option or ""))}
        text = self._view().scan(
            str(varexp or ""),
            str(selection or ""),
            entries=_count(nentries),
            first_entry=int(firstentry),
            width=settings.get("colsize"),
            precision=settings.get("precision"),
            file=sys.stdout,
            aliases=self._aliases or None,
        )
        found = re.search(r"==> (\d+) selected", text)
        if found is not None:
            return int(found.group(1))
        return len({line.split("*")[1] for line in text.splitlines()[3:-1] if "*" in line})

    def Print(self, option: str = "") -> None:
        """``TTree::Print``: the tree, then every branch, in ROOT's table."""
        layout = self._layout()
        record, packed = self._record(layout)
        lines = tree_lines(
            self._name, self._title, self.GetEntries(), layout, self._tree_key, record, packed
        )
        wanted = str(option or "")
        pattern = wanted if wanted and wanted not in ("all", "toponly") else "*"
        count = 0
        for branch in layout:
            if fnmatch.fnmatchcase(branch.name, pattern):
                more, count = element_lines(branch, count)
                lines.extend(more)
            else:
                count += len(branch.walk())
        print("\n".join(lines))

    def Show(self, entry: int = -1, lenmax: int = 20) -> None:
        """``TTree::Show``: every leaf's values in one entry - the last read, unless told."""
        if entry != -1:
            self.GetEntry(int(entry))
        at = max(self._read_entry, 0)
        self._read_entry = at
        leaves = [
            (leaf, self._batch.value(leaf.column, at))
            for leaf in self._leaves()
            if self._active(leaf.column)
        ]
        print("\n".join(show_lines(at, leaves)))

    def SetAlias(self, alias: str, expression: str) -> bool:
        self._aliases[alias] = expression
        return True

    def GetAlias(self, alias: str) -> str | None:
        return self._aliases.get(alias)

    def SetWeight(self, weight: float = 1.0, option: str = "") -> None:
        self._weight = float(weight)

    def GetWeight(self) -> float:
        return self._weight

    def SetEstimate(self, estimate: int = 1_000_000) -> None:
        self._estimate = int(estimate) if estimate > 0 else self._estimate

    def GetEstimate(self) -> int:
        return self._estimate

    def SetScanField(self, count: int = 50) -> None:
        """How many rows ``Scan`` pauses after in ROOT; a script never waits here."""
