"""A frame's column of collections of objects: an ``RVec`` of a class, read and written.

A tree holds a ``std::vector<ROOT::Math::XYZTVector>`` as one branch per
coordinate - ``tracks.fCoordinates.fX`` and its three fellows - so what a
frame reads as one entry's collection has to be made again from them, and
what ``Snapshot`` writes has to be split back into them. This is the ROOT
layer's half of that: the engine hands it the class a branch holds and the
members of each entry, and gets back the objects, ROOT's branch declaration
for writing them, and the columns such a branch is filled from.

The engine reaches this through :func:`kinds`, declared as the provider of
``rdf.objects``. A class this layer knows nothing about stays what it is, a
dictionary of its members per entry.
"""

from __future__ import annotations

from typing import Any

from ...wbranch import Collection, column_keys
from ...wclasses import harvested
from ..trees.objects import LORENTZ, Lorentzes
from ..trees.rebuild import rebuilt

__all__ = ["kinds"]


def kinds(classname: str) -> FourVectors | None:
    """How a collection of ``classname`` is read and written, or ``None`` for a class this
    layer makes no objects of."""
    return FourVectors(classname) if _four_vectors(classname) else None


def _four_vectors(classname: str) -> bool:
    """Is this a collection of GenVector's Cartesian four-vectors?

    A branch spells the class out, ``vector<ROOT::Math::LorentzVector<
    ROOT::Math::PxPyPzE4D<double> > >``; a column a macro filled spells the
    typedef ``vector<ROOT::Math::XYZTVector>``.
    """
    return classname.startswith("vector<") and (
        "PxPyPzE4D" in classname or "XYZTVector" in classname
    )


class FourVectors:
    """A ``std::vector<ROOT::Math::XYZTVector>``: ROOT reads it as an ``RVec`` of the element
    class, and writes it with each coordinate in a branch of its own."""

    __slots__ = ("classname",)

    def __init__(self, classname: str) -> None:
        self.classname = classname

    def __getstate__(self) -> str:
        return self.classname  # a worker process is sent the class, which is all it holds

    def __setstate__(self, state: str) -> None:
        self.classname = state

    def column_type(self) -> str:
        """The C++ type of the column, an ``RVec`` of the class one element is."""
        element = self.classname[len("vector<") : -1]
        return f"ROOT::VecOps::RVec<{element}>"

    def read(self, rows: list[dict[str, Any]]) -> list[Any]:
        """Each entry's collection, made again from the coordinate columns under the branch."""
        return [rebuilt(self.classname, row, None) for row in rows]

    def pack(self, rows: list[Any]) -> list[list[tuple[float, ...]]]:
        """Each entry's collection as numbers, which a worker process can be sent."""
        return [[tuple(one._cartesian()) for one in row] for row in rows]

    def unpack(self, packed: list[list[tuple[float, ...]]]) -> list[Any]:
        """The collections those numbers were."""
        from ..core.genvector import XYZTVector

        return [[XYZTVector(*one) for one in row] for row in packed]

    def spec(self, name: str) -> Collection:
        """The branch ROOT declares for the column: the collection, split level and all."""
        layout = harvested(LORENTZ)
        assert layout is not None  # harvested from the 6.40 donor, see winfo
        return Collection(layout)

    def columns(self, name: str, values: Any) -> dict[str, Any]:
        """The columns a tree takes the collections from, by the names it gave them: each
        entry's length, and one column per coordinate."""
        kind = Lorentzes()
        keys = column_keys(name, self.spec(name).build(name, 32000))
        return kind.columns(keys, [kind.entry(each) for each in values])
