"""Which of ROOT's own classes come back as a Python class, and which one.

A file describes a histogram the way it describes any other class, and read
member by member that is what it would be: a dictionary. For the classes
analysis actually handles - histograms, profiles, graphs, efficiencies,
sparse histograms, the lists that draw several at once, entry lists, the
functions a fit is made with, and canvases with the pads and drawing
classes they hold -
that dictionary is handed to the class that knows what the members mean.
This is the one table saying which, used for a key of a file and for an
object met inside another alike, so the two can never disagree.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .canvas import CANVASES
from .efficiency import EFFICIENCIES, Efficiency
from .entries import ENTRY_LISTS, EntryList
from .function import FUNCTIONS
from .function.function import dress as function
from .graph import GRAPHS, Graph
from .hist import HISTOGRAMS, Histogram
from .profile import PROFILES, Profile
from .sparse import SPARSE, SparseHistogram
from .stacks import COLLECTIONS

__all__ = ["CLASSES", "dress"]

#: Every class that comes back as more than a dictionary, and what it comes
#: back as; each is called with the class name and the members.
CLASSES: dict[str, Callable[[str, dict[str, Any]], Any]] = {
    **dict.fromkeys(HISTOGRAMS, Histogram),
    **dict.fromkeys(PROFILES, Profile),
    **dict.fromkeys(GRAPHS, Graph),
    **dict.fromkeys(EFFICIENCIES, Efficiency),
    **dict.fromkeys(SPARSE, SparseHistogram),
    **dict.fromkeys(ENTRY_LISTS, EntryList),
    **dict.fromkeys(FUNCTIONS, function),
    **COLLECTIONS,
    **CANVASES,
}


def dress(classname: str, value: Any) -> Any:
    """``value`` as the class ``classname`` comes back as, or as it is."""
    made = CLASSES.get(classname)
    if made is None or not isinstance(value, dict):
        return value
    return made(classname, value)
