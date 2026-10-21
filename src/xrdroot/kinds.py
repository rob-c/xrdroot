"""Which of ROOT's own classes come back as a Python class, and which one.

A file describes a histogram the way it describes any other class, and read
member by member that is what it would be: a dictionary. For the classes
analysis actually handles - histograms, profiles, graphs, efficiencies,
sparse histograms, the lists that draw several at once, entry lists, the
functions a fit is made with, pictures, and canvases with the pads and drawing
classes they hold -
that dictionary is handed to the class that knows what the members mean.
This is the one table saying which, used for a key of a file and for an
object met inside another alike, so the two can never disagree.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from .efficiency import EFFICIENCIES, Efficiency
from .entries import ENTRY_LISTS, EntryList
from .function import FUNCTIONS
from .function.function import dress as function
from .graph import GRAPHS, Graph
from .hist import HISTOGRAMS, Histogram
from .image import IMAGES, Image
from .plugins import keys, lookup, lookup_loaded, require
from .profile import PROFILES, Profile
from .sparse import SPARSE, SparseHistogram
from .stacks import COLLECTIONS

__all__ = ["CLASSES", "dress"]

#: Every class that comes back as more than a dictionary, and what it comes
#: back as; each is called with the class name and the members.
CLASSES: dict[str, Callable[[str, dict[str, Any]], Any]]
_CLASSES: dict[str, Callable[[str, dict[str, Any]], Any]] = {
    **dict.fromkeys(HISTOGRAMS, Histogram),
    **dict.fromkeys(PROFILES, Profile),
    **dict.fromkeys(GRAPHS, Graph),
    **dict.fromkeys(EFFICIENCIES, Efficiency),
    **dict.fromkeys(SPARSE, SparseHistogram),
    **dict.fromkeys(ENTRY_LISTS, EntryList),
    **dict.fromkeys(FUNCTIONS, function),
    **COLLECTIONS,
    **dict.fromkeys(IMAGES, Image),
}


#: Image entries keep their original position after the lazy canvas entries.
_IMAGE_NAMES = tuple(IMAGES)
#: A published or observed public binding governs subsequent lookup, including deletion.
_canvas_materialized = False


def __getattr__(name: str) -> Any:
    """Materialize the public dictionary only when its complete values are requested."""
    global _canvas_materialized
    if name != "CLASSES" or _canvas_materialized:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    canvases = _canvas_classes()
    if canvases is None:
        canvases = {key: require("canvas.dress", key) for key in keys("canvas.dress")}
    images = {key: _CLASSES.pop(key) for key in _IMAGE_NAMES}
    _CLASSES.update(canvases)
    _CLASSES.update(images)
    _canvas_materialized = True
    globals()["CLASSES"] = _CLASSES
    return _CLASSES


def __dir__() -> list[str]:
    return sorted({*globals(), "CLASSES"})


def _canvas_classes() -> dict[str, Callable[[str, dict[str, Any]], Any]] | None:
    """Inspect caller changes only when the graphics registry has already been loaded."""
    current = lookup_loaded("canvas.classes")
    return None if current is None else cast(
        dict[str, Callable[[str, dict[str, Any]], Any]], current()
    )


def _active_classes() -> dict[str, Callable[[str, dict[str, Any]], Any]]:
    """An observed public binding remains authoritative, including its later deletion."""
    global _canvas_materialized
    current = globals()
    if "CLASSES" in current:
        _canvas_materialized = True
        return cast(dict[str, Callable[[str, dict[str, Any]], Any]], current["CLASSES"])
    if _canvas_materialized:
        raise NameError("name 'CLASSES' is not defined", name="CLASSES")
    return _CLASSES


def _known(classname: str) -> bool:
    """Recognize dressing targets without importing a higher component for membership."""
    if classname in _active_classes():
        return True
    if _canvas_materialized:
        return False
    canvases = _canvas_classes()
    return classname in keys("canvas.dress") if canvases is None else classname in canvases


def dress(classname: str, value: Any) -> Any:
    """``value`` as the class ``classname`` comes back as, or as it is."""
    made = _active_classes().get(classname)
    if made is None and not _canvas_materialized:
        canvases = _canvas_classes()
        made = lookup("canvas.dress", classname) if canvases is None else canvases.get(classname)
    if made is None or not isinstance(value, dict):
        return value
    return made(classname, value)
