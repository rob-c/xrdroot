"""ROOT 7's graphics: ``RCanvas``, its pads and frames, the primitives on them, and styles.

A canvas is a scene - its primitives in the order they were drawn, each
with its attributes - which ROOT draws in a web browser. Run in batch,
as xrdroot runs, there is no browser: the scene is built and kept as
ROOT's is, and what would show it, ``Show`` and ``SaveAs``, does what
ROOT's batch mode does without one (:mod:`.pads` says how). The names
are ``ROOT::Experimental``'s, and are also the namespace's own, which is
where a macro's ``using namespace ROOT::Experimental`` finds them.
"""

from __future__ import annotations

from typing import Any

from ...errors import UnsupportedFeatureError
from .attrs import RColor
from .axes import RAttrAxis, RAttrAxisLabels, RAttrAxisTicks, RAttrAxisTitle, RAttrLineEnding
from .drawables import (
    RAxisDrawable,
    RBox,
    RFont,
    RFrameTitle,
    RLine,
    RMarker,
    RPave,
    RPaveText,
    RText,
    TObjectDrawable,
)
from .kinds import (
    RAttrBorder,
    RAttrFill,
    RAttrFont,
    RAttrLine,
    RAttrMargins,
    RAttrMarker,
    RAttrText,
)
from .lengths import RPadExtent, RPadLength, RPadPos
from .pads import RCanvas, RFrame, RPad
from .style import RStyle

__all__ = [
    "RCanvas", "RPad", "RFrame", "RStyle", "RColor", "RPadLength", "RPadPos", "RPadExtent",
    "RLine", "RBox", "RText", "RMarker", "RPave", "RPaveText", "RFrameTitle", "RAxisDrawable",
    "TObjectDrawable", "RFont", "RAttrLine", "RAttrFill", "RAttrBorder", "RAttrFont",
    "RAttrText", "RAttrMarker", "RAttrMargins", "RAttrAxis", "RAttrAxisTitle", "RAttrAxisTicks",
    "RAttrAxisLabels", "RAttrLineEnding", "RWebWindowsManager",
]  # fmt: skip


class RWebWindowsManager:
    """``RWebWindowsManager``: the server of ROOT 7's web windows - of which, in batch, there
    are none to serve; asking for it is harmless, asking it to serve one is refused."""

    _instance: RWebWindowsManager | None = None

    @staticmethod
    def Instance() -> RWebWindowsManager:
        if RWebWindowsManager._instance is None:
            RWebWindowsManager._instance = RWebWindowsManager()
        return RWebWindowsManager._instance

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        raise UnsupportedFeatureError(
            f"RWebWindowsManager::{name} is not supported: it serves ROOT 7's windows to a web "
            f"browser, and xrdroot runs in batch, with no browser and no window to serve.")
