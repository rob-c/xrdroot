"""``RooStatsConfig``: the switches every RooStats calculator's likelihoods and fits read.

``useLikelihoodOffset`` offsets each likelihood by its first value -
``UseNLLOffset(true)`` - and ``useEvalErrorWall`` keeps Minuit off the
parameter values where the density cannot be evaluated. ``NLLOffsetMode``
is ``"initial"`` when the minimum is to be read back without the offset.
"""

from __future__ import annotations

__all__ = ["GetGlobalRooStatsConfig", "NLLOffsetMode", "SetNLLOffsetMode", "UseNLLOffset"]


class RooStatsConfig:
    """The switches, as ROOT's defaults set them."""

    def __init__(self) -> None:
        self.useLikelihoodOffset = False
        self.useEvalErrorWall = True
        self.nllOffsetMode = ""


#: The one configuration every calculator shares.
CONFIG = RooStatsConfig()


def GetGlobalRooStatsConfig() -> RooStatsConfig:
    return CONFIG


def UseNLLOffset(on: bool) -> None:
    CONFIG.useLikelihoodOffset = bool(on)


def SetNLLOffsetMode(mode: str) -> None:
    """``"initial"`` offsets every likelihood by its first value, and reads minima back without
    it; anything else offsets none."""
    CONFIG.nllOffsetMode = str(mode)
    CONFIG.useLikelihoodOffset = CONFIG.nllOffsetMode == "initial"


def NLLOffsetMode() -> str:
    return CONFIG.nllOffsetMode
