"""A small stand-in for ROOT, for running translated macros without xrdroot.pyroot.

The translator's output names ROOT's classes as ``ROOT.TH1F`` and so on;
``ROOT.bind(fake())`` points those names here instead. What is here is only
what the tests' macros use, done simply and deterministically: histograms
that count, a random generator that is a fixed sequence, canvases that
remember what was drawn, and the address contract - anything with
``.value`` is written through.
"""

from __future__ import annotations

import math
import types
from typing import Any

__all__ = ["fake"]


class Named:
    def __init__(self, name: str = "", title: str = "", *rest: Any) -> None:
        self.name = name
        self.title = title
        self.rest = rest
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def GetName(self) -> str:
        return self.name

    def GetTitle(self) -> str:
        return self.title

    def __getattr__(self, method: str) -> Any:
        if method.startswith("_"):
            raise AttributeError(method)

        def record(*args: Any) -> Any:
            self.calls.append((method, args))
            return self

        return record


class TH1(Named):
    def __init__(self, name: str = "", title: str = "", nbins: int = 1, low: float = 0.0,
                 high: float = 1.0, *rest: Any) -> None:
        super().__init__(name, title)
        self.nbins, self.low, self.high = int(nbins), float(low), float(high)
        self.counts = [0.0] * (self.nbins + 2)
        self.values: list[float] = []

    def Fill(self, x: Any, *rest: Any) -> int:
        value = float(x)
        self.values.append(value)
        width = (self.high - self.low) / self.nbins
        index = 0 if value < self.low else self.nbins + 1 if value >= self.high else (
            1 + int((value - self.low) / width)
        )
        self.counts[index] += 1
        return index

    def GetEntries(self) -> float:
        return float(len(self.values))

    def GetMean(self) -> float:
        return sum(self.values) / len(self.values) if self.values else 0.0

    def GetNbinsX(self) -> int:
        return self.nbins

    def GetBinContent(self, index: int) -> float:
        return self.counts[int(index)]

    def SetBinContent(self, index: int, value: float) -> None:
        self.counts[int(index)] = float(value)


class TRandom3(Named):
    """A generator whose numbers are a fixed, easily predicted sequence."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__("random")
        self.state = 0

    def Rndm(self) -> float:
        self.state = (self.state * 1103515245 + 12345) % 2**31
        return self.state / 2**31

    def Uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        return low + (high - low) * self.Rndm()

    def Gaus(self, mean: float = 0.0, sigma: float = 1.0) -> float:
        return mean + sigma * (self.Rndm() - 0.5)

    def Rannor(self, a: Any, b: Any) -> None:
        a.value = self.Rndm() - 0.5
        b.value = self.Rndm() - 0.5


class TMath:
    @staticmethod
    def Pi() -> float:
        return math.pi

    @staticmethod
    def Sqrt(x: float) -> float:
        return math.sqrt(x)

    @staticmethod
    def Abs(x: Any) -> Any:
        return abs(x)

    @staticmethod
    def Power(x: float, y: float) -> float:
        return float(x) ** y

    @staticmethod
    def Permute(n: int, a: Any) -> bool:
        """The next permutation of ``a[:n]`` in place, as TMath::Permute, false after the last."""
        items = list(a[:n])
        i = n - 2
        while i >= 0 and items[i] >= items[i + 1]:
            i -= 1
        if i < 0:
            return False
        j = n - 1
        while items[j] <= items[i]:
            j -= 1
        items[i], items[j] = items[j], items[i]
        items[i + 1 :] = reversed(items[i + 1 :])
        a[:n] = items
        return True


class Vector(list):  # type: ignore[type-arg]
    def push_back(self, value: Any) -> None:
        self.append(value)

    def size(self) -> int:
        return len(self)

    def at(self, index: int) -> Any:
        return self[index]


class _Template:
    def __init__(self, kind: type) -> None:
        self.kind = kind

    def __getitem__(self, args: Any) -> type:
        return self.kind


def fake() -> types.SimpleNamespace:
    """A fresh fake ROOT namespace, with a canvas list tests can look at."""
    canvases: list[Named] = []

    def canvas(*args: Any) -> Named:
        made = Named(*args)
        canvases.append(made)
        return made

    std = types.SimpleNamespace(vector=_Template(Vector))
    return types.SimpleNamespace(
        TH1F=TH1, TH1D=TH1, TH1I=TH1, TRandom3=TRandom3, gRandom=TRandom3(), TMath=TMath,
        TCanvas=canvas, TGraph=Named, TF1=Named, TLegend=Named, TFile=Named,
        kRed=632, kBlue=600, kGreen=416, gPad=Named("pad"), gStyle=Named("style"),
        std=std, canvases=canvases,
    )
