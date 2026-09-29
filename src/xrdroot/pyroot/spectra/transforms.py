"""``TSpectrumTransform`` and ``TSpectrum2Transform``: ROOT's orthogonal transforms of spectra.

    >>> t = TSpectrumTransform(16)
    >>> t.SetTransformType(t.kTransformHaar, 0)
    >>> t.Transform(source, dest)                 # dest: the Haar coefficients  # doctest: +SKIP

Each keeps ROOT's settings by ROOT's setters, refused as ROOT refuses them
- with ROOT's message, the setting left as it was - and transforms, filters
or enhances by :mod:`xrdroot.spectrum.transform`, writing into ``dest`` as
the C++ writes through its pointer. The 1-D class changes itself as ROOT's
does when it runs: a cosine or sine transform doubles its size, and a
cosine or sine mixed transform raises its degree, every time.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...spectrum import transformtypes as kinds
from ...spectrum.transform import Settings, enhance, filter_zonal, source_length, transform
from ...spectrum.transform2 import Settings2, enhance2, filter_zonal2, has_imaginary, transform2
from ..core.objects import TNamed, TObject
from .arrays import matrix_out, vector_in, vector_out

__all__ = ["TSpectrum2Transform", "TSpectrumTransform"]


class _Kinds:
    """ROOT's enum of transforms and directions, which both classes carry."""

    kTransformHaar = kinds.HAAR
    kTransformWalsh = kinds.WALSH
    kTransformCos = kinds.COS
    kTransformSin = kinds.SIN
    kTransformFourier = kinds.FOURIER
    kTransformHartley = kinds.HARTLEY
    kTransformFourierWalsh = kinds.FOURIER_WALSH
    kTransformFourierHaar = kinds.FOURIER_HAAR
    kTransformWalshHaar = kinds.WALSH_HAAR
    kTransformCosWalsh = kinds.COS_WALSH
    kTransformCosHaar = kinds.COS_HAAR
    kTransformSinWalsh = kinds.SIN_WALSH
    kTransformSinHaar = kinds.SIN_HAAR
    kTransformForward = kinds.FORWARD
    kTransformInverse = kinds.INVERSE


def power_of_two(size: int) -> bool:
    """Is ``size`` a power of two, as ROOT's doubling loop finds it?"""
    n = 1
    while n < size:
        n *= 2
    return n == size


def _valid_type(owner: Any, kind: int, degree: int, sizes: tuple[int, ...]) -> bool:
    """ROOT's check of ``SetTransformType``: a known kind, a mixed one's degree in range."""
    if kind < kinds.HAAR or kind > kinds.SIN_HAAR:
        owner.Error("TSpectrumTransform", "Invalid type of transform")
        return False
    if kind >= kinds.FOURIER_WALSH and (degree < 1 or any(degree > _bits(n) for n in sizes)):
        owner.Error("TSpectrumTransform", "Invalid degree of mixed transform")
        return False
    return True


def _bits(size: int) -> int:
    """How many doublings of one reach ``size``: its log2, rounded up."""
    count, n = 0, 1
    while n < size:
        count, n = count + 1, n * 2
    return count


def _valid_size(owner: Any, *sizes: int) -> bool:
    """ROOT's check of a constructor's lengths: positive, and powers of two."""
    if any(size <= 0 for size in sizes):
        owner.Error("TSpectrumTransform", "Invalid length, must be > than 0")
        return False
    if not all(power_of_two(size) for size in sizes):
        owner.Error("TSpectrumTransform", "Invalid length, must be power of 2")
        return False
    return True


def _direction(owner: Any, direction: int) -> bool:
    """ROOT's check of ``SetDirection``."""
    if direction not in (kinds.FORWARD, kinds.INVERSE):
        owner.Error("TSpectrumTransform", "Wrong direction")
        return False
    return True


def _source(values: Any, needed: int, what: str) -> Any:
    """``needed`` numbers from ``values``, refused if it holds fewer, as ROOT would read past it."""
    found = vector_in(values, needed)
    if len(found) < needed:
        raise ValueError(
            f"{what} reads {needed} numbers from its source, which holds only {len(found)}."
        )
    return found


def _sized(size: int, what: str) -> None:
    """Refuse to run a transform whose constructor refused its length, which ROOT never set."""
    if size <= 0:
        raise ValueError(
            f"{what} has no length to transform: its constructor refused the one it was given, "
            "and ROOT would run on whatever its unset length happened to be."
        )


class TSpectrumTransform(_Kinds, TNamed):
    """ROOT's ``TSpectrumTransform``: 1-D transforms, zonal filtering and enhancement."""

    def __init__(self, size: int | None = None) -> None:
        self._settings = Settings()
        if size is None:
            super().__init__()
            return
        super().__init__("SpectrumTransform", "Miroslav Morhac transformer")
        size = int(size)
        if _valid_size(self, size):
            self._settings = Settings(size=size, xmin=size // 4, xmax=size - 1)

    def SetTransformType(self, transType: int, degree: int) -> None:
        """``SetTransformType``: the kind of transform, and a mixed one's degree."""
        if _valid_type(self, int(transType), int(degree), (self._settings.size,)):
            self._settings.kind, self._settings.degree = int(transType), int(degree)

    def SetRegion(self, xmin: int, xmax: int) -> None:
        """``SetRegion``: the channels ``FilterZonal`` and ``Enhance`` change."""
        if xmin < 0 or xmax < xmin or xmax >= self._settings.size:
            self.Error("TSpectrumTransform", "Wrong range")
            return
        self._settings.xmin, self._settings.xmax = int(xmin), int(xmax)

    def SetDirection(self, direction: int) -> None:
        """``SetDirection``: ``kTransformForward`` or ``kTransformInverse``."""
        if _direction(self, int(direction)):
            self._settings.direction = int(direction)

    def SetFilterCoeff(self, filterCoeff: float) -> None:
        """``SetFilterCoeff``: what ``FilterZonal`` puts in the region."""
        self._settings.filter_coeff = float(filterCoeff)

    def SetEnhanceCoeff(self, enhanceCoeff: float) -> None:
        """``SetEnhanceCoeff``: what ``Enhance`` multiplies the region by."""
        self._settings.enhance_coeff = float(enhanceCoeff)

    def _read(self, source: Any, method: str) -> Any:
        """The source a call reads, once its degree is known to be one ROOT survives."""
        settings = self._settings
        _sized(settings.size, f"TSpectrumTransform::{method}")
        grown = settings.degree + 1
        if settings.kind >= kinds.COS_WALSH and kinds.power2(grown) > 2 * settings.size:
            raise ValueError(
                f"TSpectrumTransform::{method} would raise this cosine or sine mixed transform's "
                f"degree to {grown}, past what {settings.size} channels allow: ROOT, which raises "
                "it by one on every call, would divide by zero."
            )
        return _source(source, source_length(settings), f"TSpectrumTransform::{method}")

    def Transform(self, source: Any, destVector: Any) -> None:
        """``Transform``: the coefficients of ``source``, or inverse its spectrum, written out."""
        vector_out(destVector, transform(self._settings, self._read(source, "Transform")))

    def FilterZonal(self, source: Any, destVector: Any) -> None:
        """``FilterZonal``: ``source`` with the region's coefficients set to one value."""
        vector_out(destVector, filter_zonal(self._settings, self._read(source, "FilterZonal")))

    def Enhance(self, source: Any, destVector: Any) -> None:
        """``Enhance``: ``source`` with the region's coefficients multiplied up."""
        vector_out(destVector, enhance(self._settings, self._read(source, "Enhance")))


class TSpectrum2Transform(_Kinds, TObject):
    """ROOT's ``TSpectrum2Transform``: 2-D transforms, zonal filtering and enhancement.

    Unlike the 1-D class it is a plain ``TObject``, so its name is its class's.
    """

    #: What ``ClassDef`` says of the class, which is its title.
    CLASS_TITLE = "Spectrum2 Transformer, it calculates classic orthogonal 2D transforms"

    def __init__(self, sizeX: int | None = None, sizeY: int | None = None) -> None:
        super().__init__()
        self._settings = Settings2()
        if sizeX is None:
            return
        nx, ny = int(sizeX), int(sizeY if sizeY is not None else 0)
        if _valid_size(self, nx, ny):
            self._settings = Settings2(nx, ny, xmin=nx // 4, xmax=nx - 1, ymin=ny // 4, ymax=ny - 1)

    def SetTransformType(self, transType: int, degree: int) -> None:
        """``SetTransformType``: the kind of transform, a mixed one's degree within both axes."""
        sizes = (self._settings.sizex, self._settings.sizey)
        if _valid_type(self, int(transType), int(degree), sizes):
            self._settings.kind, self._settings.degree = int(transType), int(degree)

    def SetRegion(self, xmin: int, xmax: int, ymin: int, ymax: int) -> None:
        """``SetRegion``: the rectangle of coefficients ``FilterZonal`` and ``Enhance`` change."""
        settings = self._settings
        for low, high, size in ((xmin, xmax, settings.sizex), (ymin, ymax, settings.sizey)):
            if low < 0 or high < low or high >= size:
                self.Error("TSpectrumTransform", "Wrong range")
                return
        settings.xmin, settings.xmax = int(xmin), int(xmax)
        settings.ymin, settings.ymax = int(ymin), int(ymax)

    def SetDirection(self, direction: int) -> None:
        """``SetDirection``: ``kTransformForward`` or ``kTransformInverse``."""
        if _direction(self, int(direction)):
            self._settings.direction = int(direction)

    def SetFilterCoeff(self, filterCoeff: float) -> None:
        """``SetFilterCoeff``: what ``FilterZonal`` puts in the region."""
        self._settings.filter_coeff = float(filterCoeff)

    def SetEnhanceCoeff(self, enhanceCoeff: float) -> None:
        """``SetEnhanceCoeff``: what ``Enhance`` multiplies the region by."""
        self._settings.enhance_coeff = float(enhanceCoeff)

    def _read(self, source: Any, method: str, imaginary: bool) -> Any:
        """``sizeX`` rows of the source - twice ``sizeY`` wide when an imaginary part is read."""
        settings = self._settings
        what = f"TSpectrum2Transform::{method}"
        _sized(settings.sizex, what)
        width = settings.sizey * (2 if imaginary else 1)
        rows = [_source(source[i], width, what) for i in range(settings.sizex)]
        return np.array(rows).reshape(settings.sizex, width)

    def Transform(self, fSource: Any, fDest: Any) -> None:
        """``Transform``: the coefficients - or, inverse, the spectrum - into ``fDest``."""
        settings = self._settings
        wide = has_imaginary(settings.kind) and settings.direction == kinds.INVERSE
        matrix_out(fDest, transform2(settings, self._read(fSource, "Transform", wide)))

    def FilterZonal(self, fSource: Any, fDest: Any) -> None:
        """``FilterZonal``: the region's coefficients set; nothing written if nothing is left."""
        _written(fDest, filter_zonal2(self._settings, self._read(fSource, "FilterZonal", False)))

    def Enhance(self, fSource: Any, fDest: Any) -> None:
        """``Enhance``: the region's coefficients multiplied; nothing written if none left."""
        _written(fDest, enhance2(self._settings, self._read(fSource, "Enhance", False)))


def _written(target: Any, values: Any) -> None:
    """Write ``values`` into ``target`` - unless there are none, when ROOT writes nothing."""
    if values is not None:
        matrix_out(target, values)
