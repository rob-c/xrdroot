"""The engine's small helpers: named constants, NaN packing, component selection,
renaming on import, and the table of classes.

The NaN bits are RooFit's ``RooNaNPacker``: a quiet NaN, the tag ``0x321AB``
above bit 32, and the payload as a ``float`` below it, so ``1.5`` packs to
``0x7ffb21ab3fc00000``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from xrdroot.roofit import nanpack, registry, selection
from xrdroot.roofit.editing import rename_all
from xrdroot.roofit.names import named_constant
from xrdroot.roofit.pdfs.basic import RooGaussian
from xrdroot.roofit.variables import RooRealVar


def test_a_colour_name_plus_or_minus_a_number_is_its_value() -> None:
    """``LineColor("kAzure+2")`` is ``kAzure`` (860) plus two, spaces or not."""
    assert named_constant("kAzure+2") == 862
    assert named_constant(" kRed - 9 ") == 623
    assert named_constant("kDashed") == 2
    assert named_constant("42") == 42
    assert named_constant("-3+1") == -2


def test_a_name_that_is_no_constant_is_refused() -> None:
    """A misspelt colour, or something that is no expression at all, is an error."""
    with pytest.raises(ValueError, match='"kPurple" is not a valid color name'):
        named_constant("kPurple")
    with pytest.raises(ValueError, match="is not a valid color name or style"):
        named_constant("red")


def test_a_payload_is_packed_into_the_low_bits_of_a_tagged_quiet_nan() -> None:
    """The bits are RooFit's: quiet NaN, the magic tag, then the payload as a ``float``."""
    packed = nanpack.pack(1.5)
    assert isinstance(packed, float) and math.isnan(packed)
    assert hex(int(np.float64(packed).view(np.uint64))) == "0x7ffb21ab3fc00000"
    many = nanpack.pack([1.5, -2.25])
    assert [hex(int(one)) for one in many.view(np.uint64)] == [
        "0x7ffb21ab3fc00000",
        "0x7ffb21abc0100000",
    ]


def test_a_payload_comes_back_out_of_the_nan_that_carries_it() -> None:
    """``unpackNaN`` of a packed NaN is its payload, of anything else zero."""
    assert nanpack.unpack(nanpack.pack(1.5)) == 1.5
    assert nanpack.unpack(float("nan")) == 0.0
    assert nanpack.unpack(2.0) == 0.0
    found = nanpack.unpack(np.array([nanpack.pack(-2.25), 1.0, float("nan")]))
    assert found.tolist() == [-2.25, 0.0, 0.0]
    assert nanpack.tagged([nanpack.pack(3.0), float("nan"), 1.0]).tolist() == [True, False, False]


def test_only_the_selected_components_are_active_while_a_selection_holds() -> None:
    """``Components("bkg")``: inside the selection only ``bkg`` contributes, after it all do."""
    signal, background = RooRealVar("sig", "s", 1.0), RooRealVar("bkg", "b", 1.0)
    assert selection.active(signal)
    with selection.selecting({"bkg"}):
        assert not selection.active(signal)
        assert selection.active(background)
        with selection.selecting(None):
            assert selection.active(signal)
        assert not selection.active(signal)
    assert selection.active(signal)


def test_a_selection_is_undone_when_the_curve_fails() -> None:
    """An error while sampling must not leave the other components switched off."""
    signal = RooRealVar("sig", "s", 1.0)
    with pytest.raises(RuntimeError), selection.selecting(set()):
        raise RuntimeError("sampling failed")
    assert selection.active(signal)


def test_renaming_on_import_suffixes_every_node_but_the_variables(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``RenameAllNodes="ws"`` renames ``g`` to ``g_ws`` and says so; ``x`` keeps its name."""
    x = RooRealVar("x", "x", 0, -1, 1)
    gauss = RooGaussian("g", "g", x, RooRealVar("m", "m", 0, -1, 1), RooRealVar("s", "s", 1, 0, 2))
    rename_all(gauss, "ws", {})
    assert (gauss.GetName(), x.GetName()) == ("g_ws", "x")
    assert capsys.readouterr().out.endswith(
        "INFO:ObjectHandling -- RooWorkspace::import(w) Resolving name conflict in workspace"
        " by changing name of imported node  g to g_ws\n"
    )


def test_every_class_in_the_table_is_found_by_its_root_name() -> None:
    """The factory and ``import ROOT`` share this table, so each entry must load."""
    table = registry.classes()
    assert len(table) == sum(len(names) for names in registry.MODULES.values())
    assert all(klass.__name__ == name for name, klass in table.items())
    assert registry.find("RooGaussian") is RooGaussian


def test_the_factory_may_leave_off_the_roo_prefix() -> None:
    """``Gaussian::g(...)`` finds ``RooGaussian``; a name that is no class finds nothing."""
    assert registry.find("Gaussian") is RooGaussian
    assert registry.find("Nonsense") is None
