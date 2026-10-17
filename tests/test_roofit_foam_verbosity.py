"""TFoam's chat output reports existing cell estimates without changing any random draw."""

from pathlib import Path

import numpy as np

from xrdroot.random import TRandom3
from xrdroot.roofit import printing
from xrdroot.roofit.generation.foam import Foam


def test_verbose_cell_diagnostics_match_root_character_for_character(monkeypatch, capsys):
    monkeypatch.setattr(printing, "PRECISION", [6])
    foam = Foam(
        lambda p: 1 + p[:, 0] * p[:, 0], 1, TRandom3(7), n_cells=5, n_sampl=20, chat_level=2
    )
    foam.initialize()
    expected = (Path(__file__).parent / "data/roofit-foam-chat-6.40.txt").read_text()
    assert capsys.readouterr().out == expected
    assert printing.PRECISION[0] == 8


def test_diagnostics_do_not_consume_random_numbers_or_change_the_foam(monkeypatch, capsys):
    monkeypatch.setattr(printing, "PRECISION", [6])

    def density(points):
        return 1 + points[:, 0] * points[:, 0]

    quiet = Foam(density, 1, TRandom3(17), n_cells=30, n_sampl=20)
    shown = Foam(density, 1, TRandom3(17), n_cells=30, n_sampl=20, chat_level=1)
    quiet.initialize()
    shown.initialize()
    assert "11\n" in capsys.readouterr().out
    assert quiet.n_calls == shown.n_calls
    assert quiet.prime == shown.prime
    assert np.array_equal(quiet.make_event(), shown.make_event())
    assert quiet.rng.rndm() == shown.rng.rndm()


def test_large_foams_report_a_dimension_marker_and_their_cell_checkpoint(capsys):
    from types import SimpleNamespace

    from xrdroot.roofit.generation.foamverbosity import progress

    # TFoam prints a dot for ten or more dimensions and a checkpoint at cell 10000.
    progress(SimpleNamespace(cells=[None] * 10001, dim=10))
    assert capsys.readouterr().out == ".|10000\n"
