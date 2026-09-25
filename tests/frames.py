"""Files the RDataFrame tests read, written with this library into a temporary directory.

``xyn`` is go-hep's ``mktree``: a hundred entries in which ``x`` runs from 0
to 99, ``y`` is ``-x`` and ``n`` the entry number, so every expected value in
the tests ported from go-hep's ``rdf`` package can be worked out by hand.
``muons`` is shaped like CMS NanoAOD's muon collection - a count, and a
``Muon_`` column per property holding each entry's muons - drawn from a
seeded generator, so a dimuon analysis over it can be checked against the
same arithmetic written out in NumPy.
"""

from __future__ import annotations

import pathlib

import numpy as np

import xrdroot
from xrdroot import Jagged

DATA = pathlib.Path(__file__).parent / "data"


def write_xyn(path: pathlib.Path, entries: int = 100, basket_size: int = 32_000) -> str:
    """go-hep's tree of ``x = i``, ``y = -i`` and ``n = i``, at ``path``."""
    x = np.arange(entries, dtype=np.float64)
    with xrdroot.create(path) as f:
        columns = {"x": np.float64, "y": np.float64, "n": np.int32}
        tree = f.tree("tree", columns, basket_size=basket_size)
        tree.extend({"x": x, "y": -x, "n": np.arange(entries, dtype=np.int32)})
    return str(path)


def muon_columns(entries: int, seed: int = 7) -> dict[str, object]:
    """NanoAOD-like muons: how many, and each one's pt, eta, phi, mass and charge."""
    rng = np.random.default_rng(seed)
    counts = rng.poisson(2.0, entries).astype(np.int32)
    total = int(counts.sum())
    offsets = np.zeros(entries + 1, np.int64)
    np.cumsum(counts, out=offsets[1:])

    def rows(values: np.ndarray) -> Jagged:
        return Jagged(values, offsets)

    return {
        "nMuon": counts,
        "Muon_pt": rows(rng.exponential(20.0, total).astype(np.float32) + 3),
        "Muon_eta": rows(rng.normal(0.0, 1.5, total).astype(np.float32)),
        "Muon_phi": rows(rng.uniform(-np.pi, np.pi, total).astype(np.float32)),
        "Muon_mass": rows(np.full(total, 0.105658, np.float32)),
        "Muon_charge": rows(rng.choice(np.array([-1, 1], np.int32), total)),
    }


def write_muons(path: pathlib.Path, entries: int = 5000, *, rntuple: bool = False) -> str:
    """A file of NanoAOD-like muons, as a tree or an RNTuple called ``Events``."""
    with xrdroot.create(path) as f:
        f.write("Events", muon_columns(entries), rntuple=rntuple)
    return str(path)
