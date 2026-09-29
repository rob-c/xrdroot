"""ROOT's spectrum classes: ``TSpectrum``, ``TSpectrum2`` and their fits and transforms.

Each class keeps ROOT's names, argument orders and defaults, and does its
arithmetic in :mod:`xrdroot.spectrum`. Where ROOT takes a ``Double_t *`` or
a ``Double_t **`` and writes its answer back into it, so do these
(:mod:`.arrays`): a NumPy array, an ``array.array``, a list, or a list of
rows, filled in place.
"""
