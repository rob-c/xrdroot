"""Merging and copying ROOT files: ``hadd`` and ``rootcp``, without ROOT.

:func:`merge` is ``hadd``: many files made into one, histograms added up,
trees concatenated with their baskets copied across as they are.
:func:`copy` is ``rootcp`` - and, given a cut, ``TTree::CopyTree``: objects
taken from one file into another, trees whole or cut down to the entries and
columns wanted.

The pieces are here in the order they build on each other: :mod:`.infos`
carries one file's description of its classes into another, :mod:`.records`
copies a record as it was, :mod:`.baskets` moves a tree's baskets,
:mod:`.trees` and :mod:`.ntuples` write a tree or an RNTuple from others,
:mod:`.objects` adds up what ``hadd`` adds up, and :mod:`.files` and
:mod:`.copying` are the two functions that walk files doing all of it.
"""

from __future__ import annotations

from .copying import copy
from .files import Merged, MergeWarning, merge

__all__ = ["merge", "copy", "Merged", "MergeWarning"]
