"""Reading ROOT files, in Python, over the same connection as everything else.

    >>> import xrdroot
    >>> with xrdroot.open_root("root://eos.example.org//store/events.root") as f:  # doctest: +SKIP
    ...     tree = f["Events"]
    ...     for batch in tree.iterate(["pt", "eta"], step=10_000):
    ...         train(batch)

ROOT is how experimental physics stores its data, and until now getting at it
from Python meant a C++ toolchain or a wheel with a compiler behind it. This
reads the format itself - keys, directories, trees, baskets and all four of
ROOT's compression algorithms - into NumPy, and it reads it a basket at a
time, so a tree on the other side of the world costs the entries you asked for
rather than the file. ``tree.arrays(library="pd")`` - or ``ak``, ``pa`` and
``pl`` - hands the columns to pandas, Awkward, Arrow or Polars, and every
histogram speaks the plotting protocol ``hist`` and ``mplhep`` share.

Numbers, strings, jagged rows, STL containers and the members ROOT splits a
C++ class into are all read, and a split object can be asked for whole. So are
the objects ROOT's own kit writes beside a tree: a histogram comes back as a
:class:`Histogram`, with its bins and its edges where you would look for them,
and a graph as a :class:`Graph` you can walk a point at a time; a profile is a
:class:`Profile`, an efficiency an :class:`Efficiency` with ROOT's intervals,
and a sparse histogram a :class:`SparseHistogram`. :func:`chain` reads a tree
written as many files as one, a tree reads its friends beside it, and an
:class:`EntryList` picks the entries to read. :func:`compile_formula` makes a
``TTree::Draw`` expression, evaluated over whole columns at once, and
``tree.arrays(["Sum$(pt > 30)"], cut="n > 1")`` reads them by their text.
:class:`RDataFrame` is ROOT's declarative analysis over a tree, a chain or an
RNTuple: lazy, in one pass, with C++ expressions and ``ROOT::VecOps``
evaluated over whole batches of entries, and shared across processes by
:func:`EnableImplicitMT`. :data:`gRandom` and :class:`TRandom3` are ROOT's
random numbers to the bit, an array at a time, with the rest of ROOT's
generators in :mod:`xrdroot.random`. A :class:`Function` is ROOT's ``TF1``: a
``TFormula`` or a Python model, read from a file or hung on a histogram by
a fit, evaluated, differentiated and integrated over whole arrays, and
``h.fit("gaus")`` is ``TH1::Fit`` - ROOT's options, starting values and
chi-squares, Minuit through iminuit - with :mod:`xrdroot.fit` beneath it and
its :class:`FitResult` handed back. :data:`gROOT` and :data:`gDirectory` are
ROOT's session - the open files, where you are in them, a name looked up
the way ROOT's prompt looks it up - for the shell ``xrdroot`` starts and the
macros it runs, beside ``xrdroot ls``, ``dump``, ``diff`` and the rest of
ROOT's command-line kit (``%load_ext xrdroot`` brings the prompt's commands
into IPython). A saved ``TCanvas`` is a :class:`Canvas`
of pads, each with what it drew and the option it drew it with, and
``c.save("c1.png")`` draws it as ROOT did.
What it does not do is every ROOT class ever written: one whose layout the
file does not describe, or one that streams itself in some way of its own, is
refused by name with the class in the message, because a plausible misreading
of physics data is worse than a refusal.

It writes, too: :func:`create` makes a new ROOT file anywhere this library
can put bytes, holding trees - from a dict of arrays or any DataFrame -
histograms and graphs - read from another file, made by ``hist`` or
:func:`numpy.histogram`, or built from plain numbers with
:meth:`Histogram.new` and :meth:`Graph.new` - along with strings and arrays,
in directories of their own if their names say so; and :func:`update` adds
to a file that is already there. And everything drawable draws itself,
with ROOT's options, defaults and colours: ``.plot()`` onto matplotlib
axes, a plotly or a bokeh figure, or into characters with nothing installed
at all, with :mod:`xrdroot.plot` for ratio plots, comparisons, stacks and
experiments' styles - and a notebook shows each as its picture.

:func:`merge` is ``hadd``: many files made into one, histograms added up and
trees concatenated with their baskets copied across as they are; and
:func:`copy` is ``rootcp``, or ``TTree::CopyTree`` given a cut.

:mod:`xrdml` turns what comes out into tensors, if PyTorch or TensorFlow
is there; it is a separate package that builds on this one.
"""

from __future__ import annotations

from typing import Any

from . import fit, plot, stats
from .canvas import Canvas
from .chain import Chain, ChainedBranch, chain
from .efficiency import Efficiency
from .entries import EntryList
from .errors import FormatError, ROOTError, UnsupportedFeatureError
from .file import Directory, Key, ROOTFile, open_root
from .fit import FitResult
from .formula import Formula, FormulaError, compile_formula
from .function import Function
from .graph import Graph
from .hist import Axis, Histogram
from .merging import Merged, MergeWarning, copy, merge
from .profile import Profile
from .random import TRandom3, gRandom
from .rdf import EnableImplicitMT, RDataFrame, RunGraphs
from .rntuple import RField, RNTuple, WritableRNTuple
from .session import gDirectory, gROOT
from .slicing import loc, overflow, rebin, underflow
from .sparse import SparseHistogram
from .stacks import MultiGraph, Stack
from .tree import Branch, Group, Jagged, TTree
from .writer import WritableDirectory, WritableFile, create
from .wtree import WritableTree
from .wupdate import update

__all__ = [
    # opening
    "open_root",
    "chain",
    "ROOTFile",
    "Directory",
    "Key",
    # writing
    "create",
    "update",
    "WritableFile",
    "WritableDirectory",
    "WritableTree",
    "WritableRNTuple",
    # merging and copying, as hadd and rootcp do
    "merge",
    "copy",
    "Merged",
    "MergeWarning",
    # data
    "TTree",
    "Branch",
    "Group",
    "Jagged",
    "RNTuple",
    "RField",
    "Chain",
    "ChainedBranch",
    "EntryList",
    "Histogram",
    "Axis",
    "Profile",
    "Efficiency",
    "SparseHistogram",
    "Stack",
    "Graph",
    "MultiGraph",
    "Function",
    "Canvas",
    # fitting
    "fit",
    "FitResult",
    # indexing, as UHI does it
    "loc",
    "rebin",
    "underflow",
    "overflow",
    # statistics
    "stats",
    # drawing
    "plot",
    # analysis
    "RDataFrame",
    "RunGraphs",
    "EnableImplicitMT",
    # random numbers
    "TRandom3",
    "gRandom",
    # the session
    "gROOT",
    "gDirectory",
    # expressions
    "compile_formula",
    "Formula",
    "FormulaError",
    # errors
    "ROOTError",
    "FormatError",
    "UnsupportedFeatureError",
]


def load_ipython_extension(ipython: Any) -> None:
    """``%load_ext xrdroot``: ROOT's prompt commands, and ``%root_ls`` and friends, in IPython."""
    from .cli.magics import load

    load(ipython)
