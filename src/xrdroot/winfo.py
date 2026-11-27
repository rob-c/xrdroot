"""What the classes this library writes look like, harvested rather than typed.

Every entry here was read out of the streamer information ROOT itself wrote
into the files under ``tests/data``, by a walk of the same records
``read_streamers`` reads. The tree, graph and collection classes come from
``gauss-h1.root``, ``gauss-h2.root``, ``graphs.root`` and
``small-flat-tree.root``, all written by ROOT 6.08; three leaf classes no 6.08
donor happens to contain - ``TLeafO``, ``TLeafB`` and ``TLeafS`` - come from
``leaves.root`` instead, which ROOT 6.28 wrote, and the two vintages describe
every leaf class they share identically down to the checksum.

The histogram family is one vintage newer. A profile, an efficiency and a
three-dimensional histogram have no 6.08 donor here, and a ``TProfile`` built
on one ``TH1D`` beside a ``TH1F`` built on another would be a file describing
two different ``TH1`` classes by the one name. So all of it comes from files
ROOT 6.22 and 6.24 wrote: ``TH1``, ``TH1D``, ``TH2``, ``TH2D``, ``TProfile``
and ``TProfile2D`` from ``tprofile.root``, ``TH1F`` from ``tgme.root``,
``TEfficiency`` from ``tconfidence-level.root``, and ``TH3``, ``TH3D``,
``TAtt3D`` and ``TProfile3D`` from ``uproot-issue-227b.root``. That vintage
adds ``fStatOverflows`` to the end of ``TH1``, where a reader of the older
layout steps over it.

The functions a fit is made with are of that vintage too, and from the same
donors: ``TF1``, ``TFormula`` and ``TF1Parameters`` from ``tgme.root`` - whose
``TMultiGraph`` ROOT fitted with a ``pol1`` - and ``TF1AbsComposition``,
which ``tgme.root`` never needed, from ``tformula.root``, which ROOT 6.24
wrote as well. That ``TFormula`` is version 13, the last without
``fNumber``; ROOT 6.34 reads it. ``TF2`` and ``TF3`` have no donor here at
all, so they are read - a file describes them - but not written.

``TObjString``, a ``TObject`` and the ``TString`` it wraps, comes from
``streamers.root``, the same at every vintage.

That ``TMultiGraph`` is the donor of its own class too: version 2, which
holds nothing but a name and pointers - to the list of its graphs, its
functions and the histogram it draws its axes on - so the graphs in it are
described as ``graphs.root`` describes them.

``TNtuple`` and ``TNtupleD`` are derived the same way. The one donor here,
``tntuple.root``, is of the 6.24 vintage and builds its ``TNtuple`` on a
``TTree`` of version 20, where this writer's is 19; but either class is only
a ``TTree`` base and ``Int_t fNvar``, exactly as that donor and ROOT 6.40
itself describe them, and the checksum - worked out as for the histograms,
now with the one member counted too - comes out as ROOT's for the version-20
base (``0xb8a51cab`` and ``0x8de8d873``), so over the version-19 base it is
what ROOT would make of it there.

Eleven histogram classes - ``TH1C``, ``TH1S`` and ``TH1I``, the same three
and ``TH2F`` in two dimensions, and the other four ``TH3`` classes - have no
donor of that vintage here. Each is nothing but two bases, a ``TH1``, ``TH2``
or ``TH3`` and the ``TArray`` it keeps its bins in, so its elements are those
bases exactly as the harvested classes carry them, and its checksum is what
ROOT's ``TClass::GetCheckSum`` makes of them: the class name, then each base's
name and checksum. That calculation reproduces every harvested checksum in
this table, which the tests hold it to, and the seven of the eleven that
ROOT's own streamer dump in go-hep contains come out as ROOT has them.

The branches of objects come from three donors. ``TBranchElement`` and
``TLeafElement`` are the 6.08 vintage's, from ``small-evnt-tree-fullsplit.root``,
built on the same ``TBranch`` and ``TLeaf`` as everything else here.
``TBranchObject`` and ``TLeafObject`` have no 6.08 donor; they come from
``tlv-split00.root``, which ROOT 6.24 wrote, along with ``TLorentzVector``
and ``TVector3``. Its ``TBranchObject`` names its ``TBranch`` base at version
13, and a ``TBranch`` streams itself by the version in front of its own bytes,
so the version 12 this writer gives it is read as one. ``TClonesArray``,
``TLine`` and the ``TAttBBox2D`` it derives from, and GenVector's
``LorentzVector<PxPyPzE4D<double> >`` and its coordinates, come from
``object-branches-6.40.root``, which ROOT 6.40 wrote from ``object_branches.C``.

A class a *program* declares is not here, even where ROOT's own record of one can be read
out of a file. An instance of a template the program itself declares -
``roottest/root/io/bigevent``'s ``Event::fTemplateMember``, a ``template2<template1<int> >`` -
is described by that declaration: ROOT writes its record as version 1, no elements, checksum
``0xf1819316``, and writes *objects* of it as a foreign class of no ``ClassDef`` - version 0
and that checksum after it, which is what the declaration reproduces to the byte.
``pyroot/core/declared._reach`` stops at a name this table holds, so an entry here would take
the class away from the declaration and write the member's record as ``40000002 0001`` where
ROOT writes ``40000006 0000`` and ``f1819316``.

The arrays ROOT builds its histograms on - ``TArray``, ``TArrayF``, ``TArrayI``,
``TArrayD`` - are the one family deliberately absent from this table although a program's
class can hold one: ``roottest/root/io/bigevent``'s ``Event`` has ``TArrayF fArrayF`` by
value and ``TArrayI *fArrayI``, and its ``Track`` a ``TArrayI fInts``. Each streams itself,
``TArray::Streamer`` writing version 1, ``fN`` and then that many values, and
``TArrayF::Streamer`` writing nothing of its own - so a member of one is written as an
*object*: a ``TStreamerObjectAny`` of type 62, type 69 behind a pointer, naming the class,
with those bytes inside it, which is what the declaration's ``Catalogue`` makes of the
member.

They are absent for two measured reasons. This table is what the *writer* walks for the
classes a file describes (:func:`~.writer._closure`, over the bases each element names), and
``TArrayF`` is a base of every ``TH1F``: an entry here would put ``TArrayF``, and ``TArray``
through it, into the streamer record of every file holding a histogram, where ROOT names
neither. And a file ROOT 6.40.04 writes describes them only when a tree carries them: a
``TTree`` with a branch element of a ``ClassDef`` class holding the five arrays says
``TArray``, ``TArrayC``, ``TArrayS``, ``TArrayI``, ``TArrayF`` and ``TArrayD``, where a bare
``TObject`` holding a ``TArrayF`` and a ``TArrayI *``, written with ``WriteObject``, makes a
file of one ``TStreamerInfo``, the holder's. Their own descriptions - the version, ``fN``
and the counted ``fArray``, as ROOT 6.40.04's own file says them - are
:data:`ARRAY_LAYOUTS`, and that is what a file which carries them must be told.

The counting word is where the vintages part, and it is what the ``Element`` tuple has to
spell exactly. ROOT 6.40.04 says ``int``: streamer type 6, four bytes, and the base's
checksum in the second maximum index of an array's ``TArray`` element. ``g4-like.root``, an
older ROOT's, says ``Int_t`` instead: type 3, no size, no base checksum, and titles the
counted pointer of each array ``[fN] Array of fN integers`` - a different checksum for every
one of them (``0x007021b2`` against ``0x03f126b3`` for ``TArray`` alone).

Writing carries these descriptions back out verbatim, checksums and all, so a
file written here says exactly what a ROOT of that vintage would say about the
same classes, and anything reading it - ROOT, this library, or another - walks
the bytes by the same map they were written from.

Regenerate from the donor files rather than editing: a hand-edited layout is a
guess, and a guess about layout is how physics data gets misread.
"""

from __future__ import annotations

__all__ = ["ARRAY_LAYOUTS", "INFOS", "SUBVERSIONS", "WRITER_VERSION"]

#: The ROOT version the donor descriptions came from, written into the header.
WRITER_VERSION = 60806

#: The record version each ``TStreamerElement`` subclass wrote itself with.
SUBVERSIONS = {
    "TStreamerBase": 3,
    "TStreamerBasicType": 2,
    "TStreamerBasicPointer": 2,
    "TStreamerString": 2,
    "TStreamerObject": 2,
    "TStreamerObjectAny": 2,
    "TStreamerObjectPointer": 2,
    "TStreamerObjectAnyPointer": 1,
    "TStreamerSTL": 3,
    "TStreamerSTLstring": 2,
    "TStreamerLoop": 2,
}

#: One element of one class: its ``TStreamerElement`` subclass, name, title,
#: streamer type, size, array length, array dimensions, the five maximum
#: indices, the type name, and whatever extra fields the subclass adds - the
#: base class version for a ``TStreamerBase``, the counter's version, name
#: and class for a ``TStreamerBasicPointer``, and the kind of container and
#: of what it holds for a ``TStreamerSTL``.
Element = tuple[str, str, str, int, int, int, int, tuple[int, ...], str, tuple[object, ...]]

#: What ROOT 6.40.04 says a file describes of the arrays it builds its histograms on - the
#: ``TArray`` a class of a program's holds by value as ``TArrayF fArrayF`` or by pointer as
#: ``TArrayI *fArrayI``. They are not in :data:`INFOS`; the module docstring says why, and
#: what the counted ``fArray`` of each is worth. The donor is a ``TTree`` written by ROOT
#: 6.40.04 through ACLiC, whose branch element is a ``ClassDef`` class holding one of each:
#:
#:   root -l -b -q -e 't.Branch("h", "HoldAll", &h); t.Fill(); t.Write()'
#:
#: An older ROOT's file, ``g4-like.root``, spells the count ``Int_t`` and titles each counted
#: pointer ``[fN] Array of fN integers`` - different bytes and checksums - so a reader is
#: handed whichever file's word it read, and this is what the writer must say.
ARRAY_LAYOUTS: dict[str, tuple[int, int, tuple[Element, ...]]] = {
    "TArray": (0x007021B2, 1, (
        ("TStreamerBasicType", "fN", "Number of array elements",
         6, 4, 0, 0, (0, 0, 0, 0, 0), "int", ()),
    )),
    "TArrayC": (0xAE879936, 1, (
        ("TStreamerBase", "TArray", "Abstract array base class",
         0, 0, 0, 0, (0, 0x007021B2, 0, 0, 0), "BASE", (1,)),
        ("TStreamerBasicPointer", "fArray", "[fN] Array of fN chars",
         41, 1, 0, 0, (0, 0, 0, 0, 0), "char*", (1, "fN", "TArray")),
    )),
    "TArrayS": (0x035C9314, 1, (
        ("TStreamerBase", "TArray", "Abstract array base class",
         0, 0, 0, 0, (0, 0x007021B2, 0, 0, 0), "BASE", (1,)),
        ("TStreamerBasicPointer", "fArray", "[fN] Array of fN shorts",
         42, 2, 0, 0, (0, 0, 0, 0, 0), "short*", (1, "fN", "TArray")),
    )),
    "TArrayI": (0xD9D571C7, 1, (
        ("TStreamerBase", "TArray", "Abstract array base class",
         0, 0, 0, 0, (0, 0x007021B2, 0, 0, 0), "BASE", (1,)),
        ("TStreamerBasicPointer", "fArray", "[fN] Array of fN 32 bit integers",
         43, 4, 0, 0, (0, 0, 0, 0, 0), "int*", (1, "fN", "TArray")),
    )),
    "TArrayF": (0x5A0BF6F1, 1, (
        ("TStreamerBase", "TArray", "Abstract array base class",
         0, 0, 0, 0, (0, 0x007021B2, 0, 0, 0), "BASE", (1,)),
        ("TStreamerBasicPointer", "fArray", "[fN] Array of fN floats",
         45, 4, 0, 0, (0, 0, 0, 0, 0), "float*", (1, "fN", "TArray")),
    )),
    "TArrayD": (0x7139EF34, 1, (
        ("TStreamerBase", "TArray", "Abstract array base class",
         0, 0, 0, 0, (0, 0x007021B2, 0, 0, 0), "BASE", (1,)),
        ("TStreamerBasicPointer", "fArray", "[fN] Array of fN doubles",
         48, 8, 0, 0, (0, 0, 0, 0, 0), "double*", (1, "fN", "TArray")),
    )),
}

#: Every class this library can write: its checksum, version and elements,
#: exactly as the donor files describe them.
INFOS: dict[str, tuple[int, int, tuple[Element, ...]]] = {
    "TH1C": (0x36f6e4ad, 3, (
        ('TStreamerBase', 'TH1', '1-Dim histogram base class',
         0, 0, 0, 0, (0, 473383108, 0, 0, 0), 'BASE', (8,)),
        ('TStreamerBase', 'TArrayC', 'Array of chars',
         0, 0, 0, 0, (0, -1366845130, 0, 0, 0), 'BASE', (1,)),
    )),
    "TH1S": (0x8c4d9dcb, 3, (
        ('TStreamerBase', 'TH1', '1-Dim histogram base class',
         0, 0, 0, 0, (0, 473383108, 0, 0, 0), 'BASE', (8,)),
        ('TStreamerBase', 'TArrayS', 'Array of shorts',
         0, 0, 0, 0, (0, 56398612, 0, 0, 0), 'BASE', (1,)),
    )),
    "TH1I": (0x627564f6, 3, (
        ('TStreamerBase', 'TH1', '1-Dim histogram base class',
         0, 0, 0, 0, (0, 473383108, 0, 0, 0), 'BASE', (8,)),
        ('TStreamerBase', 'TArrayI', 'Array of ints',
         0, 0, 0, 0, (0, -640323129, 0, 0, 0), 'BASE', (1,)),
    )),
    "TH1F": (0xe2939644, 3, (
        ('TStreamerBase', 'TH1', '1-Dim histogram base class',
         0, 0, 0, 0, (0, 473383108, 0, 0, 0), 'BASE', (8,)),
        ('TStreamerBase', 'TArrayF', 'Array of floats',
         0, 0, 0, 0, (0, 1510733553, 0, 0, 0), 'BASE', (1,)),
    )),
    "TH1D": (0xf9b1569f, 3, (
        ('TStreamerBase', 'TH1', '1-Dim histogram base class',
         0, 0, 0, 0, (0, 473383108, 0, 0, 0), 'BASE', (8,)),
        ('TStreamerBase', 'TArrayD', 'Array of doubles',
         0, 0, 0, 0, (0, 1899622196, 0, 0, 0), 'BASE', (1,)),
    )),
    "TH2C": (0xbd0010fe, 4, (
        ('TStreamerBase', 'TH2', '2-Dim histogram base class',
         0, 0, 0, 0, (0, 25310335, 0, 0, 0), 'BASE', (5,)),
        ('TStreamerBase', 'TArrayC', 'Array of chars',
         0, 0, 0, 0, (0, -1366845130, 0, 0, 0), 'BASE', (1,)),
    )),
    "TH2S": (0x1256ca1c, 4, (
        ('TStreamerBase', 'TH2', '2-Dim histogram base class',
         0, 0, 0, 0, (0, 25310335, 0, 0, 0), 'BASE', (5,)),
        ('TStreamerBase', 'TArrayS', 'Array of shorts',
         0, 0, 0, 0, (0, 56398612, 0, 0, 0), 'BASE', (1,)),
    )),
    "TH2I": (0xe87e9147, 4, (
        ('TStreamerBase', 'TH2', '2-Dim histogram base class',
         0, 0, 0, 0, (0, 25310335, 0, 0, 0), 'BASE', (5,)),
        ('TStreamerBase', 'TArrayI', 'Array of ints',
         0, 0, 0, 0, (0, -640323129, 0, 0, 0), 'BASE', (1,)),
    )),
    "TH2F": (0x689cc295, 4, (
        ('TStreamerBase', 'TH2', '2-Dim histogram base class',
         0, 0, 0, 0, (0, 25310335, 0, 0, 0), 'BASE', (5,)),
        ('TStreamerBase', 'TArrayF', 'Array of floats',
         0, 0, 0, 0, (0, 1510733553, 0, 0, 0), 'BASE', (1,)),
    )),
    "TH2D": (0x7fba82f0, 4, (
        ('TStreamerBase', 'TH2', '2-Dim histogram base class',
         0, 0, 0, 0, (0, 25310335, 0, 0, 0), 'BASE', (5,)),
        ('TStreamerBase', 'TArrayD', 'Array of doubles',
         0, 0, 0, 0, (0, 1899622196, 0, 0, 0), 'BASE', (1,)),
    )),
    "TH3C": (0xa1ff8d94, 4, (
        ('TStreamerBase', 'TH3', '3-Dim histogram base class',
         0, 0, 0, 0, (0, 1121076319, 0, 0, 0), 'BASE', (6,)),
        ('TStreamerBase', 'TArrayC', 'Array of chars',
         0, 0, 0, 0, (0, -1366845130, 0, 0, 0), 'BASE', (1,)),
    )),
    "TH3S": (0xf75646b2, 4, (
        ('TStreamerBase', 'TH3', '3-Dim histogram base class',
         0, 0, 0, 0, (0, 1121076319, 0, 0, 0), 'BASE', (6,)),
        ('TStreamerBase', 'TArrayS', 'Array of shorts',
         0, 0, 0, 0, (0, 56398612, 0, 0, 0), 'BASE', (1,)),
    )),
    "TH3I": (0xcd7e0ddd, 4, (
        ('TStreamerBase', 'TH3', '3-Dim histogram base class',
         0, 0, 0, 0, (0, 1121076319, 0, 0, 0), 'BASE', (6,)),
        ('TStreamerBase', 'TArrayI', 'Array of ints',
         0, 0, 0, 0, (0, -640323129, 0, 0, 0), 'BASE', (1,)),
    )),
    "TH3F": (0x4d9c3f2b, 4, (
        ('TStreamerBase', 'TH3', '3-Dim histogram base class',
         0, 0, 0, 0, (0, 1121076319, 0, 0, 0), 'BASE', (6,)),
        ('TStreamerBase', 'TArrayF', 'Array of floats',
         0, 0, 0, 0, (0, 1510733553, 0, 0, 0), 'BASE', (1,)),
    )),
    "TH3D": (0x64b9ff86, 4, (
        ('TStreamerBase', 'TH3', '3-Dim histogram base class',
         0, 0, 0, 0, (0, 1121076319, 0, 0, 0), 'BASE', (6,)),
        ('TStreamerBase', 'TArrayD', 'Array of doubles',
         0, 0, 0, 0, (0, 1899622196, 0, 0, 0), 'BASE', (1,)),
    )),
    "TH1": (0x1c3740c4, 8, (
        ('TStreamerBase', 'TNamed', 'The basis for a named object (name, title)',
         67, 0, 0, 0, (0, -541636036, 0, 0, 0), 'BASE', (1,)),
        ('TStreamerBase', 'TAttLine', 'Line attributes',
         0, 0, 0, 0, (0, -1811462839, 0, 0, 0), 'BASE', (2,)),
        ('TStreamerBase', 'TAttFill', 'Fill area attributes',
         0, 0, 0, 0, (0, -2545006, 0, 0, 0), 'BASE', (2,)),
        ('TStreamerBase', 'TAttMarker', 'Marker attributes',
         0, 0, 0, 0, (0, 689802220, 0, 0, 0), 'BASE', (2,)),
        ('TStreamerBasicType', 'fNcells', 'number of bins(1D), cells (2D) +U/Overflows',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerObject', 'fXaxis', 'X axis descriptor',
         61, 216, 0, 0, (0, 0, 0, 0, 0), 'TAxis', ()),
        ('TStreamerObject', 'fYaxis', 'Y axis descriptor',
         61, 216, 0, 0, (0, 0, 0, 0, 0), 'TAxis', ()),
        ('TStreamerObject', 'fZaxis', 'Z axis descriptor',
         61, 216, 0, 0, (0, 0, 0, 0, 0), 'TAxis', ()),
        ('TStreamerBasicType', 'fBarOffset', '(1000*offset) for bar charts or legos',
         2, 2, 0, 0, (0, 0, 0, 0, 0), 'short', ()),
        ('TStreamerBasicType', 'fBarWidth', '(1000*width) for bar charts or legos',
         2, 2, 0, 0, (0, 0, 0, 0, 0), 'short', ()),
        ('TStreamerBasicType', 'fEntries', 'Number of entries',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fTsumw', 'Total Sum of weights',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fTsumw2', 'Total Sum of squares of weights',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fTsumwx', 'Total Sum of weight*X',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fTsumwx2', 'Total Sum of weight*X*X',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fMaximum', 'Maximum value for plotting',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fMinimum', 'Minimum value for plotting',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fNormFactor', 'Normalization factor',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerObjectAny', 'fContour', 'Array to display contour levels',
         62, 24, 0, 0, (0, 0, 0, 0, 0), 'TArrayD', ()),
        ('TStreamerObjectAny', 'fSumw2', 'Array of sum of squares of weights',
         62, 24, 0, 0, (0, 0, 0, 0, 0), 'TArrayD', ()),
        ('TStreamerString', 'fOption', 'histogram options',
         65, 24, 0, 0, (0, 0, 0, 0, 0), 'TString', ()),
        ('TStreamerObjectPointer', 'fFunctions', '->Pointer to list of functions (fits and user)',
         63, 8, 0, 0, (0, 0, 0, 0, 0), 'TList*', ()),
        ('TStreamerBasicType', 'fBufferSize', 'fBuffer size',
         6, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicPointer', 'fBuffer', '[fBufferSize] entry buffer',
         48, 8, 0, 0, (0, 0, 0, 0, 0), 'double*', (8, 'fBufferSize', 'TH1')),
        ('TStreamerBasicType', 'fBinStatErrOpt', 'option for bin statistical errors',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'TH1::EBinErrorOpt', ()),
        ('TStreamerBasicType', 'fStatOverflows',
         'per object flag to use under/overflows in statistics',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'TH1::EStatOverflows', ()),
    )),
    "TH2": (0x0182347f, 5, (
        ('TStreamerBase', 'TH1', '1-Dim histogram base class',
         0, 0, 0, 0, (0, 473383108, 0, 0, 0), 'BASE', (8,)),
        ('TStreamerBasicType', 'fScalefactor', 'Scale factor',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fTsumwy', 'Total Sum of weight*Y',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fTsumwy2', 'Total Sum of weight*Y*Y',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fTsumwxy', 'Total Sum of weight*X*Y',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
    )),
    "TH3": (0x42d2445f, 6, (
        ('TStreamerBase', 'TH1', '1-Dim histogram base class',
         0, 0, 0, 0, (0, 473383108, 0, 0, 0), 'BASE', (8,)),
        ('TStreamerBase', 'TAtt3D', '3D attributes',
         0, 0, 0, 0, (0, 30074, 0, 0, 0), 'BASE', (1,)),
        ('TStreamerBasicType', 'fTsumwy', 'Total Sum of weight*Y',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fTsumwy2', 'Total Sum of weight*Y*Y',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fTsumwxy', 'Total Sum of weight*X*Y',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fTsumwz', 'Total Sum of weight*Z',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fTsumwz2', 'Total Sum of weight*Z*Z',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fTsumwxz', 'Total Sum of weight*X*Z',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fTsumwyz', 'Total Sum of weight*Y*Z',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
    )),
    "TAtt3D": (0x0000757a, 1, (
    )),
    "TProfile": (0x4bedee54, 7, (
        ('TStreamerBase', 'TH1D', '1-Dim histograms (one double per channel)',
         0, 0, 0, 0, (0, -105818465, 0, 0, 0), 'BASE', (3,)),
        ('TStreamerObjectAny', 'fBinEntries', 'number of entries per bin',
         62, 24, 0, 0, (0, 0, 0, 0, 0), 'TArrayD', ()),
        ('TStreamerBasicType', 'fErrorMode', 'Option to compute errors',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'EErrorType', ()),
        ('TStreamerBasicType', 'fYmin', 'Lower limit in Y (if set)',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fYmax', 'Upper limit in Y (if set)',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fTsumwy', 'Total Sum of weight*Y',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fTsumwy2', 'Total Sum of weight*Y*Y',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerObjectAny', 'fBinSumw2', 'Array of sum of squares of weights per bin',
         62, 24, 0, 0, (0, 0, 0, 0, 0), 'TArrayD', ()),
    )),
    "TProfile2D": (0x36a142ac, 8, (
        ('TStreamerBase', 'TH2D', '2-Dim histograms (one double per channel)',
         0, 0, 0, 0, (0, 2142929648, 0, 0, 0), 'BASE', (4,)),
        ('TStreamerObjectAny', 'fBinEntries', 'number of entries per bin',
         62, 24, 0, 0, (0, 0, 0, 0, 0), 'TArrayD', ()),
        ('TStreamerBasicType', 'fErrorMode', 'Option to compute errors',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'EErrorType', ()),
        ('TStreamerBasicType', 'fZmin', 'Lower limit in Z (if set)',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fZmax', 'Upper limit in Z (if set)',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fTsumwz', 'Total Sum of weight*Z',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fTsumwz2', 'Total Sum of weight*Z*Z',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerObjectAny', 'fBinSumw2', 'Array of sum of squares of weights per bin',
         62, 24, 0, 0, (0, 0, 0, 0, 0), 'TArrayD', ()),
    )),
    "TProfile3D": (0xf60c6814, 8, (
        ('TStreamerBase', 'TH3D', '3-Dim histograms (one double per channel)',
         0, 0, 0, 0, (0, 1689911174, 0, 0, 0), 'BASE', (4,)),
        ('TStreamerObjectAny', 'fBinEntries', 'number of entries per bin',
         62, 24, 0, 0, (0, 0, 0, 0, 0), 'TArrayD', ()),
        ('TStreamerBasicType', 'fErrorMode', 'Option to compute errors',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'EErrorType', ()),
        ('TStreamerBasicType', 'fTmin', 'Lower limit in T (if set)',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fTmax', 'Upper limit in T (if set)',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fTsumwt', 'Total Sum of weight*T',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fTsumwt2', 'Total Sum of weight*T*T',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerObjectAny', 'fBinSumw2', 'Array of sum of squares of weights per bin',
         62, 24, 0, 0, (0, 0, 0, 0, 0), 'TArrayD', ()),
    )),
    "TEfficiency": (0x52931aeb, 2, (
        ('TStreamerBase', 'TNamed', 'The basis for a named object (name, title)',
         67, 0, 0, 0, (0, -541636036, 0, 0, 0), 'BASE', (1,)),
        ('TStreamerBase', 'TAttLine', 'Line attributes',
         0, 0, 0, 0, (0, -1811462839, 0, 0, 0), 'BASE', (2,)),
        ('TStreamerBase', 'TAttFill', 'Fill area attributes',
         0, 0, 0, 0, (0, -2545006, 0, 0, 0), 'BASE', (2,)),
        ('TStreamerBase', 'TAttMarker', 'Marker attributes',
         0, 0, 0, 0, (0, 689802220, 0, 0, 0), 'BASE', (2,)),
        ('TStreamerBasicType', 'fBeta_alpha',
         'global parameter for prior beta distribution (default = 1)',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fBeta_beta',
         'global parameter for prior beta distribution (default = 1)',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerSTL', 'fBeta_bin_params',
         'parameter for prior beta distribution different bin by bin',
         500, 24, 0, 0, (0, 0, 0, 0, 0), 'vector<pair<double,double> >', (1, 61)),
        ('TStreamerBasicType', 'fConfLevel', 'confidence level (default = 0.683, 1 sigma)',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerObjectPointer', 'fFunctions', '->pointer to list of functions',
         63, 8, 0, 0, (0, 0, 0, 0, 0), 'TList*', ()),
        ('TStreamerObjectPointer', 'fPassedHistogram',
         'histogram for events which passed certain criteria',
         64, 8, 0, 0, (0, 0, 0, 0, 0), 'TH1*', ()),
        ('TStreamerBasicType', 'fStatisticOption',
         'defines how the confidence intervals are determined',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'TEfficiency::EStatOption', ()),
        ('TStreamerObjectPointer', 'fTotalHistogram', 'histogram for total number of events',
         64, 8, 0, 0, (0, 0, 0, 0, 0), 'TH1*', ()),
        ('TStreamerBasicType', 'fWeight', 'weight for all events (default = 1)',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
    )),
    "TF1": (0x7223ffd8, 12, (
        ('TStreamerBase', 'TNamed', 'The basis for a named object (name, title)',
         67, 0, 0, 0, (0, -541636036, 0, 0, 0), 'BASE', (1,)),
        ('TStreamerBase', 'TAttLine', 'Line attributes',
         0, 0, 0, 0, (0, -1811462839, 0, 0, 0), 'BASE', (2,)),
        ('TStreamerBase', 'TAttFill', 'Fill area attributes',
         0, 0, 0, 0, (0, -2545006, 0, 0, 0), 'BASE', (2,)),
        ('TStreamerBase', 'TAttMarker', 'Marker attributes',
         0, 0, 0, 0, (0, 689802220, 0, 0, 0), 'BASE', (2,)),
        ('TStreamerBasicType', 'fXmin', 'Lower bounds for the range',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fXmax', 'Upper bounds for the range',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fNpar', 'Number of parameters',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicType', 'fNdim', 'Function dimension',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicType', 'fNpx', 'Number of points used for the graphical representation',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicType', 'fType', '',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'TF1::EFType', ()),
        ('TStreamerBasicType', 'fNpfits', 'Number of points used in the fit',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicType', 'fNDF', 'Number of degrees of freedom in the fit',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicType', 'fChisquare', 'Function fit chisquare',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fMinimum', 'Minimum value for plotting',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fMaximum', 'Maximum value for plotting',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerSTL', 'fParErrors', 'Array of errors of the fNpar parameters',
         500, 24, 0, 0, (0, 0, 0, 0, 0), 'vector<double>', (1, 8)),
        ('TStreamerSTL', 'fParMin', 'Array of lower limits of the fNpar parameters',
         500, 24, 0, 0, (0, 0, 0, 0, 0), 'vector<double>', (1, 8)),
        ('TStreamerSTL', 'fParMax', 'Array of upper limits of the fNpar parameters',
         500, 24, 0, 0, (0, 0, 0, 0, 0), 'vector<double>', (1, 8)),
        ('TStreamerSTL', 'fSave', 'Array of fNsave function values',
         500, 24, 0, 0, (0, 0, 0, 0, 0), 'vector<double>', (1, 8)),
        ('TStreamerBasicType', 'fNormalized', 'Normalization option (false by default)',
         18, 1, 0, 0, (0, 0, 0, 0, 0), 'bool', ()),
        ('TStreamerBasicType', 'fNormIntegral', 'Integral of the function before being normalized',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerObjectPointer', 'fFormula',
         'Pointer to TFormula in case when user define formula',
         64, 8, 0, 0, (0, 0, 0, 0, 0), 'TFormula', ()),
        ('TStreamerObjectAnyPointer', 'fParams',
         'Pointer to Function parameters object (exists only for not-formula functions)',
         69, 8, 0, 0, (0, 0, 0, 0, 0), 'TF1Parameters', ()),
        ('TStreamerObjectPointer', 'fComposition', 'Pointer to composition (NSUM or CONV)',
         64, 8, 0, 0, (0, 0, 0, 0, 0), 'TF1AbsComposition', ()),
    )),
    "TFormula": (0x3d29ef01, 13, (
        ('TStreamerBase', 'TNamed', 'The basis for a named object (name, title)',
         67, 0, 0, 0, (0, -541636036, 0, 0, 0), 'BASE', (1,)),
        ('TStreamerSTL', 'fClingParameters', 'parameter values',
         500, 24, 0, 0, (0, 0, 0, 0, 0), 'vector<double>', (1, 8)),
        ('TStreamerBasicType', 'fAllParametersSetted',
         'flag to control if all parameters are setted',
         18, 1, 0, 0, (0, 0, 0, 0, 0), 'bool', ()),
        ('TStreamerSTL', 'fParams', '|| list of  parameter names',
         500, 48, 0, 0, (0, 0, 0, 0, 0), 'map<TString,int,TFormulaParamOrder>', (4, 61)),
        ('TStreamerString', 'fFormula', 'string representing the formula expression',
         65, 24, 0, 0, (0, 0, 0, 0, 0), 'TString', ()),
        ('TStreamerBasicType', 'fNdim', 'Dimension - needed for lambda expressions',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerSTL', 'fLinearParts', 'vector of linear functions',
         500, 24, 0, 0, (0, 0, 0, 0, 0), 'vector<TObject*>', (1, 63)),
        ('TStreamerBasicType', 'fVectorized',
         'whether we should use vectorized or regular variables',
         18, 1, 0, 0, (0, 0, 0, 0, 0), 'bool', ()),
    )),
    "TF1Parameters": (0xd703028e, 1, (
        ('TStreamerSTL', 'fParameters', 'parameter values',
         500, 24, 0, 0, (0, 0, 0, 0, 0), 'vector<double>', (1, 8)),
        ('TStreamerSTL', 'fParNames', 'parameter names',
         500, 24, 0, 0, (0, 0, 0, 0, 0), 'vector<string>', (1, 61)),
    )),
    "TF1AbsComposition": (0xa71d6529, 1, (
        ('TStreamerBase', 'TObject', 'Basic ROOT object',
         66, 0, 0, 0, (0, -1877229523, 0, 0, 0), 'BASE', (1,)),
    )),
    "TGraph": (0x05f7f465, 4, (
        ('TStreamerBase', 'TNamed', 'The basis for a named object (name, title)',
         67, 0, 0, 0, (0, -541636036, 0, 0, 0), 'BASE', (1,)),
        ('TStreamerBase', 'TAttLine', 'Line attributes',
         0, 0, 0, 0, (0, -1811462839, 0, 0, 0), 'BASE', (2,)),
        ('TStreamerBase', 'TAttFill', 'Fill area attributes',
         0, 0, 0, 0, (0, -2545006, 0, 0, 0), 'BASE', (2,)),
        ('TStreamerBase', 'TAttMarker', 'Marker attributes',
         0, 0, 0, 0, (0, 689802220, 0, 0, 0), 'BASE', (2,)),
        ('TStreamerBasicType', 'fNpoints', 'Number of points <= fMaxSize',
         6, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicPointer', 'fX', '[fNpoints] array of X points',
         48, 8, 0, 0, (0, 0, 0, 0, 0), 'double*', (4, 'fNpoints', 'TGraph')),
        ('TStreamerBasicPointer', 'fY', '[fNpoints] array of Y points',
         48, 8, 0, 0, (0, 0, 0, 0, 0), 'double*', (4, 'fNpoints', 'TGraph')),
        ('TStreamerObjectPointer', 'fFunctions', 'Pointer to list of functions (fits and user)',
         64, 8, 0, 0, (0, 0, 0, 0, 0), 'TList*', ()),
        ('TStreamerObjectPointer', 'fHistogram', 'Pointer to histogram used for drawing axis',
         64, 8, 0, 0, (0, 0, 0, 0, 0), 'TH1F*', ()),
        ('TStreamerBasicType', 'fMinimum', 'Minimum value for plotting along y',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fMaximum', 'Maximum value for plotting along y',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
    )),
    "TGraphErrors": (0x2a7ce30f, 3, (
        ('TStreamerBase', 'TGraph', 'Graph graphics class',
         0, 0, 0, 0, (0, 100136037, 0, 0, 0), 'BASE', (4,)),
        ('TStreamerBasicPointer', 'fEX', '[fNpoints] array of X errors',
         48, 8, 0, 0, (0, 0, 0, 0, 0), 'double*', (4, 'fNpoints', 'TGraph')),
        ('TStreamerBasicPointer', 'fEY', '[fNpoints] array of Y errors',
         48, 8, 0, 0, (0, 0, 0, 0, 0), 'double*', (4, 'fNpoints', 'TGraph')),
    )),
    "TGraphAsymmErrors": (0xcc46af3b, 3, (
        ('TStreamerBase', 'TGraph', 'Graph graphics class',
         0, 0, 0, 0, (0, 100136037, 0, 0, 0), 'BASE', (4,)),
        ('TStreamerBasicPointer', 'fEXlow', '[fNpoints] array of X low errors',
         48, 8, 0, 0, (0, 0, 0, 0, 0), 'double*', (4, 'fNpoints', 'TGraph')),
        ('TStreamerBasicPointer', 'fEXhigh', '[fNpoints] array of X high errors',
         48, 8, 0, 0, (0, 0, 0, 0, 0), 'double*', (4, 'fNpoints', 'TGraph')),
        ('TStreamerBasicPointer', 'fEYlow', '[fNpoints] array of Y low errors',
         48, 8, 0, 0, (0, 0, 0, 0, 0), 'double*', (4, 'fNpoints', 'TGraph')),
        ('TStreamerBasicPointer', 'fEYhigh', '[fNpoints] array of Y high errors',
         48, 8, 0, 0, (0, 0, 0, 0, 0), 'double*', (4, 'fNpoints', 'TGraph')),
    )),
    "TKDE": (3391967380, 3, (
        #: Every element measured off the streamer info ROOT 6.40 writes for a TKDE
        #: (``TFile::GetStreamerInfoList`` on a file the oracle wrote, round 78): the
        #: TNamed base, five vector<double>, vector<bool> and the option words.
        ('TStreamerBase', 'TNamed', 'The basis for a named object (name, title)',
         0, 0, 0, 0, (0, 100136037, 0, 0, 0), 'BASE', (4,)),
        ('TStreamerSTL', 'fData', 'Data events',
         300, 24, 0, 0, (0, 0, 0, 0, 0), 'vector<double>', (1, 8)),
        ('TStreamerSTL', 'fEvents', 'Original data storage',
         300, 24, 0, 0, (0, 0, 0, 0, 0), 'vector<double>', (1, 8)),
        ('TStreamerSTL', 'fEventWeights', 'Original data weights',
         300, 24, 0, 0, (0, 0, 0, 0, 0), 'vector<double>', (1, 8)),
        ('TStreamerBasicType', 'fKernelType', '',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'TKDE::EKernelType', ()),
        ('TStreamerBasicType', 'fIteration', '',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'TKDE::EIteration', ()),
        ('TStreamerBasicType', 'fMirror', '',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'TKDE::EMirror', ()),
        ('TStreamerBasicType', 'fBinning', '',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'TKDE::EBinning', ()),
        ('TStreamerBasicType', 'fUseMirroring', '',
         18, 1, 0, 0, (0, 0, 0, 0, 0), 'bool', ()),
        ('TStreamerBasicType', 'fMirrorLeft', '',
         18, 1, 0, 0, (0, 0, 0, 0, 0), 'bool', ()),
        ('TStreamerBasicType', 'fMirrorRight', '',
         18, 1, 0, 0, (0, 0, 0, 0, 0), 'bool', ()),
        ('TStreamerBasicType', 'fAsymLeft', '',
         18, 1, 0, 0, (0, 0, 0, 0, 0), 'bool', ()),
        ('TStreamerBasicType', 'fAsymRight', '',
         18, 1, 0, 0, (0, 0, 0, 0, 0), 'bool', ()),
        ('TStreamerBasicType', 'fUseBins', '',
         18, 1, 0, 0, (0, 0, 0, 0, 0), 'bool', ()),
        ('TStreamerBasicType', 'fNewData', 'Flag to control when new data are given',
         18, 1, 0, 0, (0, 0, 0, 0, 0), 'bool', ()),
        ('TStreamerBasicType', 'fUseMinMaxFromData',
         'Flag top control if min and max must be used from data',
         18, 1, 0, 0, (0, 0, 0, 0, 0), 'bool', ()),
        ('TStreamerBasicType', 'fNBins', 'Number of bins for binned data option',
         13, 4, 0, 0, (0, 0, 0, 0, 0), 'unsigned int', ()),
        ('TStreamerBasicType', 'fNEvents', 'Data\'s number of events',
         13, 4, 0, 0, (0, 0, 0, 0, 0), 'unsigned int', ()),
        ('TStreamerBasicType', 'fSumOfCounts', 'Data sum of weights',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fUseBinsNEvents',
         'If the algorithm is allowed to use automatic (relaxed) binning this is the minimum '
         'number of events to do so',
         13, 4, 0, 0, (0, 0, 0, 0, 0), 'unsigned int', ()),
        ('TStreamerBasicType', 'fMean', 'Data mean',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fSigma', 'Data std deviation',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fSigmaRob', 'Data std deviation (robust estimation)',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fXMin', 'Data minimum value',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fXMax', 'Data maximum value',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fRho', 'Adjustment factor for sigma',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fAdaptiveBandwidthFactor',
         'Geometric mean of the kernel density estimation from the data for adaptive iteration',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fWeightSize', 'Caches the weight size',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerSTL', 'fCanonicalBandwidths', '',
         300, 24, 0, 0, (0, 0, 0, 0, 0), 'vector<double>', (1, 8)),
        ('TStreamerSTL', 'fKernelSigmas2', '',
         300, 24, 0, 0, (0, 0, 0, 0, 0), 'vector<double>', (1, 8)),
        ('TStreamerSTL', 'fBinCount', 'Number of events per bin for binned data option',
         300, 24, 0, 0, (0, 0, 0, 0, 0), 'vector<double>', (1, 8)),
        ('TStreamerSTL', 'fSettedOptions', 'User input options flag',
         300, 24, 0, 0, (0, 0, 0, 0, 0), 'vector<bool>', (1, 18)),
    )),
    "TGraphMultiErrors": (0x1a63990a, 1, (
        ('TStreamerBase', 'TGraph', 'Graph graphics class',
         0, 0, 0, 0, (0, 100136037, 0, 0, 0), 'BASE', (4,)),
        ('TStreamerBasicType', 'fNYErrors', 'The amount of different y-errors',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicType', 'fSumErrorsMode',
         'How y errors are summed: kOnlyFirst = Only First; kSquareSum = Squared Sum; kSum =',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicPointer', 'fExL', '[fNpoints] array of X low errors',
         48, 8, 0, 0, (0, 0, 0, 0, 0), 'double*', (4, 'fNpoints', 'TGraph')),
        ('TStreamerBasicPointer', 'fExH', '[fNpoints] array of X high errors',
         48, 8, 0, 0, (0, 0, 0, 0, 0), 'double*', (4, 'fNpoints', 'TGraph')),
        ('TStreamerSTL', 'fEyL', 'two dimensional array of Y low errors',
         500, 24, 0, 0, (0, 0, 0, 0, 0), 'vector<TArrayD>', (1, 61)),
        ('TStreamerSTL', 'fEyH', 'two dimensional array of Y high errors',
         500, 24, 0, 0, (0, 0, 0, 0, 0), 'vector<TArrayD>', (1, 61)),
        ('TStreamerSTL', 'fAttFill', 'the AttFill attributes of the different errors',
         500, 24, 0, 0, (0, 0, 0, 0, 0), 'vector<TAttFill>', (1, 61)),
        ('TStreamerSTL', 'fAttLine', 'the AttLine attributes of the different errors',
         500, 24, 0, 0, (0, 0, 0, 0, 0), 'vector<TAttLine>', (1, 61)),
    )),
    "TMultiGraph": (0xe0893cd5, 2, (
        ('TStreamerBase', 'TNamed', 'The basis for a named object (name, title)',
         67, 0, 0, 0, (0, -541636036, 0, 0, 0), 'BASE', (1,)),
        ('TStreamerObjectPointer', 'fGraphs', 'Pointer to list of TGraphs',
         64, 8, 0, 0, (0, 0, 0, 0, 0), 'TList*', ()),
        ('TStreamerObjectPointer', 'fFunctions', 'Pointer to list of functions (fits and user)',
         64, 8, 0, 0, (0, 0, 0, 0, 0), 'TList*', ()),
        ('TStreamerObjectPointer', 'fHistogram', 'Pointer to histogram used for drawing axis',
         64, 8, 0, 0, (0, 0, 0, 0, 0), 'TH1F*', ()),
        ('TStreamerBasicType', 'fMaximum', 'Maximum value for plotting along y',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fMinimum', 'Minimum value for plotting along y',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
    )),
    "TTree": (0x58a396eb, 19, (
        ('TStreamerBase', 'TNamed', 'The basis for a named object (name, title)',
         67, 0, 0, 0, (0, -541636036, 0, 0, 0), 'BASE', (1,)),
        ('TStreamerBase', 'TAttLine', 'Line attributes',
         0, 0, 0, 0, (0, -1811462839, 0, 0, 0), 'BASE', (2,)),
        ('TStreamerBase', 'TAttFill', 'Fill area attributes',
         0, 0, 0, 0, (0, -2545006, 0, 0, 0), 'BASE', (2,)),
        ('TStreamerBase', 'TAttMarker', 'Marker attributes',
         0, 0, 0, 0, (0, 689802220, 0, 0, 0), 'BASE', (2,)),
        ('TStreamerBasicType', 'fEntries', 'Number of entries',
         16, 8, 0, 0, (0, 0, 0, 0, 0), 'Long64_t', ()),
        ('TStreamerBasicType', 'fTotBytes',
         'Total number of bytes in all branches before compression',
         16, 8, 0, 0, (0, 0, 0, 0, 0), 'Long64_t', ()),
        ('TStreamerBasicType', 'fZipBytes',
         'Total number of bytes in all branches after compression',
         16, 8, 0, 0, (0, 0, 0, 0, 0), 'Long64_t', ()),
        ('TStreamerBasicType', 'fSavedBytes', 'Number of autosaved bytes',
         16, 8, 0, 0, (0, 0, 0, 0, 0), 'Long64_t', ()),
        ('TStreamerBasicType', 'fFlushedBytes', 'Number of auto-flushed bytes',
         16, 8, 0, 0, (0, 0, 0, 0, 0), 'Long64_t', ()),
        ('TStreamerBasicType', 'fWeight', 'Tree weight (see TTree::SetWeight)',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fTimerInterval', 'Timer interval in milliseconds',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicType', 'fScanField', 'Number of runs before prompting in Scan',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicType', 'fUpdate', 'Update frequency for EntryLoop',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicType', 'fDefaultEntryOffsetLen',
         'Initial Length of fEntryOffset table in the basket buffers',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicType', 'fNClusterRange',
         "Number of Cluster range in addition to the one defined by 'AutoFlush'",
         6, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicType', 'fMaxEntries',
         'Maximum number of entries in case of circular buffers',
         16, 8, 0, 0, (0, 0, 0, 0, 0), 'Long64_t', ()),
        ('TStreamerBasicType', 'fMaxEntryLoop', 'Maximum number of entries to process',
         16, 8, 0, 0, (0, 0, 0, 0, 0), 'Long64_t', ()),
        ('TStreamerBasicType', 'fMaxVirtualSize', 'Maximum total size of buffers kept in memory',
         16, 8, 0, 0, (0, 0, 0, 0, 0), 'Long64_t', ()),
        ('TStreamerBasicType', 'fAutoSave',
         'Autosave tree when fAutoSave entries written or -fAutoSave (compressed) bytes produced',
         16, 8, 0, 0, (0, 0, 0, 0, 0), 'Long64_t', ()),
        ('TStreamerBasicType', 'fAutoFlush',
         'Auto-flush tree when fAutoFlush entries written or -fAutoFlush (compressed) by'
         'tes produced',
         16, 8, 0, 0, (0, 0, 0, 0, 0), 'Long64_t', ()),
        ('TStreamerBasicType', 'fEstimate', 'Number of entries to estimate histogram limits',
         16, 8, 0, 0, (0, 0, 0, 0, 0), 'Long64_t', ()),
        ('TStreamerBasicPointer', 'fClusterRangeEnd',
         '[fNClusterRange] Last entry of a cluster range.',
         56, 8, 0, 0, (0, 0, 0, 0, 0), 'Long64_t*', (19, 'fNClusterRange', 'TTree')),
        ('TStreamerBasicPointer', 'fClusterSize',
         '[fNClusterRange] Number of entries in each cluster for a given range.',
         56, 8, 0, 0, (0, 0, 0, 0, 0), 'Long64_t*', (19, 'fNClusterRange', 'TTree')),
        ('TStreamerObject', 'fBranches', 'List of Branches',
         61, 64, 0, 0, (0, 0, 0, 0, 0), 'TObjArray', ()),
        ('TStreamerObject', 'fLeaves', 'Direct pointers to individual branch leaves',
         61, 64, 0, 0, (0, 0, 0, 0, 0), 'TObjArray', ()),
        ('TStreamerObjectPointer', 'fAliases',
         'List of aliases for expressions based on the tree branches.',
         64, 8, 0, 0, (0, 0, 0, 0, 0), 'TList*', ()),
        ('TStreamerObjectAny', 'fIndexValues', 'Sorted index values',
         62, 24, 0, 0, (0, 0, 0, 0, 0), 'TArrayD', ()),
        ('TStreamerObjectAny', 'fIndex', 'Index of sorted values',
         62, 24, 0, 0, (0, 0, 0, 0, 0), 'TArrayI', ()),
        ('TStreamerObjectPointer', 'fTreeIndex', 'Pointer to the tree Index (if any)',
         64, 8, 0, 0, (0, 0, 0, 0, 0), 'TVirtualIndex*', ()),
        ('TStreamerObjectPointer', 'fFriends', 'pointer to list of friend elements',
         64, 8, 0, 0, (0, 0, 0, 0, 0), 'TList*', ()),
        ('TStreamerObjectPointer', 'fUserInfo',
         'pointer to a list of user objects associated to this Tree',
         64, 8, 0, 0, (0, 0, 0, 0, 0), 'TList*', ()),
        ('TStreamerObjectPointer', 'fBranchRef', 'Branch supporting the TRefTable (if any)',
         64, 8, 0, 0, (0, 0, 0, 0, 0), 'TBranchRef*', ()),
    )),
    "TNtuple": (0xa5e66297, 2, (
        ('TStreamerBase', 'TTree', 'Tree descriptor (the main ROOT I/O class)',
         0, 0, 0, 0, (0, 1487116011, 0, 0, 0), 'BASE', (19,)),
        ('TStreamerBasicType', 'fNvar', 'Number of columns',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
    )),
    "TNtupleD": (0x7b2a1e5f, 1, (
        ('TStreamerBase', 'TTree', 'Tree descriptor (the main ROOT I/O class)',
         0, 0, 0, 0, (0, 1487116011, 0, 0, 0), 'BASE', (19,)),
        ('TStreamerBasicType', 'fNvar', 'Number of columns',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
    )),
    "TBranch": (0x59108cb8, 12, (
        ('TStreamerBase', 'TNamed', 'The basis for a named object (name, title)',
         67, 0, 0, 0, (0, -541636036, 0, 0, 0), 'BASE', (1,)),
        ('TStreamerBase', 'TAttFill', 'Fill area attributes',
         0, 0, 0, 0, (0, -2545006, 0, 0, 0), 'BASE', (2,)),
        ('TStreamerBasicType', 'fCompress', 'Compression level and algorithm',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicType', 'fBasketSize', 'Initial Size of  Basket Buffer',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicType', 'fEntryOffsetLen',
         'Initial Length of fEntryOffset table in the basket buffers',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicType', 'fWriteBasket', 'Last basket number written',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicType', 'fEntryNumber',
         'Current entry number (last one filled in this branch)',
         16, 8, 0, 0, (0, 0, 0, 0, 0), 'Long64_t', ()),
        ('TStreamerBasicType', 'fOffset', 'Offset of this branch',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicType', 'fMaxBaskets', 'Maximum number of Baskets so far',
         6, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicType', 'fSplitLevel', 'Branch split level',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicType', 'fEntries', 'Number of entries',
         16, 8, 0, 0, (0, 0, 0, 0, 0), 'Long64_t', ()),
        ('TStreamerBasicType', 'fFirstEntry', 'Number of the first entry in this branch',
         16, 8, 0, 0, (0, 0, 0, 0, 0), 'Long64_t', ()),
        ('TStreamerBasicType', 'fTotBytes',
         'Total number of bytes in all leaves before compression',
         16, 8, 0, 0, (0, 0, 0, 0, 0), 'Long64_t', ()),
        ('TStreamerBasicType', 'fZipBytes', 'Total number of bytes in all leaves after compression',
         16, 8, 0, 0, (0, 0, 0, 0, 0), 'Long64_t', ()),
        ('TStreamerObject', 'fBranches', '-> List of Branches of this branch',
         61, 64, 0, 0, (0, 0, 0, 0, 0), 'TObjArray', ()),
        ('TStreamerObject', 'fLeaves', '-> List of leaves of this branch',
         61, 64, 0, 0, (0, 0, 0, 0, 0), 'TObjArray', ()),
        ('TStreamerObject', 'fBaskets', '-> List of baskets of this branch',
         61, 64, 0, 0, (0, 0, 0, 0, 0), 'TObjArray', ()),
        ('TStreamerBasicPointer', 'fBasketBytes', '[fMaxBaskets] Length of baskets on file',
         43, 4, 0, 0, (0, 0, 0, 0, 0), 'int*', (12, 'fMaxBaskets', 'TBranch')),
        ('TStreamerBasicPointer', 'fBasketEntry',
         '[fMaxBaskets] Table of first entry in each basket',
         56, 8, 0, 0, (0, 0, 0, 0, 0), 'Long64_t*', (12, 'fMaxBaskets', 'TBranch')),
        ('TStreamerBasicPointer', 'fBasketSeek', '[fMaxBaskets] Addresses of baskets on file',
         56, 8, 0, 0, (0, 0, 0, 0, 0), 'Long64_t*', (12, 'fMaxBaskets', 'TBranch')),
        ('TStreamerString', 'fFileName',
         'Name of file where buffers are stored ("" if in same file as Tree header)',
         65, 24, 0, 0, (0, 0, 0, 0, 0), 'TString', ()),
    )),
    "TLeaf": (0x6d1e8152, 2, (
        ('TStreamerBase', 'TNamed', 'The basis for a named object (name, title)',
         67, 0, 0, 0, (0, -541636036, 0, 0, 0), 'BASE', (1,)),
        ('TStreamerBasicType', 'fLen', 'Number of fixed length elements',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicType', 'fLenType', 'Number of bytes for this data type',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicType', 'fOffset', 'Offset in ClonesArray object (if one)',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicType', 'fIsRange', '(=kTRUE if leaf has a range, kFALSE otherwise)',
         18, 1, 0, 0, (0, 0, 0, 0, 0), 'bool', ()),
        ('TStreamerBasicType', 'fIsUnsigned', '(=kTRUE if unsigned, kFALSE otherwise)',
         18, 1, 0, 0, (0, 0, 0, 0, 0), 'bool', ()),
        ('TStreamerObjectPointer', 'fLeafCount',
         'Pointer to Leaf count if variable length (we do not own the counter)',
         64, 8, 0, 0, (0, 0, 0, 0, 0), 'TLeaf*', ()),
    )),
    "TLeafO": (0x02ae48d3, 1, (
        ('TStreamerBase', 'TLeaf', 'Leaf: description of a Branch data type',
         0, 0, 0, 0, (0, 1830715730, 0, 0, 0), 'BASE', (2,)),
        ('TStreamerBasicType', 'fMinimum', 'Minimum value if leaf range is specified',
         18, 1, 0, 0, (0, 0, 0, 0, 0), 'bool', ()),
        ('TStreamerBasicType', 'fMaximum', 'Maximum value if leaf range is specified',
         18, 1, 0, 0, (0, 0, 0, 0, 0), 'bool', ()),
    )),
    "TLeafB": (0x0f1e4b5e, 1, (
        ('TStreamerBase', 'TLeaf', 'Leaf: description of a Branch data type',
         0, 0, 0, 0, (0, 1830715730, 0, 0, 0), 'BASE', (2,)),
        ('TStreamerBasicType', 'fMinimum', 'Minimum value if leaf range is specified',
         1, 1, 0, 0, (0, 0, 0, 0, 0), 'char', ()),
        ('TStreamerBasicType', 'fMaximum', 'Maximum value if leaf range is specified',
         1, 1, 0, 0, (0, 0, 0, 0, 0), 'char', ()),
    )),
    "TLeafS": (0x150ceecf, 1, (
        ('TStreamerBase', 'TLeaf', 'Leaf: description of a Branch data type',
         0, 0, 0, 0, (0, 1830715730, 0, 0, 0), 'BASE', (2,)),
        ('TStreamerBasicType', 'fMinimum', 'Minimum value if leaf range is specified',
         2, 2, 0, 0, (0, 0, 0, 0, 0), 'short', ()),
        ('TStreamerBasicType', 'fMaximum', 'Maximum value if leaf range is specified',
         2, 2, 0, 0, (0, 0, 0, 0, 0), 'short', ()),
    )),
    "TLeafI": (0x7e6aae19, 1, (
        ('TStreamerBase', 'TLeaf', 'Leaf: description of a Branch data type',
         0, 0, 0, 0, (0, 1830715730, 0, 0, 0), 'BASE', (2,)),
        ('TStreamerBasicType', 'fMinimum', 'Minimum value if leaf range is specified',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicType', 'fMaximum', 'Maximum value if leaf range is specified',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
    )),
    "TLeafL": (0xde320862, 1, (
        ('TStreamerBase', 'TLeaf', 'Leaf: description of a Branch data type',
         0, 0, 0, 0, (0, 1830715730, 0, 0, 0), 'BASE', (2,)),
        ('TStreamerBasicType', 'fMinimum', 'Minimum value if leaf range is specified',
         16, 8, 0, 0, (0, 0, 0, 0, 0), 'Long64_t', ()),
        ('TStreamerBasicType', 'fMaximum', 'Maximum value if leaf range is specified',
         16, 8, 0, 0, (0, 0, 0, 0, 0), 'Long64_t', ()),
    )),
    "TLeafF": (0x3add9d72, 1, (
        ('TStreamerBase', 'TLeaf', 'Leaf: description of a Branch data type',
         0, 0, 0, 0, (0, 1830715730, 0, 0, 0), 'BASE', (2,)),
        ('TStreamerBasicType', 'fMinimum', 'Minimum value if leaf range is specified',
         5, 4, 0, 0, (0, 0, 0, 0, 0), 'float', ()),
        ('TStreamerBasicType', 'fMaximum', 'Maximum value if leaf range is specified',
         5, 4, 0, 0, (0, 0, 0, 0, 0), 'float', ()),
    )),
    "TLeafD": (0x118e8776, 1, (
        ('TStreamerBase', 'TLeaf', 'Leaf: description of a Branch data type',
         0, 0, 0, 0, (0, 1830715730, 0, 0, 0), 'BASE', (2,)),
        ('TStreamerBasicType', 'fMinimum', 'Minimum value if leaf range is specified',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fMaximum', 'Maximum value if leaf range is specified',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
    )),
    "TLeafC": (0xfbe3b2f3, 1, (
        ('TStreamerBase', 'TLeaf', 'Leaf: description of a Branch data type',
         0, 0, 0, 0, (0, 1830715730, 0, 0, 0), 'BASE', (2,)),
        ('TStreamerBasicType', 'fMinimum', 'Minimum value if leaf range is specified',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicType', 'fMaximum', 'Maximum value if leaf range is specified',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
    )),
    "TObjArray": (0xa99e6552, 3, (
        ('TStreamerBase', 'TSeqCollection', 'Sequenceable collection ABC',
         0, 0, 0, 0, (0, -60015674, 0, 0, 0), 'BASE', (0,)),
        ('TStreamerBasicType', 'fLowerBound', 'Lower bound of the array',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicType', 'fLast', 'Last element in array containing an object',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
    )),
    "TNamed": (0xdfb74a3c, 1, (
        ('TStreamerBase', 'TObject', 'Basic ROOT object',
         66, 0, 0, 0, (0, -1877229523, 0, 0, 0), 'BASE', (1,)),
        ('TStreamerString', 'fName', 'object identifier',
         65, 24, 0, 0, (0, 0, 0, 0, 0), 'TString', ()),
        ('TStreamerString', 'fTitle', 'object title',
         65, 24, 0, 0, (0, 0, 0, 0, 0), 'TString', ()),
    )),
    "TObject": (0x901bc02d, 1, (
        ('TStreamerBasicType', 'fUniqueID', 'object unique identifier',
         13, 4, 0, 0, (0, 0, 0, 0, 0), 'unsigned int', ()),
        ('TStreamerBasicType', 'fBits', 'bit field status word',
         15, 4, 0, 0, (0, 0, 0, 0, 0), 'unsigned int', ()),
    )),
    "TAttLine": (0x94074549, 2, (
        ('TStreamerBasicType', 'fLineColor', 'Line color',
         2, 2, 0, 0, (0, 0, 0, 0, 0), 'short', ()),
        ('TStreamerBasicType', 'fLineStyle', 'Line style',
         2, 2, 0, 0, (0, 0, 0, 0, 0), 'short', ()),
        ('TStreamerBasicType', 'fLineWidth', 'Line width',
         2, 2, 0, 0, (0, 0, 0, 0, 0), 'short', ()),
    )),
    "TAttFill": (0xffd92a92, 2, (
        ('TStreamerBasicType', 'fFillColor', 'Fill area color',
         2, 2, 0, 0, (0, 0, 0, 0, 0), 'short', ()),
        ('TStreamerBasicType', 'fFillStyle', 'Fill area style',
         2, 2, 0, 0, (0, 0, 0, 0, 0), 'short', ()),
    )),
    "TAttMarker": (0x291d8bec, 2, (
        ('TStreamerBasicType', 'fMarkerColor', 'Marker color',
         2, 2, 0, 0, (0, 0, 0, 0, 0), 'short', ()),
        ('TStreamerBasicType', 'fMarkerStyle', 'Marker style',
         2, 2, 0, 0, (0, 0, 0, 0, 0), 'short', ()),
        ('TStreamerBasicType', 'fMarkerSize', 'Marker size',
         5, 4, 0, 0, (0, 0, 0, 0, 0), 'float', ()),
    )),
    "TAxis": (0x5a496e70, 10, (
        ('TStreamerBase', 'TNamed', 'The basis for a named object (name, title)',
         67, 0, 0, 0, (0, -541636036, 0, 0, 0), 'BASE', (1,)),
        ('TStreamerBase', 'TAttAxis', 'Axis attributes',
         0, 0, 0, 0, (0, 1550843710, 0, 0, 0), 'BASE', (4,)),
        ('TStreamerBasicType', 'fNbins', 'Number of bins',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicType', 'fXmin', 'low edge of first bin',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fXmax', 'upper edge of last bin',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerObjectAny', 'fXbins', 'Bin edges array in X',
         62, 24, 0, 0, (0, 0, 0, 0, 0), 'TArrayD', ()),
        ('TStreamerBasicType', 'fFirst', 'first bin to display',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicType', 'fLast', 'last bin to display',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicType', 'fBits2', 'second bit status word',
         12, 2, 0, 0, (0, 0, 0, 0, 0), 'unsigned short', ()),
        ('TStreamerBasicType', 'fTimeDisplay', 'on/off displaying time values instead of numerics',
         18, 1, 0, 0, (0, 0, 0, 0, 0), 'bool', ()),
        ('TStreamerString', 'fTimeFormat', 'Date&time format, ex: 09/12/99 12:34:00',
         65, 24, 0, 0, (0, 0, 0, 0, 0), 'TString', ()),
        ('TStreamerObjectPointer', 'fLabels', 'List of labels',
         64, 8, 0, 0, (0, 0, 0, 0, 0), 'THashList*', ()),
        ('TStreamerObjectPointer', 'fModLabs', 'List of modified labels',
         64, 8, 0, 0, (0, 0, 0, 0, 0), 'TList*', ()),
    )),
    "TAttAxis": (0x5c6fff3e, 4, (
        ('TStreamerBasicType', 'fNdivisions', 'Number of divisions(10000*n3 + 100*n2 + n1)',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicType', 'fAxisColor', 'Color of the line axis',
         2, 2, 0, 0, (0, 0, 0, 0, 0), 'short', ()),
        ('TStreamerBasicType', 'fLabelColor', 'Color of labels',
         2, 2, 0, 0, (0, 0, 0, 0, 0), 'short', ()),
        ('TStreamerBasicType', 'fLabelFont', 'Font for labels',
         2, 2, 0, 0, (0, 0, 0, 0, 0), 'short', ()),
        ('TStreamerBasicType', 'fLabelOffset', 'Offset of labels',
         5, 4, 0, 0, (0, 0, 0, 0, 0), 'float', ()),
        ('TStreamerBasicType', 'fLabelSize', 'Size of labels',
         5, 4, 0, 0, (0, 0, 0, 0, 0), 'float', ()),
        ('TStreamerBasicType', 'fTickLength', 'Length of tick marks',
         5, 4, 0, 0, (0, 0, 0, 0, 0), 'float', ()),
        ('TStreamerBasicType', 'fTitleOffset', 'Offset of axis title',
         5, 4, 0, 0, (0, 0, 0, 0, 0), 'float', ()),
        ('TStreamerBasicType', 'fTitleSize', 'Size of axis title',
         5, 4, 0, 0, (0, 0, 0, 0, 0), 'float', ()),
        ('TStreamerBasicType', 'fTitleColor', 'Color of axis title',
         2, 2, 0, 0, (0, 0, 0, 0, 0), 'short', ()),
        ('TStreamerBasicType', 'fTitleFont', 'Font for axis title',
         2, 2, 0, 0, (0, 0, 0, 0, 0), 'short', ()),
    )),
    "THashList": (0xcc7e49c1, 0, (
        ('TStreamerBase', 'TList', 'Doubly linked list',
         0, 0, 0, 0, (0, 1774568379, 0, 0, 0), 'BASE', (5,)),
    )),
    "TList": (0x69c5c3bb, 5, (
        ('TStreamerBase', 'TSeqCollection', 'Sequenceable collection ABC',
         0, 0, 0, 0, (0, -60015674, 0, 0, 0), 'BASE', (0,)),
    )),
    "TSeqCollection": (0xfc6c3bc6, 0, (
        ('TStreamerBase', 'TCollection', 'Collection abstract base class',
         0, 0, 0, 0, (0, 1474546588, 0, 0, 0), 'BASE', (3,)),
    )),
    "TCollection": (0x57e3cb9c, 3, (
        ('TStreamerBase', 'TObject', 'Basic ROOT object',
         66, 0, 0, 0, (0, -1877229523, 0, 0, 0), 'BASE', (1,)),
        ('TStreamerString', 'fName', 'name of the collection',
         65, 24, 0, 0, (0, 0, 0, 0, 0), 'TString', ()),
        ('TStreamerBasicType', 'fSize', 'number of elements in collection',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
    )),
    "TString": (0x00017419, 2, (
    )),
    "TObjString": (0x9c8e4800, 1, (
        ('TStreamerBase', 'TObject', 'Basic ROOT object',
         66, 0, 0, 0, (0, -1877229523, 0, 0, 0), 'BASE', (1,)),
        ('TStreamerString', 'fString', 'wrapped TString',
         65, 24, 0, 0, (0, 0, 0, 0, 0), 'TString', ()),
    )),
    # TBits as ROOT 6.40 describes it (tests/data/references-6.40.root).
    "TBits": (0x0e763c48, 1, (
        ('TStreamerBase', 'TObject', 'Basic ROOT object',
         66, 0, 0, 0, (0, -1877229523, 0, 0, 0), 'BASE', (1,)),
        ('TStreamerBasicType', 'fNbits', 'Highest bit set + 1',
         13, 4, 0, 0, (0, 0, 0, 0, 0), 'unsigned int', ()),
        ('TStreamerBasicType', 'fNbytes', 'Number of UChars in fAllBits',
         13, 4, 0, 0, (0, 0, 0, 0, 0), 'unsigned int', ()),
        ('TStreamerBasicPointer', 'fAllBits', '[fNbytes] array of UChars',
         51, 8, 0, 0, (0, 0, 0, 0, 0), 'unsigned char*', (1, 'fNbytes', 'TBits')),
    )),
    "TBranchElement": (0xe6037caf, 10, (
        ('TStreamerBase', 'TBranch', 'Branch descriptor',
         0, 0, 0, 0, (0, 1494256824, 0, 0, 0), 'BASE', (12,)),
        ('TStreamerString', 'fClassName', 'Class name of referenced object',
         65, 24, 0, 0, (0, 0, 0, 0, 0), 'TString', ()),
        ('TStreamerString', 'fParentName', 'Name of parent class',
         65, 24, 0, 0, (0, 0, 0, 0, 0), 'TString', ()),
        ('TStreamerString', 'fClonesName', 'Name of class in TClonesArray (if any)',
         65, 24, 0, 0, (0, 0, 0, 0, 0), 'TString', ()),
        ('TStreamerBasicType', 'fCheckSum', 'CheckSum of class',
         13, 4, 0, 0, (0, 0, 0, 0, 0), 'unsigned int', ()),
        ('TStreamerBasicType', 'fClassVersion', 'Version number of class',
         2, 2, 0, 0, (0, 0, 0, 0, 0), 'short', ()),
        ('TStreamerBasicType', 'fID', 'element serial number in fInfo',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicType', 'fType', 'branch type',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicType', 'fStreamerType', 'branch streamer type',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicType', 'fMaximum', 'Maximum entries for a TClonesArray or variable array',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerObjectPointer', 'fBranchCount', 'pointer to primary branchcount branch',
         64, 8, 0, 0, (0, 0, 0, 0, 0), 'TBranchElement*', ()),
        ('TStreamerObjectPointer', 'fBranchCount2', 'pointer to secondary branchcount branch',
         64, 8, 0, 0, (0, 0, 0, 0, 0), 'TBranchElement*', ()),
    )),
    "TLeafElement": (0xa04f8893, 1, (
        ('TStreamerBase', 'TLeaf', 'Leaf: description of a Branch data type',
         0, 0, 0, 0, (0, 1830715730, 0, 0, 0), 'BASE', (2,)),
        ('TStreamerBasicType', 'fID', 'element serial number in fInfo',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicType', 'fType', 'leaf type',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
    )),
    "TLorentzVector": (0xe3dec1a1, 4, (
        ('TStreamerBase', 'TObject', 'Basic ROOT object',
         66, 0, 0, 0, (0, -1877229523, 0, 0, 0), 'BASE', (1,)),
        ('TStreamerObject', 'fP', '3 vector component',
         61, 40, 0, 0, (0, 0, 0, 0, 0), 'TVector3', ()),
        ('TStreamerBasicType', 'fE', 'time or energy of (x,y,z,t) or (px,py,pz,e)',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
    )),
    "TVector3": (0xabb6be1e, 3, (
        ('TStreamerBase', 'TObject', 'Basic ROOT object',
         66, 0, 0, 0, (0, -1877229523, 0, 0, 0), 'BASE', (1,)),
        ('TStreamerBasicType', 'fX', '',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fY', '',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fZ', '',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
    )),
    "TBranchObject": (0xa4720f49, 1, (
        ('TStreamerBase', 'TBranch', 'Branch descriptor',
         0, 0, 0, 0, (0, 278366892, 0, 0, 0), 'BASE', (13,)),
        ('TStreamerString', 'fClassName', 'Class name of referenced object',
         65, 24, 0, 0, (0, 0, 0, 0, 0), 'TString', ()),
    )),
    "TLeafObject": (0x26ba7c4c, 4, (
        ('TStreamerBase', 'TLeaf', 'Leaf: description of a Branch data type',
         0, 0, 0, 0, (0, 1830715730, 0, 0, 0), 'BASE', (2,)),
        ('TStreamerBasicType', 'fVirtual',
         'Support for polymorphism, when set classname is written with object.',
         18, 1, 0, 0, (0, 0, 0, 0, 0), 'bool', ()),
    )),
    "ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >": (0x5251e516, 1, (
        ('TStreamerObjectAny', 'fCoordinates', 'internal coordinate system',
         62, 32, 0, 0, (0, 0, 0, 0, 0), 'ROOT::Math::PxPyPzE4D<double>', ()),
    )),
    "ROOT::Math::PxPyPzE4D<double>": (0x568bb968, 1, (
        ('TStreamerBasicType', 'fX', '',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fY', '',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fZ', '',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fT', '',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
    )),
    "TClonesArray": (0xb6c0ca63, 4, (
        ('TStreamerBase', 'TObjArray', 'An array of objects',
         0, 0, 0, 0, (0, -1449237166, 0, 0, 0), 'BASE', (3,)),
    )),
    "TLine": (0x2a08f634, 3, (
        ('TStreamerBase', 'TObject', 'Basic ROOT object',
         66, 0, 0, 0, (0, -1877229523, 0, 0, 0), 'BASE', (1,)),
        ('TStreamerBase', 'TAttLine', 'Line attributes',
         0, 0, 0, 0, (0, -1811462839, 0, 0, 0), 'BASE', (2,)),
        ('TStreamerBase', 'TAttBBox2D', '2D bounding box attributes',
         0, 0, 0, 0, (0, 2443772, 0, 0, 0), 'BASE', (0,)),
        ('TStreamerBasicType', 'fX1', 'X of 1st point',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fY1', 'Y of 1st point',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fX2', 'X of 2nd point',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fY2', 'Y of 2nd point',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
    )),
    "TAttBBox2D": (0x002549fc, 0, (
    )),

    # The class a program's own class reaches as a member rather than as a base - a
    # ``TDatime`` held by value, which is what ``roottest/root/io/bigevent``'s ``Event``
    # carries - exactly as the oracle says its own ``TStreamerInfo`` holds it:
    #
    #   root -l -b -q -e 'for (auto n : {"TDatime"}) { auto *c = TClass::GetClass(n);
    #     printf("%s %#x v%d\n", n, (unsigned)c->GetCheckSum(),
    #     c->GetStreamerInfo()->GetClassVersion());
    #     c->GetStreamerInfo()->GetElements()->ls(); }'
    #
    # ``TArray`` and the arrays built on it, which that test reaches next as ``fArrayF`` and
    # ``fArrayI``, are in :data:`~.wdeclared.CARRIED_LAYOUTS` instead: each streams version 1,
    # ``fN`` then the values, and this table is what a file describes - which ROOT 6.40.04
    # writes without a word about them (see the module docstring).
    "TDatime": (0xb44671ee, 1, (
        ('TStreamerBasicType', 'fDatime', 'Date (relative to 1995) + time',
         13, 4, 0, 0, (0, 0, 0, 0, 0), 'unsigned int', ()),
    )),
    "TAttText": (0x2c0902d2, 2, (
        ('TStreamerBasicType', 'fTextAngle', 'Text angle',
         5, 4, 0, 0, (0, 0, 0, 0, 0), 'float', ()),
        ('TStreamerBasicType', 'fTextSize', 'Text size',
         5, 4, 0, 0, (0, 0, 0, 0, 0), 'float', ()),
        ('TStreamerBasicType', 'fTextAlign', 'Text alignment',
         2, 2, 0, 0, (0, 0, 0, 0, 0), 'short', ()),
        ('TStreamerBasicType', 'fTextColor', 'Text color',
         2, 2, 0, 0, (0, 0, 0, 0, 0), 'short', ()),
        ('TStreamerBasicType', 'fTextFont', 'Text font',
         2, 2, 0, 0, (0, 0, 0, 0, 0), 'short', ()),
    )),
    "TText": (0x3e81d43d, 3, (
        ('TStreamerBase', 'TNamed', 'The basis for a named object (name, title)',
         67, 0, 0, 0, (0, -541636036, 0, 0, 0), 'BASE', (1,)),
        ('TStreamerBase', 'TAttText', 'Text attributes',
         0, 0, 0, 0, (0, 738788050, 0, 0, 0), 'BASE', (2,)),
        ('TStreamerBase', 'TAttBBox2D', '2D bounding box attributes',
         0, 0, 0, 0, (0, 2443772, 0, 0, 0), 'BASE', (0,)),
        ('TStreamerBasicType', 'fX', 'X position of text (left,center,etc..)',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fY', 'Y position of text (left,center,etc..)',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
    )),
    "TBox": (0x65c15337, 3, (
        ('TStreamerBase', 'TObject', 'Basic ROOT object',
         66, 0, 0, 0, (0, -1877229523, 0, 0, 0), 'BASE', (1,)),
        ('TStreamerBase', 'TAttLine', 'Line attributes',
         0, 0, 0, 0, (0, -1811462839, 0, 0, 0), 'BASE', (2,)),
        ('TStreamerBase', 'TAttFill', 'Fill area attributes',
         0, 0, 0, 0, (0, -2545006, 0, 0, 0), 'BASE', (2,)),
        ('TStreamerBase', 'TAttBBox2D', '2D bounding box attributes',
         0, 0, 0, 0, (0, 2443772, 0, 0, 0), 'BASE', (0,)),
        ('TStreamerBasicType', 'fX1', 'X of 1st point',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fY1', 'Y of 1st point',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fX2', 'X of 2nd point',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fY2', 'Y of 2nd point',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
    )),
    "TPave": (0xb32728bf, 3, (
        ('TStreamerBase', 'TBox', 'Box class',
         0, 0, 0, 0, (0, 1707168567, 0, 0, 0), 'BASE', (3,)),
        ('TStreamerBasicType', 'fX1NDC', 'X1 point in NDC coordinates',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fY1NDC', 'Y1 point in NDC coordinates',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fX2NDC', 'X2 point in NDC coordinates',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fY2NDC', 'Y2 point in NDC coordinates',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fBorderSize', 'window box bordersize in pixels',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicType', 'fInit', '(=0 if transformation to NDC not yet done)',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicType', 'fShadowColor', "Color of the pave's shadow",
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicType', 'fCornerRadius', 'Corner radius in case of option arc',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerString', 'fOption', 'Pave style',
         65, 24, 0, 0, (0, 0, 0, 0, 0), 'TString', ()),
        ('TStreamerString', 'fName', 'Pave name',
         65, 24, 0, 0, (0, 0, 0, 0, 0), 'TString', ()),
    )),
    "TLatex": (0xef2c5136, 2, (
        ('TStreamerBase', 'TText', 'Text',
         0, 0, 0, 0, (0, 1048695869, 0, 0, 0), 'BASE', (3,)),
        ('TStreamerBase', 'TAttLine', 'Line attributes',
         0, 0, 0, 0, (0, -1811462839, 0, 0, 0), 'BASE', (2,)),
        ('TStreamerBasicType', 'fLimitFactorSize', 'lower bound for subscripts/superscripts size',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicType', 'fOriginSize', 'Font size of the starting font',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
    )),
    "TPaveText": (0x9cd2638d, 2, (
        ('TStreamerBase', 'TPave', 'Pave. A box with shadowing',
         0, 0, 0, 0, (0, -1289279297, 0, 0, 0), 'BASE', (3,)),
        ('TStreamerBase', 'TAttText', 'Text attributes',
         0, 0, 0, 0, (0, 738788050, 0, 0, 0), 'BASE', (2,)),
        ('TStreamerString', 'fLabel', 'Label written at the top of the pavetext',
         65, 24, 0, 0, (0, 0, 0, 0, 0), 'TString', ()),
        ('TStreamerBasicType', 'fLongest', 'Length of the longest line',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicType', 'fMargin', 'Text margin',
         5, 4, 0, 0, (0, 0, 0, 0, 0), 'float', ()),
        ('TStreamerObjectPointer', 'fLines', 'List of labels',
         64, 8, 0, 0, (0, 0, 0, 0, 0), 'TList*', ()),
    )),
    "TVirtualPaveStats": (0x4f46652c, 0, (
    )),
    "TPaveStats": (0x5b80dacc, 5, (
        ('TStreamerBase', 'TPaveText', 'PaveText. A Pave with several lines of text.',
         0, 0, 0, 0, (0, -1663933555, 0, 0, 0), 'BASE', (2,)),
        ('TStreamerBase', 'TVirtualPaveStats', 'Abstract interface for TPaveStats',
         0, 0, 0, 0, (0, 1330013484, 0, 0, 0), 'BASE', (0,)),
        ('TStreamerBasicType', 'fOptFit', 'option Fit',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicType', 'fOptStat', 'option Stat',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerString', 'fFitFormat', 'Printing format for fit parameters',
         65, 24, 0, 0, (0, 0, 0, 0, 0), 'TString', ()),
        ('TStreamerString', 'fStatFormat', 'Printing format for stats',
         65, 24, 0, 0, (0, 0, 0, 0, 0), 'TString', ()),
        ('TStreamerObjectPointer', 'fParent', 'owner of this TPaveStats',
         64, 8, 0, 0, (0, 0, 0, 0, 0), 'TObject*', ()),
    )),
    "TBranchRef": (0x2360b3fd, 1, (
        ('TStreamerBase', 'TBranch', 'Branch descriptor',
         0, 0, 0, 0, (0, 278366892, 0, 0, 0), 'BASE', (13,)),
        ('TStreamerObjectPointer', 'fRefTable', 'pointer to the TRefTable',
         64, 8, 0, 0, (0, 0, 0, 0, 0), 'TRefTable*', ()),
    )),
    "TRefTable": (0x8c895b85, 3, (
        ('TStreamerBase', 'TObject', 'Basic ROOT object',
         66, 0, 0, 0, (0, -1877229523, 0, 0, 0), 'BASE', (1,)),
        ('TStreamerBasicType', 'fSize', 'dummy for backward compatibility',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerObjectPointer', 'fParents',
          'array of Parent objects  (eg TTree branch) holding the referenced objects', 64, 8, 0, 0,
          (0, 0, 0, 0, 0), 'TObjArray*', ()),
        ('TStreamerObjectPointer', 'fOwner', 'Object owning this TRefTable',
         64, 8, 0, 0, (0, 0, 0, 0, 0), 'TObject*', ()),
        ('TStreamerSTL', 'fProcessGUIDs', 'UUIDs of TProcessIDs used in fParentIDs',
         500, 24, 0, 0, (0, 0, 0, 0, 0), 'vector<string>', (1, 61)),
    )),
}

#: The matrix classes, harvested from a donor written by ROOT 6.40.04 - the donor holds one
#: ``TMatrixD``, and the harvest tool (``tools/harvest_infos.py``) reports both classes and
#: their elements. ``TMatrixD`` is the typedef of ``TMatrixT<double>``, which is the name
#: ROOT puts in the file, so both are here: the base carries the shape, the derived one a
#: pointer to the values counted by ``fNelems``.
INFOS.update({
    "TMatrixTBase<double>": (2333786657, 5, (
        ('TStreamerBase', 'TObject', 'Basic ROOT object',
         66, 0, 0, 0, (0, -1877229523, 0, 0, 0), 'BASE', (1,)),
        ('TStreamerBasicType', 'fNrows', '', 3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicType', 'fNcols', '', 3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicType', 'fRowLwb', '', 3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicType', 'fColLwb', '', 3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicType', 'fNelems', '', 6, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicType', 'fNrowIndex', '', 3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicType', 'fTol', '', 8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
    )),
    "TMatrixT<double>": (135074716, 4, (
        ('TStreamerBase', 'TMatrixTBase<double>', 'Matrix base class',
         0, 0, 0, 0, (0, -1961180639, 0, 0, 0), 'BASE', (5,)),
        ('TStreamerBasicPointer', 'fElements', 'Matrix elements', 48, 8, 0, 0,
         (0, 0, 0, 0, 0), 'double*', (4, 'fNelems', 'TMatrixTBase<double>')),
    )),
})

#: The random engines, harvested from a donor written by ROOT 6.40.04 - one ``TRandom3``
#: seeded at 7. Three classes are needed for the one: the engine, its ``TRandom`` base (which
#: itself has ``TNamed`` and ``ROOT::Math::TRandomEngine`` bases) and the no-member interface.
INFOS.update({
    "ROOT::Math::TRandomEngine": (3688732995, 1, ()),
    "TRandom": (235880363, 3, (
        ('TStreamerBase', 'TNamed', 'The basis for a named object (name, title)',
         67, 0, 0, 0, (0, -541636036, 0, 0, 0), 'BASE', (1,)),
        ('TStreamerBase', 'ROOT::Math::TRandomEngine', '',
         0, 0, 0, 0, (0, -606234301, 0, 0, 0), 'BASE', (1,)),
        ('TStreamerBasicType', 'fSeed', 'Random number generator seed',
         13, 4, 0, 0, (0, 0, 0, 0, 0), 'unsigned int', ()),
    )),
    "TRandom3": (2185949371, 2, (
        ('TStreamerBase', 'TRandom', 'Simple Random number generator (periodicity = 10**9)',
         0, 0, 0, 0, (0, 235880363, 0, 0, 0), 'BASE', (3,)),
        ('TStreamerBasicType', 'fMt', '', 33, 4, 624, 0, (624, 0, 0, 0, 0), 'unsigned int', ()),
        ('TStreamerBasicType', 'fCount624', '', 3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
    )),
})

#: The single-precision matrix, harvested from a donor holding one ``TMatrixF`` - the same
#: shape as the double one with four-byte members and streamer type 45 for its elements.
INFOS.update({
    "TMatrixTBase<float>": (2142390597, 5, (
        ('TStreamerBase', 'TObject', 'Basic ROOT object',
         66, 0, 0, 0, (0, -1877229523, 0, 0, 0), 'BASE', (1,)),
        ('TStreamerBasicType', 'fNrows', '', 3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicType', 'fNcols', '', 3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicType', 'fRowLwb', '', 3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicType', 'fColLwb', '', 3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicType', 'fNelems', '', 6, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicType', 'fNrowIndex', '', 3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicType', 'fTol', '', 5, 4, 0, 0, (0, 0, 0, 0, 0), 'float', ()),
    )),
    "TMatrixT<float>": (3272986191, 4, (
        ('TStreamerBase', 'TMatrixTBase<float>', 'Matrix base class',
         0, 0, 0, 0, (0, 2142390597, 0, 0, 0), 'BASE', (5,)),
        ('TStreamerBasicPointer', 'fElements', '[fNelems] elements', 45, 4, 0, 0,
         (0, 0, 0, 0, 0), 'float*', (4, 'fNelems', 'TMatrixTBase<float>')),
    )),
})
