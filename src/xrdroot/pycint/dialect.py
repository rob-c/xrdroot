"""ROOT's dialect of C++: what a macro written against ROOT uses without declaring.

A ROOT macro is C++ in ROOT's dialect. Its typedefs (``Int_t``, ``Double32_t``)
are the types they stand for, a name spelled like ROOT's classes (``TH1F``,
``RooRealVar``, ``RVec``) is a type, its headers define ``ClassDef`` and
``R__LOAD_LIBRARY``, its unit tests have ``ROOT_EXPECT_ERROR``, and every name
the macro does not declare is ROOT's: ``ROOT.TH1F``, found in
:mod:`xrdroot.pyroot` through :data:`~.runtime.root.ROOT`. :data:`ROOT` says
all of that to :mod:`xrdroot.pycxx` as a :class:`~xrdroot.pycxx.dialect.Dialect`,
with the hooks of :mod:`.emit` for the patterns of ROOT's that need code.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..pycxx import gtest_macros
from ..pycxx.ctype import CType
from ..pycxx.dialect import Dialect, Prelude
from . import emit, returned
from .runtime import ROOT as HOST
from .runtime import __all__ as RUNTIME_NAMES
from .typedefs import TYPEDEFS

__all__ = ["ROOT", "dialect", "TYPEDEFS", "PREDEFINED", "ROOT_VERSION"]

#: ROOT's typedefs that are pointers to what a macro never sees inside - ``TXMLEngine``'s
#: handles - which a macro copies, compares and passes by reference as the pointers they are.
POINTER_TYPEDEFS = frozenset({"XMLNodePointer_t", "XMLAttrPointer_t", "XMLNsPointer_t",
                              "XMLDocPointer_t"})  # fmt: skip


#: What a ROOT class name looks like, when nothing declared it: TH1F, RooRealVar, RVec...
CLASS = re.compile(r"^(?:T[A-Z]|Roo[A-Z]|R[A-Z][a-z]|TMVA|Math[A-Z])\w*$")
#: What a ROOT typedef looks like, when nothing declared it: ``Int_t``, ``Color_t``.
TYPEDEF = re.compile(r"^[A-Z]\w*_t$")

#: The namespaces whose members named with a capital are types rather than functions.
TYPE_NAMESPACES = frozenset({"ROOT", "RooFit", "RooStats", "TMVA", "Math", "RDF", "VecOps"})

#: ROOT's templates, which begin a template argument list after them whatever else is known.
TEMPLATES = frozenset(
    """RVec TMatrixT TMatrixTSym TVectorT TMatrixTSparse TParameter TTreeReaderValue
    TTreeReaderArray RResultPtr RNode TArrayT RVecOps TNDArrayT THnT THnSparseT RHist RTensor
    TMatrixTRow TMatrixTColumn TMatrixTDiag dynamic_range_cast static_range_cast""".split()
)

#: ROOT's version as Cling reports it, ``ROOT_VERSION(6, 40, 4)``.
ROOT_VERSION = (6 << 16) | (40 << 8) | 4

#: The macros defined before a macro's first line, past the language's and the platform's:
#: Cling's and ROOT's own.
#: ``R__HAS_MATHMORE`` and ``R__HAS_VDT`` are RConfigure.h's for a ROOT built with MathMore
#: and VDT, as the ROOT releases (and the one tutorials are compared against) are: its macros
#: take those branches.
PREDEFINED = f"""
#define __CLING__ 1
#define R__UNIX 1
#define R__LINUX 1
#define R__USE_IMT 1
#define R__HAS_TMVACPU 1
#define R__HAS_MATHMORE 1
#define R__HAS_VDT 1
#define ROOT_VERSION_CODE {ROOT_VERSION}
#define ROOT_VERSION(a, b, c) (((a) << 16) + ((b) << 8) + (c))
#define R__LOAD_LIBRARY(x)
#define R__ADD_INCLUDE_PATH(x)
#define R__ADD_LIBRARY_PATH(x)
#define R__CLASS_VERSION(version) public: static Version_t Class_Version() {{ return version; }}
#define R__CLASS_INFO(name) static TClass *Class() {{ return TClass::GetClass(#name); }} \
   static const char *Class_Name() {{ return #name; }} \
   virtual TClass *IsA() const {{ return TClass::GetClass(#name); }} \
   virtual void Streamer(TBuffer &b) {{ TClass::GetClass(#name)->Streamer(this, b); }}
#define ClassDef(name, version) R__CLASS_VERSION(version) R__CLASS_INFO(name)
#define ClassDefOverride(name, version) R__CLASS_VERSION(version) R__CLASS_INFO(name)
#define ClassDefNV(name, version) R__CLASS_VERSION(version) R__CLASS_INFO(name)
#define ClassDefInline(name, version) R__CLASS_VERSION(version) R__CLASS_INFO(name)
#define ClassDefInlineOverride(name, version) R__CLASS_VERSION(version) R__CLASS_INFO(name)
#define ClassDefInlineNV(name, version) R__CLASS_VERSION(version) R__CLASS_INFO(name)
#define ClassImp(name)
#define ClassImpUnique(name, key)
#define templateClassImp(name)
#define NamespaceImp(name)
#define R__EXTERN extern
#define BIT(n) (1ULL << (n))
#define SETBIT(n, i) ((n) |= BIT(i))
#define CLRBIT(n, i) ((n) &= ~BIT(i))
#define TESTBIT(n, i) ((bool)(((n) & BIT(i)) != 0))
#define _R__UNIQUE_(X) X
#define R__DEPRECATED(maj, min, reason)
#define RQ_OBJECT(name)
#define Q_OBJECT
#define R__CLING_PTRCHECK(on)
#define _QUOTE_(name) #name
"""

#: ``ROOT/TestSupport.hxx``'s macros, past GoogleTest's own: the check of the diagnostics a
#: statement prints, which call ``::testing::internal``'s ``ExpectDiag``.
DIAGNOSTIC_MACROS = r"""
#define ROOT_EXPECT_DIAG(level, e, where, expected, full) \
   ::testing::internal::ExpectDiag([&]() { e; }, level, where, expected, full, __FILE__, __LINE__)
#define ROOT_EXPECT_NODIAG(e) ::testing::internal::ExpectNoDiag([&]() { e; }, __FILE__, __LINE__)
"""


#: The levels of ROOT's diagnostics that ``ROOT_EXPECT_<level>`` checks for, as ROOT names them.
DIAGNOSTICS = {"ERROR": "kError", "WARNING": "kWarning", "INFO": "kInfo", "SYSERROR": "kSysError"}


def _diagnostics() -> list[str]:
    """``ROOT_EXPECT_ERROR`` and kin, each also ``_PARTIAL``: the message need only be in it."""
    lines = []
    for word, level in DIAGNOSTICS.items():
        for suffix, full in (("", "true"), ("_PARTIAL", "false")):
            body = f"ROOT_EXPECT_DIAG({level}, e, where, expected, {full})"
            lines.append(f"#define ROOT_EXPECT_{word}{suffix}(e, where, expected) {body}")
    return lines


#: What including ``gtest/gtest.h`` or ``ROOT/TestSupport.hxx`` defines in a ROOT unit test:
#: GoogleTest's macros, and ``ROOT_EXPECT_ERROR`` and its kin, since every one may use them.
TESTS = Prelude(
    gtest_macros.PRELUDE + DIAGNOSTIC_MACROS + "\n".join(_diagnostics()) + "\n",
    "TEST",
    gtest_macros.REFUSED,
    "<gtest>",
)

#: The headers whose including defines those.
TEST_HEADERS = ("gtest/gtest.h", "gmock/gmock.h", "ROOT/TestSupport.hxx", "ROOTUnitTestSupport.h")

#: Headers of the libraries whose programs are not macros ROOT runs, and what each is.
OUT_OF_SCOPE = (
    (re.compile(r"^(?:Q[A-Z]\w*|Qt\w+/.*)$"), "a Qt widget program, which builds against Qt"),
    (re.compile(r"^(?:sycl/.*|CL/sycl\.hpp)$"), "a SYCL program, which runs on a device queue"),
)

#: ROOT methods that write through references, by name and number of arguments: which ones.
OUT_PARAMETERS: dict[tuple[str, int], tuple[int, ...]] = {
    ("Rannor", 2): (0, 1),
    ("Sphere", 4): (0, 1, 2),
    ("Circle", 3): (0, 1),
    ("GetRandom2", 2): (0, 1),
    ("GetRandom2", 3): (0, 1),
    ("GetRandom3", 3): (0, 1, 2),
    ("GetRandom3", 4): (0, 1, 2),
    ("GetPoint", 3): (1, 2),
    ("GetPoint", 4): (1, 2, 3),
    ("GetRange", 2): (0, 1),
    ("GetRange", 4): (0, 1, 2, 3),
    ("GetRange", 6): (0, 1, 2, 3, 4, 5),
    ("GetParLimits", 3): (1, 2),
    ("GetParameter", 3): (1, 2),
    ("GetBinXYZ", 4): (1, 2, 3),
    ("GetObject", 2): (1,),
    ("HLS2RGB", 6): (3, 4, 5),
    ("RGB2HLS", 6): (3, 4, 5),
    ("HSV2RGB", 6): (3, 4, 5),
    ("RGB2HSV", 6): (3, 4, 5),
    ("GetRGB", 3): (0, 1, 2),
    ("GetHLS", 3): (0, 1, 2),
    ("AbsPixeltoXY", 4): (2, 3),
    ("GetPadPar", 4): (0, 1, 2, 3),
    ("GetMinimumAndMaximum", 2): (0, 1),
    ("mnstat", 6): (0, 1, 2, 3, 4, 5),
    ("GetStats", 4): (0, 1, 2, 3),
    ("GetSigma", 2): (0, 1),
    ("GetSigmaX", 2): (0, 1),
    ("GetSigmaY", 2): (0, 1),
    ("GetRo", 2): (0, 1),
    ("GetBackgroundParameters", 6): (0, 1, 2, 3, 4, 5),
    ("GetTailParameters", 6): (0, 1, 2, 3, 4, 5),
    ("GetTailParameters", 16): tuple(range(16)),
    ("GetAngles", 3): (0, 1, 2),
    ("ShiftToNext", 1): (0,),
    ("GetEntryAndTree", 2): (1,),
    ("ShiftToNext", 2): (0,),
    ("GetAngles", 6): (0, 1, 2, 3, 4, 5),
    ("Det", 2): (0, 1),
    ("Determinant", 2): (0, 1),
    ("Solve", 2): (1,),
}


#: TMath's functions that return an integer; the rest of TMath returns a double - but for
#: those overloaded on their arguments, which return what they are given.
TMATH_INTEGERS = frozenset(
    "Nint FloorNint CeilNint BinarySearch LocMin LocMax Even Odd Permute".split()
)
TMATH_OVERLOADED = frozenset("Abs Sign Min Max Hypot".split())

#: What ROOT's decompositions give back by value - matrices and vectors, which ``auto x =
#: y`` then copies, as C++ copies them - by class and method.
RETURNED = {
    **{("TDecompQRH", name): "TMatrixD" for name in ("GetOrthogonalMatrix", "GetQ", "GetR")},
    **{("TDecompQRH", name): "TVectorD" for name in ("GetUp", "GetW")},
    **{("TDecompSVD", name): "TMatrixD" for name in ("GetU", "GetV", "Invert")},
    ("TDecompSVD", "GetSig"): "TVectorD",
    **{("TDecompLU", name): "TMatrixD" for name in ("GetLU", "Invert")},
    ("TDecompChol", "GetU"): "TMatrixD",
    **{(kind, "Invert"): "TMatrixDSym" for kind in ("TDecompChol", "TDecompBK")},
}

#: ROOT's getters, whose call leaves the object as it was: a loop bound using them is fixed.
READ_ONLY = frozenset(
    """GetN GetNbinsX GetNbinsY GetNbinsZ GetEntries GetEntriesFast GetSize GetLast GetNpar
    GetNdim GetNrows GetNcols GetNoElements GetXmin GetXmax GetBinContent GetNumberOfPoints
    GetListOfKeys numEntries getSize GetNumberOfBins GetNofBins""".split()
)

#: What a view of part of a matrix is called after: ``TMatrixDRow``, ``TMatrixTDiag<double>``.
VIEWS = ("Row", "Column", "Diag", "Sub", "Flat", "Row_const", "Column_const", "Diag_const")

#: ROOT's classes a variable of is set by ``=`` in place: a matrix's view, which ``view =
#: value`` writes through, and those whose ``operator=`` takes a number or a label rather than
#: making the object anew - ``RooRealVar x(...); x = 3;`` (rf402).
ASSIGNED = re.compile(
    rf"^(?:TMatrix.*(?:{'|'.join(VIEWS)})|RooRealVar|RooCategory)$", re.S
)

#: ``ROOT::RVecF`` and its kind: ``RVec``s whose element type their name gives.
TYPED_VECTORS = frozenset(
    f"RVec{kind}" for kind in ("B", "C", "D", "F", "I", "L", "LL", "U", "UL", "ULL")
)

#: ROOT's numeric aliases identify the same RVec specializations as their template spellings.
VECTOR_ALIASES = {"RVecC": "char", "RVecD": "double", "RVecF": "float", "RVecI": "int",
                  "RVecL": "long", "RVecLL": "long long", "RVecU": "unsigned int",
                  "RVecUL": "unsigned long", "RVecULL": "unsigned long long"}  # fmt: skip


@contextmanager
def _session() -> Iterator[None]:
    """A macro runs as cling runs it: compiled RooFit code streams a null pointer as ``0``."""
    from ..roofit.printing import NULL_POINTER

    before, NULL_POINTER[0] = NULL_POINTER[0], "0"
    try:
        yield
    finally:
        NULL_POINTER[0] = before


def _loaded(module: dict[str, Any], made: Any) -> None:
    """A macro's names from a header a loaded library was built from are the library's."""
    from .library.loading import rebind

    rebind(module, made)


def _data(obj: str, args: list[str]) -> str:
    """``s.Data()`` of a string: the string, which is what a ``TString``'s holds."""
    return obj


#: ROOT's dialect of C++.
ROOT = Dialect(
    name="root",
    translator="xrdroot.pycint",
    typedefs=TYPEDEFS,
    pointer_typedefs=POINTER_TYPEDEFS,
    class_pattern=CLASS,
    typedef_pattern=TYPEDEF,
    type_namespaces=TYPE_NAMESPACES,
    templates=TEMPLATES,
    literal_suffixes=frozenset({"_normal", "_px", "_user"}),
    predefined=PREDEFINED,
    preludes=dict.fromkeys(TEST_HEADERS, TESTS),
    out_of_scope=OUT_OF_SCOPE,
    host="ROOT",
    host_namespace="ROOT",
    runtime="xrdroot.pycint.runtime",
    runtime_names=frozenset(RUNTIME_NAMES),
    host_object=HOST,
    constants={"kTRUE": "True", "kFALSE": "False", "kNPOS": "npos"},
    functions={"R__ASSERT": "cassert"},
    member_pattern=re.compile(r"^f[A-Z]\w*$"),
    calls={"Format": emit.formatted},
    call_classes=frozenset({"TString"}),
    member_statement=emit.changing,
    out_parameters=OUT_PARAMETERS,
    returns={"Form": CType("char", pointer=1)},
    numeric_namespaces={"TMath": (TMATH_INTEGERS, TMATH_OVERLOADED)},
    method_returns=RETURNED,
    field_makers=frozenset({"MakeField"}),
    read_only=READ_ONLY,
    string_classes=frozenset({"TString"}),
    string_methods={"Data": _data},
    guards=frozenset({"TFile", "TMemFile"}),
    value_readers=frozenset({"TTreeReaderValue"}),
    array_readers=frozenset({"TTreeReaderArray"}),
    held_results=frozenset({"RResultPtr", "RDF::RResultPtr", "ROOT::RDF::RResultPtr"}),
    assigned_pattern=ASSIGNED,
    containers=frozenset({"RVec", *TYPED_VECTORS}),
    typed_containers=TYPED_VECTORS,
    vector="RVec",
    vector_aliases=VECTOR_ALIASES,
    vector_bool="ROOT::RVecB",
    vector_namespaces=frozenset({"ROOT", "ROOT::VecOps", "VecOps"}),
    vector_spellings=frozenset({"RVec", "ROOT::RVec", "ROOT::VecOps::RVec"}),
    counted_loop=emit.counted_loop,
    range_loop=emit.range_loop,
    spelling=returned.spelling,
    shown=returned.shown,
    status=returned.status,
    session=_session,
    loaded=_loaded,
    sources=(Path(__file__).parent,),
)


def dialect() -> Any:
    """:data:`ROOT`, for a caller below this package that reaches it through a provider."""
    return ROOT
