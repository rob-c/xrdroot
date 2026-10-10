"""ROOT 6.40's HistFactory classes as its dictionaries lay them out, member by member.

What the writer streams a ``Measurement`` by (:mod:`.streamed`), and what
the reader falls back on for a file ROOT wrote: ROOT describes in a file
only the classes it streamed whole, and the ones it streamed member-wise
inside a vector - a ``HistoSys`` is always one of those - go undescribed,
so a file ROOT wrote cannot be read by its own account alone.
"""

from __future__ import annotations

from ..streamers import Member

__all__ = ["CHECKSUMS", "LAYOUTS", "described", "full", "kind_type"]

#: ROOT 6.40's checksums of HistFactory's classes without a ``ClassDef``, and of the pairs
#: its maps hold; the ``Measurement`` itself is version 3.
CHECKSUMS: dict[str, int] = {
    "Channel": 0xD0B92F90, "Data": 0xE5DA1CB4, "HistRef": 0xE4043B25,
    "StatErrorConfig": 0x2C1E9C30, "Sample": 0x7A684602, "OverallSys": 0xD650479F,
    "NormFactor": 0x983E10B6, "HistoSys": 0xF3D0913B, "HistoFactor": 0x94DD4875,
    "ShapeSys": 0x6BA8A5FB, "ShapeFactor": 0xE3AC4CE0, "StatError": 0xE7C9D11C,
    "HistogramUncertaintyBase": 0xF91EB892, "PreprocessFunction": 0x6740BEA3,
    "Asimov": 0xEC1B334B, "pair<string,double>": 0xFA9428B2, "pair<string,bool>": 0x0B5FB752,
}  # fmt: skip

#: Each class's members in order and what each is: a string, a number, a vector of strings,
#: a map, another class ("Data"), a vector of one ("[Sample]"), a base class or a histogram.
LAYOUTS: dict[str, tuple[tuple[str, str], ...]] = {
    "Measurement": (
        ("fOutputFilePrefix", "string"), ("fPOI", "strings"), ("fLumi", "double"),
        ("fLumiRelErr", "double"), ("fBinLow", "int"), ("fBinHigh", "int"),
        ("fExportOnly", "bool"), ("fInterpolationScheme", "string"), ("fChannels", "[Channel]"),
        ("fConstantParams", "strings"), ("fParamValues", "map"),
        ("fFunctionObjects", "[PreprocessFunction]"), ("fAsimovDatasets", "[Asimov]"),
        ("fGammaSyst", "map"), ("fUniformSyst", "map"), ("fLogNormSyst", "map"),
        ("fNoSyst", "map"),
    ),
    "Channel": (
        ("fName", "string"), ("fInputFile", "string"), ("fHistoPath", "string"),
        ("fData", "Data"), ("fAdditionalData", "[Data]"),
        ("fStatErrorConfig", "StatErrorConfig"), ("fSamples", "[Sample]"),
    ),
    "Data": (("fName", "string"), ("fInputFile", "string"), ("fHistoName", "string"),
             ("fHistoPath", "string"), ("fhData", "HistRef")),
    "HistRef": (("fHist", "TH1*"),),
    "StatErrorConfig": (("fRelErrorThreshold", "double"), ("fConstraintType", "int")),
    "Sample": (
        ("fName", "string"), ("fInputFile", "string"), ("fHistoName", "string"),
        ("fHistoPath", "string"), ("fChannelName", "string"),
        ("fOverallSysList", "[OverallSys]"), ("fNormFactorList", "[NormFactor]"),
        ("fHistoSysList", "[HistoSys]"), ("fHistoFactorList", "[HistoFactor]"),
        ("fShapeSysList", "[ShapeSys]"), ("fShapeFactorList", "[ShapeFactor]"),
        ("fStatError", "StatError"), ("fNormalizeByTheory", "bool"),
        ("fStatErrorActivate", "bool"), ("fhNominal", "HistRef"), ("fhCountingHist", "TH1*"),
    ),
    "OverallSys": (("fName", "string"), ("fLow", "double"), ("fHigh", "double")),
    "NormFactor": (("fName", "string"), ("fVal", "double"), ("fLow", "double"),
                   ("fHigh", "double")),
    "HistogramUncertaintyBase": (
        ("fName", "string"), ("fInputFileLow", "string"), ("fHistoNameLow", "string"),
        ("fHistoPathLow", "string"), ("fInputFileHigh", "string"),
        ("fHistoNameHigh", "string"), ("fHistoPathHigh", "string"), ("fhLow", "TH1*"),
        ("fhHigh", "TH1*"),
    ),
    "HistoSys": (("HistogramUncertaintyBase", "base"),),
    "HistoFactor": (("HistogramUncertaintyBase", "base"),),
    "ShapeSys": (("HistogramUncertaintyBase", "base"), ("fConstraintType", "int")),
    "ShapeFactor": (("HistogramUncertaintyBase", "base"), ("fConstant", "bool"),
                    ("fHasInitialShape", "bool"), ("fVal", "double"), ("fLow", "double"),
                    ("fHigh", "double")),
    "StatError": (("HistogramUncertaintyBase", "base"), ("fActivate", "bool"),
                  ("fUseHisto", "bool")),
    "PreprocessFunction": (("fName", "string"), ("fExpression", "string"),
                           ("fDependents", "string")),
    "Asimov": (("fName", "string"), ("fParamsToFix", "mapbool"), ("fParamValsToSet", "map")),
}  # fmt: skip

#: ROOT's streamer type of a base, a number, an object pointer, a whole object and an STL
#: container or string, and the type names its STL members are declared with.
BASE, NUMBER, POINTER, OBJECT, STL = 0, {"double": 8, "int": 3, "bool": 18}, 64, 62, 500
STL_TYPES = {"string": "string", "strings": "vector<string>", "map": "map<string,double>",
             "mapbool": "map<string,bool>"}  # fmt: skip


def full(name: str) -> str:
    """A HistFactory class's name with its namespaces."""
    return f"RooStats::HistFactory::{name}"


def kind_type(member: str, kind: str) -> tuple[int, str]:
    """A member's streamer type and the type name a ``TStreamerElement`` declares it with."""
    if kind == "base":
        return BASE, "BASE"
    if kind in NUMBER:
        enum = member == "fConstraintType"
        return NUMBER[kind], full("Constraint::Type") if enum else kind
    if kind == "TH1*":
        return POINTER, "TH1*"
    if kind in STL_TYPES:
        return STL, STL_TYPES[kind]
    if kind.startswith("["):
        return STL, f"vector<{full(kind[1:-1])}>"
    return OBJECT, full(kind)


def described() -> dict[str, dict[str, Member]]:
    """Every HistFactory class as a file's streamer information would describe it."""
    found: dict[str, dict[str, Member]] = {}
    for classname, layout in LAYOUTS.items():
        members = [Member("TNamed", "", 67, "BASE", 0)] if classname == "Measurement" else []
        for member, kind in layout:
            stype, typename = kind_type(member, kind)
            name = full(member) if kind == "base" else member
            members.append(Member(name, "", stype, typename, 0))
        found[full(classname)] = {one.name: one for one in members}
    return found
