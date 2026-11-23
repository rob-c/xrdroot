"""Persist model links from the workspace graph rather than the live-object registry.

Workspace imports hold copies whose Python identities need not be in RooAbsArg's
live registry. Deriving their clients from the owned graph preserves ROOT's
cache invalidation after a file is reopened or a workspace is cloned.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ...errors import UnsupportedFeatureError
from ..shared import NULL_IDENTITY, PROPERTIES, PROPERTIES_LOCK, UUID_FIELDS, SharedBinnings
from .processids import SESSION
from .records import collection, named, put, record, refcounts
from .stream import Streamed


class Nodes:
    """A model node is written once, including proxies pointing back at its owner."""

    def __init__(self, process_id: int = 0) -> None:
        self.nodes: dict[int, Streamed] = {}
        self.proxies: list[Streamed] = []
        self.owners: dict[int, Any] = {}
        self.process_id = process_id
        self.shared: dict[int, Streamed] = {}
        self.property_maps: dict[int, tuple[Any, SharedBinnings]] = {}

    def node(self, obj: Any) -> Streamed:
        found = self.nodes.get(id(obj))
        if found is not None:
            return found
        method = self._model(obj)
        made = record(obj.ClassName())
        self.nodes[id(obj)] = made
        self.owners[id(made)] = obj
        named(made, obj.GetName(), obj.GetTitle())
        put(made, "_boolAttrib", sorted(obj._attributes))
        put(made, "_stringAttrib", dict(obj._strings))
        put(made, "_valueDirty", True)
        put(made, "_shapeDirty", True)
        self._values(obj, made)
        method(obj, made)
        return made

    def _values(self, obj: Any, made: Streamed) -> None:
        if obj.InheritsFrom("RooAbsReal") and not obj.InheritsFrom("RooAbsCategory"):
            put(made, "_value", float(obj.getVal()))
            put(made, "_unit", obj.getUnit())
            put(made, "_label", obj._plot_label)
            put(made, "_forceNumInt", obj.getForceNumInt())
            config = obj.specialIntegratorConfig()
            if config is not None:
                put(made, "_specIntegratorConfig", self.numint_config(config))
        if obj.InheritsFrom("RooAbsPdf"):
            put(made, "_selectComp", True)
            put(made, "_normRange", obj._norm_range or "")

    def _model(self, obj: Any) -> Callable[[Any, Streamed], None]:
        methods = {
            "RooRealVar": self.variable,
            "RooGaussian": self.gaussian,
            "RooChebychev": self.chebychev,
            "RooAddPdf": self.add_pdf,
            "RooConstVar": self.constant,
            "RooCategory": self.category,
            "RooPolynomial": self.polynomial,
            "RooRecursiveFraction": self.recursive_fraction,
        }
        method = methods.get(obj.ClassName())
        if method is None:
            raise UnsupportedFeatureError(
                f"Writing the workspace class {obj.ClassName()} is not implemented. "
                "Save a workspace containing supported RooFit classes instead."
            )
        return method

    def proxy(self, owner: Streamed, classname: str, name: str, target: Any,
              many: bool = False, value_server: bool = True) -> Streamed:
        made = record(classname)
        uid = SESSION.uid(self.owners[id(owner)], classname, name)
        identity = {"fUniqueID": uid, "fBits": 16, "pid": self.process_id}
        if many:
            put(made, "_list", [self.node(one) for one in target])
            put(made, "_name", name)
            put(made, "TObject", identity)
            put(made, "_defValueServer", value_server)
        else:
            put(made, "TNamed", {"fName": name, "fTitle": "", "TObject": identity})
            put(made, "_arg", self.node(target))
            put(made, "_valueServer", value_server)
            put(made, "_isFund", target.isFundamental())
        put(made, "_owner", owner)
        self.proxies.append(made)
        return made

    def variable(self, obj: Any, made: Streamed) -> None:
        put(made, "_fast", True)
        put(made, "_error", obj._error)
        put(made, "_asymErrLo", obj._asym[0])
        put(made, "_asymErrHi", obj._asym[1])
        put(made, "_binning", self.binning(obj.getBinning()))
        properties = self.shared_properties(obj)
        shared = self.shared.get(id(properties))
        if shared is None:
            shared = record("RooRealVarSharedProperties")
            put(shared, "_altBinning", {name: self.binning(binning)
                                      for name, binning in properties.items()})
            identity = shared.get("_uuid")
            identity.m.update(zip(UUID_FIELDS, properties.key() or NULL_IDENTITY, strict=True))
            self.shared[id(properties)] = shared
        put(made, "_sharedProp", shared)

    def shared_properties(self, obj: Any) -> SharedBinnings:
        values = obj._shared
        if not isinstance(values, SharedBinnings):
            held = self.property_maps.get(id(values))
            if held is None:
                # Hold the original map until serialization ends, so an id cannot be reused.
                held = (values, SharedBinnings(values))
                self.property_maps[id(values)] = held
            values = held[1]
            obj._shared = values
        with PROPERTIES_LOCK:
            key = values.key()
            if key is not None:
                existing = PROPERTIES.get(key)
                if existing is not None:
                    values = existing
                    obj._shared = values
                else:
                    PROPERTIES[key] = values
        return values

    def binning(self, obj: Any) -> Streamed:
        if obj.isParameterized():
            raise UnsupportedFeatureError(
                "Writing parameterized workspace binnings is not implemented. "
                "Save a workspace with fixed bin boundaries instead."
            )
        classname = type(obj).__name__
        made = record(classname)
        named(made, obj.GetName(), "")
        low, high = max(obj.lowBound(), -1e30), min(obj.highBound(), 1e30)
        if classname == "RooRangeBinning":
            made.m["_range"] = [low, high]
        elif obj.isUniform():
            count = obj.numBins()
            made.m.update(_xlo=low, _xhi=high, _nbins=count, _binw=(high - low) / count)
        else:
            edges = list(map(float, obj.array()))
            owns_low, owns_high = getattr(obj, "_streamed_bounds",
                                         (low not in obj._edges, high not in obj._edges))
            made.m.update(_xlo=low, _xhi=high, _nbins=obj.numBins(), _boundaries=edges,
                          _ownBoundLo=owns_low, _ownBoundHi=owns_high)
        return made

    def constant(self, obj: Any, made: Streamed) -> None:
        put(made, "_fast", True)

    def category(self, obj: Any, made: Streamed) -> None:
        if obj._ranges:
            raise UnsupportedFeatureError(
                "Writing workspace category named ranges is not implemented. "
                "Save a category without named ranges instead of losing its range definitions."
            )
        states = obj.states()
        put(made, "_currentIndex", obj.getIndex())
        put(made, "_stateNames", dict(sorted(states.items())))
        put(made, "_insertionOrder", list(states))

    def numint_config(self, config: Any) -> Streamed:
        from .records import linked

        made = record("RooNumIntConfig")
        made.m.update(_epsAbs=config.epsAbs(), _epsRel=config.epsRel(),
                      _printEvalCounter=config.printEvalCounter())
        for name, method in config._methods.items():
            put(made, "_" + name, self.node(method))
        made.m["_configSets"] = linked([self.listed(list(section), section.GetName(), True)
                                      for section in config._sections])
        return made

    def gaussian(self, obj: Any, made: Streamed) -> None:
        for name in ("x", "mean", "sigma"):
            put(made, name, self.proxy(made, "RooTemplateProxy<RooAbsReal>",
                                       name, getattr(obj, name)))

    def chebychev(self, obj: Any, made: Streamed) -> None:
        put(made, "_x", self.proxy(made, "RooTemplateProxy<RooAbsReal>", "x", obj.x))
        put(made, "_coefList", self.proxy(made, "RooCollectionProxy<RooArgList>",
                                         "coefList", obj.coefs, True))
        if obj._reference:
            reference = record("TNamed")
            reference.m.update(fName=obj._reference, fTitle="")
            put(made, "_refRangeName", reference)

    def polynomial(self, obj: Any, made: Streamed) -> None:
        put(made, "_x", self.proxy(made, "RooTemplateProxy<RooAbsReal>", "x", obj.x))
        put(made, "_coefList", self.proxy(made, "RooCollectionProxy<RooArgList>",
                                         "coefList", obj.coefs, True))
        put(made, "_lowestOrder", obj._lowest)

    def recursive_fraction(self, obj: Any, made: Streamed) -> None:
        put(made, "_list", self.proxy(made, "RooCollectionProxy<RooArgList>",
                                     "list", obj.fractions, True))

    def add_pdf(self, obj: Any, made: Streamed) -> None:
        if obj.getStringAttribute("ref_coef_norm"):
            raise UnsupportedFeatureError(
                "Writing RooAddPdf fixed coefficient normalization is not implemented. "
                "Remove the reserved reference-normalization attribute before saving the model."
            )
        put(made, "_refCoefNorm", self.proxy(
            made, "RooCollectionProxy<RooArgSet>", "!refCoefNorm", [], True, False))
        put(made, "_pdfList", self.proxy(made, "RooCollectionProxy<RooArgList>",
                                        "!pdfs", obj.pdfs, True))
        put(made, "_coefList", self.proxy(made, "RooCollectionProxy<RooArgList>",
                                         "!coefficients", obj.coefs, True))
        made.m.update(_haveLastCoef=obj._have_last, _allExtendable=obj._all_extendable,
                      _recursive=obj._recursive)

    def finalize_links(self, objects: list[Any]) -> None:
        for obj in objects:
            made = self.nodes[id(obj)]
            clients = [self.nodes[id(parent)] for parent in objects
                       if any(child is obj for child in parent.servers())]
            put(made, "_serverList", refcounts([self.node(one) for one in obj.servers()]))
            put(made, "_clientList", refcounts(clients))
            put(made, "_clientListValue", refcounts(clients))
            self._proxy_references(made)

    def _proxy_references(self, made: Streamed) -> None:
        identifiers = []
        for proxy in self.proxies:
            if proxy.get("_owner") is made:
                title = proxy.get("TNamed")
                identity = title["TObject"] if title else proxy.get("TObject")
                identifiers.append(identity["fUniqueID"])
        held = record("RooRefArray")
        held.m.update(uids=identifiers, pid=self.process_id)
        put(made, "_proxyList", held)

    def listed(self, objects: list[Any], name: str = "", owned: bool = False) -> Streamed:
        return collection("RooArgSet", [self.node(one) for one in objects], name, owned)
