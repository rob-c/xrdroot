"""``RooCustomizer``: copies of a model with some of its nodes split by a category, or replaced.

Given a model and a category, ``build("A")`` makes the model for state ``A``: each node to
be split stood in for by its ``_A`` specialisation - one the caller made, from the set handed
in of every specialisation, or else a clone so named and titled ``(A)`` - and each node above
them cloned with the same suffix, so that the model, and its other states' copies, stay as
they were. Given a name instead of a category, it only replaces nodes, its copies suffixed
with that name. RooFit finds which input a node stands for by its ``ORIGNAME:<name>``
attribute; xrdroot sets and reads the same one.
"""

from __future__ import annotations

from typing import Any

from .arg import graph_changed
from .collections import RooArgList
from .messages import ERROR, INFO, WARNING, log

__all__ = ["RooCustomizer", "redirect"]


def _standing_for(one: Any, given: list[Any]) -> Any:
    """The node of ``given`` made to stand for ``one`` - carrying ``ORIGNAME:<its name>`` -
    or ``one`` itself."""
    wanted = f"ORIGNAME:{one.GetName()}"
    return next((node for node in given if node.getAttribute(wanted)), one)


def redirect(node: Any, given: list[Any]) -> None:
    """``RooAbsArg::redirectServers(given, false, true)``: each input of ``node`` one of
    ``given`` stands for replaced by it - in its inputs, and in the attributes that hold
    them, which a clone shares with what it was cloned from until then."""
    swapped: dict[int, Any] = {}
    for proxy in node._proxies:
        old = proxy.target
        if proxy.many:
            proxy.target = RooArgList([_standing_for(one, given) for one in old])
        else:
            proxy.target = _standing_for(old, given)
        swapped[id(old)] = proxy.target
    for key, value in list(vars(node).items()):
        if id(value) in swapped:
            node.__dict__[key] = swapped[id(value)]
    node.__dict__.pop("_norm_cache", None)  # its key is the leaves' values, not the nodes
    graph_changed()


def _marked(node: Any, original: Any) -> Any:
    """``node`` marked as standing for ``original``, as RooFit marks a specialisation."""
    node.setAttribute(f"ORIGNAME:{original.GetName()}")
    if not node.getStringAttribute("origName"):
        node.setStringAttribute("origName", original.GetName())
    return node


def _rule(rules: list[tuple[Any, Any]], node: Any) -> Any:
    """What the first rule for a node of ``node``'s name gives it, or ``None``."""
    return next((given for one, given in rules if one.GetName() == node.GetName()), None)


def _ordered(top: Any) -> tuple[list[Any], list[Any]]:
    """``top``'s leaves and its branches - ``top`` itself first - each in RooFit's order."""
    walked = list(top._walk())
    return [node for node in walked if node.isFundamental()], [
        node for node in walked if node.isDerived()
    ]


def _reconnect(clones: list[Any], stand_ins: list[Any], substitutes: list[Any]) -> None:
    """Each clone's inputs redirected to the clones, then the stand-ins, then the substitutes."""
    for clone in clones:
        for given in (clones, stand_ins, substitutes):
            redirect(clone, given)


def _error(text: str) -> None:
    log(None, ERROR, "InputArguments", text)


def _clone(node: Any, name: str, title: Any = None) -> Any:
    made = node.clone(name)
    made.removeStringAttribute("factory_tag")
    if title is not None:
        made.SetTitle(title)
    return _marked(made, node)


class RooCustomizer:
    """A model's customiser: ``RooCustomizer(model, category, made[, every])`` splits nodes by
    ``category``'s states, ``RooCustomizer(model, name)`` - a "sterile" one - only replaces."""

    def __init__(self, pdf: Any, category: Any, made: Any = None, every: Any = None) -> None:
        self._master = pdf
        self._sterile = isinstance(category, str)
        self._name = str(category) if self._sterile else ""
        self._category: Any = None if self._sterile else category
        self._made, self._every = made, every
        self._splits: list[tuple[Any, Any]] = []
        self._replacements: list[tuple[Any, Any]] = []
        self._branches: list[Any] = []

    def GetName(self) -> str:
        return str(self._master.GetName())

    def splitArg(self, arg: Any, category: Any) -> None:
        if any(one.GetName() == arg.GetName() for one, _ in self._splits):
            _error(
                f"RooCustomizer({self.GetName()}) ERROR: multiple "
                f"splitting rules defined for {arg.GetName()} only using first rule"
            )
        elif self._sterile:
            _error(
                f"RooCustomizer::splitArg({self._name}) ERROR "
                "cannot set spitting rules on this sterile customizer"
            )
        else:
            self._splits.append((arg, category))

    def splitArgs(self, args: Any, category: Any) -> None:
        for one in args:
            self.splitArg(one, category)

    def replaceArg(self, original: Any, substitute: Any) -> None:
        if any(one.GetName() == original.GetName() for one, _ in self._replacements):
            _error(
                f"RooCustomizer({self.GetName()}) ERROR: multiple "
                f"replacement rules defined for {original.GetName()} only using first rule"
            )
            return
        self._replacements.append((original, substitute))

    def cloneBranchList(self) -> list[Any]:
        return self._branches

    def cloneLeafList(self) -> Any:
        return self._made

    def build(self, state: Any = False, verbose: bool = False) -> Any:
        """The model for ``state`` of the category; or - given only ``verbose``, as a sterile
        customiser is - its copy suffixed with the customiser's name, if it has one."""
        if not isinstance(state, str):
            return self._build(self._name or None, bool(state))
        if self._sterile:
            _error(
                f"RooCustomizer::build({self._name}) ERROR cannot "
                "use leaf spitting build() on this sterile customizer"
            )
            return None
        if self._category.setLabel(state):
            _error(
                f"RooCustomizer::build({self.GetName()}): ERROR "
                f"label '{state}' not defined for master splitting category "
                f"{self._category.GetName()}"
            )
            return None
        return self._build(state, verbose)

    def _say(self, text: str, verbose: bool, level: int = INFO) -> None:
        if verbose:
            log(None, level, "ObjectHandling", f"RooCustomizer::build({self.GetName()}){text}")

    def _build(self, state: str | None, verbose: bool) -> Any:
        """``RooCustomizer::doBuild``: every node the rules name - leaves first, then branches,
        each in RooFit's order - stood in for, and every branch above them cloned and
        reconnected to the clones, the stand-ins and the substitutes."""
        leaves, branches = _ordered(self._master)
        split, stand_ins, replaced, substitutes = self._ruled(leaves + branches, state, verbose)
        pairs = self._cloned(branches, split, replaced, state, verbose)
        _reconnect([clone for _, clone in pairs], stand_ins, substitutes)
        self._branches.extend(clone for _, clone in pairs)
        return next((clone for one, clone in pairs if one is self._master), self._master)

    def _ruled(self, nodes: list[Any], state: str | None, verbose: bool) -> tuple[list[Any], ...]:
        """The nodes split and their stand-ins, and the nodes replaced and their substitutes."""
        split: list[Any] = []
        stand_ins: list[Any] = []
        replaced: list[Any] = []
        substitutes: list[Any] = []
        for node in nodes:
            category = _rule(self._splits, node)
            if category is not None:
                stand_ins.append(self._specialised(node, category, state, verbose))
                split.append(node)
            substitute = _rule(self._replacements, node)
            if substitute is not None:
                self._say(f": tree node {node.GetName()} will be replaced by "
                          f"{substitute.GetName()}", verbose)  # fmt: skip
                substitutes.append(_marked(substitute, node))
                replaced.append(node)
        return split, stand_ins, replaced, substitutes

    def _specialised(self, node: Any, category: Any, state: str | None, verbose: bool) -> Any:
        """The node standing for ``node`` in ``category``'s state: the one made already, or a
        clone so named and titled, added to the sets of what was made."""
        label = category.getCurrentLabel()
        self._say(
            f": tree node {node.GetName()} is split by category {category.GetName()}", verbose
        )
        name = f"{node.GetName()}_{label}" if state else node.GetName()
        known = self._every if self._every is not None else self._made
        found = known.find(name)
        if found is not None:
            self._say(f" Adding existing node specialization {name} to clonedMasterNodes", verbose)
            return _marked(found, node)
        if node.isDerived():
            self._say(
                f": WARNING: branch node {node.GetName()} is split but has no pre-defined "
                "specializations",
                True,
                WARNING,
            )
        made = _clone(node, name, f"{node.GetTitle()} ({label})")
        self._made.add(made)
        if self._every is not None:
            self._every.add(made)
        return made

    def _cloned(
        self,
        branches: list[Any],
        split: list[Any],
        replaced: list[Any],
        state: str | None,
        verbose: bool,
    ) -> list[tuple[Any, Any]]:
        """Each branch above a node split or replaced, and its clone, suffixed with ``state``."""
        made: list[tuple[Any, Any]] = []
        for branch in branches:
            name = branch.GetName()
            done = next(
                (
                    why
                    for why, nodes in (("split", split), ("replaced", replaced))
                    if any(one.GetName() == name for one in nodes)
                ),
                None,
            )
            if done is not None:
                self._say(f" Branch node {name} is already {done}", verbose)
                continue
            why = "split" if branch.dependsOn(split) else "replaced"
            if not branch.dependsOn(split + replaced):
                continue
            self._say(
                f" Branch node {branch.ClassName()}::{name} cloned: depends on a {why} parameter",
                verbose,
            )
            made.append((branch, _clone(branch, f"{name}_{state}" if state else name)))
        return made
