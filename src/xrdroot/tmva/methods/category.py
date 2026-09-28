"""``Category``: a method per region of the events, each trained and asked only there.

``AddMethod(cut, variables, type, title, options)`` gives each region - a
cut on the variables and spectators - its own method, over its own
variables, trained on a data set of its own: the loader's trees again, with
the region's cut added and the requested numbers scaled by the cut's
efficiency, as ``CreateCategoryDSI`` has it. The main data set gains a
spectator per region, ``<title>_cat<i> := <cut>``, which says for every
event which region's method answers it; an event in none is answered 0.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..dataset import DataSet, DataSetInfo, Events
from ..method import CLASSIFICATION, REGRESSION, Method
from ..variables import VariableInfo
from ..xmlfile import Node, children

__all__ = ["MethodCategory"]


class CategoryLoader:
    """A region's loader: its declarations, and its data set built from the main loader's trees."""

    def __init__(self, loader: Any, info: DataSetInfo) -> None:
        self.loader, self.info = loader, info
        self._dataset: DataSet | None = None

    def GetName(self) -> str:
        return self.info.name

    def dataset(self) -> DataSet:
        if self._dataset is None:
            from ..building import create_dataset

            self.loader.dataset()
            self._dataset = create_dataset(self.info, self.loader.inputs)
        return self._dataset


def category_info(primary: DataSetInfo, cut: str, variables: str, title: str) -> DataSetInfo:
    """``CreateCategoryDSI``: the region's variables, the main spectators, and the cut added."""
    made = DataSetInfo(f"{title}_dsi")
    made.targets = list(primary.targets)
    made.spectators = list(primary.spectators)
    known = [*primary.variables, *primary.spectators]
    for name in (piece for piece in variables.split(":") if piece):
        found = [info for info in known if info.label == name]
        if not found:
            raise made.log.fatal(f"The variable {name} was not found and could not be added ")
        made.variables.extend(found)
    if not variables:
        made.variables = list(primary.variables)
    for info in primary.classes:
        added = made.AddClass(info.name)
        added.weight, added.cut = info.weight, info.cut
        made.AddCut(cut, info.name)
    made.split_options = f"{primary.split_options}:ScaleWithPreselEff"
    made.normalization = primary.normalization
    return made


class MethodCategory(Method):
    """``TMVA::MethodCategory``: its regions, each a cut, variables and a method of its own."""

    type_name = "Category"
    analyses = frozenset({CLASSIFICATION, REGRESSION})
    evaluation_headers = 1

    def process_options(self) -> None:
        #: Each region: its cut, its variables, its method, its variables' columns, its spectator.
        self.subs: list[dict[str, Any]] = []

    def AddMethod(
        self, cut: Any, variables: Any, kind: Any, title: Any, options: Any = ""
    ) -> Method:
        """``AddMethod``: a region's method, booked over the region's own data set."""
        from ..factory import _method_name
        from ..weightfile import method_class

        name, title = _method_name(kind), str(title)
        if self.base_directory is not None:
            self.base_directory()
        self.log.info(f"Adding sub-classifier: {name}::{title}")
        info = category_info(self.dsi, str(cut), str(variables), title)
        made = method_class(name)(
            self.job, title, info, str(options), self.weight_dir, self.analysis
        )
        made.loader = CategoryLoader(self.loader, info) if self.loader is not None else None
        made.setup()
        self._region(str(cut), str(variables), made)
        return made

    def _region(self, cut: str, variables: str, made: Method) -> None:
        known = [*self.dsi.variables, *self.dsi.spectators]
        labels = [v.label for v in known]
        columns = [labels.index(v.label) for v in made.dsi.variables]
        spectator = VariableInfo(
            f"{self.name}_cat{len(self.subs) + 1}:={cut}", f"{self.name}:{made.name}", "pass", "C"
        )
        found = [i for i, s in enumerate(self.dsi.spectators) if s.label == spectator.label]
        if not found:
            self.dsi.spectators.append(spectator)
            found = [len(self.dsi.spectators) - 1]
        self.subs.append(
            {
                "cut": cut,
                "variables": variables,
                "method": made,
                "columns": columns,
                "spectator": found[0],
            }
        )

    def train(self, events: Events) -> None:
        from ..training import evaluate_sample

        what = "Regression" if self.analysis == REGRESSION else "Classification"
        self.log.info(f"Train all sub-classifiers for {what} ...")
        for sub in self.subs:
            method = sub["method"]
            train = method.loader.dataset().train
            self.log.info(f"Train method: {method.name} for {what}")
            method.raw_train = train
            method.train(method.handler.prepare(train) if method.handler.transforms else train)
            evaluate_sample(method, train, "training", method.dsi.name)
            if self.output is not None:
                from ..training import method_directory

                where = f"{method_directory(self)}/Method_{method.type_name}/{method.name}"
                method.monitoring(self.output, where)
            self.log.info("Training finished")
        if self.analysis != REGRESSION:
            self._rank()

    def _rank(self) -> None:
        from ..ranking import print_ranking

        self.log.info("Begin ranking of input variables...")
        for sub in self.subs:
            found = sub["method"].ranking()
            if found is None:
                self.log.info(f"No variable ranking supplied by classifier: {sub['method'].name}")
            else:
                print_ranking(sub["method"].name, *found)

    def _sub_events(self, events: Events, sub: dict[str, Any]) -> Events:
        table = np.column_stack([events.values, events.spectators])
        return events.with_values(table[:, sub["columns"]])

    def _passes(self, events: Events, sub: dict[str, Any]) -> Any:
        """``PassesCut``: the region's spectator, or - for a reader, which has none - its cut."""
        spectators = np.asarray(events.spectators)
        if spectators.ndim == 2 and sub["spectator"] < spectators.shape[1]:
            return spectators[:, sub["spectator"]] > 0.5
        from ...formula import compile_formula

        known = [*self.dsi.variables, *self.dsi.spectators]
        table = np.column_stack([events.values, spectators])
        columns = {info.label: table[:, i] for i, info in enumerate(known[: table.shape[1]])}
        found = compile_formula(sub["cut"], list(columns)).evaluate(columns)
        return np.broadcast_to(np.asarray(found, dtype=bool), (len(events),))

    def mva(self, events: Events) -> Any:
        output = np.zeros(len(events))
        kind = getattr(self, "sample_kind", "")
        for sub in self.subs:
            method = sub["method"]
            if kind:
                method.log.header(
                    f"[{method.dsi.name}] : Evaluation of {method.name} on {kind} "
                    f"sample ({len(events)} events)"
                )
            mask = self._passes(events, sub)
            if mask.any():
                values = np.asarray(method.mva(self._sub_events(events, sub)), dtype=np.float64)
                output[mask] = values.reshape(len(events), -1)[mask, 0]
        return output

    def evaluate(self, values: Any) -> Any:
        raise self.log.fatal("Category answers events, which carry the spectators of its regions")

    # -- the weight file ------------------------------------------------------------------

    def add_weights(self, node: Node) -> None:
        """``AddWeightsXMLTo``: every region's cut and variables, and its method's whole state."""
        weights = node.add("Weights", NSubMethods=len(self.subs))
        for index, sub in enumerate(self.subs):
            method = sub["method"]
            made = weights.add(
                "SubMethod",
                Index=index,
                Method=f"{method.type_name}::{method.name}",
                Cut=sub["cut"],
                Variables=sub["variables"],
            )
            made.children.extend(method.to_xml().children)

    def read_weights(self, node: Any) -> None:
        from ..weightfile import read_method_node

        self.subs = []
        self.log.info("Recreating sub-classifiers from XML-file ")
        for item in children(node, "SubMethod"):
            kind, _, title = str(item.get("Method")).partition("::")
            info = category_info(self.dsi, str(item.get("Cut")), str(item.get("Variables")), title)
            made = read_method_node(item, kind, title, info, self.job)
            self._region(str(item.get("Cut")), str(item.get("Variables")), made)
