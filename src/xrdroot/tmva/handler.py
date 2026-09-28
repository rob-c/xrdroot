"""``TransformationHandler``: a chain of transformations, and what TMVA prints about it.

A method's ``VarTransform=G,D`` and each of the Factory's ``Transformations``
is a handler: the transformations named, created with TMVA's
``Create Transformation "D" with events from all classes.`` and its
variable-selection listing, prepared one after the other on the training
events - each on what the one before made of them - and then described by
the table TMVA prints of each variable's mean, RMS and range.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .dataset import DataSetInfo, Events
from .log import Logger
from .transforms import Transform, make_transform

__all__ = ["TransformationHandler", "parse_definition", "stats_table"]


def parse_definition(definition: str) -> list[tuple[str, int | str]]:
    """``"G,D"`` or ``"D_Signal+N"``: each transformation's name and the class it is made from."""
    depth, pieces, current = 0, [], ""
    for character in definition:
        depth += (character == "(") - (character == ")")
        if character in ",+" and depth == 0:
            pieces.append(current)
            current = ""
        else:
            current += character
    pieces.append(current)
    parsed: list[tuple[str, int | str]] = []
    for piece in pieces:
        name = piece.split("(")[0].strip()
        if not name:
            continue
        letter, _, cls = name.partition("_")
        parsed.append((letter, cls if cls and cls != "AllClasses" else -1))
    return parsed


def stats_table(labels: list[str], stats: Any) -> list[str]:
    """``TransformationHandler::CalcStats``'s table: a header line, then each variable's row."""
    width = max([8, *(len(label) for label in labels)])
    column = width + 2
    rule = "-" * (width + 4 * column + 11)
    head = (
        "Variable".rjust(width)
        + "  "
        + "Mean".rjust(column)
        + " "
        + "RMS".rjust(column)
        + "   "
        + "[        Min ".rjust(column)
        + "  "
        + "    Max ]".rjust(column)
    )
    rows = [head, rule]
    for index, label in enumerate(labels):
        mean, rms, low, high = (float(part[index]) for part in stats)
        rows.append(
            f"{label.rjust(width)}:{_g(mean, column)}{_g(rms, column)}   "
            f"[{_g(low, column)}{_g(high, column)} ]"
        )
    rows.append(rule)
    return rows


def _g(value: float, width: int) -> str:
    return format(value, "#11.5g").rjust(width)


def weighted_stats(values: Any, weights: Any) -> tuple[Any, Any, Any, Any]:
    """Each column's weighted mean and RMS, and its range, as ``CalcStats`` takes them."""
    total = float(np.sum(weights))
    mean = (weights @ values) / total
    rms = np.sqrt(np.maximum((weights @ (values * values)) / total - mean * mean, 0.0))
    return mean, rms, values.min(axis=0), values.max(axis=0)


class TransformationHandler:
    """The transformations one method - or the Factory - applies, in order."""

    def __init__(self, dsi: DataSetInfo, caller: str) -> None:
        self.dsi = dsi
        self.caller = caller
        self.transforms: list[Transform] = []
        self.log = Logger(f"TFHandler_{caller}")
        self.stats: tuple[Any, Any, Any, Any] | None = None

    @property
    def name(self) -> str:
        """``GetName``: the transformations' names joined by ``_`` - ``Gauss_Deco``."""
        return "_".join(transform.name for transform in self.transforms)

    def create(self, definition: str, log: Logger) -> None:
        """``CreateVariableTransforms``: each transformation of ``definition`` made, and said so."""
        if definition in ("", "None"):
            return
        for letter, cls in parse_definition(definition):
            reference = self._reference(cls, log)
            transform = make_transform(letter, reference)
            if reference >= 0:
                log.header(
                    f'[{self.dsi.name}] : Create Transformation "{letter}" with reference class '
                    f"{self.dsi.classes[reference].name}=({reference})"
                )
            else:
                log.header(
                    f'[{self.dsi.name}] : Create Transformation "{letter}" '
                    "with events from all classes."
                )
            log.info("")
            self._selection(transform, log)
            self.transforms.append(transform)

    def _reference(self, cls: int | str, log: Logger) -> int:
        if isinstance(cls, int):
            return cls
        info = self.dsi.GetClassInfo(cls)
        if info is None:
            raise log.fatal(
                f"Dataset[{self.dsi.name}] : Class {cls} not known for variable transformation, "
                "please check."
            )
        return info.number

    def _selection(self, transform: Transform, log: Logger) -> None:
        """The ``Transformation, Variable selection :`` listing of what goes in and out."""
        log.info("Transformation, Variable selection : ")
        for variable in self.dsi.variables:
            log.info(
                f"Input : variable '{variable.label}' <---> Output : variable '{variable.label}'"
            )
        if transform.targets:
            for target in self.dsi.targets:
                log.info(f"Input : target '{target.label}' <---> Output : target '{target.label}'")

    def prepare(self, events: Events) -> Events:
        """``CalcTransformations``: each prepared in turn, the result described; the events made."""
        current = events
        nclasses = self.dsi.GetNClasses()
        for transform in self.transforms:
            announced = transform.announce()
            if announced:
                Logger(transform.__class__.__name__).info(announced)
            transform.prepare(current, nclasses)
            current = self._through(transform, current, None)
        self.print_stats(current)
        return current

    def _through(self, transform: Transform, events: Events, cls: int | None) -> Events:
        made = events.with_values(transform.apply(events.values, cls))
        apply_targets = getattr(transform, "apply_targets", None)
        if apply_targets is not None and events.targets.shape[1]:
            made.targets = apply_targets(events.targets, events.values.shape[1], cls)
        return made

    def apply(self, events: Events, cls: int | None = None) -> Events:
        """``events`` through every transformation, with class ``cls``'s parameters if given."""
        for transform in self.transforms:
            events = self._through(transform, events, cls)
        return events

    def apply_values(self, values: Any, cls: int | None = None) -> Any:
        for transform in self.transforms:
            values = transform.apply(values, cls)
        return values

    def inverse_targets(self, targets: Any, cls: int | None = None) -> Any:
        """A regression's output taken back through the target transformations, last first."""
        nvar = self.dsi.GetNVariables()
        for transform in reversed(self.transforms):
            inverse = getattr(transform, "inverse_targets", None)
            if inverse is not None:
                targets = inverse(targets, nvar, cls)
        return targets

    def print_stats(self, events: Events) -> None:
        """The table of each variable's (and target's) mean, RMS and range, weighted."""
        values = np.column_stack([events.values, events.targets])
        self.stats = weighted_stats(values, events.weights)
        labels = [v.label for v in self.dsi.variables] + [t.label for t in self.dsi.targets]
        lines = stats_table(labels, self.stats)
        self.log.header(lines[0])
        for line in lines[1:]:
            self.log.info(line)
