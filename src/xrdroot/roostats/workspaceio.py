"""RooStats objects reconstructed from the records of a RooFit workspace.

Reading ModelConfig belongs here so RooFit's reader can keep its registered
maker without importing the statistical engine until that maker is called.
"""

from __future__ import annotations

from typing import Any

from ..roofit.io.build import Builder, name_of, title_of
from ..roofit.io.stream import Streamed


def make_model_config(builder: Builder, record: Streamed) -> Any:
    """The names of the density and the sets, in the workspace being read."""
    from .modelconfig import ModelConfig

    made = ModelConfig(name_of(record), title_of(record))
    for key, member in (("Pdf", "fPdfName"), ("PriorPdf", "fPriorPdfName"),
                        ("ProtoData", "fProtoDataName"), ("Snapshot", "fSnapshotName"),
                        ("POI", "fPOIName"), ("NuisParams", "fNuisParamsName"),
                        ("ConstrainedParams", "fConstrParamsName"),
                        ("Observables", "fObservablesName"),
                        ("ConditionalObservables", "fConditionalObsName"),
                        ("GlobalObservables", "fGlobalObsName"),
                        ("ExternalConstraints", "fExtConstraintsName")):  # fmt: skip
        made._names[key] = str(record.get(member) or "")
    return made

