"""Persisted numeric integration controls retain their private precision and methods."""

from __future__ import annotations

from typing import Any

from ..collections import RooArgSet
from ..numintconfig import RooNumIntConfig
from .build import Builder, maker
from .stream import Streamed


@maker("RooNumIntConfig")
def _numint_config(builder: Builder, record: Streamed) -> Any:
    made = RooNumIntConfig(RooNumIntConfig.defaultConfig())
    made.setEpsAbs(float(record.get("_epsAbs")))
    made.setEpsRel(float(record.get("_epsRel")))
    made.setPrintEvalCounter(bool(record.get("_printEvalCounter")))
    for name in made._methods:
        made._methods[name] = builder.node(record.get("_" + name))
    sections = []
    for section in record.get("_configSets").get("items"):
        copied = RooArgSet(builder.nodes(section.get("_list")))
        copied.SetName(section.get("_name"))
        sections.append(copied)
    made._sections = sections
    return made
