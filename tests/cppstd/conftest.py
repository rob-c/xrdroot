"""Restore generic state as one unit; integration fixtures remain in the parent tree."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Any

import numpy as np
import pytest

from xrdroot.cppstd import (
    cmath,
    containers,
    istreams,
    objects,
    printf,
    stdlib,
    streams,
    units,
    vectorassign,
    vectorvalue,
)


def _mappings() -> list[dict[Any, Any]]:
    return [
        containers.NUMBERS,
        containers.TYPE_ALIASES,
        containers.PYTHON_TYPES,
        containers.TEMPLATES,
        containers._OPAQUE,
        stdlib._KINDS,
        streams.FORMATTERS,
        vectorvalue._KINDS,
        vectorassign._LAYOUTS,
        vectorassign._CAPACITIES,
        istreams.KINDS,
        istreams.PATTERNS,
        objects.DTYPES,
        printf.BITS,
        printf.CONVERTERS,
        units.DURATIONS,
    ]


def _restore_mapping(target: dict[Any, Any], saved: dict[Any, Any]) -> None:
    target.clear()
    target.update(saved)


def _restore_mappings(mappings: list[tuple[dict[Any, Any], dict[Any, Any]]]) -> None:
    for target, saved in mappings:
        _restore_mapping(target, saved)


@contextmanager
def _preserved_state() -> Iterator[None]:
    # Factory caches and provenance registries must describe the same concrete classes.
    mappings = [(target, dict(target)) for target in _mappings()]
    caches = [
        (template, template._made, dict(template._made))
        for template in containers.TEMPLATES.values()
    ]
    resolver = containers.TYPE_RESOLVER
    constructor = cmath.VECTOR_RESULT
    outputs = [
        (stream, dict(vars(stream))) for stream in (streams.cout, streams.cerr, streams.clog)
    ]
    random_state = np.random.get_state()
    try:
        yield
    finally:
        _restore_mappings(mappings)
        for template, cache, saved in caches:
            template._made = cache
            _restore_mapping(cache, saved)
        containers.set_type_resolver(resolver)
        cmath.set_vector_result(constructor)
        for stream, saved in outputs:
            _restore_mapping(vars(stream), saved)
        np.random.set_state(random_state)


@pytest.fixture(autouse=True)
def _cppstd_state_restored() -> Iterator[None]:
    with _preserved_state():
        yield


@pytest.fixture
def cppstd_state_context() -> Callable[[], AbstractContextManager[None]]:
    """Exercise restoration through the same context the autouse fixture uses."""
    return _preserved_state
