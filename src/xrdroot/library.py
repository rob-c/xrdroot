"""Columns handed over to the rest of scientific Python, in the shape each wants.

What a tree reads into is NumPy: an array per column, :class:`~.tree.Jagged`
for rows of varying length, and a list for columns of strings or objects.
That is already what most code wants. The rest of the ecosystem keeps tables
its own way - pandas as a frame, Awkward as nested arrays, Arrow and Polars as
columnar tables with real list types - and each of those is one keyword away:

    >>> tree.arrays(["pt", "eta"], library="pd")        # doctest: +SKIP

None of them is a dependency. Each is imported only when asked for, and one
that is not installed is refused with the ``pip install`` that fixes it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from .errors import UnsupportedFeatureError

if TYPE_CHECKING:
    from .tree import Jagged

__all__ = ["LIBRARIES", "convert", "awkward_list", "arrow_list"]

#: Every name a library can be asked for by, against the one it is known as.
LIBRARIES = {
    "np": "np",
    "numpy": "np",
    "pd": "pd",
    "pandas": "pd",
    "ak": "ak",
    "awkward": "ak",
    "pa": "pa",
    "pyarrow": "pa",
    "arrow": "pa",
    "pl": "pl",
    "polars": "pl",
}

#: The package each library is, for the refusal that says how to get it.
PACKAGES = {"pd": "pandas", "ak": "awkward", "pa": "pyarrow", "pl": "polars"}


def _module(name: str) -> Any:
    """Import one of the libraries, or refuse with the command that installs it."""
    import importlib

    package = PACKAGES[name]
    try:
        return importlib.import_module(package)
    except ImportError:
        raise UnsupportedFeatureError(
            f"library={name!r} needs {package}, which is not installed: pip install "
            f"{package} - or leave library out, and the columns come back as NumPy arrays"
        ) from None


def _jagged(value: Any) -> bool:
    from .tree import Jagged

    return isinstance(value, Jagged)


def awkward_list(rows: Jagged) -> Any:
    """Rows of varying length as an Awkward Array, over the same buffers."""
    ak = _module("ak")
    layout = ak.contents.ListOffsetArray(
        ak.index.Index64(rows.offsets), ak.contents.NumpyArray(rows.content)
    )
    return ak.Array(layout)


def arrow_list(rows: Jagged) -> Any:
    """Rows of varying length as an Arrow ``large_list``, whose offsets are 64-bit."""
    pa = _module("pa")
    return pa.LargeListArray.from_arrays(pa.array(rows.offsets), pa.array(rows.content))


def _to_awkward(value: Any) -> Any:
    ak = _module("ak")
    if _jagged(value):
        return awkward_list(value)
    if isinstance(value, np.ndarray):
        return ak.Array(ak.contents.NumpyArray(value))
    return ak.Array(value)


def _to_arrow(value: Any) -> Any:
    pa = _module("pa")
    if _jagged(value):
        return arrow_list(value)
    if isinstance(value, np.ndarray) and value.ndim > 1:
        width = int(np.prod(value.shape[1:]))
        return pa.FixedSizeListArray.from_arrays(pa.array(value.reshape(-1)), width)
    return pa.array(value)


def _to_pandas(value: Any) -> Any:
    """One column as pandas keeps it: numbers as they are, rows as objects."""
    if _jagged(value):
        return list(value)
    if isinstance(value, np.ndarray) and value.ndim > 1:
        return list(value)
    return value


def _numpy(columns: Mapping[str, Any]) -> dict[str, Any]:
    return dict(columns)


def _pandas(columns: Mapping[str, Any]) -> Any:
    pd = _module("pd")
    return pd.DataFrame({name: _to_pandas(value) for name, value in columns.items()})


def _awkward(columns: Mapping[str, Any]) -> Any:
    ak = _module("ak")
    return ak.zip({name: _to_awkward(value) for name, value in columns.items()}, depth_limit=1)


def _arrow(columns: Mapping[str, Any]) -> Any:
    pa = _module("pa")
    return pa.table({name: _to_arrow(value) for name, value in columns.items()})


def _polars(columns: Mapping[str, Any]) -> Any:
    pl = _module("pl")
    return pl.from_arrow(_arrow(columns))


#: How each library turns a mapping of columns into its own kind of table.
CONVERTERS: dict[str, Callable[[Mapping[str, Any]], Any]] = {
    "np": _numpy,
    "pd": _pandas,
    "ak": _awkward,
    "pa": _arrow,
    "pl": _polars,
}


def convert(columns: Mapping[str, Any], library: str) -> Any:
    """Columns read from one range of a tree, in the table ``library`` keeps.

    ``np`` - the default everywhere - is a dict of what the branches gave,
    ``pd`` a :class:`pandas.DataFrame`, ``ak`` an Awkward record array,
    ``pa`` a :class:`pyarrow.Table` and ``pl`` a :class:`polars.DataFrame`.
    """
    known = LIBRARIES.get(library)
    if known is None:
        raise ValueError(
            f"library={library!r} is not one of np, pd, ak, pa and pl (or numpy, pandas, "
            f"awkward, pyarrow and polars, spelled out)"
        )
    return CONVERTERS[known](columns)
