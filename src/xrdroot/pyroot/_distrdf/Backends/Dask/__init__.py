"""``ROOT.RDF.Distributed.Dask``: a distributed ``RDataFrame`` on a Dask cluster."""

from __future__ import annotations

import warnings
from typing import Any

__all__ = ["RDataFrame"]


def RDataFrame(*args: Any, **kwargs: Any) -> Any:
    """A distributed frame of ``args``' data on the Dask client given as ``executor`` - or,
    as before ROOT 6.36, as ``daskclient``, which is warned about."""
    from . import Backend

    daskclient = kwargs.pop("daskclient", None)
    executor = kwargs.pop("executor", None)
    if daskclient is not None:
        warnings.warn(
            "The keyword argument 'daskclient' is not necessary anymore and will be removed "
            "in a future release. To provide the distributed.Client object, use 'executor' "
            "instead.",
            FutureWarning,
            stacklevel=2,
        )
        if executor is not None:
            raise ValueError(
                "Both the 'daskclient' and 'executor' keyword arguments were provided. This is "
                "not supported. Please provide only the 'executor' argument."
            )
        executor = daskclient
    return Backend.DaskBackend(daskclient=executor).make_dataframe(*args, **kwargs)
