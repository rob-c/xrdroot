"""The user's own TorchScript model: loaded, saved, and handed to the loop that trains it.

PyMVA's ``PyTorch`` method has no learner of its own. The user's Python file
builds a model, scripts it (``torch.jit.script``) and saves it, and hands
TMVA the training loop, the loss, the optimiser and the prediction function
in one dict, ``load_model_custom_objects``; TMVA then calls them on the
events as NumPy arrays. Only the handful of PyTorch calls below are used, so
that the whole of it is exercised in the tests against a small stand-in for
PyTorch where PyTorch itself is not installed.
"""

from __future__ import annotations

import importlib
import importlib.util
import runpy
from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = [
    "available",
    "best_checkpoint",
    "load",
    "loader",
    "module",
    "predictions",
    "save",
    "schedule",
    "user_code",
]


def available() -> bool:
    """Is PyTorch installed?"""
    return importlib.util.find_spec("torch") is not None


def module() -> Any:
    """PyTorch itself, which the method holds from the moment it is booked."""
    return importlib.import_module("torch")


def user_code(path: str) -> dict[str, Any]:
    """The names the user's file defines, run as ``__main__`` as ROOT runs it.

    A user's file is a script, not a module: ROOT runs it with ``PyRun_SimpleFile``,
    so what it does at the top level - build and save the model - happens here too.
    """
    return runpy.run_path(path, run_name="__main__")


def load(torch: Any, path: str) -> Any:
    """The TorchScript model saved at ``path``."""
    return torch.jit.load(path)


def save(torch: Any, model: Any, path: str) -> None:
    """``model`` scripted and written where the method says its trained model goes."""
    torch.jit.save(torch.jit.script(model), path)


def loader(torch: Any, values: Any, targets: Any, batch_size: int) -> Any:
    """One of PyTorch's dataloaders over events and what they are trained towards.

    The events keep the order they came in, as ROOT's ``shuffle=False`` keeps it: a
    validation sample is the tail of the training sample, and must stay that tail.
    """
    dataset = torch.utils.data.TensorDataset(torch.Tensor(values), torch.Tensor(targets))
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


def best_checkpoint(torch: Any, path: str) -> Callable[[Any, float, float], float]:
    """ROOT's ``SaveBestOnly``: the model scripted and written whenever validation improves.

    The user's loop hands this ``(model, current, best)`` each epoch and takes back the
    best loss so far, which is where ROOT's own snippet keeps it.
    """

    def save_best(model: Any, current: float, best: float) -> float:
        if current <= best:
            save(torch, model, path)
            return float(current)
        return float(best)

    return save_best


def schedule(text: str) -> tuple[Callable[..., None] | None, dict[int, float] | None]:
    """ROOT's ``LearningRateSchedule``: ``"50,0.01;70,0.005"`` as epochs and learning rates.

    The loop is the user's, so the rate is set on whatever optimiser it hands the
    function back - the instance it made, as ROOT's own snippet expects. An empty
    option is the pair of ``None`` ROOT's own snippet leaves, which the loop unpacks.
    """
    if not text:
        return None, None
    steps = {
        int(epoch): float(rate)
        for epoch, rate in (setting.split(",") for setting in text.split(";") if setting)
    }

    def applied(optimizer: Any, epoch: int, rates: Any = None) -> None:
        wanted = steps if rates is None else rates
        if epoch in wanted:
            for group in optimizer.param_groups:
                group["lr"] = float(wanted[epoch])

    return applied, steps


def predictions(predict: Any, model: Any, values: Any) -> np.ndarray:
    """The user's own prediction function: a row per event, a column per output.

    The events are handed over in the single precision ROOT holds them in, which is
    what the model was trained on, and the answer comes back reshaped as
    ``GetMvaValues`` gathers it: one row of outputs an event.
    """
    events = np.ascontiguousarray(values, dtype=np.float32)
    found = np.asarray(predict(model, events), dtype=np.float64)
    return found.reshape(len(events), -1)
