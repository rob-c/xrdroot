"""The user's own Keras model: loaded, trained by Keras, and asked for its predictions.

PyMVA's ``PyKeras`` method has no learner of its own, as its ``PyTorch``
method has none: the user's Python script builds a Keras model, compiles it
and saves it, and TMVA loads that file, hands the events to ``model.fit`` as
NumPy arrays of single precision, and asks ``model.predict`` for a row of
outputs an event. Keras is TensorFlow's own high-level API, so TensorFlow is
what is imported here. Only the handful of calls below are used, so that the
whole of it is exercised in the tests against a small stand-in for
TensorFlow where TensorFlow itself is not installed.
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
    "checkpoint",
    "compiled",
    "early_stopping",
    "fit",
    "learning_rate",
    "load",
    "memory_growth",
    "module",
    "predictions",
    "save",
    "schedule",
    "tensor_board",
    "threads",
    "user_code",
]

#: One sample of events as Keras takes them: the values, what they train towards, their weights.
Sample = tuple[Any, Any, Any]


def available() -> bool:
    """Is TensorFlow installed, and so Keras with it?"""
    return importlib.util.find_spec("tensorflow") is not None


def module() -> Any:
    """TensorFlow itself, whose ``keras`` the method calls."""
    return importlib.import_module("tensorflow")


def user_code(path: str) -> dict[str, Any]:
    """The names the user's file defines, run as ``__main__`` as ROOT runs it."""
    return runpy.run_path(path, run_name="__main__")


def load(tensorflow: Any, path: str, custom_objects: Any = None) -> Any:
    """The Keras model saved at ``path`` - a ``.keras`` file, or an ``.h5`` one."""
    return tensorflow.keras.models.load_model(path, custom_objects=custom_objects)


def save(model: Any, path: str) -> None:
    """``model`` written where the method says its trained model goes."""
    model.save(path)


def compiled(model: Any) -> bool:
    """Was the model compiled before it was saved? Keras will not train one that was not."""
    return getattr(model, "optimizer", None) is not None


def fit(
    model: Any,
    train: Sample,
    valid: Sample,
    batch_size: int,
    epochs: int,
    verbose: int,
    callbacks: list[Any],
) -> dict[str, list[tuple[int, float]]]:
    """Train the model on the events, validating on the last of them, as ROOT trains it.

    The history comes back the way TMVA records one - each quantity a list of
    ``(epoch, value)``, epochs counted from one - so that the Factory writes it
    into its output file as ROOT's ``TrainingHistory`` does.
    """
    history = model.fit(
        train[0],
        train[1],
        sample_weight=train[2],
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        validation_data=valid,
        callbacks=callbacks,
    )
    return {
        str(name): [(epoch + 1, float(value)) for epoch, value in enumerate(points)]
        for name, points in dict(history.history).items()
    }


def predictions(model: Any, values: Any, verbose: int = 0) -> np.ndarray:
    """What the model answers for a row of events: one row of outputs an event.

    The events go over in the single precision the model was trained on, and the
    answers come back in the precision the rest of TMVA counts in.
    """
    events = np.ascontiguousarray(values, dtype=np.float32)
    found = np.asarray(model.predict(events, verbose=verbose), dtype=np.float64)
    return found.reshape(len(events), -1)


def checkpoint(tensorflow: Any, path: str, verbose: int) -> Any:
    """``SaveBestOnly``: the model written whenever the validation loss improves."""
    return tensorflow.keras.callbacks.ModelCheckpoint(
        path, monitor="val_loss", verbose=verbose, save_best_only=True, mode="auto"
    )


def early_stopping(tensorflow: Any, tries: int, verbose: int) -> Any:
    """``TriesEarlyStopping``: training stopped once the validation loss stops improving."""
    return tensorflow.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=tries, verbose=verbose, mode="auto"
    )


def learning_rate(tensorflow: Any, rate_for: Callable[[int, float], float], verbose: int) -> Any:
    """``LearningRateSchedule``: the rate set by ``rate_for`` as the epochs go by."""
    return tensorflow.keras.callbacks.LearningRateScheduler(rate_for, verbose=verbose)


def tensor_board(tensorflow: Any, log_dir: str) -> Any:
    """``TensorBoard``: the training logged where ``log_dir`` says, to be watched as it runs.

    ROOT also asks for ``batch_size`` and ``write_grads``, which Keras 3 dropped
    from this callback; what is asked for here is what both Keras 2 and 3 have.
    """
    return tensorflow.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False
    )


def schedule(text: str) -> Callable[[int, float], float] | None:
    """ROOT's ``LearningRateSchedule``: ``"10,0.5;20,0.1"`` as epochs and their rates.

    The rate in force at an epoch is the first one set at an epoch still to
    come, and past the last of them the rate is left as it was - which is what
    ROOT's own generated ``schedule(epoch, lr)`` does.
    """
    if not text:
        return None
    steps = [
        (int(epoch), float(rate))
        for epoch, rate in (setting.split(",") for setting in text.split(";") if setting)
    ]

    def rate_for(epoch: int, rate: float) -> float:
        for boundary, wanted in steps:
            if epoch < boundary:
                return wanted
        return rate

    return rate_for


def threads(tensorflow: Any, count: int) -> None:
    """``NumThreads``: TensorFlow's own pools sized as asked, while they can still be sized.

    A pool that has already run cannot be resized, and TensorFlow refuses with a
    RuntimeError - while still reporting the pool as unset, so asking first is no
    guard at all. The count is left alone then, rather than a traceback coming
    out of the booking: ROOT builds a session configuration on TensorFlow 2.15
    instead of resizing, and does not fail there either.
    """
    if count <= 0:
        return
    if tensorflow.config.threading.get_intra_op_parallelism_threads() != 0:
        return
    try:
        tensorflow.config.threading.set_intra_op_parallelism_threads(count)
        tensorflow.config.threading.set_inter_op_parallelism_threads(count)
    except RuntimeError:
        return


def memory_growth(tensorflow: Any, options: str) -> None:
    """``GpuOptions=allow_growth=True``: the GPU's memory taken as it is needed, not all at once.

    ROOT applies this one and no other, so neither does this.
    """
    if "allow_growth=True" not in [part.strip() for part in options.split(",")]:
        return
    for device in tensorflow.config.list_physical_devices("GPU"):
        tensorflow.config.experimental.set_memory_growth(device, True)
