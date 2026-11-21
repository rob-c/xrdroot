"""PyMVA's ``PyKeras`` method: the user's compiled Keras model, trained by Keras.

TensorFlow is not a dependency of xrdroot's tests, so most of what the method does is
exercised against a small stand-in for TensorFlow - a module whose ``keras.models``
records what it was asked to load and hands back a model that writes down what it was
fitted to and answers with the row of outputs the test asked for, and whose
``keras.callbacks`` record the callbacks ROOT's options ask for. Two tests then drive
the real TensorFlow, where it is installed, as ROOT's own PyMVA tests do.
"""

from __future__ import annotations

import importlib.machinery
import os
import sys
import types
from typing import Any, ClassVar

import numpy as np
import pytest

import xrdroot
import xrdroot.pyroot as ROOT
from test_tmva_learn_bdt import QUIET, multiclass_loader, regression_loader
from tmvasupport import loader, session, weights
from xrdroot.tmva import TMVAError, kerasmodel
from xrdroot.tmva.methods import REGISTRY
from xrdroot.tmva.methods.pymva import MethodPyKeras

__all__ = ["fake_tensorflow", "session"]


class Model:
    """A Keras model: what it was fitted to, and the row of outputs it answers with."""

    #: Keras keeps the optimiser it was compiled with, and has none before it is compiled.
    optimizer: Any = "adam"

    def __init__(self, outputs: tuple[float, ...]) -> None:
        self.outputs = outputs
        self.fits: list[dict[str, Any]] = []
        self.saved: list[str] = []

    def fit(self, values: Any, targets: Any, **named: Any) -> Any:
        self.fits.append({"values": values, "targets": targets, **named})
        return types.SimpleNamespace(history={"loss": [0.5, 0.25]})

    def predict(self, values: Any, verbose: int = 0) -> Any:
        assert values.dtype == np.float32, "Keras is handed the events in single precision"
        return np.tile(np.asarray(self.outputs, dtype=np.float32), (len(values), 1))

    def save(self, path: str) -> None:
        self.saved.append(path)


class Uncompiled(Model):
    """A model saved before it was compiled, which Keras will not train."""

    optimizer = None


class Models:
    """``keras.models``: what was loaded, and the model a load hands back."""

    loaded: ClassVar[list[tuple[str, Any]]] = []
    #: The models handed back, in the order they were loaded.
    handed: ClassVar[list[Model]] = []
    #: The model class a load makes, and the outputs it answers with.
    kind: ClassVar[type[Model]] = Model
    outputs: ClassVar[tuple[float, ...]] = (0.25, 0.75)

    def load_model(self, path: str, custom_objects: Any = None) -> Any:
        Models.loaded.append((path, custom_objects))
        made = Models.kind(Models.outputs)
        Models.handed.append(made)
        return made


class Callbacks:
    """``keras.callbacks``: each callback ROOT's options ask for, with what it was built of."""

    made: ClassVar[list[tuple[str, dict[str, Any]]]] = []

    def ModelCheckpoint(self, path: str, **named: Any) -> Any:
        return self._record("ModelCheckpoint", path=path, **named)

    def EarlyStopping(self, **named: Any) -> Any:
        return self._record("EarlyStopping", **named)

    def LearningRateScheduler(self, schedule: Any, **named: Any) -> Any:
        return self._record("LearningRateScheduler", schedule=schedule, **named)

    def TensorBoard(self, **named: Any) -> Any:
        return self._record("TensorBoard", **named)

    def _record(self, kind: str, **named: Any) -> Any:
        Callbacks.made.append((kind, named))
        return (kind, named)


class Threading:
    """``tf.config.threading``: the pool sizes asked for, and whether they can still be set."""

    intra: ClassVar[int] = 0
    asked: ClassVar[list[tuple[str, int]]] = []

    def get_intra_op_parallelism_threads(self) -> int:
        return Threading.intra

    #: Whether the pools have run already, which makes TensorFlow refuse to resize them.
    raises: ClassVar[bool] = False

    def set_intra_op_parallelism_threads(self, count: int) -> None:
        if Threading.raises:
            raise RuntimeError("Intra op parallelism cannot be modified after initialization.")
        Threading.asked.append(("intra", count))
        Threading.intra = count

    def set_inter_op_parallelism_threads(self, count: int) -> None:
        Threading.asked.append(("inter", count))


class Experimental:
    """``tf.config.experimental``: the GPUs whose memory was let grow."""

    grown: ClassVar[list[Any]] = []

    def set_memory_growth(self, device: Any, growth: bool) -> None:
        Experimental.grown.append((device, growth))


class Config:
    """``tf.config``: the thread pools, the devices, and what may be done to them."""

    def __init__(self) -> None:
        self.threading = Threading()
        self.experimental = Experimental()

    def list_physical_devices(self, kind: str) -> list[str]:
        return ["gpu:0", "gpu:1"] if kind == "GPU" else []


def fake_tensorflow() -> types.ModuleType:
    """The stand-in: ``tensorflow.keras`` and ``tensorflow.config`` and nothing else."""
    tensorflow = types.ModuleType("tensorflow")
    tensorflow.__spec__ = importlib.machinery.ModuleSpec("tensorflow", None)
    tensorflow.keras = types.SimpleNamespace(models=Models(), callbacks=Callbacks())
    tensorflow.config = Config()
    return tensorflow


@pytest.fixture
def keras(monkeypatch):
    """The stand-in, where ``import tensorflow`` finds it."""
    made = fake_tensorflow()
    monkeypatch.setitem(sys.modules, "tensorflow", made)
    return made


@pytest.fixture(autouse=True)
def _forgotten_records() -> None:
    """The stand-in's records of the last test, so that each test reads only its own."""
    Models.loaded.clear()
    Models.handed.clear()
    Models.kind, Models.outputs = Model, (0.25, 0.75)
    Callbacks.made.clear()
    Threading.intra, Threading.asked = 0, []
    Experimental.grown.clear()


def model_file(path: str) -> None:
    """A model file where the options say one is, as the user's own script would write it."""
    with open(path, "wb"):
        pass


def booked(options: str, analysis: str = "Classification") -> Any:
    """A PyKeras method booked through the Factory, which is what reads its options."""
    factory = ROOT.TMVA.Factory("job", QUIET + analysis)
    data = {
        "Classification": loader,
        "Regression": regression_loader,
        "Multiclass": multiclass_loader,
    }[analysis]()
    return factory.BookMethod(data, ROOT.TMVA.Types.kPyKeras, "PyKeras", options)


def trained(options: str, analysis: str = "Classification") -> Any:
    """The method, trained on its loader's training events as the Factory trains it."""
    method = booked(
        f"FilenameModel=model.keras:FilenameTrainedModel=trained.keras:{options}", analysis
    )
    model_file("model.keras")
    method.train(method.handler.prepare(method.loader.dataset().train))
    return method


def loads() -> list[Model]:
    """The models the stand-in handed back, in the order they were loaded."""
    return Models.handed


# -- the TensorFlow calls ---------------------------------------------------------------


def test_tensorflow_is_available_when_it_can_be_imported(keras, monkeypatch):
    assert kerasmodel.available()
    monkeypatch.delitem(sys.modules, "tensorflow")
    monkeypatch.setattr(sys, "path", [])
    assert not kerasmodel.available()


def test_the_users_file_is_run_as_the_script_root_runs_it(tmp_path):
    path = tmp_path / "user.py"
    path.write_text("name = __name__\n")
    assert kerasmodel.user_code(str(path))["name"] == "__main__"


def test_a_model_is_saved_and_loaded_again_through_keras(keras):
    made = Model((0.5, 0.5))
    kerasmodel.save(made, "trained.keras")
    assert made.saved == ["trained.keras"]
    loaded = kerasmodel.load(keras, "model.keras", {"MyLayer": "layer"})
    assert Models.loaded == [("model.keras", {"MyLayer": "layer"})]
    assert isinstance(loaded, Model)


def test_a_model_is_compiled_exactly_when_it_has_an_optimiser(keras):
    assert kerasmodel.compiled(Model((0.5, 0.5)))
    assert not kerasmodel.compiled(Uncompiled((0.5, 0.5)))


def test_the_events_go_to_keras_in_single_precision_with_their_weights_beside_them(keras):
    made = Model((0.25, 0.75))
    train = (np.zeros((4, 3)), np.ones((4, 2)), np.full(4, 0.5))
    valid = (np.zeros((2, 3)), np.ones((2, 2)), np.ones(2))
    history = kerasmodel.fit(made, train, valid, 8, 3, 0, [])
    called = made.fits[0]
    assert called["values"] is train[0] and called["targets"] is train[1]
    assert called["sample_weight"] is train[2]
    assert called["batch_size"] == 8 and called["epochs"] == 3 and called["verbose"] == 0
    assert called["validation_data"] is valid and called["callbacks"] == []
    assert history == {"loss": [(1, 0.5), (2, 0.25)]}


def test_the_answer_is_a_row_of_outputs_for_each_event(keras):
    found = kerasmodel.predictions(Model((0.25, 0.75)), np.zeros((5, 4), dtype=np.float64), 0)
    assert found.shape == (5, 2) and found.dtype == np.float64
    assert found == pytest.approx(np.tile([0.25, 0.75], (5, 1)))


def test_save_best_only_asks_keras_to_keep_the_model_of_the_best_validation_loss(keras):
    made = kerasmodel.checkpoint(keras, "trained.keras", 0)
    assert made == (
        "ModelCheckpoint",
        {
            "path": "trained.keras",
            "monitor": "val_loss",
            "verbose": 0,
            "save_best_only": True,
            "mode": "auto",
        },
    )


def test_early_stopping_waits_the_epochs_the_option_names(keras):
    made = kerasmodel.early_stopping(keras, 3, 1)
    assert made == (
        "EarlyStopping",
        {"monitor": "val_loss", "patience": 3, "verbose": 1, "mode": "auto"},
    )


def test_a_learning_rate_scheduler_is_keras_callback_over_roots_schedule(keras):
    rate_for = kerasmodel.schedule("10,0.5")
    made = kerasmodel.learning_rate(keras, rate_for, 0)
    assert made == (
        "LearningRateScheduler",
        {"schedule": rate_for, "verbose": 0},
    )


def test_tensor_board_logs_where_the_option_says(keras):
    made = kerasmodel.tensor_board(keras, "logs")
    assert made == (
        "TensorBoard",
        {"log_dir": "logs", "histogram_freq": 0, "write_graph": True, "write_images": False},
    )


def test_a_learning_rate_schedule_is_read_the_way_root_reads_it():
    rate_for = kerasmodel.schedule("10,0.5;20,0.1")
    assert rate_for is not None
    assert [rate_for(epoch, 1.0) for epoch in (0, 9, 10, 19, 20, 30)] == [
        0.5,
        0.5,
        0.1,
        0.1,
        1.0,
        1.0,
    ]


def test_no_learning_rate_schedule_is_no_scheduler_at_all():
    assert kerasmodel.schedule("") is None


def test_a_thread_count_is_given_to_tensorflows_pools_while_they_can_be_sized(keras):
    kerasmodel.threads(keras, 0)
    assert Threading.asked == []
    kerasmodel.threads(keras, 2)
    assert Threading.asked == [("intra", 2), ("inter", 2)]
    kerasmodel.threads(keras, 4)
    assert Threading.asked == [("intra", 2), ("inter", 2)]


def test_a_pool_that_has_already_run_is_left_alone_rather_than_failing_the_booking(keras):
    """TensorFlow refuses to resize a pool that has run, and still reports it unset.

    Asking first is therefore no guard: the setter raises after the context exists,
    and that must not become a traceback out of BookMethod. ROOT on TensorFlow 2.15
    builds a session configuration instead of resizing and does not fail either.
    """
    Threading.intra = 0
    Threading.asked = []
    Threading.raises = True
    try:
        kerasmodel.threads(keras, 2)
    finally:
        Threading.raises = False
    assert Threading.asked == []


def test_memory_growth_is_asked_of_every_gpu_when_the_option_says_so(keras):
    kerasmodel.memory_growth(keras, "")
    assert Experimental.grown == []
    kerasmodel.memory_growth(keras, "allow_growth=False")
    assert Experimental.grown == []
    kerasmodel.memory_growth(keras, "allow_growth=True")
    assert Experimental.grown == [("gpu:0", True), ("gpu:1", True)]


# -- booking and training ---------------------------------------------------------------


def test_pykeras_is_a_method_of_the_registry_with_a_model_file_of_its_own(session):
    assert REGISTRY["PyKeras"] is MethodPyKeras
    assert booked("V").trained_model == os.path.join(
        "dataset", "weights", "TrainedModel_PyKeras.keras"
    )


def test_a_keras_method_without_tensorflow_is_refused_with_the_extra_that_installs_it(
    monkeypatch,
):
    monkeypatch.setattr(kerasmodel, "available", lambda: False)
    with pytest.raises(xrdroot.UnsupportedFeatureError, match=r"pip install 'xrdroot\[pymva\]'"):
        booked("V")


def test_the_events_and_their_one_hot_targets_are_what_keras_is_trained_on(session, keras):
    method = trained("BatchSize=8:NumEpochs=3:Verbose=0")
    events = method.handler.prepare(method.loader.dataset().train)
    cut = len(events) - len(events) // 5
    called = loads()[0].fits[0]
    assert called["values"].shape == (cut, 4) and called["values"].dtype == np.float32
    assert called["values"] == pytest.approx(np.asarray(events.values[:cut], dtype=np.float32))
    assert called["targets"].shape == (cut, 2) and called["targets"].sum() == pytest.approx(cut)
    assert called["sample_weight"] == pytest.approx(np.asarray(events.weights[:cut], np.float32))
    assert called["validation_data"][0].shape == (len(events) - cut, 4)
    assert called["batch_size"] == 8 and called["epochs"] == 3 and called["verbose"] == 0


def test_a_regression_is_trained_on_the_targets_themselves(session, keras):
    method = trained("Verbose=0", "Regression")
    called = loads()[0].fits[0]
    events = method.handler.prepare(method.loader.dataset().train)
    assert called["targets"].shape == (len(events) - len(events) // 5, 1)
    assert called["targets"][:, 0] == pytest.approx(
        np.asarray(events.targets[: len(called["targets"])], dtype=np.float32)[:, 0]
    )


def test_the_training_history_is_recorded_an_epoch_at_a_time(session, keras):
    method = trained("Verbose=0")
    assert method.history == {"loss": [(1, 0.5), (2, 0.25)]}


def test_save_best_only_is_a_callback_and_no_model_is_written_at_the_end(session, keras):
    method = trained("Verbose=0")
    assert [kind for kind, _ in Callbacks.made] == ["ModelCheckpoint"]
    assert Callbacks.made[0][1]["path"] == "trained.keras"
    assert loads()[0].saved == []
    assert method.trained_model == "trained.keras"


def test_without_save_best_only_the_last_model_is_written_at_the_end(session, keras):
    trained("Verbose=0:!SaveBestOnly")
    assert Callbacks.made == []
    assert loads()[0].saved == ["trained.keras"]


def test_the_other_options_are_the_callbacks_root_makes_of_them(session, keras):
    trained(
        "Verbose=2:TriesEarlyStopping=4:LearningRateSchedule=10,0.5:"
        "TensorBoard=logs"
    )
    kinds = [kind for kind, _ in Callbacks.made]
    assert kinds == ["ModelCheckpoint", "EarlyStopping", "LearningRateScheduler", "TensorBoard"]
    assert Callbacks.made[1][1]["patience"] == 4 and Callbacks.made[1][1]["verbose"] == 2


def test_continuing_training_starts_from_the_trained_model(session, keras):
    method = booked("ContinueTraining:FilenameModel=model.keras:FilenameTrainedModel=trained.keras")
    model_file("trained.keras")
    method.train(method.loader.dataset().train)
    assert [path for path, _ in Models.loaded] == ["trained.keras"]


def test_a_model_the_options_name_but_nothing_wrote_is_refused_with_its_path(session, keras):
    method = booked("FilenameModel=absent.keras")
    with pytest.raises(xrdroot.UnsupportedFeatureError, match=r'"absent.keras".*does not exist'):
        method.train(method.loader.dataset().train)


def test_a_model_saved_before_it_was_compiled_is_refused_with_what_to_do(session, keras):
    Models.kind = Uncompiled
    method = booked("FilenameModel=model.keras")
    model_file("model.keras")
    with pytest.raises(xrdroot.UnsupportedFeatureError, match=r"call model.compile"):
        method.train(method.loader.dataset().train)


def test_a_user_file_that_is_not_there_is_refused_with_its_path(session, keras):
    method = booked("FilenameModel=model.keras:UserCode=nowhere.py")
    model_file("model.keras")
    with pytest.raises(xrdroot.UnsupportedFeatureError, match=r'"nowhere.py" does not exist'):
        method.train(method.loader.dataset().train)


def test_the_user_files_objects_are_what_the_model_is_loaded_with(session, keras, tmp_path):
    (tmp_path / "user.py").write_text('load_model_custom_objects = {"MyLayer": "layer"}\n')
    method = booked("FilenameModel=model.keras:UserCode=user.py")
    model_file("model.keras")
    method.train(method.loader.dataset().train)
    assert Models.loaded[0] == ("model.keras", {"MyLayer": "layer"})


@pytest.mark.parametrize("size, share", [("20%", 0.2), ("0.25", 0.25), ("25", 25.0)])
def test_a_validation_size_is_a_share_of_the_training_events_or_a_number_of_them(
    session, keras, size, share
):
    method = trained(f"Verbose=0:ValidationSize={size}")
    events = method.handler.prepare(method.loader.dataset().train)
    wanted = int(len(events) * share) if share < 1.0 else int(share)
    assert len(loads()[0].fits[0]["validation_data"][0]) == wanted


@pytest.mark.parametrize("size", ["0", "0%", "100%", "0.1%", "notanumber", "-1", "200"])
def test_a_validation_size_that_leaves_keras_no_events_is_refused(session, keras, size):
    method = booked(f"FilenameModel=model.keras:ValidationSize={size}")
    model_file("model.keras")
    with pytest.raises(TMVAError, match=r"Validation size|Cannot parse number"):
        method.train(method.loader.dataset().train)


def test_a_thread_count_and_a_gpu_option_are_applied_at_booking(session, keras):
    booked("NumThreads=1:GpuOptions=allow_growth=True")
    assert Threading.asked == [("intra", 1), ("inter", 1)]
    assert Experimental.grown == [("gpu:0", True), ("gpu:1", True)]


# -- evaluation -------------------------------------------------------------------------


def test_a_classification_answers_with_the_signal_classs_chance(session, keras):
    method = trained("Verbose=0")
    assert method.evaluate(np.zeros((4, 4))) == pytest.approx(np.full(4, 0.25))


def test_a_multiclass_answers_with_every_classs_chance(session, keras):
    Models.outputs = (0.25, 0.5, 0.25)
    method = trained("Verbose=0", "Multiclass")
    assert np.shape(method.evaluate(np.zeros((4, 4)))) == (4, 3)


def test_a_regression_answers_with_the_targets_it_was_trained_on(session, keras):
    Models.outputs = (1.5,)
    method = trained("Verbose=0", "Regression")
    assert np.asarray(method.evaluate(np.zeros((2, 2)))).reshape(2, 1) == pytest.approx(
        np.full((2, 1), 1.5)
    )


def test_the_weight_file_holds_no_weights_and_the_model_is_loaded_from_the_file_it_names(
    session, keras
):
    method = trained("Verbose=0:!SaveBestOnly")
    method.write_weight_file()
    # The stand-in writes nothing, so what the fit said it saved is put there by hand.
    model_file("trained.keras")
    with open(weights("PyKeras")) as source:
        written = source.read()
    assert "<Weights>" not in written and "FilenameTrainedModel" in written
    read = ROOT.TMVA.Reader("!Color:Silent")
    for name in ("var1", "var2", "var3", "var4"):
        read.AddVariable(name, np.zeros(1, dtype=np.float32))
    read.BookMVA("PyKeras", weights("PyKeras"))
    assert read.EvaluateMVA("PyKeras") == pytest.approx(0.25)


def test_a_model_that_is_no_longer_there_is_refused_when_it_is_first_asked_for(session, keras):
    method = booked("FilenameModel=model.keras:FilenameTrainedModel=gone.keras")
    model_file("model.keras")
    with pytest.raises(xrdroot.UnsupportedFeatureError, match=r'"gone.keras".*does not exist'):
        method.evaluate(np.zeros((1, 4)))


# -- the real TensorFlow, where it is installed -----------------------------------------


def real_model(tensorflow: Any, nvar: int, nout: int, last: str, loss: str) -> Any:
    """A compiled Keras model of ``nvar`` inputs, saved where the method expects its own."""
    tensorflow.keras.utils.set_random_seed(1234)
    made = tensorflow.keras.models.Sequential(
        [
            tensorflow.keras.Input(shape=(nvar,)),
            tensorflow.keras.layers.Dense(16, activation="relu"),
            tensorflow.keras.layers.Dense(nout, activation=last),
        ]
    )
    made.compile(loss=loss, optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.01))
    made.save("model.keras")
    return made


def test_the_real_keras_trains_the_users_model_to_tell_signal_from_background(session):
    tensorflow = pytest.importorskip("tensorflow")
    real_model(tensorflow, 4, 2, "softmax", "categorical_crossentropy")
    method = booked(
        "!H:!V:VarTransform=N:FilenameModel=model.keras:FilenameTrainedModel=trained.keras:"
        "NumEpochs=10:BatchSize=32:!SaveBestOnly:Verbose=0"
    )
    method.train(method.handler.prepare(method.loader.dataset().train))
    assert os.path.exists("trained.keras")
    test = method.loader.dataset().test
    found = np.asarray(method.evaluate(method.handler.apply(test).values)).ravel()
    signal = found[test.classes == method.dsi.GetSignalClassIndex()].mean()
    background = found[test.classes != method.dsi.GetSignalClassIndex()].mean()
    assert signal > background and signal > 0.5 > background


def test_the_real_keras_scores_the_model_it_wrote_not_the_last_epoch_it_trained(session):
    """ROOT reloads FilenameTrainedModel to evaluate, so the Factory scores the file.

    With the default SaveBestOnly that file is the epoch with the lowest validation
    loss, and the model training left in memory is the last epoch's - a different
    model, giving every event a different answer. The method must let its own go.
    """
    tensorflow = pytest.importorskip("tensorflow")
    real_model(tensorflow, 4, 2, "softmax", "categorical_crossentropy")
    method = booked(
        "!H:!V:VarTransform=N:FilenameModel=model.keras:FilenameTrainedModel=trained.keras:"
        "NumEpochs=8:BatchSize=32:Verbose=0"
    )
    method.train(method.handler.prepare(method.loader.dataset().train))
    test = method.loader.dataset().test
    values = method.handler.apply(test).values
    found = np.asarray(method.evaluate(values))
    written = kerasmodel.load(tensorflow, "trained.keras", None)
    rows = np.asarray(kerasmodel.predictions(written, values, 0))
    assert np.array_equal(found, rows[:, method.dsi.GetSignalClassIndex()])


def test_the_real_keras_multiclass_answers_a_softmax_row_as_one_output_per_class(session):
    tensorflow = pytest.importorskip("tensorflow")
    real_model(tensorflow, 4, 3, "softmax", "categorical_crossentropy")
    method = booked(
        "!H:!V:VarTransform=N:FilenameModel=model.keras:FilenameTrainedModel=trained.keras:"
        "NumEpochs=10:BatchSize=32:!SaveBestOnly:Verbose=0",
        "Multiclass",
    )
    method.train(method.handler.prepare(method.loader.dataset().train))
    test = method.loader.dataset().test
    found = np.asarray(method.evaluate(method.handler.apply(test).values))
    assert found.shape == (len(test), 3)
    # A softmax row is a distribution over the classes, one output per class for TMVA.
    assert found.sum(axis=1) == pytest.approx(np.ones(len(test)))
    # Each class's own events are scored highest by its own output, above a third of the row.
    for index in range(3):
        scored = found[test.classes == index].mean(axis=0)
        assert scored.argmax() == index and scored[index] > 0.4


def test_the_real_keras_regression_learns_a_target_that_is_smooth_in_two_variables(session):
    tensorflow = pytest.importorskip("tensorflow")
    real_model(tensorflow, 2, 1, "linear", "mean_squared_error")
    method = booked(
        "!H:!V:VarTransform=N:FilenameModel=model.keras:FilenameTrainedModel=trained.keras:"
        "NumEpochs=20:BatchSize=32:!SaveBestOnly:Verbose=0",
        "Regression",
    )
    method.train(method.handler.prepare(method.loader.dataset().train))
    test = method.loader.dataset().test
    found = np.asarray(method.evaluate(method.handler.apply(test).values)).ravel()
    targets = test.targets.ravel()
    assert float(np.mean((found - targets) ** 2)) < 0.1 * float(np.var(targets))
