"""PyMVA's ``PyTorch`` method: the user's TorchScript model, trained by the user's own loop.

PyTorch is not a dependency of xrdroot's tests, so most of what the method does is
exercised against a small stand-in for PyTorch - a module whose ``jit`` records what it
was asked to save and hands back what it was asked to load, whose dataloaders keep the
events they were made of, whose ``Tensor`` is NumPy - with the user's file replaced by
one that writes down the arguments TMVA calls its loop with. Two tests then drive the
real PyTorch, where it is installed, as ROOT's own PyMVA tests do.
"""

from __future__ import annotations

import ast
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
from xrdroot.tmva import TMVAError, torchscript

__all__ = ["session"]

#: A user's file: its loop writes down what it was called with, so the test can read it.
FAKE_USER = '''
import numpy as np


def fit(model, train_loader, val_loader, num_epochs, batch_size, optimizer, criterion,
        save_best, scheduler):
    with open("calls.log", "a") as out:
        out.write(repr(dict(
            model=model, num_epochs=num_epochs, batch_size=batch_size, optimizer=optimizer,
            criterion=criterion, save_best=save_best is not None, scheduler=scheduler,
            train=len(train_loader), valid=len(val_loader),
        )) + "\\n")
    if save_best is not None:
        save_best(model, 0.5, 0.9)
    return model


def predict(model, values, batch_size=32):
    return np.tile([0.25, 0.75], (len(values), 1))


load_model_custom_objects = {"criterion": "mse", "train_func": fit, "predict_func": predict}
'''


class Tensor(np.ndarray):
    """A PyTorch tensor as a NumPy array, which is all the method reads of one."""


class Loader:
    """``DataLoader``: the events it was made of, in the order they were given."""

    made: ClassVar[list[Loader]] = []

    def __init__(self, dataset: Any, batch_size: int, shuffle: bool) -> None:
        self.dataset, self.batch_size, self.shuffle = dataset, batch_size, shuffle
        Loader.made.append(self)

    def __len__(self) -> int:
        return len(self.dataset[0])


class Jit:
    """``torch.jit``: what was saved, and what a load hands back."""

    saved: ClassVar[list[tuple[Any, str]]] = []
    loaded: ClassVar[list[str]] = []

    def script(self, model: Any) -> Any:
        return ("scripted", model)

    def save(self, model: Any, path: str) -> None:
        Jit.saved.append((model, path))

    def load(self, path: str) -> Any:
        Jit.loaded.append(path)
        return ("loaded", path)


def fake_torch() -> types.ModuleType:
    """The stand-in: ``torch``, ``torch.jit``, ``torch.utils.data`` and ``torch.optim``."""
    torch = types.ModuleType("torch")
    torch.__spec__ = importlib.machinery.ModuleSpec("torch", None)
    torch.jit = Jit()
    torch.utils = types.SimpleNamespace(
        data=types.SimpleNamespace(TensorDataset=lambda x, y: (x, y), DataLoader=Loader)
    )
    torch.Tensor = lambda values: np.asarray(values, dtype=np.float32).view(Tensor)
    torch.optim = types.SimpleNamespace(SGD="sgd")
    return torch


@pytest.fixture
def torch(monkeypatch):
    """The stand-in, where ``import torch`` finds it."""
    made = fake_torch()
    monkeypatch.setitem(sys.modules, "torch", made)
    return made


@pytest.fixture(autouse=True)
def _forgotten_records() -> None:
    """The stand-in's records of the last test, so that each test reads only its own."""
    Loader.made.clear()
    Jit.saved.clear()
    Jit.loaded.clear()


def user_code(tmp_path: Any, text: str = FAKE_USER, name: str = "user.py") -> str:
    """The user's file, written where the test runs, and the name the option takes."""
    (tmp_path / name).write_text(text)
    return name


def calls() -> list[dict[str, Any]]:
    """What the user's loop was called with, as it wrote it down."""
    if not os.path.exists("calls.log"):
        return []
    with open("calls.log") as source:
        return [ast.literal_eval(line) for line in source if line.strip()]


def model_file(path: str) -> None:
    """A model file where the options say one is, as the user's own file would have written."""
    with open(path, "wb"):
        pass


def booked(options: str, analysis: str = "Classification") -> Any:
    """A PyTorch method booked through the Factory, which is what reads its options."""
    factory = ROOT.TMVA.Factory("job", QUIET + analysis)
    data = {
        "Classification": loader,
        "Regression": regression_loader,
        "Multiclass": multiclass_loader,
    }[analysis]()
    return factory.BookMethod(data, ROOT.TMVA.Types.kPyTorch, "PyTorch", options)


# -- the PyTorch calls ------------------------------------------------------------------


def test_pytorch_is_available_when_it_can_be_imported(torch, monkeypatch):
    assert torchscript.available()
    monkeypatch.delitem(sys.modules, "torch")
    monkeypatch.setattr(sys, "path", [])
    assert not torchscript.available()


def test_the_users_file_is_run_as_the_script_root_runs_it(tmp_path):
    """ROOT runs it with ``PyRun_SimpleFile``, so what it does at the top level happens."""
    path = tmp_path / "user.py"
    path.write_text("name = __name__\n")
    assert torchscript.user_code(str(path))["name"] == "__main__"


def test_a_model_is_saved_scripted_and_loaded_again_through_torchs_jit(torch):
    torchscript.save(torch, "model", "trained.pt")
    assert Jit.saved == [(("scripted", "model"), "trained.pt")]
    assert torchscript.load(torch, "trained.pt") == ("loaded", "trained.pt")


def test_a_loader_keeps_the_events_in_the_order_they_came_in(torch):
    values, targets = np.zeros((3, 2)), np.ones((3, 2))
    torchscript.loader(torch, values, targets, 8)
    made = Loader.made[0]
    assert made.batch_size == 8 and made.shuffle is False
    assert np.array_equal(made.dataset[0], values) and np.array_equal(made.dataset[1], targets)


def test_only_a_better_validation_loss_writes_the_model(torch):
    save_best = torchscript.best_checkpoint(torch, "trained.pt")
    assert save_best("model", 0.5, 0.9) == 0.5 and len(Jit.saved) == 1
    assert save_best("model", 0.95, 0.5) == 0.5 and len(Jit.saved) == 1
    assert save_best("model", 0.5, 0.5) == 0.5 and len(Jit.saved) == 2


def test_a_learning_rate_schedule_sets_the_rate_at_the_epochs_it_names():
    schedule, steps = torchscript.schedule("1,0.5;3,0.1")
    assert steps == {1: 0.5, 3: 0.1}
    trainer = types.SimpleNamespace(param_groups=[{"lr": 1.0}])
    schedule(trainer, 0)
    assert trainer.param_groups[0]["lr"] == 1.0
    schedule(trainer, 3)
    assert trainer.param_groups[0]["lr"] == pytest.approx(0.1)
    schedule(trainer, 1, {1: 0.25})
    assert trainer.param_groups[0]["lr"] == pytest.approx(0.25)


def test_no_learning_rate_schedule_is_the_pair_of_nothings_the_users_loop_unpacks():
    assert torchscript.schedule("") == (None, None)


def test_the_answer_is_the_users_prediction_a_row_for_each_event():
    def predict(model: Any, values: Any) -> Any:
        assert values.dtype == np.float32
        return np.zeros((len(values), 3))

    found = torchscript.predictions(predict, None, np.zeros((5, 2)))
    assert found.shape == (5, 3) and found.dtype == np.float64


# -- what the user's file hands TMVA ----------------------------------------------------


def test_a_pytorch_method_without_pytorch_is_refused_with_the_extra_that_installs_it(monkeypatch):
    monkeypatch.setattr(torchscript, "available", lambda: False)
    with pytest.raises(xrdroot.UnsupportedFeatureError, match=r"pip install 'xrdroot\[pymva\]'"):
        booked("V")


def test_a_user_file_that_is_not_named_is_refused_with_the_option_that_names_it(session, torch):
    method = booked("BatchSize=8")
    with pytest.raises(xrdroot.UnsupportedFeatureError, match=r"UserCode=<file>"):
        method.train(method.loader.dataset().train)


def test_a_user_file_that_is_not_there_is_refused_with_its_path(session, torch):
    method = booked("UserCode=nowhere.py")
    with pytest.raises(xrdroot.UnsupportedFeatureError, match=r'"nowhere.py" does not exist'):
        method.train(method.loader.dataset().train)


def test_a_user_file_that_leaves_no_custom_objects_is_refused_with_what_it_must_leave(
    session, torch, tmp_path
):
    text = "fit = None\n"
    method = booked(f"UserCode={user_code(tmp_path, text)}")
    with pytest.raises(xrdroot.UnsupportedFeatureError, match=r"load_model_custom_objects"):
        method.train(method.loader.dataset().train)


def test_a_user_file_that_leaves_out_a_key_is_refused_by_that_key(session, torch, tmp_path):
    text = 'load_model_custom_objects = {"train_func": None}\n'
    method = booked(f"UserCode={user_code(tmp_path, text)}")
    said = r"leaves no criterion \(the loss function\), predict_func"
    with pytest.raises(xrdroot.UnsupportedFeatureError, match=said):
        method.train(method.loader.dataset().train)


# -- training ---------------------------------------------------------------------------

def test_the_loop_is_given_the_events_split_by_the_validation_size(session, torch, tmp_path):
    method = booked(
        f"UserCode={user_code(tmp_path)}:BatchSize=8:NumEpochs=3:!SaveBestOnly:"
        "FilenameModel=model.pt"
    )
    model_file("model.pt")
    trained = method.loader.dataset().train
    method.train(trained)
    called = calls()[0]
    assert called["num_epochs"] == 3 and called["batch_size"] == 8
    assert called["scheduler"] == (None, None) and called["save_best"] is False
    assert called["optimizer"] == "sgd" and called["criterion"] == "mse"
    assert called["train"] == len(trained) - len(trained) // 5
    assert called["valid"] == len(trained) // 5


def test_the_best_model_is_written_where_the_options_name_it(session, torch, tmp_path):
    method = booked(
        f"UserCode={user_code(tmp_path)}:FilenameModel=model.pt:"
        "FilenameTrainedModel=trained.pt"
    )
    model_file("model.pt")
    method.train(method.loader.dataset().train)
    assert calls()[0]["save_best"] is True
    assert Jit.saved == [(("scripted", ("loaded", "model.pt")), "trained.pt")]


def test_a_loop_that_saves_no_best_model_has_the_last_one_saved_for_it(session, torch, tmp_path):
    method = booked(
        f"UserCode={user_code(tmp_path)}:!SaveBestOnly:FilenameModel=model.pt:"
        "FilenameTrainedModel=trained.pt"
    )
    model_file("model.pt")
    method.train(method.loader.dataset().train)
    assert calls()[0]["save_best"] is False
    assert Jit.saved == [(("scripted", ("loaded", "model.pt")), "trained.pt")]


def test_the_trained_model_goes_beside_the_weights_when_the_options_name_no_file(
    session, torch
):
    assert booked("V").trained_model == os.path.join(
        "dataset", "weights", "TrainedModel_PyTorch.pt"
    )


def test_continuing_training_starts_from_the_trained_model(session, torch, tmp_path):
    method = booked(
        f"UserCode={user_code(tmp_path)}:ContinueTraining:FilenameModel=model.pt:"
        "FilenameTrainedModel=trained.pt"
    )
    model_file("trained.pt")
    method.train(method.loader.dataset().train)
    assert Jit.loaded == ["trained.pt"]


def test_a_model_the_options_name_but_nothing_wrote_is_refused_with_its_path(
    session, torch, tmp_path
):
    method = booked(f"UserCode={user_code(tmp_path)}:FilenameModel=absent.pt")
    with pytest.raises(xrdroot.UnsupportedFeatureError, match=r'"absent.pt".*does not exist'):
        method.train(method.loader.dataset().train)


@pytest.mark.parametrize(
    "size, share",
    [("20%", 0.2), ("0.25", 0.25), ("25", 25.0)],
)
def test_a_validation_size_is_a_share_of_the_training_events_or_a_number_of_them(
    session, torch, tmp_path, size, share
):
    method = booked(f"UserCode={user_code(tmp_path)}:ValidationSize={size}:FilenameModel=m.pt")
    model_file("m.pt")
    trained = method.loader.dataset().train
    method.train(trained)
    wanted = int(len(trained) * share) if share < 1.0 else int(share)
    assert calls()[0]["valid"] == wanted


@pytest.mark.parametrize("size", ["0", "0%", "100%", "0.1%", "notanumber", "-1", "200"])
def test_a_validation_size_that_leaves_the_loop_no_events_is_refused(
    session, torch, tmp_path, size
):
    method = booked(f"UserCode={user_code(tmp_path)}:ValidationSize={size}:FilenameModel=m.pt")
    model_file("m.pt")
    with pytest.raises(TMVAError, match=r"Validation size|Cannot parse number"):
        method.train(method.loader.dataset().train)


# -- evaluation -------------------------------------------------------------------------

def test_a_classification_answers_with_the_signal_classs_chance(session, torch, tmp_path):
    method = booked(f"UserCode={user_code(tmp_path)}:FilenameModel=model.pt")
    model_file("model.pt")
    method.train(method.loader.dataset().train)
    assert method.evaluate(np.zeros((4, 4))) == pytest.approx(np.full(4, 0.25))


def test_a_multiclass_answers_with_every_classs_chance(session, torch, tmp_path):
    text = FAKE_USER.replace("[0.25, 0.75]", "[0.25, 0.5, 0.25]")
    method = booked(f"UserCode={user_code(tmp_path, text)}:FilenameModel=model.pt", "Multiclass")
    model_file("model.pt")
    method.train(method.loader.dataset().train)
    assert np.shape(method.evaluate(np.zeros((4, 4)))) == (4, 3)


def test_a_regression_answers_with_the_targets_it_was_trained_on(session, torch, tmp_path):
    text = FAKE_USER.replace("[0.25, 0.75]", "[1.5]")
    method = booked(f"UserCode={user_code(tmp_path, text)}:FilenameModel=model.pt", "Regression")
    model_file("model.pt")
    method.train(method.loader.dataset().train)
    assert method.evaluate(np.zeros((2, 2))) == pytest.approx(np.full((2, 1), 1.5))


def test_the_weight_file_holds_no_weights_and_the_model_is_loaded_from_the_file_it_names(
    session, torch, tmp_path
):
    method = booked(f"UserCode={user_code(tmp_path)}:FilenameModel=model.pt")
    model_file("model.pt")
    method.train(method.loader.dataset().train)
    model_file("TrainedModel_PyTorch.pt")
    method.write_weight_file()
    with open(weights("PyTorch")) as source:
        written = source.read()
    assert "<Weights>" not in written and "FilenameTrainedModel" in written
    assert Jit.saved[0][1] == os.path.join("dataset", "weights", "TrainedModel_PyTorch.pt")
    read = ROOT.TMVA.Reader("!Color:Silent")
    for name in ("var1", "var2", "var3", "var4"):
        read.AddVariable(name, np.zeros(1, dtype=np.float32))
    read.BookMVA("PyTorch", weights("PyTorch"))
    assert read.EvaluateMVA("PyTorch") == pytest.approx(0.25)


def test_a_model_that_is_no_longer_there_is_refused_when_it_is_first_asked_for(
    session, torch, tmp_path
):
    method = booked(f"UserCode={user_code(tmp_path)}:FilenameTrainedModel=gone.pt")
    with pytest.raises(xrdroot.UnsupportedFeatureError, match=r'"gone.pt".*does not exist'):
        method.evaluate(np.zeros((1, 4)))


def test_a_method_xrdroot_does_not_have_is_refused_by_name_where_pytorch_is_trained(session):
    with pytest.raises(xrdroot.UnsupportedFeatureError, match="TMlpANN"):
        ROOT.TMVA.Factory("job", QUIET + "Classification").BookMethod(loader(), "TMlpANN", "k", "")


# -- the real PyTorch, where it is installed --------------------------------------------

#: The user's file ROOT's own PyMVA tests hand TMVA, in miniature: a model scripted and
#: saved, and the loop, loss, optimiser and prediction beside it in one dict.
REAL_USER = '''
import torch
from torch import nn

model = nn.Sequential(nn.Linear({nvar}, 16), nn.ReLU(), nn.Linear(16, {nout}), {last})
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD


def fit(model, train_loader, val_loader, num_epochs, batch_size, optimizer, criterion,
        save_best, scheduler):
    trainer = optimizer(model.parameters(), lr=0.1)
    best = None
    for _ in range(num_epochs):
        model.train()
        for X, y in train_loader:
            trainer.zero_grad()
            criterion(model(X), y).backward()
            trainer.step()
        model.eval()
        with torch.no_grad():
            loss = sum(criterion(model(X), y).item() for X, y in val_loader) / len(val_loader)
        best = loss if best is None else best
        if save_best is not None:
            best = save_best(model, loss, best)
    return model


def predict(model, values, batch_size=32):
    model.eval()
    with torch.no_grad():
        return model(torch.Tensor(values)).numpy()


load_model_custom_objects = {{"optimizer": optimizer, "criterion": criterion,
                             "train_func": fit, "predict_func": predict}}
torch.jit.save(torch.jit.script(model), "model.pt")
'''


def real_user(tmp_path: Any, nvar: int, nout: int, last: str = "nn.Softmax(dim=1)") -> str:
    """The real PyTorch user's file for a network of ``nvar`` inputs and ``nout`` outputs."""
    text = REAL_USER.format(nvar=nvar, nout=nout, last=last)
    return user_code(tmp_path, text, "real_user.py")


def test_the_real_pytorch_trains_the_users_model_to_tell_signal_from_background(
    session, tmp_path
):
    pytest.importorskip("torch")
    import torch as real

    real.set_num_threads(1)
    # The user's file builds and saves the model, which the method then loads and trains.
    real_user(tmp_path, 4, 2)
    options = (
        "!H:!V:FilenameModel=model.pt:FilenameTrainedModel=trained.pt:NumEpochs=5:"
        "BatchSize=32:UserCode=real_user.py"
    )
    method = booked(options + ":VarTransform=D,G")
    method.train(method.handler.prepare(method.loader.dataset().train))
    assert os.path.exists("trained.pt")
    signal = method.evaluate(np.array([[1.0, 1.0, 1.0, 1.0]]))[0]
    background = method.evaluate(np.array([[-1.0, -1.0, -1.0, -1.0]]))[0]
    assert signal > background and signal > 0.5


def test_the_real_pytorchs_regression_learns_a_target_that_is_linear_in_two_variables(
    session, tmp_path
):
    """The target is normalised, so that a plain loop of ROOT's own learning rate learns it."""
    pytest.importorskip("torch")
    import torch as real

    real.set_num_threads(1)
    real_user(tmp_path, 2, 1, last="nn.Identity()")
    options = (
        "!H:!V:VarTransform=N:FilenameModel=model.pt:FilenameTrainedModel=trained.pt:"
        "NumEpochs=20:BatchSize=32:!SaveBestOnly:UserCode=real_user.py"
    )
    method = booked(options, "Regression")
    method.train(method.handler.prepare(method.loader.dataset().train))
    test = method.loader.dataset().test
    found = np.asarray(method.evaluate(method.handler.apply(test).values)).ravel()
    targets = test.targets.ravel()
    assert float(np.mean((found - targets) ** 2)) < 0.1 * float(np.var(targets))
