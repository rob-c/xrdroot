"""TMVA's DNN CPU backend under NumPy: its matrices, its loss functions and their gradients.

TMVA's own DNN tests check the backend against a reference written in the test, so the
formulas here are checked against the same reference the tests use - a few values worked
out by hand, and one of ROOT's own tests run end to end.
"""

from __future__ import annotations

import numpy as np
import pytest

from xrdroot import tmva
from xrdroot.pycxx.ctype import CType
from xrdroot.pyroot import tmva as pyroot_tmva
from xrdroot.tmva import dnn


def test_a_matrix_is_made_of_zeros_or_of_what_it_copies() -> None:
    zeros = dnn.matrix_type("double")(2, 3)
    assert (zeros.GetNrows(), zeros.GetNcols(), zeros.GetNoElements()) == (2, 3, 6)
    assert zeros.GetSize() == 6
    copied = dnn.matrix_type("double")(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert copied(1, 0) == 3.0
    copied.__setcall__(0, 1, 9.5)
    assert copied.values[0, 1] == 9.5
    from_other = dnn.matrix_type("double")(copied)
    assert from_other(0, 1) == 9.5
    assert np.asarray(copied).tolist() == [[1.0, 9.5], [3.0, 4.0]]
    assert dnn.matrix_type("double")(copied).values is not copied.values


def test_a_matrix_holds_the_precision_of_its_architecture() -> None:
    single = dnn.matrix_type("float")(1, 1)
    assert single.values.dtype == np.float32
    assert dnn.matrix_type("float")(np.array([[1 / 3]])).values.dtype == np.float32
    assert dnn.matrix_type("double")(1, 1).values.dtype == np.float64
    assert dnn.matrix_type("something else")(1, 1).values.dtype == np.float64
    assert dnn.matrix_type("Float_t")(1, 1).values.dtype == np.float32


def test_a_matrix_reads_a_tmatrix_its_elements_and_hands_them_back() -> None:
    class TMatrixT:
        values = np.array([[1.0, 2.0], [3.0, 4.0]])

    made = dnn.matrix_type("double")(TMatrixT())
    assert made.GetMatrixArray().tolist() == [1.0, 3.0, 2.0, 4.0]
    assert np.asarray(made, dtype=np.float32).dtype == np.float32
    copied = np.asarray(made, copy=True)
    copied[0, 0] = 99.0
    assert made(0, 0) == 1.0
    assert repr(made) == "<TCpuMatrix 2x2>"
    assert dnn.matrix_type("double")["double"] is not None


def test_a_matrix_prints_its_shape_and_its_elements(capsys: pytest.CaptureFixture[str]) -> None:
    matrix = dnn.matrix_type("double")(np.array([[1.5, -2.0]]))
    matrix.Print()
    assert capsys.readouterr().out == "Matrix 1x2\n  1.5 -2\n"


def test_the_mean_squared_error_is_the_weighted_mean_of_the_squared_differences() -> None:
    y = np.array([[1.0, 1.0], [1.0, 1.0]])
    x = np.array([[0.0, 0.5], [2.0, 1.0]])
    ones = np.ones((2, 2))
    assert dnn.mean_squared_error(y, x, ones) == pytest.approx((1 + 0.25 + 1 + 0) / 4)
    halved = np.array([[0.5], [2.0]])
    assert dnn.mean_squared_error(y, x, halved) == pytest.approx((0.5 + 0.125 + 2 + 0) / 4)


def test_the_mean_squared_errors_gradient_is_written_into_the_matrix_it_is_given() -> None:
    y = np.array([[1.0, 1.0], [1.0, 1.0]])
    x = np.array([[0.0, 0.5], [2.0, 1.0]])
    weights = np.array([[1.0], [3.0]])
    written = dnn.matrix_type("double")(2, 2)
    dnn.mean_squared_error_gradients(written, y, x, weights)
    assert written.values.tolist() == [[-0.5, -0.25], [1.5, 0.0]]


def test_the_cross_entropy_is_the_sigmoids_and_is_finite_far_from_zero() -> None:
    y = np.ones((1, 3))
    x = np.array([[0.0, 100.0, -100.0]])
    ones = np.ones((1, 3))
    expected = (np.log(2.0) + 100.0) / 3.0
    assert dnn.cross_entropy(y, x, ones) == pytest.approx(expected)
    assert np.isfinite(dnn.cross_entropy(y, x, ones))


def test_the_cross_entropys_gradient_is_the_sigmoid_less_the_truth() -> None:
    y = np.array([[0.0, 1.0]])
    x = np.array([[0.0, 0.0]])
    written = dnn.matrix_type("double")(1, 2)
    dnn.cross_entropy_gradients(written, y, x, np.ones((1, 2)))
    assert written.values.tolist() == [[0.25, -0.25]]


def test_the_softmax_cross_entropy_is_taken_over_each_row() -> None:
    y = np.array([[1.0, 0.0]])
    x = np.array([[0.0, 0.0]])
    ones = np.array([[1.0]])
    assert dnn.softmax_cross_entropy(y, x, ones) == pytest.approx(np.log(2.0))
    # one row's weights, not one element's: TMVA weights the loss per event
    twice = np.array([[2.0]])
    assert dnn.softmax_cross_entropy(y, x, twice) == pytest.approx(2.0 * np.log(2.0))


def test_the_softmax_cross_entropys_gradient_follows_the_softmax() -> None:
    y = np.array([[0.0, 1.0]])
    x = np.array([[0.0, 0.0]])
    written = dnn.matrix_type("double")(1, 2)
    dnn.softmax_cross_entropy_gradients(written, y, x, np.array([[2.0]]))
    assert written.values.tolist() == [[1.0, -1.0]]


def test_every_loss_function_of_tmvas_has_its_gradient() -> None:
    assert sorted(dnn.LOSSES) == sorted(dnn.LOSS_GRADIENTS)
    assert sorted(dnn.LOSSES) == ["CrossEntropy", "MeanSquaredError", "SoftmaxCrossEntropy"]
    y = np.array([[0.5, 0.5]])
    x = np.array([[0.25, -0.25]])
    ones = np.ones((1, 2))
    for name, loss in dnn.LOSSES.items():
        assert np.isfinite(loss(y, x, ones)), name
        written = dnn.matrix_type("double")(1, 2)
        dnn.LOSS_GRADIENTS[name](written, y, x, ones)
        assert np.all(np.isfinite(written.values)), name


def test_the_enumerations_are_tmvas_own_letters() -> None:
    assert pyroot_tmva.ELossFunction.kMeanSquaredError.name == "kMeanSquaredError"
    assert pyroot_tmva.EActivationFunction.kFastTanh == 7
    assert pyroot_tmva.EOutputFunction.kSoftmax == ord("M")
    assert pyroot_tmva.ELossFunction(ord("S")) is pyroot_tmva.ELossFunction.kSoftmaxCrossEntropy


def test_a_loss_function_is_named_by_its_architecture_as_tmva_names_it() -> None:
    evaluate = pyroot_tmva.DNN.evaluate
    y, x = np.zeros((2, 2)), np.ones((2, 2))
    weights = np.ones((2, 2))
    for architecture in ("TCpu<double>", pyroot_tmva.DNN.TCpuMatrix):
        assert evaluate[architecture](
            pyroot_tmva.ELossFunction.kMeanSquaredError, y, x, weights) == pytest.approx(1.0)
    written = pyroot_tmva.DNN.TCpuMatrix(2, 2)
    pyroot_tmva.DNN.evaluateGradients["TCpu<double>"](
        written, pyroot_tmva.ELossFunction.kMeanSquaredError, y, x, weights)
    assert written.values.tolist() == [[0.5, 0.5], [0.5, 0.5]]


def test_the_dnn_vocabulary_is_reached_under_tmva_and_by_its_tensor_type() -> None:
    assert pyroot_tmva.TMVA.DNN is pyroot_tmva.DNN
    assert pyroot_tmva.DNN.ATensor.Scalar_t is float
    assert pyroot_tmva.DNN.ATensor["float"] is pyroot_tmva.DNN.ATensor
    assert float(pyroot_tmva.DNN.ATensor.Scalar_t(2)) == 2.0
    assert repr(pyroot_tmva.DNN) == "<namespace TMVA::DNN>"


def test_a_name_tmva_has_and_this_does_not_is_refused_by_that_name() -> None:
    with pytest.raises(AttributeError, match=r"ROOT has TMVA::DNN::TCpu; xrdroot\.pyroot"):
        assert pyroot_tmva.DNN.TCpu
    with pytest.raises(AttributeError, match=r"ROOT has TMVA::DNN::TCpu"):
        assert pyroot_tmva.TMVA.DNN.TCpu
    assert tmva is not None


def test_a_dnn_templates_members_are_the_types_tmva_declares(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``Architecture::Matrix_t`` of ``TCpu<double>``: a ``TCpuMatrix<double>``, made here."""
    from xrdroot.pycint import translate
    from xrdroot.pycint.execute import run_source

    source = """
    using namespace TMVA::DNN;
    template <typename Architecture>
    auto made(size_t n) -> typename Architecture::Scalar_t {
       using Matrix_t = typename Architecture::Matrix_t;
       Matrix_t m(n, n);
       return (typename Architecture::Scalar_t) m.GetNrows();
    }
    void t() { printf("%d\\n", (int) made<TCpu<double>>(3)); }
    """
    assert "ROOT.TCpuMatrix['double'](n, n)" in translate(source, "t.C")
    assert run_source(source, "t.C") is None
    assert capsys.readouterr().out == "3\n"
    assert "ROOT.TCpuMatrix['float'](n, n)" in translate(
        source.replace("double", "float"), "t.C")


def test_a_member_of_a_dnn_template_instance_is_reached_through_the_instance() -> None:
    from xrdroot.pycint import translate

    source = """
    template <typename T> void f() { T::SetRandomSeed(3); }
    void t() { f<TCpu<double>>(); }
    """
    assert "ROOT.TCpu['double'].SetRandomSeed(3)" in translate(source, "t.C")


@pytest.mark.parametrize(
    ("name", "args", "expected"),
    [
        ("TCpu", ["float"], {"Scalar_t": "float", "Matrix_t": "TCpuMatrix<float>"}),
        ("TCuda", [], {"Scalar_t": "double", "Tensor_t": "TCpuTensor<double>"}),
        ("TReference", ["float"], {"HostBuffer_t": "TCpuBuffer<float>"}),
        ("TCpuMatrix", ["float"], {"Scalar_t": "float", "Tensor_t": "TCpuTensor<float>"}),
        ("TCpuTensor", ["double"], {"Matrix_t": "TCpuMatrix<double>"}),
        ("TDeepNet", [], {"Architecture_t": "TCpu", "Layer_t": "VGeneralLayer<TCpu>"}),
        ("TNet", ["TCuda"], {"Architecture_t": "TCuda"}),
        ("TNet", ["TCpu", "LabLayer"], {"Layer_t": "LabLayer"}),
        ("THist", [], None),
    ],
)
def test_the_member_aliases_of_every_dnn_class_tmva_names(
    name: str, args: list[str], expected: dict[str, str] | None
) -> None:
    """TMVA's DNN headers' aliases as data: what a test that names them gets."""
    from xrdroot.pycint.dialect import _dnn_members

    found = _dnn_members(name, [CType(arg) for arg in args])
    if expected is None:
        assert found is None
        return
    assert found is not None
    for member, spelled in expected.items():
        value = found[member]
        text = value.name + (f"<{','.join(arg.name for arg in value.args)}>" if value.args else "")
        assert text == spelled, member
