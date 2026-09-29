"""The framework's parts asked directly: root finding, norms, weight files, option dumps."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from tmvasupport import classify, loader, session, weights
from xrdroot.tmva import efficiency, evaluation, hists, weightfile
from xrdroot.tmva.log import Logger
from xrdroot.tmva.optiondump import DL_OPTIONS, print_parsed

__all__ = ["session"]


def test_brent_s_method_finds_a_cube_root_by_interpolation():
    found = efficiency.root(lambda x: x**3, -1.0, 2.0, 2.0)
    assert found == pytest.approx(2.0 ** (1 / 3), 1e-12)


def test_a_root_is_not_looked_for_where_the_ends_do_not_straddle_it(capsys):
    assert efficiency.root(lambda x: x, 1.0, 2.0, 5.0) == 1.0
    assert "<Root> initial interval w/o root: (a=1, b=2), refValue = 5" in capsys.readouterr().out


def test_a_jump_that_is_never_reached_exhausts_the_iterations(capsys):
    found = efficiency.root(lambda x: -1.0 if x <= 0 else 1.0, -1.0, 3.0, 0.0)
    assert abs(found) < 1e-20
    assert "maximum iterations (100) reached" in capsys.readouterr().out


def test_efficiencies_of_a_classifier_whose_signal_lies_below_are_counted_from_below():
    rng = np.random.default_rng(3)
    values = np.concatenate((rng.normal(-1, 1, 300), rng.normal(1, 1, 300)))
    signal = np.arange(600) < 300
    weights = np.ones(600)
    low, high = float(values.min()), float(values.max())
    made = efficiency.Efficiencies("MVA_x", low, high, positive=False)
    made.test(values, signal, weights, low, high + 1e-5)
    made.train(values, signal, weights, low, high + 1e-5)
    eff, error = made.efficiency(0.1)
    assert 0.5 < eff < 0.9 and error > 0
    assert made.training_efficiency(0.1) == pytest.approx(eff, abs=0.05)
    assert 0.7 < made.area() < 1.0


def test_a_sample_of_no_weight_has_no_mean_or_spread():
    values, signal = np.array([1.0, 2.0]), np.array([False, False])
    stats = evaluation.compute_stat(values, signal, np.ones(2))
    assert stats[0] == 0.0 and stats[2] == 0.0 and stats[1] == 1.5


def test_an_empty_histogram_is_not_normalised_and_a_negative_one_keeps_its_scale():
    empty = hists.book("e", "e", 4, 0.0, 1.0)
    assert evaluation.norm_hist(empty) == 1.0
    negative = hists.book("n", "n", 4, 0.0, 1.0)
    negative.fill(np.array([0.5]), weight=np.array([-2.0]))
    assert evaluation.norm_hist(negative) == -0.5
    assert hists.bins(negative)[3] == -2.0
    assert evaluation.separation_of_hists(empty, negative) == 0.0


def test_a_test_without_output_densities_has_no_separation():
    made = evaluation.ClassifierTest(1.0, 0.0, 1.0, 1.0, -1.0, 1.0, True)
    assert made.separation() == 0.0
    assert made.significance() == pytest.approx(1 / np.sqrt(2))


def test_the_option_dump_lists_the_given_options_and_the_defaults(capsys):
    print_parsed(
        Logger("DL"), "!H:V:Layout=DENSE|8|RELU:Architecture:!BatchLayout:RandomSeed=", DL_OPTIONS
    )
    printed = capsys.readouterr().out
    assert printed.count("Parsing option string: ") == 2
    assert '    V: "True" [Verbose output' in printed
    assert '    H: "False" [Print method-specific help message]' in printed
    assert '    Layout: "DENSE|8|RELU" [Layout of the network.]' in printed
    assert '    Architecture: "True" [Which architecture' in printed
    assert '    BatchLayout: "False" [The Layout of the batch]' in printed
    assert '    RandomSeed: "False" [Random seed' in printed
    assert "    <none>" in printed and 'Boost_num: "0"' in printed


#: A weight file of the least a method's state can be: no transformations, no analysis named.
BARE = """<?xml version="1.0"?>
<MethodSetup Method="Fisher::Bare">
  <GeneralInfo><Info name="Creator" value="me"/></GeneralInfo>
  <Options>
    <Option name="Method" modified="Yes">Fisher</Option>
    <Option name="Array" modified="Yes" size="2">1,2</Option>
  </Options>
  <Variables NVar="1">
    <Variable VarIndex="0" Expression="x" Label="x" Title="x" Unit="" Internal="x" Type="F"
              Min="-1" Max="1"/>
  </Variables>
  <Classes NClass="2"><Class Name="Signal" Index="0"/><Class Name="Background" Index="1"/></Classes>
  <Weights NCoeff="2">
    <Coefficient Index="0" Value="0.5"/><Coefficient Index="1" Value="2"/>
  </Weights>
</MethodSetup>
"""


def test_a_bare_weight_file_is_read_with_its_own_declarations(session, tmp_path):
    path = tmp_path / "bare.weights.xml"
    path.write_text(BARE)
    root = ET.parse(path).getroot()
    assert weightfile.options_text(root) == "Method=Fisher"
    assert weightfile._analysis(root) == 0
    assert weightfile._analysis(ET.fromstring("<MethodSetup/>")) == 0
    assert weightfile.read_variables(ET.fromstring("<MethodSetup/>"), "Variables") == []
    method = weightfile.read_method(str(path))
    assert method.GetName() == "Bare" and method.GetMethodTypeName() == "Fisher"
    assert [v.label for v in method.dsi.variables] == ["x"]
    assert [c.name for c in method.dsi.classes] == ["Signal", "Background"]
    assert weightfile.options_text(ET.fromstring("<MethodSetup/>")) == ""


def test_a_weight_file_is_written_where_the_method_is_when_it_has_no_directory(session):
    factory = classify([(ROOT.TMVA.Types.kLD, "LD", "")], "!V:Silent", output="")
    method = factory.GetMethod("dataset", "LD")
    method.weight_dir = ""
    method.write_weight_file()
    assert weightfile.read_method("job_LD.weights.xml").GetMethodName() == "LD"
    assert weights("LD").endswith(method.weight_file)


def test_an_option_given_twice_is_warned_of_when_the_method_is_booked(session, capsys):
    factory = ROOT.TMVA.Factory("job", "!V:AnalysisType=Classification")
    factory.BookMethod(loader(), "LD", "LD", "VarTransform=N:VarTransform=D")
    printed = capsys.readouterr().out
    assert "Value for option VarTransform was previously set to N" in printed


def test_a_factory_given_its_options_before_a_file_takes_them_as_its_options(session):
    factory = ROOT.TMVA.Factory("job", "!V:Silent:AnalysisType=Regression", None)
    assert factory.analysis == 1 and factory.output.silent
    cv = ROOT.TMVA.CrossValidation("job", loader(), "!V:Silent:NumFolds=3", None)
    assert cv.GetNumFolds() == 3 and cv.output_file is None


def test_a_factory_left_to_find_its_analysis_finds_three_classes_multiclass(session):
    from test_tmva_frame_multiclass import three_classes

    factory = ROOT.TMVA.Factory("job", "!V:Silent")
    factory.BookMethod(three_classes(), "BDT", "BDT", "NTrees=5:BoostType=Grad")
    assert factory.analysis == 2
