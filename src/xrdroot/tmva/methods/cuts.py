"""``Cuts``: TMVA's rectangular cut optimisation, by Monte Carlo sampling or its genetic algorithm.

For every signal efficiency in 100 bins the method finds the box of cuts
- a lower and an upper edge per variable - with the least background
efficiency, by trying boxes: ``FitMethod=MC`` draws ``SampleSize`` of them
at random, ``GA`` evolves them with TMVA's genetic algorithm towards TMVA's
estimator. ``VarProp=FSmart`` (``FMax``, ``FMin``) keeps one edge of a
variable's cut at infinity, the side decided by which class has the larger
mean. What it answers is a table, not an output: an event passes or fails
the cuts of the signal efficiency asked for (``EvaluateMVA(name, effS)``),
and TMVA's efficiencies are read from the table rather than from an output
distribution. Everything here is TMVA's, the Monte Carlo's draws included.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import hists
from ..cutsfit import MAX_CUT, CutTable, Sample, draw_mc
from ..dataset import Events
from ..genetic import CxxVector, GeneticFitter, Interval
from ..log import Logger
from ..method import Method
from ..xmlfile import Node, children, number

__all__ = ["MethodCuts"]

#: ``EFitMethodType`` and ``EEffMethod``, as a weight file writes them.
FIT_METHODS = {"MC": 0, "GA": 1, "SA": 2, "MINUIT": 3, "EventScan": 4, "MCEvents": 5}
#: How the optimisation method is described, by its name.
FIT_NAMES = {"MC": "Monte Carlo", "EventScan": "Full Event Scan (slow)", "MINUIT": "MINUIT"}
#: What reading a weight file says of each fitter.
READ_NAMES = {
    0: "sample of MC events",
    5: "sample of MC-Event events",
    1: "Genetic Algorithm",
    2: "Simulated Annealing algorithm",
    4: "Full Event Scan",
}


class MethodCuts(Method):
    """``TMVA::MethodCuts``."""

    type_name = "Cuts"
    defaults = {
        "FitMethod": "GA",
        "EffMethod": "EffSel",
        "SampleSize": 100000,
        "Sigma": -1.0,
        "Seed": 100,
    }

    def process_options(self) -> None:
        nvar = self.dsi.GetNVariables()
        self.properties = self.options.array("VarProp", nvar, "NotEnforced")
        self.range_min = self.options.array("CutRangeMin", nvar, -1.0)
        self.range_max = self.options.array("CutRangeMax", nvar, -1.0)
        self.fit = str(self.opt("FitMethod"))
        if self.fit not in ("MC", "GA"):
            raise self.log.fatal(
                f"FitMethod={self.fit} is a fit method of TMVA's Cuts xrdroot does not have; "
                "it has MC and GA"
            )
        self.table = CutTable(nvar=nvar)
        self.test_signal_eff = -1.0

    def booked(self) -> None:
        self.log.info(f'Use optimization method: "{FIT_NAMES.get(self.fit, "Genetic Algorithm")}"')
        self.log.info('Use efficiency computation method: "Event Selection"')
        for info, prop in zip(self.dsi.variables, self.properties):
            if prop != "NotEnforced":
                self.log.info(f"Use \"{prop}\" cuts for variable: '{info.label}'")

    # -- training -----------------------------------------------------------------------

    def _intervals(self, sample: Sample) -> list[Interval]:
        """Each variable's edge and width intervals, from its range and its ``VarProp``."""
        values, signal, weights = sample.values, sample.signal, sample.weights
        self.forced = []
        intervals = []
        for index in range(values.shape[1]):
            column = values[:, index]
            low, high = float(column.min()), float(column.max())
            eps = 0.01 * (high - low)
            low, high = low - eps, high + eps
            if abs(self.range_min[index] - self.range_max[index]) >= 1.0e-300:
                low, high = max(low, self.range_min[index]), min(high, self.range_max[index])
            prop = self.properties[index]
            if prop == "FSmart":
                mean_s = np.average(column[signal], weights=weights[signal])
                mean_b = np.average(column[~signal], weights=weights[~signal])
                prop = "FMax" if mean_s > mean_b else "FMin"
            self.forced.append(prop)
            nbins = int(high - low) + 1 if self.dsi.variables[index].vartype == "I" else 0
            if prop == "FMin":
                intervals += [Interval(low, low, nbins), Interval(0, high - low, nbins)]
            elif prop == "FMax":
                intervals += [Interval(low, high, nbins), Interval(high - low, high - low, nbins)]
            else:
                intervals += [Interval(low, high, nbins), Interval(0, high - low, nbins)]
        self.intervals = intervals
        return intervals

    def train(self, events: Events) -> None:
        if self.handler.transforms:
            self.handler.print_stats(events)
        sample = Sample(
            events.values, events.classes == self.dsi.GetSignalClassIndex(), events.weights
        )
        intervals = self._intervals(sample)
        self.table = CutTable(nvar=len(self.forced))
        if self.fit == "MC":
            Logger("FitterBase").header("<MCFitter> Sampling, please be patient ...")
            self._monte_carlo(sample, intervals)
            Logger("FitterBase").info("Elapsed time: 0 sec                           ")
        else:
            self.sample = sample
            GeneticFitter(self, f"{self.name}Fitter_GA", intervals, self.options.text).Run()
        self._force()
        eff = 0.1
        while eff < 0.95:
            self.print_cuts(eff + 0.0001)
            eff += 0.1

    def _monte_carlo(self, sample: Sample, intervals: list[Interval]) -> None:
        parameters = draw_mc(intervals, int(self.opt("SampleSize")), int(self.opt("Seed")))
        for start in range(0, len(parameters), 2000):
            block = parameters[start : start + 2000]
            lower = block[:, 0::2]
            upper = lower + block[:, 1::2]
            effs, effb = sample.efficiencies(lower, upper)
            self.table.offer_batch(effs, effb, lower, upper)

    def EstimatorFunction(self, parameters: Any) -> float:
        """``ComputeEstimator``: one box's figure of merit for the genetic algorithm - and it is kept."""
        values = np.asarray(list(parameters), dtype=np.float64)
        lower = values[0::2][None, :]
        upper = lower + values[1::2][None, :]
        effs, effb = self.sample.efficiencies(lower, upper)
        effs_, effb_ = float(effs[0]), float(effb[0])
        table = self.table
        index = int(table.bins(np.array([effs_]))[0]) - 1
        known = table.effb[min(index, table.nbins - 1)]
        left = table.effb[index - 1] if index > 0 else known
        right = table.effb[index + 1] if index < table.nbins - 1 else known
        average = known if known < effb_ else 0.5 * (left + right)
        eta = (-abs(known - average) + (1.0 - (known - effb_))) / (1.0 + effs_)
        table.offer(effs, effb, lower, upper)
        if index > 0:
            return float(eta)
        return self._penalty(lower[0], upper[0], effs_)

    def _penalty(self, lower: Any, upper: Any, effs: float) -> float:
        """The estimator of a box in the first bin: pushed towards the whole range."""
        penalty = 0.0
        for index in range(len(lower)):
            low, high = self.intervals[2 * index].low, self.intervals[2 * index].high
            width = high - low if high > low else 1.0
            penalty += ((high - upper[index]) / width) ** 2 + 4 * (
                (low - lower[index]) / width
            ) ** 2
        return 10.0 + penalty if effs < 1.0e-4 else 10.0 * (1.0 - 10.0 * effs)

    def _force(self) -> None:
        """The edges ``FMin`` and ``FMax`` keep at infinity, put there."""
        for index, prop in enumerate(self.forced):
            if prop == "FMin":
                self.table.lower[:, index] = -MAX_CUT
            if prop == "FMax":
                self.table.upper[:, index] = MAX_CUT

    # -- the cuts -----------------------------------------------------------------------

    def GetCuts(self, effs: float, cut_min: Any = None, cut_max: Any = None) -> float:
        """``GetCuts``: the cuts for a signal efficiency, put in the vectors given; its bin's low edge."""
        index = int(self.table.bins(np.array([float(effs)]))[0])
        true_effs = (index - 1) / self.table.nbins
        row = min(max(index - 1, 0), self.table.nbins - 1)
        for target, values in ((cut_min, self.table.lower[row]), (cut_max, self.table.upper[row])):
            if target is not None:
                target.clear()
                for value in values:
                    target.push_back(float(value))
        return true_effs

    def GetInputVar(self, index: int) -> str:
        return self.dsi.variables[index].expression

    def _labels(self) -> list[str]:
        """What each cut is on: the variable, or the transformed combination of them."""
        transforms = self.handler.transforms
        labels = [info.label for info in self.dsi.variables]
        if not transforms:
            return labels
        if len(transforms) > 1:
            return [f"{label} [transformed]" for label in labels]
        last = transforms[-1]
        if last.xml_name != "Decorrelation":
            return [f"{label}_[transformed]" for label in labels]
        matrix = last.which(None)
        return [
            "".join(
                f"{' + ' if value > 0 else ' - '}{abs(value):10.5g}*[{name}]"
                for value, name in zip(row, labels)
            )
            for row in matrix
        ]

    def print_cuts(self, effs: float) -> None:
        """``PrintCuts``: the cuts at a signal efficiency, as TMVA's table of them."""
        lower, upper = CxxVector(), CxxVector()
        true_effs = self.GetCuts(effs, lower, upper)
        index = int(self.table.bins(np.array([effs]))[0])
        labels = self._labels()
        width = max(len(label) for label in labels)
        rule = "-" * (20 + width + 16)
        self.log.info(rule)
        self.log.header(f"Cut values for requested signal efficiency: {true_effs:g}")
        self.log.info(f"Corresponding background efficiency       : {self.table.effb[index - 1]:g}")
        transforms = self.handler.transforms
        if len(transforms) == 1:
            self.log.info(f'Transformation applied to input variables : "{transforms[0].name}"')
        elif transforms:
            self.log.info(
                f"[ More than one (={len(transforms)})  transformations applied in transformation "
                "chain; cuts applied on transformed quantities ] "
            )
        else:
            self.log.info("Transformation applied to input variables : None")
        self.log.info(rule)
        for i, (low, high, label) in enumerate(zip(lower, upper, labels)):
            self.log.info(f"Cut[{i:2d}]: {low:10.6g} < {label:>{width}} <= {high:10.6g}")
        self.log.info(rule)

    # -- the output ---------------------------------------------------------------------

    def evaluate(self, values: Any) -> Any:
        """``GetMvaValue``: 1 for an event passing the cuts of ``test_signal_eff``, 0 otherwise."""
        if self.test_signal_eff <= 0:
            return np.zeros(len(values))
        index = int(self.table.bins(np.array([self.test_signal_eff]))[0])
        row = min(max(index, 0), self.table.nbins - 1)
        lower, upper = self.table.lower[row], self.table.upper[row]
        passed = np.all((values > lower) & (values <= upper), axis=1)
        return passed.astype(np.float64)

    def monitoring(self, output: Any, directory: str) -> None:
        self.log.info(f"{output.GetName()}:/{directory}")
        made = hists.book(
            f"{self.testvar}_effBvsSLocal", f"{self.name} efficiency of B vs S", 100, 0.0, 1.0
        )
        hists.set_bins(made, np.concatenate(([0.0], self.table.effb, [0.0])), entries=100)
        output.write(directory, made)

    def add_weights(self, node: Node) -> None:
        weights = node.add(
            "Weights", OptimisationMethod=0, FitMethod=FIT_METHODS[self.fit], nbins=self.table.nbins
        )
        for index in range(self.table.nbins):
            true_effs = index / self.table.nbins
            item = weights.add(
                "Bin",
                ibin=index + 1,
                effS=number(true_effs if true_effs > 1e-10 else 0.0),
                effB=number(self.table.effb[index]),
            )
            cuts = item.add("Cuts")
            for var in range(self.table.lower.shape[1]):
                cuts.set(f"cutMin_{var}", number(self.table.lower[index, var]))
                cuts.set(f"cutMax_{var}", number(self.table.upper[index, var]))

    def read_weights(self, node: Any) -> None:
        method = int(node.get("FitMethod", 0))
        self.log.info(f"Read cuts optimised using {READ_NAMES.get(method, 'unknown method')}")
        bins = children(node, "Bin")
        nvar = self.dsi.GetNVariables()
        self.log.info(f"Reading {len(bins)} signal efficiency bins for {nvar} variables")
        self.table = CutTable(nbins=int(node.get("nbins", len(bins))), nvar=nvar)
        for item in bins:
            index = int(item.get("ibin")) - 1
            self.table.effb[index] = float(np.float32(item.get("effB")))
            cuts = item.find("Cuts")
            for var in range(nvar):
                self.table.lower[index, var] = float(cuts.get(f"cutMin_{var}"))
                self.table.upper[index, var] = float(cuts.get(f"cutMax_{var}"))

    # -- the efficiencies ----------------------------------------------------------------

    def _sample(self, events: Events) -> Sample:
        transformed = self.handler.apply(events)
        return Sample(
            transformed.values, events.classes == self.dsi.GetSignalClassIndex(), events.weights
        )

    def _table_efficiencies(self, sample: Sample) -> tuple[Any, Any]:
        return sample.efficiencies(self.table.lower, self.table.upper)

    def _hist(self, suffix: str, title: str, contents: Any, high: float = 1.0) -> Any:
        made = hists.book(f"{self.testvar}{suffix}", title, self.table.nbins, 0.0, high)
        return hists.set_bins(
            made, np.concatenate(([0.0], contents, [0.0])), entries=self.table.nbins
        )

    def classifier_evaluation(
        self, test: Events, train: Events
    ) -> tuple[dict[str, Any], list[Any]]:
        """``MethodCuts::GetEfficiency`` and ``GetTrainingEfficiency``: the table read on each sample."""
        from ..pdf import spline1

        warn = Logger("Cuts")
        warn.warning(
            "You have asked for histogram MVA_EFF_BvsS which does not seem to exist in "
            "*Results* .. better don't use it "
        )
        if self.handler.transforms:
            self.handler.print_stats(self.handler.apply(test))
        effs, effb = self._table_efficiencies(self._sample(test))
        xs = np.concatenate(([0.0], effs, [1.0]))
        ys = np.concatenate(([0.0], effb, [1.0]))
        centres = (np.arange(1, self.table.nbins + 1) - 0.5) / np.float32(self.table.nbins)
        curve = spline1(xs, ys, centres)
        made = [
            self._hist("_effBvsS", self.testvar, curve),
            self._hist("_rejBvsS", self.testvar, 1.0 - curve),
            self._hist("_effS", f"{self.testvar} (signal)", effs, 1.000001),
            self._hist("_effB", f"{self.testvar} (background)", effb, 1.000001),
        ]
        warn.warning(
            "You have asked for histogram EFF_BVSS_TR which does not seem to exist in "
            "*Results* .. better don't use it "
        )
        if self.handler.transforms:
            self.handler.print_stats(self.handler.apply(train))
        train_s, train_b = self._table_efficiencies(self._sample(train))
        kept = self.table.bins(train_s) == np.arange(1, self.table.nbins + 1)
        training = np.where(kept, train_b, -0.1)
        made += [
            self._hist("_trainingEffBvsS", self.testvar, training),
            self._hist("_trainingRejBvsS", self.testvar, np.where(kept, 1.0 - train_b, 0.0)),
        ]
        steps = (np.arange(1, 1001) - 0.5) / np.float32(1000)
        test_curve = spline1(xs, ys, steps)
        train_curve = spline1(centres.astype(np.float64), training, steps)
        found: dict[str, Any] = {
            "sig": -1.0,
            "sep": -1.0,
            "name": self.name,
            "area": float(np.mean(1.0 - test_curve)),
        }
        found["roc"] = found["area"]
        for level in (0.01, 0.10, 0.30):
            found[f"eff{level}"] = _crossing(steps, test_curve, np.float32(level))
            found[f"train{level}"] = _crossing(steps, train_curve, np.float32(level))
        return found, made


def _crossing(effs: Any, effb: Any, reference: float) -> float:
    """The signal efficiency where the curve crosses ``reference`` - strictly, as ``MethodCuts`` scans."""
    previous_s = previous_b = 0.0
    for eff_s, eff_b in zip(effs, effb):
        if (eff_b - reference) * (previous_b - reference) < 0:
            return 0.5 * (float(eff_s) + previous_s)
        previous_s, previous_b = float(eff_s), float(eff_b)
    return 0.5 * (float(effs[-1]) + previous_s)
