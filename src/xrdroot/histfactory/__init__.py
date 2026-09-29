"""HistFactory: binned models built from histograms - samples, their systematics, channels.

A :class:`~.measurement.Measurement` holds channels, each of samples read
from ROOT files with their uncertainties - overall normalisation
systematics, shape systematics, statistical uncertainties, free factors -
and ``MakeModelAndMeasurementFast`` builds from it what ROOT's
``HistoToWorkspaceFactoryFast`` builds: a ``RooWorkspace`` of RooFit's
objects - ``RooRealSumPdf`` of ``RooHistFunc`` shapes times
``FlexibleInterpVar`` and normalisation factors, ``ParamHistFunc`` gammas
and their constraints, a ``RooSimultaneous`` over the channels - with its
``ModelConfig``, its observed and Asimov data, named and printed as ROOT
names and prints them. The model is RooFit's, on xrdroot's RooFit engine:
fits, profiles and every RooStats calculator take it as they take any
model, and read the same model back from a ROOT file.
"""
