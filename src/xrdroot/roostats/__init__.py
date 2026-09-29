"""RooStats: the statistical tools of ROOT, over the RooFit engine of :mod:`xrdroot.roofit`.

A ``ModelConfig`` says which of a model's variables are the parameters of
interest, which are nuisances and which are observed; the calculators take
one and a dataset and answer a question about it - an interval
(``ProfileLikelihoodCalculator``, ``FeldmanCousins``, ``BayesianCalculator``,
``MCMCCalculator``), a hypothesis test (``AsymptoticCalculator``,
``FrequentistCalculator``, ``HybridCalculator``) or an upper limit
(``HypoTestInverter``). Each is RooStats' algorithm step for step, fitting
with the engine's Minuit and generating toys from RooRandom's generator,
so its numbers are ROOT's where ROOT's are reproducible, and each prints
what RooStats prints on the way.
"""

from __future__ import annotations
