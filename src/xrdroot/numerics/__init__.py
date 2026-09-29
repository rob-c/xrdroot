"""The numerical routines ROOT's statistics tools lean on, ported step for step.

``ROOT::Math::Integrator`` is GSL's QAGS by default - adaptive Gauss-Kronrod
with Wynn's epsilon extrapolation (:mod:`.qags`); ``ROOT::Math::RootFinder``
is GSL's Brent solver (:mod:`.roots`); ``BrentMinimizer1D`` is MathCore's
own (:mod:`.minimize1d`). RooStats' Bayesian intervals are integrals and
roots of these, so the digits they print are these routines' digits.
"""

from __future__ import annotations
