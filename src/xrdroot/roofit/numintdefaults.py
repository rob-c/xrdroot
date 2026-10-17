"""RooFit's registered numeric integrators and their configuration parameters.

These are the ROOT 6.40 defaults, independent of the oracle installation.
Keeping registration order preserves each method category's public indices.
"""

from __future__ import annotations

from typing import Any

#: Integrator name, capabilities (1D, 2D, ND, open), dependency, parameter specifications.
PLUGINS: tuple[Any, ...] = (
    (
        "RooBinIntegrator",
        (True, True, True, False),
        "",
        (("numBins", "Number of bins in range", 100.0),),
    ),
    (
        "RooIntegrator1D",
        (True, False, False, False),
        "",
        (
            ("sumRule", "Summation Rule", ((("Midpoint", 1), ("Trapezoid", 0)), 0)),
            ("extrapolation", "Extrapolation procedure", ((("None", 0), ("Wynn-Epsilon", 1)), 1)),
            ("maxSteps", "Maximum number of steps", 20.0),
            ("minSteps", "Minimum number of steps", 999.0),
            ("fixSteps", "Fixed number of steps", 0.0),
        ),
    ),
    ("RooIntegrator2D", (False, True, False, False), "RooIntegrator1D", ()),
    (
        "RooSegmentedIntegrator1D",
        (True, False, False, False),
        "RooIntegrator1D",
        (("numSeg", "Number of segments", 3.0),),
    ),
    ("RooSegmentedIntegrator2D", (False, True, False, False), "RooSegmentedIntegrator1D", ()),
    ("RooImproperIntegrator1D", (True, False, False, True), "RooIntegrator1D", ()),
    (
        "RooMCIntegrator",
        (True, True, True, False),
        "",
        (
            (
                "samplingMode",
                "Sampling Mode",
                ((("Importance", 0), ("ImportanceOnly", 1), ("Stratified", 2)), 0),
            ),
            ("genType", "Generator Type", ((("PseudoRandom", 1), ("QuasiRandom", 0)), 0)),
            ("verbose", "Verbose flag", ((("false", 0), ("true", 1)), 0)),
            ("alpha", "Grid structure constant", 1.5),
            ("nRefineIter", "Number of refining iterations", 5.0),
            ("nRefinePerDim", "Number of refining samples (per dimension)", 1000.0),
            ("nIntPerDim", "Number of integration samples (per dimension)", 5000.0),
        ),
    ),
    (
        "RooAdaptiveIntegratorND",
        (False, True, True, False),
        "",
        (
            ("maxEval2D", "Max number of function evaluations for 2-dim integrals", 100000.0),
            ("maxEval3D", "Max number of function evaluations for 3-dim integrals", 1000000.0),
            ("maxEvalND", "Max number of function evaluations for >3-dim integrals", 10000000.0),
            ("maxWarn", "Max number of warnings on precision not reached that is printed", 5.0),
        ),
    ),
    (
        "RooAdaptiveGaussKronrodIntegrator1D",
        (True, False, False, True),
        "",
        (
            ("maxSeg", "maximum number of segments", 100.0),
            (
                "method",
                "Integration method for each segment",
                (
                    (
                        ("15Points", 1),
                        ("21Points", 2),
                        ("31Points", 3),
                        ("41Points", 4),
                        ("51Points", 5),
                        ("61Points", 6),
                        ("WynnEpsilon", 0),
                    ),
                    2,
                ),
            ),
        ),
    ),
    ("RooGaussKronrodIntegrator1D", (True, False, False, True), "", ()),
)
