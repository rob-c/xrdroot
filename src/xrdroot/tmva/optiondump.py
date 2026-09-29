"""What ``Configurable::ParseOptions`` prints of an option string under ``V``.

With ``V``, TMVA lists the options a method was given and those left at
their defaults, each with its description - once for the base
``Configurable`` (which knows only ``Boost_num``) and once for the method.
The descriptions are TMVA's, for the options of the methods that are
booked verbose in the tutorials.
"""

from __future__ import annotations

from typing import Any

from .options import Options

__all__ = ["DL_OPTIONS", "print_parsed"]

#: ``MethodBase``'s options, their defaults as printed, and their descriptions.
BASE_OPTIONS = (
    (
        "V",
        "False",
        'Verbose output (short form of "VerbosityLevel" below - overrides the latter one)',
    ),
    ("VerbosityLevel", "Default", "Verbosity level"),
    (
        "VarTransform",
        "None",
        "List of variable transformations performed before training, "
        'e.g., "D_Background,P_Signal,G,N_AllClasses" for: "Decorrelation, PCA-transformation, '
        "Gaussianisation, Normalisation, each for the given class of events ('AllClasses' denotes "
        "all events of all classes, if no class indication is given, 'All' is assumed)\"",
    ),
    ("H", "False", "Print method-specific help message"),
    ("CreateMVAPdfs", "False", "Create PDFs for classifier outputs (signal and background)"),
    (
        "IgnoreNegWeightsInTraining",
        "False",
        "Events with negative weights are ignored in the "
        "training (but are included for testing and performance evaluation)",
    ),
)
#: ``MethodDL``'s own options.
DL_OPTIONS = (
    *BASE_OPTIONS,
    ("InputLayout", "0|0|0", "The Layout of the input"),
    ("BatchLayout", "0|0|0", "The Layout of the batch"),
    ("Layout", "DENSE|(N+100)*2|SOFTSIGN,DENSE|0|LINEAR", "Layout of the network."),
    (
        "ErrorStrategy",
        "CROSSENTROPY",
        "Loss function: Mean squared error (regression) or cross entropy (binary classification).",
    ),
    ("WeightInitialization", "XAVIER", "Weight initialization strategy"),
    ("RandomSeed", "0", "Random seed used for weight initialization and batch shuffling"),
    (
        "ValidationSize",
        "20%",
        "Part of the training data to use for validation. Specify as 0.2 "
        "or 20% to use a fifth of the data set as validation set. Specify as 100 to use exactly "
        "100 events. (Default: 20%)",
    ),
    ("Architecture", "CPU", "Which architecture to perform the training on."),
    (
        "TrainingStrategy",
        "LearningRate=1e-5,Momentum=0.3,Repetitions=3,ConvergenceSteps=50,"
        "BatchSize=30,TestRepetitions=7,WeightDecay=0.0,Renormalize=L2,DropConfig=0.0,"
        "DropRepetitions=5|LearningRate=1e-4,Momentum=0.3,Repetitions=3,ConvergenceSteps=50,"
        "BatchSize=20,TestRepetitions=7,WeightDecay=0.001,Renormalize=L2,DropConfig=0.0+0.5+0.5,"
        "DropRepetitions=5,Multithreading=True",
        "Defines the training strategies.",
    ),
)


def _block(log: Any, text: str, given: list[str], rest: list[str]) -> None:
    log.info("Parsing option string: ")
    log.info(f'... "{text}"')
    log.info("The following options are set:")
    log.info("- By User:")
    for line in given or ["    <none>"]:
        log.info(line)
    log.info("- Default:")
    for line in rest:
        log.info(line)


def print_parsed(log: Any, text: str, declared: tuple[tuple[str, str, str], ...]) -> None:
    """Both of ``ParseOptions``' listings of ``text`` against the ``declared`` options."""
    _block(log, text, [], ['    Boost_num: "0" [Number of times the classifier will be boosted]'])
    options = Options(text)
    given, rest = [], []
    for name, default, description in declared:
        if options.given(name):
            value = options.text_of(name, "")
            if value == "":
                value = "True" if options.flag(name, False) else "False"
            given.append(f'    {name}: "{value}" [{description}]')
        else:
            rest.append(f'    {name}: "{default}" [{description}]')
    _block(log, text, given, rest)
