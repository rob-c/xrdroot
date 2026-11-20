"""PyMVA's interpreter: ``TMVA::PyMethodBase`` and ``TMVA::Python_Executable``.

PyMVA's methods embed a Python interpreter so that a user's Keras or PyTorch
model file can be imported and trained. xrdroot is itself a Python process, so
the interpreter PyMVA starts and stops is already there: ``PyInitialize`` and
``PyFinalize`` have nothing to do, and a test that calls them is doing nothing
more than saying which kind of method it is about to book.

What is not nothing is the executable a PyMVA test shells out to when it
generates its model file (``gSystem->Exec(TMVA::Python_Executable() + " gen.py")``):
TMVA answers with the interpreter it embeds, so xrdroot answers with the one
running it, which is the only one that can import the packages the model needs.
"""

from __future__ import annotations

import sys

__all__ = ["PyMethodBase", "Python_Executable"]


def Python_Executable() -> str:
    """The interpreter running xrdroot: the one PyMVA's model scripts must use."""
    return sys.executable


class PyMethodBase:
    """``TMVA::PyMethodBase``: the base of PyMVA's methods, whose interpreter is this process."""

    @staticmethod
    def PyInitialize() -> None:
        """Start the embedded interpreter, which xrdroot already is."""

    @staticmethod
    def PyFinalize() -> None:
        """Shut the embedded interpreter down, which belongs to whoever started xrdroot."""
