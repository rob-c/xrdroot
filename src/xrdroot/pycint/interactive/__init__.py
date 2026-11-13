"""ROOT's C++ prompt - ``root -l`` and then C++ at ``root [0]`` - on xrdroot's translator.

    >>> from xrdroot.pycint.interactive import Interpreter
    >>> prompt = Interpreter()
    >>> prompt.feed("int i = 3;")
    []
    >>> prompt.feed("i + 1")
    [Said(text='(int) 4', error=False)]

What is typed is C++ as cling takes it: declarations that stay for the
rest of the session, statements that run at once, unfinished inputs that
wait for their closing brace, and an expression without its ``;`` whose
value is printed as cling prints it. Each input is translated into Python
by :mod:`xrdroot.pycint` together with everything typed before it, and run
against :mod:`xrdroot.pyroot`, so every engine - histograms, trees, RooFit,
TMVA, RDataFrame, graphics - is there by its ROOT name. The dot commands
are cling's and ROOT's: ``.x``, ``.L``, ``.ls``, ``.q`` and the rest.

:class:`Interpreter` is the prompt without a terminal, which tests feed;
:func:`interact` is the terminal loop around it, and ``xrdroot root`` (also
``xrdroot cint``) the command line.
"""

from __future__ import annotations

from .commands import Said
from .interpreter import Interpreter
from .terminal import interact

__all__ = ["Interpreter", "Said", "interact"]
