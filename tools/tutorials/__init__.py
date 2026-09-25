"""The tutorial conformance harness: ROOT's tutorials, run by ROOT and by xrdroot.

ROOT's tutorials are its own specification by example: every one that ROOT
runs here is run again through ``python -m xrdroot run <tutorial>`` (C++
macros through the translator, PyROOT scripts with ``ROOT`` standing for
``xrdroot.pyroot``) and the two runs are compared - exit code, standard
output, and every file written. A dev tool: ROOT is never a dependency of
xrdroot, and nothing under ``src`` imports this.

Getting the tutorials
---------------------

Only ROOT's ``tutorials`` directory is needed, ideally from the same release
as the installed ROOT (``root-config --version``; tags are ``v6-40-04``)::

    git clone --filter=blob:none --no-checkout --depth 1 --branch v6-40-04 \\
        https://github.com/root-project/root.git rootsrc
    git -C rootsrc sparse-checkout set tutorials
    git -C rootsrc checkout

Running it
----------

::

    python -m tools.tutorials run --tutorials rootsrc/tutorials            # everything
    python -m tools.tutorials run --tutorials rootsrc/tutorials --only hist/ --jobs 8
    python -m tools.tutorials run --tutorials rootsrc/tutorials --oracle-only  # fill the cache
    python -m tools.tutorials report       # redraw the Markdown and HTML from results.json
    python -m tools.tutorials list --tutorials rootsrc/tutorials           # the catalogue

The oracle is the ROOT ``root-config`` finds (``--root-config`` for another);
its Python bindings are run with the ``python<X.Y>`` that ``root-config
--python-version`` names and ROOT's library directory on ``PYTHONPATH``, as
Homebrew's ROOT is used (``--root-python`` to choose another).

Where things go
---------------

``~/.cache/xrdroot-tutorials`` (or ``$XRDROOT_TUTORIALS_CACHE``, or
``--cache``) holds ``oracle/<ROOT version>/<key>/`` - each ROOT run's
``result.json`` and ``files/`` - and ``results/``: the latest
``results.json``, ``summary.md``, ``report.html``, and each xrdroot run under
``xrdroot/<tutorial>/``. Deleting the cache only costs the time to refill it;
``--refresh-oracle`` reruns ROOT without deleting anything.

The modules
-----------

:mod:`~tools.tutorials.cmake` runs ROOT's ``tutorials/CMakeLists.txt``;
:mod:`~tools.tutorials.catalog` turns that into one entry per tutorial;
:mod:`~tools.tutorials.environment` finds the installed ROOT;
:mod:`~tools.tutorials.runner` runs one tutorial in a sandbox;
:mod:`~tools.tutorials.harness` runs them all, both sides, and judges;
:mod:`~tools.tutorials.compare` and :mod:`~tools.tutorials.classify` hold
the judging; :mod:`~tools.tutorials.report` writes it up.
"""
