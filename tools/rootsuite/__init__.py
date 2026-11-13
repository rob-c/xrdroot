"""The ROOT test-suite harness: ROOT's own ctest suite, discovered from its CMake, run by xrdroot.

ROOT's CI runs five kinds of test: ``tutorial-*``, ``gtest-*`` (C++
googletest programs), ``roottest-*`` (macros and commands whose output is
diffed against ``.ref`` files), ``pyunittests-*`` (Python unittest files)
and ``test-*`` (the stress programs in ``test/``). Only the tutorials need
ROOT itself, as an oracle to compare with - :mod:`tools.tutorials` runs
those. The other four carry their own verdict: an exit code, a reference
output, a regular expression. This package finds every one of them the way
ROOT's configure step does, runs each against xrdroot, and judges it the way
ctest and ``RootTestDriver.cmake`` would - so that xrdroot's progress is
measured against the whole suite. ctest programs of no family (the
``mathcore-*`` and ``tmva-*`` programs) are listed as ``other`` and run the
same way.

The suite is configured as on a Mac with Apple silicon, with the oracle's
build options (:mod:`~tools.rootsuite.machine`), and ROOT's build model is
xrdroot's: every dictionary, test library, test executable and ACLiC step
is an ``xrdroot build`` command (:mod:`~tools.rootsuite.builds`), every
``root.exe`` command line ``xrdroot root``, and ``hadd``, ``rootls`` and
the other tools xrdroot subcommands of the same names
(:mod:`~tools.rootsuite.plan`).

It never imports xrdroot: xrdroot is run as a subprocess, in the Python
given by ``--python`` with ``--xrdroot-src`` on its path, so this can become
a project of its own.

Getting the ROOT sources
------------------------

A sparse clone of the release CI ran: ``tutorials``, ``test``, ``roottest``,
``cmake/modules``, every ``CMakeLists.txt`` (the walk starts at the top one)
and every ``*/test`` directory, except under ``interpreter/`` (cling and
LLVM, whose own tests are not xrdroot's business)::

    git clone --filter=blob:none --no-checkout --depth 1 --branch v6-40-04 \\
        https://github.com/root-project/root.git rootsrc
    git -C rootsrc sparse-checkout set --no-cone '/tutorials/' '/test/' '/roottest/' \\
        '/cmake/modules/' '**/CMakeLists.txt' '**/test/' '!/interpreter/'
    git -C rootsrc checkout

Without the module ``CMakeLists.txt`` files, every ``*/test`` directory with
a ``CMakeLists.txt`` is taken instead. ROOT's sources are only ever read:
each test runs in a copy of its directory under ``--out``.

Running it
----------

::

    python -m tools.rootsuite list --root rootsrc                      # counts against CI's
    python -m tools.rootsuite list --root rootsrc --kind gtest --tests # one line per test
    python -m tools.rootsuite run --root rootsrc --jobs 6              # the four oracle-free kinds
    python -m tools.rootsuite run --root rootsrc --only 'roottest/root/io/*' --out /tmp/io
    python -m tools.rootsuite run --root rootsrc --kind tutorial       # tools.tutorials, needs ROOT
    python -m tools.rootsuite report --results OUT/results.json        # redraw summary.md
    python -m tools.rootsuite ratchet OUT/results.json [--rerun-timeouts]
    python -m tools.rootsuite baseline OUT/results.json                # rewrite baseline.json

``--out`` defaults to ``$XRDROOT_ROOTSUITE_OUT``, else ``xrdroot-rootsuite``
in the system's temporary directory - never the repository. It receives
``results.json``, ``summary.md``, ``logs/<id>/`` (each run's command and
output, its diff, its junit XML) and ``work/`` (the copies tests run in).

Statuses: PASS; FAIL (ran, wrong exit, crash, a FAILREGEX, or a fixture it
requires was not set up); DIFF (ran, output differs from its ``.ref``);
UNSUPPORTED (xrdroot refused: a subcommand exiting 2 with its one-line
refusal, ``UnsupportedFeatureError``, an error saying xrdroot does not have
something yet, or a subcommand this xrdroot lacks - ``xrdroot build``...);
OUT-OF-SCOPE (a label in ``scope.toml``: only what no Mac runs); SKIP (the
harness cannot express the test: roottest's old Makefiles, a program this
machine lacks, a missing Python package); TIMEOUT.

The modules
-----------

:mod:`~tools.rootsuite.cmake` extends the tutorial harness's CMake
interpreter with functions, scopes and directories;
:mod:`~tools.rootsuite.rootmacros` and :mod:`~tools.rootsuite.roottestmacros`
are ROOT's test macros as what they register; :mod:`~tools.rootsuite.machine`
is the Mac they are configured for; :mod:`~tools.rootsuite.discover`
walks the tree; :mod:`~tools.rootsuite.scope` reads ``scope.toml``;
:mod:`~tools.rootsuite.builds` makes ROOT's builds ``xrdroot build``
commands; :mod:`~tools.rootsuite.plan` turns a test into command lines,
:mod:`~tools.rootsuite.buildtree` gives them the build tree they expect,
:mod:`~tools.rootsuite.runner` runs them, :mod:`~tools.rootsuite.judge`,
:mod:`~tools.rootsuite.refdiff` and :mod:`~tools.rootsuite.cases` judge them;
:mod:`~tools.rootsuite.report` and :mod:`~tools.rootsuite.ratchet` keep score.
"""
