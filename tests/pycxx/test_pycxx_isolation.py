"""The translator stands alone: importing it, and running C++ with it, loads nothing of ROOT's."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

#: What a fresh interpreter runs: the translator imported, a program translated and run, and
#: every ``xrdroot`` module then loaded listed.
SCRIPT = r"""
import sys
import xrdroot.pycxx as pycxx
from xrdroot.pycxx.runtime import gtest  # noqa: F401
names = pycxx.run_source("int f() { std::vector<int> v{1, 2}; return v.size(); }", "f.cxx")
assert names == 2, names
print("\n".join(sorted(name for name in sys.modules if name.startswith("xrdroot"))))
"""

#: The packages a plain C++ program may load: the translator, the standard library, and the
#: package itself, whose bootstrap is inert.
ALLOWED = ("xrdroot.pycxx", "xrdroot.cppstd")


def test_importing_and_running_the_translator_loads_nothing_above_it(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[2] / "src"
    environment = {**os.environ, "PYTHONPATH": str(source), "XRDROOT_CINT_CACHE": str(tmp_path)}
    done = subprocess.run(
        [sys.executable, "-c", SCRIPT], capture_output=True, text=True, env=environment,
        check=True, cwd=tmp_path,
    )
    loaded = done.stdout.split()
    assert "xrdroot.pycxx" in loaded
    beyond = [name for name in loaded if name != "xrdroot" and not name.startswith(ALLOWED)]
    assert beyond == []
