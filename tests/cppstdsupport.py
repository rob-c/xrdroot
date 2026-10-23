"""Cold child processes verify the C++ component imports only its declared dependencies."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

BOUNDARY = """
import sys

class Boundary:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('xrdclient'):
            raise AssertionError(fullname)
        if fullname.startswith('xrdroot.') and not fullname.startswith('xrdroot.cppstd'):
            raise AssertionError(fullname)

sys.meta_path.insert(0, Boundary())
"""


def isolated(program: str) -> None:
    """Run real generic behavior while rejecting imports outside the C++ component."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run(
        [sys.executable, "-c", BOUNDARY + program], env=env,
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
