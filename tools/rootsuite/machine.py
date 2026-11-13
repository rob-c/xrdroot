"""The machine ROOT's suite is configured for: macOS on ARM64, as the oracle ROOT was built.

ROOT's CMake decides which tests exist from two things: the build options
(``if(cuda)``, ``ENABLE_IF imt``) and the platform (``if(APPLE)``,
``CMAKE_SYSTEM_PROCESSOR MATCHES arm64``). Both are taken here from the Mac
xrdroot is measured on. The options are the oracle's own - ``root-config
--features`` of the ROOT at ``$XRDROOT_ORACLE_PREFIX``, else at
``~/.local/root-oracle``, found as :mod:`tools.tutorials.environment` finds
it - and, without an oracle, :data:`MACOS_ARM64_FEATURES`, what ROOT 6.40.04
from conda-forge reports on macOS ARM64. A test ROOT's CMake leaves out on
such a Mac is never registered, so it is neither run nor counted: only what
real ROOT runs there is the target.

Nothing here imports xrdroot or ROOT; ``root-config`` is a shell script.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterable
from pathlib import Path

from tools.tutorials.environment import DEFAULT_PREFIX

__all__ = [
    "MACOS_ARM64_FEATURES", "SUITE_FEATURES", "NOT_ON_MACOS", "DEFAULT_FEATURES",
    "PLATFORM", "oracle_config", "oracle_features", "default_features", "for_suite",
]  # fmt: skip

#: ``root-config --features`` of ROOT 6.40.04 from conda-forge on macOS ARM64.
MACOS_ARM64_FEATURES = frozenset(
    """asimage asimage_tiff builtin_civetweb builtin_cling builtin_openui5 check_connection clad
    cocoa curl cxx20 dataframe davix fftw3 fitsio gdml geom gviz http imt libcxx mathmore opengl
    pyroot pythia8 roofit roofit_multiprocess root7 runtime_cxxmodules shared soversion spectrum
    sqlite ssl thisroot_scripts tmva tmva-cpu tmva-cudnn tmva-pymva tpython vdt webgui xml
    xrootd""".split()
)

#: What a configure that builds the tests turns on beside an installation's options.
SUITE_FEATURES = frozenset({"testing", "roottest"})

#: Options no macOS ARM64 build of ROOT has, whatever a feature list claims: X11, io_uring,
#: CUDA, SYCL and Vc are Linux's or another processor's.
NOT_ON_MACOS = frozenset({"x11", "uring", "cuda", "sycl", "vc", "experimental_adaptivecpp"})

#: The options assumed when no oracle answers.
DEFAULT_FEATURES = MACOS_ARM64_FEATURES | SUITE_FEATURES

#: The CMake variables a configure on macOS ARM64 sets that test directories look at.
#: ``CMAKE_CXX_COMPILER_ID`` is conda-forge's clang, which built the oracle; Apple's own
#: compiler would say ``AppleClang``.
PLATFORM = {
    "APPLE": "1", "UNIX": "1", "CMAKE_HOST_APPLE": "1", "CMAKE_HOST_UNIX": "1",
    "CMAKE_SYSTEM_NAME": "Darwin", "CMAKE_HOST_SYSTEM_NAME": "Darwin",
    "CMAKE_SYSTEM_PROCESSOR": "arm64", "CMAKE_HOST_SYSTEM_PROCESSOR": "arm64",
    "ROOT_ARCHITECTURE": "macosxarm64", "ROOT_PLATFORM": "macosx",
    "CMAKE_CXX_COMPILER_ID": "Clang", "CMAKE_C_COMPILER_ID": "Clang",
    "CMAKE_SHARED_LIBRARY_SUFFIX": ".dylib", "CMAKE_SHARED_LIBRARY_PREFIX": "lib",
}  # fmt: skip


def oracle_config(prefix: Path | None = None) -> Path | None:
    """The oracle's ``root-config``: under ``prefix``, else the default prefix, if it is there."""
    for candidate in (prefix, DEFAULT_PREFIX):
        if candidate is not None:
            config = Path(candidate).expanduser() / "bin" / "root-config"
            if config.is_file():
                return config
    return None


def oracle_features(config: Path | None) -> frozenset[str] | None:
    """``root-config --features`` of the oracle, or ``None`` when it does not answer."""
    if config is None:
        return None
    try:
        done = subprocess.run([str(config), "--features"], capture_output=True, text=True,
                              timeout=60, env=dict(os.environ))
    except (OSError, subprocess.TimeoutExpired):
        return None
    features = frozenset(done.stdout.split())
    return features if done.returncode == 0 and features else None


def for_suite(features: Iterable[str]) -> frozenset[str]:
    """An installation's options as a test configure on this Mac sees them."""
    return (frozenset(features) - NOT_ON_MACOS) | SUITE_FEATURES


def default_features(prefix: Path | None = None) -> frozenset[str]:
    """The oracle's options when there is an oracle, else the committed macOS ARM64 ones."""
    found = oracle_features(oracle_config(prefix))
    return for_suite(found) if found is not None else DEFAULT_FEATURES
