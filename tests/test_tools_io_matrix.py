"""The tree round-trip tool is the I/O reference point, so it needs a guard in the suite.

The oracle is a real ROOT installation; where there is none the test skips, and says so,
rather than passing on nothing.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from tools.io_matrix import TREE_COLUMNS, TREE_ENTRIES, tree_round_trip

#: Where the project installs ROOT 6.40.04 for comparison (see AGENTS.md, "Setup").
ORACLE = Path.home() / ".local" / "root-oracle"

needs_root = pytest.mark.skipif(
    not (ORACLE / "bin" / "python").exists(),
    reason=f"no ROOT oracle at {ORACLE}: the tree round-trip is checked by ROOT's own reader",
)


@needs_root
def test_a_tree_of_every_leaf_shape_survives_root_cloning_it(tmp_path):
    """ROOT opens our tree, copies it with ``CloneTree``, and the copy still holds our values."""
    finding = tree_round_trip(str(ORACLE), tmp_path)
    assert finding["root_read_ok"], finding["root_why"]
    assert finding["entries"] == len(TREE_ENTRIES)
    assert set(TREE_COLUMNS) <= set(finding["branches"])
    assert finding["values_match"], finding["read_back"]


@needs_root
def test_the_written_values_are_what_xrdroot_reads_back_before_root_sees_them(tmp_path):
    """The tool's own reading of its file, so a change in the writer fails here first."""
    from tools.io_matrix import _same, _tree_read, _tree_written

    written = tmp_path / "written.root"
    _tree_written(written)
    got = _tree_read(written)
    assert len(got) == len(TREE_ENTRIES)
    assert all(_same(row, want) for row, want in zip(got, TREE_ENTRIES, strict=True))


@needs_root
def test_root_writes_a_tree_of_every_leaf_shape_and_xrdroot_reads_it(tmp_path):
    """The other direction is the harder one: ROOT picks the layouts, and we must decode them."""
    finding = tree_round_trip(str(ORACLE), tmp_path)
    assert finding["root_wrote_ok"], finding["root_wrote_why"]
    assert finding["reverse_values_match"], finding["reverse_read_back"]


def test_the_tools_module_is_runnable_without_a_root_installation_installed():
    """Importing the tool must not need ROOT: `--classes` works on xrdroot alone."""
    assert os.path.exists(Path(__file__).resolve().parents[1] / "tools" / "io_matrix.py")


def test_a_dump_line_with_roots_own_print_around_it_is_still_read():
    """ROOT prints ``Info in <...>`` to the dump's stream, so a line can hold both."""
    from tools.harvest_infos import _json_in

    assert _json_in('Info in <TFile::TFile>: {"class": "TFoam"}') == {"class": "TFoam"}
    assert _json_in('{"class": "TFoam"}') == {"class": "TFoam"}
    assert _json_in("no object here") is None
