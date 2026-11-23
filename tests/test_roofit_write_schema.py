"""Writer descriptions can be regenerated from the committed native donor."""

from pathlib import Path

from tools import roofit_write_schema

from xrdroot.roofit.io import layouts

DATA = Path(__file__).parent / "data/roofit-write-workspace-6.40.root"


def test_the_committed_writer_layouts_are_exactly_the_native_donors_descriptions(tmp_path):
    output = tmp_path / "layouts.py"
    assert roofit_write_schema.main([str(DATA), str(output)]) == 0
    assert output.read_bytes() == Path(layouts.__file__).read_bytes()
    descriptors, entries = roofit_write_schema.harvest(DATA)
    assert descriptors == layouts.LAYOUTS
    assert entries == layouts.descriptions()
