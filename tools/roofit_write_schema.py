"""Harvest workspace writer layouts from a ROOT-written donor without importing ROOT."""

from __future__ import annotations

import argparse
import base64
import json
import pprint
import textwrap
import zlib
from pathlib import Path
from typing import Any

from xrdroot import open_root
from xrdroot.merging.infos import read_entries


def harvest(filename: Path) -> tuple[dict[str, Any], dict[tuple[str, int], bytes]]:
    """Native descriptors retain their versions and declaration order."""
    with open_root(filename) as source:
        entries = read_entries(source._source)
        infos = source._source.streamers()
    layouts = {
        name: (version, tuple((member.name, member.stype, member.typename, member.length)
                              for member in infos[name].values()))
        for name, version in entries
    }
    layouts.update({"RooRefArray": (1, ()), "RooWorkspace::CodeRepo": (2, ())})
    return layouts, entries


def encoded(entries: dict[tuple[str, int], bytes]) -> str:
    rows = [[name, version, base64.b85encode(raw).decode()]
            for (name, version), raw in entries.items()]
    raw = json.dumps(rows, separators=(",", ":")).encode()
    return base64.b85encode(zlib.compress(raw)).decode()


def render(filename: Path) -> str:
    layouts, entries = harvest(filename)
    data = pprint.pformat(layouts, width=100, sort_dicts=False)
    blob = "".join("    " + repr(line) + "\n" for line in
                   textwrap.wrap(encoded(entries), 90, break_on_hyphens=False))
    return f'''"""ROOT 6.40.04 workspace layouts and relocatable streamer descriptions.

Harvested from a native workspace by ``tools/roofit_write_schema.py``.
No ROOT installation is needed to use these descriptions. Each member tuple
is its name, ROOT streamer type, C++ type and fixed array length.
"""

from __future__ import annotations

import base64
import json
import zlib
from functools import cache

#: Class versions and their persisted members, in declaration order.
LAYOUTS: dict[str, tuple[int, tuple[tuple[str, int, str, int], ...]]] = {data}

#: Native streamer entries, compressed together; class references are relocated.
ENTRIES = (
{blob})


@cache
def descriptions() -> dict[tuple[str, int], bytes]:
    """Self-contained ROOT class descriptions may be carried into any output file."""
    rows = json.loads(zlib.decompress(base64.b85decode(ENTRIES)))
    return {{(name, version): base64.b85decode(raw) for name, version, raw in rows}}
'''


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("donor", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    args.output.write_text(render(args.donor))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
