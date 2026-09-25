"""The ``xrdroot`` command: one subcommand per module, each ``add_parser`` and ``run``.

A PLACEHOLDER, kept as small as it can be: the shell and CLI work owns this
package, and its own ``__init__`` replaces this one when the two are put
together. All that is relied on here is its contract - :data:`COMMANDS`
names the subcommands, and each is the module ``xrdroot.cli.<name>`` with
``add_parser(subparsers)`` and ``run(args) -> int`` - so that ``merge`` and
``cp`` can be tested through the parser they will be run through.
"""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Sequence

__all__ = ["COMMANDS", "main"]

#: The subcommands, each the module of that name in this package.
COMMANDS = ["merge", "cp"]


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv`` and run the subcommand it names; what that returns is the exit status."""
    parser = argparse.ArgumentParser(prog="xrdroot")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in COMMANDS:
        importlib.import_module(f"{__name__}.{name}").add_parser(subparsers)
    args = parser.parse_args(argv)
    status: int = importlib.import_module(f"{__name__}.{args.command}").run(args)
    return status
