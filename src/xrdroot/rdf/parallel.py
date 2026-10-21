"""ROOT's configured implicit pool is bounded before any event work starts."""

from __future__ import annotations

import os
import re
import sys

#: strtol(base0) consumes a signed integer prefix, including C octal spelling.
_INTEGER_PREFIX = re.compile(r"[ \t\n\v\f\r]*([+-]?(?:0[xX][0-9a-fA-F]+|0[0-7]*|[1-9][0-9]*))")


def _integer_prefix(value: str) -> int | None:
    match = _INTEGER_PREFIX.match(value)
    if match is None:
        return None
    token = match[1]
    digits = token.lstrip("+-")
    base = 16 if digits.lower().startswith("0x") else 8 if digits.startswith("0") else 10
    return int(token, base)


def _cap_error(parsed: int | None) -> None:
    namespace = sys.modules.get("xrdroot.pyroot")
    ignore = getattr(namespace, "__dict__", {}).get("gErrorIgnoreLevel", -1)
    if int(ignore) > 3000:  # ROOT's kError; importing its namespace would invert the layers.
        return
    reason = ("cannot parse number in environment variable ROOT_MAX_THREADS" if parsed is None
              else f"environment variable ROOT_MAX_THREADS must be >= 1, but set to {parsed}")
    print(f"Error in <ROOT::Internal::LogicalCPUBandwidthControl()>: {reason}; ignoring.",
          file=sys.stderr)


def _configured_cap() -> int | None:
    value = os.environ.get("ROOT_MAX_THREADS")
    if value is None:
        return None
    parsed = _integer_prefix(value)
    if parsed is not None and parsed >= 1:
        return parsed
    _cap_error(parsed)
    return None


def configured_workers(requested: int) -> int:
    """A request is a hint bounded by hardware and ROOT's environment cap.

    Invalid caps are diagnosed and ignored, as ROOT's task arena does. This
    process pool has no external TBB arena or scheduler to attach to.
    """
    hardware = os.cpu_count() or 1
    cap = _configured_cap()
    wanted = int(requested)
    return min(wanted if wanted > 0 else hardware, hardware, cap or hardware)
