"""The harness's results as JSON to track, Markdown to read, and HTML to browse.

The JSON is the record: a ``schema`` string that changes only when its shape
does, the settings and versions the run was made with, counts overall and
per area, the failure reasons ranked, and one entry per tutorial. Markdown
and HTML are drawn from it alone, so ``report`` can redraw them from an old
run's ``results.json``.

The ranking is the point of the exercise: every FAIL and UNSUPPORTED
tutorial is blocked by its first failure, so a reason's count is how many
tutorials fixing it would move past that point - what to build next.
"""

from __future__ import annotations

import html
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .classify import STATUSES

__all__ = ["SCHEMA", "document", "ranked", "markdown", "page", "write"]

#: The results' shape; a consumer tracking trends checks this before reading on.
SCHEMA = "xrdroot-tutorials/1"

#: The statuses a ranked reason is drawn from: xrdroot's own failures.
BLOCKING = ("FAIL", "UNSUPPORTED")


def counts(records: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    found = Counter(record["status"] for record in records)
    return {status: found.get(status, 0) for status in STATUSES}


def areas(records: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["area"]].append(record)
    return {area: counts(grouped[area]) for area in sorted(grouped)}


def ranked(records: Iterable[Mapping[str, Any]], statuses: Sequence[str]) -> list[dict[str, Any]]:
    """Reasons among records of these statuses, most tutorials first, then by name."""
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if record["status"] in statuses:
            grouped[record["reason"]].append(record)
    rows = [
        {
            "reason": reason,
            "count": len(members),
            "statuses": dict(Counter(member["status"] for member in members)),
            "tutorials": sorted(member["path"] for member in members),
        }
        for reason, members in grouped.items()
    ]
    return sorted(rows, key=lambda row: (-row["count"], row["reason"]))


def document(records: Sequence[Mapping[str, Any]], meta: Mapping[str, Any]) -> dict[str, Any]:
    """The whole run, in the stable schema."""
    return {
        "schema": SCHEMA,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "meta": dict(meta),
        "counts": counts(records),
        "areas": areas(records),
        "reasons": ranked(records, BLOCKING),
        "diffs": ranked(records, ("DIFF",)),
        "skips": ranked(records, ("SKIP",)),
        "oracle_failures": ranked(records, ("ORACLE-FAIL",)),
        "tutorials": sorted((dict(record) for record in records), key=lambda r: r["path"]),
    }


# --- Markdown --------------------------------------------------------------


def _cell(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_cell(cell) for cell in row) + " |" for row in rows]
    return lines


def _examples(paths: Sequence[str], shown: int = 3) -> str:
    more = f" (+{len(paths) - shown} more)" if len(paths) > shown else ""
    return ", ".join(paths[:shown]) + more


def _reason_rows(rows: Sequence[Mapping[str, Any]], top: int) -> list[list[Any]]:
    return [
        [index, row["reason"], row["count"], _examples(row["tutorials"])]
        for index, row in enumerate(rows[:top], start=1)
    ]


def markdown(doc: Mapping[str, Any], top: int = 30) -> str:
    """A summary to read: totals, per area, then the reasons, ranked."""
    meta, total = doc["meta"], sum(doc["counts"].values())
    lines = [
        "# ROOT tutorials on xrdroot",
        "",
        f"Generated {doc['generated']} against ROOT {meta.get('root_version') or '(none)'}; "
        f"xrdroot {meta.get('xrdroot_commit') or '?'}; {total} tutorials.",
        "",
    ]
    lines += _table(STATUSES, [[doc["counts"][status] for status in STATUSES]])
    lines += ["", "## By area", ""]
    lines += _table(["area", *STATUSES], [[a, *c.values()] for a, c in doc["areas"].items()])
    sections = (
        ("Top failure reasons (tutorials each would unblock)", "reasons"),
        ("Output differences", "diffs"),
        ("Why ROOT itself failed", "oracle_failures"),
        ("Why tutorials were skipped", "skips"),
    )
    for title, key in sections:
        lines += ["", f"## {title}", ""]
        lines += _table(["#", "reason", "tutorials", "e.g."], _reason_rows(doc[key], top))
    return "\n".join(lines) + "\n"


# --- HTML ------------------------------------------------------------------

_STYLE = """
:root { --bg:#fbfbfa; --fg:#1d1d1b; --muted:#6b6b66; --line:#e3e2dd; --card:#ffffff;
  --PASS:#2f7d4f; --DIFF:#b7791f; --FAIL:#b83232; --UNSUPPORTED:#7c4dbd; --SKIP:#8a8a84;
  --ORACLE-FAIL:#35659e; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#171716;
  --fg:#ecebe6; --muted:#a3a29c; --line:#34332f; --card:#1f1f1d; } }
:root[data-theme="dark"] { --bg:#171716; --fg:#ecebe6; --muted:#a3a29c; --line:#34332f;
  --card:#1f1f1d; }
body { background:var(--bg); color:var(--fg); font:14px/1.45 system-ui, sans-serif;
  margin:0 auto; max-width:1100px; padding:24px 16px; }
h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:16px; margin:28px 0 8px; }
.muted { color:var(--muted); } .tiles { display:flex; flex-wrap:wrap; gap:8px; margin:16px 0; }
.tile { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:8px 14px;
  min-width:90px; } .tile b { display:block; font-size:22px; }
.wrap { overflow-x:auto; } table { border-collapse:collapse; width:100%; }
th, td { text-align:left; padding:4px 8px; border-bottom:1px solid var(--line);
  vertical-align:top; } th { color:var(--muted); font-weight:600; }
.status { font-weight:700; } td.num { text-align:right; font-variant-numeric:tabular-nums; }
input { font:inherit; padding:6px 8px; width:100%; max-width:420px; box-sizing:border-box;
  background:var(--card); color:var(--fg); border:1px solid var(--line); border-radius:6px; }
details summary { cursor:pointer; } pre { white-space:pre-wrap; font-size:12px; margin:4px 0; }
"""

_SCRIPT = """
const box = document.getElementById('filter');
box.addEventListener('input', () => {
  const q = box.value.toLowerCase();
  document.querySelectorAll('#tutorials tbody tr').forEach(tr => {
    tr.style.display = tr.textContent.toLowerCase().includes(q) ? '' : 'none';
  });
});
"""


def _e(text: Any) -> str:
    return html.escape(str(text))


def _status(status: str) -> str:
    return f'<span class="status" style="color:var(--{status})">{_e(status)}</span>'


def _html_table(header: Sequence[str], rows: Iterable[Sequence[str]], ident: str = "") -> str:
    head = "".join(f"<th>{_e(h)}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    attr = f' id="{ident}"' if ident else ""
    table = f"<table{attr}><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
    return f'<div class="wrap">{table}</div>'


def _reasons_html(rows: Sequence[Mapping[str, Any]], top: int) -> str:
    return _html_table(
        ["#", "reason", "tutorials", "which"],
        (
            [str(i), _e(row["reason"]), str(row["count"]), _e(_examples(row["tutorials"], 5))]
            for i, row in enumerate(rows[:top], start=1)
        ),
    )


def _detail(record: Mapping[str, Any]) -> str:
    details = [line for line in record.get("details", []) if line]
    if not details:
        return _e(record.get("reason", ""))
    body = "".join(f"<pre>{_e(line)}</pre>" for line in details[:8])
    return f"<details><summary>{_e(record.get('reason', ''))}</summary>{body}</details>"


def _runtime(side: Any) -> str:
    return f"{side['runtime']:.1f}" if side else ""


def page(doc: Mapping[str, Any], top: int = 40) -> str:
    """A self-contained page: tiles, the ranked reasons, and every tutorial, filterable."""
    meta = doc["meta"]
    tiles = "".join(
        f'<div class="tile"><span style="color:var(--{s})">{_e(s)}</span><b>{n}</b></div>'
        for s, n in doc["counts"].items()
    )
    area_rows = ([_e(a), *(str(n) for n in c.values())] for a, c in doc["areas"].items())
    tutorial_rows = (
        [_e(r["path"]), _status(r["status"]), _detail(r), _runtime(r.get("oracle")),
         _runtime(r.get("xrdroot"))]
        for r in doc["tutorials"]
    )  # fmt: skip
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tutorial conformance</title><style>{_STYLE}</style></head><body>
<h1>ROOT tutorials on xrdroot</h1>
<div class="muted">{_e(doc["generated"])} · ROOT {_e(meta.get("root_version") or "none")} ·
xrdroot {_e(meta.get("xrdroot_commit") or "?")} · {_e(meta.get("tutorials_dir", ""))}</div>
<div class="tiles">{tiles}</div>
<h2>Top failure reasons</h2>{_reasons_html(doc["reasons"], top)}
<h2>Output differences</h2>{_reasons_html(doc["diffs"], top)}
<h2>By area</h2>{_html_table(["area", *STATUSES], area_rows)}
<h2>ROOT itself failed</h2>{_reasons_html(doc["oracle_failures"], top)}
<h2>Skipped</h2>{_reasons_html(doc["skips"], top)}
<h2>Every tutorial</h2><input id="filter" placeholder="filter: path, status, reason">
{_html_table(["tutorial", "status", "reason", "ROOT s", "xrdroot s"], tutorial_rows, "tutorials")}
<script>{_SCRIPT}</script></body></html>
"""


def write(doc: Mapping[str, Any], directory: Path) -> list[Path]:
    """``results.json``, ``summary.md`` and ``report.html`` in the directory."""
    directory.mkdir(parents=True, exist_ok=True)
    written = {
        "results.json": json.dumps(doc, indent=1, sort_keys=False) + "\n",
        "summary.md": markdown(doc),
        "report.html": page(doc),
    }
    for name, text in written.items():
        (directory / name).write_text(text, encoding="utf-8")
    return [directory / name for name in written]
