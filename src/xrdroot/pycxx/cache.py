"""Translations kept on disk, so a program run again is not translated again.

A translation is filed under the hash of everything it depends on: the
program's text and path, the text of every local header it read, the
translator's own source, and the dialect it was translated in with the
source of the code that defines it - so a changed program, a changed header
or a new version of the translator each translate afresh, and nothing stale
is ever run. The directory is ``$XRDROOT_CINT_CACHE``, or
``~/.cache/xrdroot/cint``. A cache that cannot be written is simply not used.
"""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

from .dialect import Dialect, active
from .translation import Translation

__all__ = ["directory", "key", "load", "save"]

#: The environment variable naming the cache directory.
ENVIRONMENT = "XRDROOT_CINT_CACHE"


def directory() -> Path:
    """Where translations are kept."""
    chosen = os.environ.get(ENVIRONMENT)
    if chosen:
        return Path(chosen)
    return Path.home() / ".cache" / "xrdroot" / "cint"


@lru_cache(maxsize=8)
def _translator_digest(sources: tuple[Path, ...] = ()) -> str:
    """A hash of the translator's own code, and of the code defining a dialect: a new
    translator never runs an old translation."""
    digest = hashlib.sha256()
    for here in (Path(__file__).parent, *sources):
        for path in sorted(here.glob("*.py")):
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _hash(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8", "surrogatepass"))
        digest.update(b"\0")
    return digest.hexdigest()


def key(source: str, file: str, dialect: Dialect | None = None) -> str:
    """The name a translation of ``source`` from ``file`` in ``dialect`` is filed under."""
    chosen = dialect or active()
    return _hash(_translator_digest(chosen.sources), chosen.name, file, source)


def _included_hash(paths: list[str]) -> str:
    texts = []
    for path in paths:
        try:
            texts.append(Path(path).read_text(encoding="utf-8", errors="replace"))
        except OSError:
            texts.append("")
    return _hash(*paths, *texts)


def load(source: str, file: str, dialect: Dialect | None = None) -> Translation | None:
    """The translation kept for ``source``, if there is one and its headers are unchanged."""
    where = directory() / f"{key(source, file, dialect)}.json"
    try:
        kept = json.loads(where.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if kept.get("included_hash") != _included_hash(kept.get("included", [])):
        return None
    source_map = {int(line): (place[0], place[1]) for line, place in kept["source_map"].items()}
    return Translation(
        kept["python"],
        file,
        source_map,
        kept["entry"],
        kept["included"],
        kept["unnamed"],
        kept["returns"],
    )


def save(
    source: str, file: str, translation: Translation, dialect: Dialect | None = None
) -> None:
    """Keep ``translation`` for the next time ``source`` is run; quietly not, if it cannot be."""
    kept = {
        "python": translation.python,
        "source_map": {str(line): list(place) for line, place in translation.source_map.items()},
        "entry": translation.entry,
        "included": translation.included,
        "unnamed": translation.unnamed,
        "returns": translation.returns,
        "included_hash": _included_hash(translation.included),
    }
    where = directory()
    try:
        where.mkdir(parents=True, exist_ok=True)
        filed = key(source, file, dialect)
        temporary = where / f".{filed}.{os.getpid()}.tmp"
        temporary.write_text(json.dumps(kept), encoding="utf-8")
        temporary.replace(where / f"{filed}.json")
    except OSError:
        return
