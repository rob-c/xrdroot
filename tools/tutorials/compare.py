"""Whether xrdroot's run of a tutorial says and writes what ROOT's did.

Standard output is compared after both sides are normalised: addresses,
dates, times, and the paths of the two runs' directories say nothing about
the tutorial, lines reporting how long something took differ run to run, and
``Processing x.C...`` is the driver's line, not the tutorial's. What is left
is compared line by line, text exactly (runs of blanks as one) and numbers
within a relative tolerance, and the first line that differs is reported.

Files are compared by kind: ROOT files key by key and value by value with
``xrdroot diff``'s own comparison, in process; images by a similarity score
against a threshold; text files as standard output is; anything else by
checksum.
"""

from __future__ import annotations

import importlib
import math
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, NamedTuple

__all__ = [
    "Tolerance",
    "normalise",
    "first_difference",
    "compare_streams",
    "compare_root_files",
    "image_similarity",
    "compare_files",
    "ImageScore",
]


@dataclass(frozen=True)
class Tolerance:
    """How near two numbers are to be equal, and how near two images."""

    rtol: float = 1e-6
    atol: float = 1e-12
    image_threshold: float = 0.95


#: Rewrites applied to every line before comparison, in order.
REWRITES = [
    (re.compile(r"\x1b\[[0-9;]*[A-Za-z]"), ""),
    (re.compile(r"\b0x[0-9a-fA-F]+\b"), "0x<addr>"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2}(\.\d+)?)?)?\b"), "<date>"),
    (
        re.compile(r"\b(Mon|Tue|Wed|Thu|Fri|Sat|Sun) (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|"
                   r"Nov|Dec) +\d+ \d{2}:\d{2}:\d{2} \d{4}\b"),
        "<date>",
    ),
    (re.compile(r"\b\d{1,2}:\d{2}:\d{2}\b"), "<time>"),
    # TStopwatch's times as the fitting tutorials print them: "RT=  0.902 s, Cpu=  0.460 s".
    (re.compile(r"\bRT= *\d+\.\d+ s, Cpu= *\d+\.\d+ s\b"), "RT=<time>, Cpu=<time>"),
    (re.compile(r"(/private)?/(var/folders|tmp)/\S*?tutorial-[^/\s]+"), "<workdir>"),
    (re.compile(r"[ \t]+"), " "),
]  # fmt: skip

#: Lines that are about the run rather than the tutorial, dropped from both sides.
DROPPED = re.compile(
    r"(?i)^\s*(processing .*\.\.\.$|info in <tunixsystem::aclic|.*\b(real|cpu|cp) ?time\b"
    r"|.*\belapsed\b|.*\btook \d|.*\bruntime\b.*\d)"
)

#: A number: integer, decimal or scientific, NaN or infinity.
NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?|nan|inf)")


def normalise(text: str, paths: Mapping[str, str] = ()) -> list[str]:  # type: ignore[assignment]
    """The lines worth comparing: rewritten, stripped, the run's own lines dropped.

    ``paths`` maps directories that differ between runs (the working directory,
    the tutorials) to the placeholder each becomes.
    """
    lines = []
    for line in text.splitlines():
        for path, placeholder in sorted(dict(paths).items(), key=lambda item: -len(item[0])):
            if path:
                line = line.replace(path, placeholder)
        for pattern, replacement in REWRITES:
            line = pattern.sub(replacement, line)
        line = line.strip()
        if line and not DROPPED.match(line):
            lines.append(line)
    return lines


def _close(one: str, two: str, tolerance: Tolerance) -> bool:
    try:
        a, b = float(one), float(two)
    except ValueError:
        return one == two
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return math.isclose(a, b, rel_tol=tolerance.rtol, abs_tol=tolerance.atol)


def same_line(one: str, two: str, tolerance: Tolerance) -> bool:
    """Equal text between the numbers, and numbers equal within the tolerance."""
    if one == two:
        return True
    if NUMBER.split(one) != NUMBER.split(two):
        return False
    first, second = NUMBER.findall(one), NUMBER.findall(two)
    return len(first) == len(second) and all(_close(a, b, tolerance) for a, b in zip(first, second))


def first_difference(
    expected: Sequence[str], actual: Sequence[str], tolerance: Tolerance
) -> str | None:
    """The first line that differs, said with its number and both sides; None if none."""
    for number, (one, two) in enumerate(zip(expected, actual), start=1):
        if not same_line(one, two, tolerance):
            return f"line {number}: ROOT {one!r}, xrdroot {two!r}"
    if len(expected) != len(actual):
        shorter = min(len(expected), len(actual))
        side, extra = ("ROOT", expected) if len(expected) > shorter else ("xrdroot", actual)
        return (
            f"line {shorter + 1}: only {side} goes on, with {extra[shorter]!r} "
            f"({len(expected)} lines and {len(actual)})"
        )
    return None


def compare_streams(
    expected: str,
    actual: str,
    paths: tuple[Mapping[str, str], Mapping[str, str]],
    tolerance: Tolerance,
) -> str | None:
    """The first difference between two runs' output, each normalised by its own paths."""
    return first_difference(normalise(expected, paths[0]), normalise(actual, paths[1]), tolerance)


# --- ROOT files ------------------------------------------------------------


def compare_root_files(expected: Path, actual: Path, tolerance: Tolerance) -> list[str]:
    """Every difference ``xrdroot diff`` finds between ROOT's file and xrdroot's."""
    try:
        from xrdroot import open_root
        from xrdroot.cli.diff import Tolerance as DiffTolerance
        from xrdroot.cli.diff import compare
    except ImportError as why:
        return [f"cannot compare ROOT files: {why}"]
    names = DiffTolerance(tolerance.atol, tolerance.rtol, "ROOT's", "xrdroot's")
    try:
        with open_root(str(expected)) as one, open_root(str(actual)) as two:
            return list(compare(one, two, names))
    except Exception as why:  # a reader failing is itself the finding
        return [f"cannot read: {type(why).__name__}: {why}"]


# --- images ----------------------------------------------------------------


class ImageScore(NamedTuple):
    """How alike two pictures are, from 0 to 1, and how that was measured."""

    score: float
    method: str
    note: str = ""


def _external_comparer() -> Callable[..., Any] | None:
    """``xrdroot.pyroot.graphics.compare_images``, when the graphics layer has one."""
    try:
        module = importlib.import_module("xrdroot.pyroot.graphics")
    except Exception:
        return None
    found = getattr(module, "compare_images", None)
    return found if callable(found) else None


def _grey(path: Path) -> Any:
    import numpy as np
    from matplotlib import image

    pixels = np.asarray(image.imread(str(path)), dtype=float)
    if pixels.max(initial=0.0) > 1.0:
        pixels = pixels / 255.0
    if pixels.ndim == 3:
        pixels = pixels[..., :3].mean(axis=2)
    return pixels


#: The side of the grid pictures are pooled to before they are compared.
POOL = 64


def _pooled(pixels: Any, grid: tuple[int, int]) -> Any:
    """The picture averaged down to ``grid`` blocks, whatever its shape."""
    import numpy as np

    rows = np.array_split(np.arange(pixels.shape[0]), grid[0])
    cols = np.array_split(np.arange(pixels.shape[1]), grid[1])
    return np.array([[pixels[np.ix_(r, c)].mean() for c in cols] for r in rows])


def _fallback_similarity(expected: Path, actual: Path) -> ImageScore:
    """One minus the mean difference of the two pictures, in grey, pooled to one grid.

    Pooling makes pictures of different sizes comparable and forgives a line
    drawn a pixel over; the size mismatch itself is said in the note.
    """
    import numpy as np

    one, two = _grey(expected), _grey(actual)
    note = "" if one.shape == two.shape else f"sizes {one.shape[::-1]} and {two.shape[::-1]}"
    grid = (min(POOL, one.shape[0], two.shape[0]), min(POOL, one.shape[1], two.shape[1]))
    score = 1.0 - float(np.abs(_pooled(one, grid) - _pooled(two, grid)).mean())
    return ImageScore(round(score, 4), "numpy-pooled-grey", note)


def _size_note(expected: Path, actual: Path) -> str:
    """What to say about two pictures' sizes: nothing when they are the same."""
    one, two = _grey(expected), _grey(actual)
    return "" if one.shape == two.shape else f"sizes {one.shape[::-1]} and {two.shape[::-1]}"


def image_similarity(expected: Path, actual: Path) -> ImageScore:
    """The graphics layer's comparison if it has one, else the NumPy fallback.

    Both pictures are decoded here first whichever scores them, so a picture
    that is not one, or two of different sizes, is said the same way either
    way - the graphics layer's score resizes silently, and a canvas saved at
    the wrong size is a difference worth reporting in its own right.
    """
    try:
        note = _size_note(expected, actual)
    except Exception as why:
        return ImageScore(0.0, "numpy-pooled-grey", f"cannot decode: {why}")
    comparer = _external_comparer()
    if comparer is None:
        return _fallback_similarity(expected, actual)
    method = "xrdroot.pyroot.graphics.compare_images"
    try:
        found = comparer(str(expected), str(actual))
    except Exception as why:
        return ImageScore(0.0, method, f"failed: {why}")
    score = found[0] if isinstance(found, tuple) else getattr(found, "score", found)
    return ImageScore(round(float(score), 4), method, note)


# --- every output ----------------------------------------------------------

#: The extensions compared as pictures.
IMAGES = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"}
#: The extensions compared as text, like standard output.
TEXT = {".txt", ".csv", ".dat", ".json", ".xml", ".tex", ".C", ".cxx", ".py", ".log", ".out",
        ".html", ".svg", ".eps", ".ps", ".md", ".h"}  # fmt: skip
#: Vector pictures, whose text holds creation dates and object addresses: only there, or not.
PRESENCE_ONLY = {".pdf", ".svg", ".eps", ".ps"}


class FileComparison(NamedTuple):
    """One output compared: its name, the kind of comparison, what differed, any score."""

    name: str
    kind: str
    differences: list[str]
    score: float | None = None


def _text_file(expected: Path, actual: Path, tolerance: Tolerance) -> list[str]:
    read = [path.read_text(encoding="utf-8", errors="replace") for path in (expected, actual)]
    found = first_difference(normalise(read[0]), normalise(read[1]), tolerance)
    return [] if found is None else [found]


def _kind(name: str) -> str:
    suffix = PurePosixPath(name).suffix
    if suffix == ".root":
        return "root"
    if suffix.lower() in IMAGES:
        return "image"
    if suffix in PRESENCE_ONLY:
        return "presence"
    return "text" if suffix in TEXT else "bytes"


def compare_file(
    name: str, expected: Path, actual: Path, sums: tuple[str, str], tolerance: Tolerance
) -> FileComparison:
    """One output both runs made, compared the way its kind is."""
    kind = _kind(name)
    if sums[0] == sums[1] or kind == "presence":
        return FileComparison(name, kind, [])
    if not (expected.is_file() and actual.is_file()):
        return FileComparison(name, kind, [f"{name}: too large to keep, and checksums differ"])
    return _BY_KIND[kind](name, expected, actual, tolerance)


def _root_file(name: str, expected: Path, actual: Path, tolerance: Tolerance) -> FileComparison:
    found = compare_root_files(expected, actual, tolerance)
    return FileComparison(name, "root", [f"{name}: {line}" for line in found[:20]])


def _image(name: str, expected: Path, actual: Path, tolerance: Tolerance) -> FileComparison:
    score = image_similarity(expected, actual)
    low = score.score < tolerance.image_threshold
    said = f"{name}: similarity {score.score} < {tolerance.image_threshold}"
    note = f" ({score.note})" if score.note else ""
    return FileComparison(name, "image", [said + note] if low else [], score.score)


def _text(name: str, expected: Path, actual: Path, tolerance: Tolerance) -> FileComparison:
    found = _text_file(expected, actual, tolerance)
    return FileComparison(name, "text", [f"{name}: {line}" for line in found])


def _bytes(name: str, expected: Path, actual: Path, tolerance: Tolerance) -> FileComparison:
    return FileComparison(name, "bytes", [f"{name}: contents differ"])


#: How each kind of output with differing checksums is compared.
_BY_KIND: dict[str, Callable[[str, Path, Path, Tolerance], FileComparison]] = {
    "root": _root_file,
    "image": _image,
    "text": _text,
    "bytes": _bytes,
}


def compare_files(
    expected: Mapping[str, str],
    actual: Mapping[str, str],
    dirs: tuple[Path, Path],
    tolerance: Tolerance,
) -> Iterator[FileComparison]:
    """Every output ROOT made, compared with xrdroot's; one ROOT made and it did not, said."""
    for name in sorted(expected):
        if name not in actual:
            yield FileComparison(name, _kind(name), [f"{name}: ROOT wrote it, xrdroot did not"])
            continue
        yield compare_file(
            name, dirs[0] / name, dirs[1] / name, (expected[name], actual[name]), tolerance
        )


def extra_files(expected: Iterable[str], actual: Iterable[str]) -> list[str]:
    """What xrdroot wrote that ROOT did not - noted, not held against it."""
    mine = set(expected)
    return sorted(name for name in actual if name not in mine)
