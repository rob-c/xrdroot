"""A Wavefront ``.obj`` file's vertices and faces, read as ``TGeoTessellated`` reads one.

``TGeoTessellated::ImportFromObjFormat`` takes the vertex lines (``v x y z
[w]``, each coordinate times ``w``) and the face lines (``f a b c [d]``,
of the ``a/t/n`` words only the vertex index, counting from 1) and leaves
everything else - normals, textures, groups - alone. A face of other than
three or four corners, or one counting from the end, it refuses, and so
does this, in ROOT's words.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["read_obj", "ObjError"]

#: The fewest faces ROOT takes as a shape.
FEWEST_FACES = 3


class ObjError(ValueError):
    """The file is not one ROOT would import: ROOT's own message says why."""


def _vertex(words: list[str]) -> tuple[float, float, float]:
    x, y, z = (float(word) for word in words[1:4])
    w = float(words[4]) if len(words) > 4 else 1.0
    return x * w, y * w, z * w


def _face(words: list[str], path: Any) -> tuple[int, ...]:
    corners = words[1:]
    if not 3 <= len(corners) <= 4:
        raise ObjError(f"Detected face having unsupported {len(corners)} vertices")
    found = tuple(int(corner.split("/")[0]) - 1 for corner in corners)
    if min(found) < 0:
        raise ObjError(f"Unsupported relative vertex index definition in {path}")
    return found


def read_obj(path: Any) -> tuple[list[tuple[float, float, float]], list[tuple[int, ...]]]:
    """The vertices and the faces - each three or four vertex indices from 0 - of ``path``."""
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []
    try:
        text = Path(path).read_text()
    except OSError:
        raise ObjError(f"Unable to open {path}") from None
    for line in text.splitlines():
        words = line.split()
        if line.startswith("v") and not line.startswith(("vt", "vn", "vp")):
            vertices.append(_vertex(words))
        elif line.startswith("f"):
            faces.append(_face(words, path))
    if len(faces) < FEWEST_FACES:
        raise ObjError(f"Not enough faces detected in {path}")
    return vertices, faces
