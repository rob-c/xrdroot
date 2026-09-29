"""A ``TASImage`` in a pad: its pixels stretched over the whole of it, as ROOT draws one."""

from __future__ import annotations

from typing import Any

__all__ = ["IMAGES", "paint_image"]


def paint_image(scene: Any, prim: Any, _option: str) -> None:
    """The picture's RGBA pixels, top row first, filling the pad from corner to corner."""
    rgba = prim.get("rgba")
    if rgba is None or not rgba.size:
        return
    limits = scene.ax.get_xlim(), scene.ax.get_ylim()
    scene.ax.imshow(rgba, extent=(0, 1, 0, 1), transform=scene.ndc, interpolation="nearest",
                    aspect="auto", zorder=scene.layer(), clip_on=False)  # fmt: skip
    scene.ax.set_xlim(limits[0])
    scene.ax.set_ylim(limits[1])


#: The painter of each picture class.
IMAGES = {"TASImage": paint_image}
