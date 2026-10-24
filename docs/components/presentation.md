# Geometry and rendering

## geom

Geometry owns shapes, matrices, placements, scene traversal and projection data
under `xrdroot.geom`. Its only direct component dependency is core. A geometry
model must be usable without importing canvas or an optional scene renderer.

Pure `View3D` projection behavior lives in `geom.projection`; the old
`canvas.view3d` names re-export the same objects. Scene point collection lives
in `geom.scene`. Shared colour decoding comes from core `colors` rather than a
canvas manager. These downward data dependencies are preferable to introducing
a provider for every shared calculation.

Optional scene exports still load pyvista or plotly lazily. The existing
`geom` extra installs those backends in this same project:

```bash
pip install -e ".[geom]"
```

Modelling, pad painting and optional scene rendering have separate import
lifetimes even though they ship together.

## graphics

Graphics owns `canvas` and `plot`: ROOT-style pad painting, text and fonts,
rasterization, colour-manager state, and matplotlib/plotly/bokeh plotting.
Its direct component dependencies are core, hist and geom.

Geometry pad painters live in `canvas.geompaint`. The old `geom.paint` path is
a lazy compatibility shim for the graphics implementation; its exact module
ownership is graphics despite its historical directory. Lower `points_of`,
`extent` and `make_view` aliases remain available at that path. Code that needs
only model data should import the owning geom modules directly.

The canvas and plot compatibility paths retain exact-object aliases for moved
projection and colour helpers. Pure HTML tables live in core `notebook`; figure
rendering stays in graphics. [Display and persistence providers](providers.md)
load higher rendering code only when a caller requests that behavior.

Select plotting extras using existing project names:

```bash
pip install -e ".[plot,plotly,bokeh,hep]"
```

Changing ownership does not accept image differences. Existing pixel, layout,
colour and numeric assertions remain intact and must pass on the integrated
source. Optional packages must still skip cleanly in tests and give the existing
install guidance when a requested backend is unavailable.
