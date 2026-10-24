# Shared primitives

## core

Core owns the small representations and algorithms shared by engines:
`errors`, buffers and codecs, numeric helpers, ROOT random generators,
`stats`, C++ type descriptions, drawing specifications and array-library
conversion. Its only direct component dependency is `cppstd`.

The shared `jagged` and `rootarrays` modules sit below I/O. Histogram slicing and
formula evaluation can use variable-length rows and primitive-array layouts
without loading a ROOT file reader. `tree.Jagged` remains the same public class
through a compatibility re-export; the implementation belongs to
`xrdroot.jagged`. Fast-path provenance checks must name the implementation while
continuing to reject altered methods and user callbacks.

```python
import numpy as np
from xrdroot.jagged import Jagged

rows = Jagged(np.array([1., 2., 3.]), [0, 2, 3])
assert list(rows[0]) == [1., 2.]
```

`library` converts engine arrays into requested third-party array types. These
libraries are optional and loaded at the point of use. Core errors retain their
approved public xrdclient inheritance; internal ownership is not an attempt to
pretend that the installed project lacks its transport dependency. Private
third-party helpers should be replaced by public Python APIs where possible.

Core's `random` package retains ROOT's `TRandom` family and exact numeric rules.
The standard C++ engines and distributions live in `cppstd`; ROOT-specific
state and adapters stay with their callers. A move must not create a second
random stream or substitute NumPy transcendental functions where the existing
libm path establishes ROOT exactness.

## Notebook and colour data

`notebook` owns pure HTML/table helpers, escaped text and notebook fallback.
The old `plot.notebook` helper paths re-export the same function objects.
Rendering a plot or figure uses the [display providers](providers.md), while
an absent HTML registration falls back to escaped `repr` output.

`colors` owns shared colour tables and decoding primitives. Geometry scene
exports can consume these values without constructing a canvas colour manager.
The existing `plot.colors` path re-exports the shared objects; canvas keeps its
own renderer state such as gradients. Shared mutable tables must retain their
identity across compatibility imports.

`plugins` is the small built-in provider registry. It records explicit targets
rather than discovering installed packages. Adding core primitives must not
introduce imports of I/O, geometry, a renderer, RooFit or the ROOT namespace.
