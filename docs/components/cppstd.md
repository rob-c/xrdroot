# C++ standard-library facilities

`xrdroot.cppstd` holds generic C++ behavior used by engines and translated
programs. It ships with the existing `xrdroot` project; there is no separate
installation or optional extra.

The component includes streams, strings, containers, arithmetic, references,
value copying, standard utilities, C formatting, generic objects, cmath and
duration literals. They can be imported without loading ROOT wrappers or the
interpreter. NumPy supplies numeric storage and array operations.
The interpreter's original module paths re-export the same objects, including
the output streams, manipulators and formatter registry.

```python
from xrdroot.cppstd import streams, strings

output = streams.ostringstream()
output << streams.fixed << streams.setprecision(2) << 1.25
assert output.str() == "1.25"
assert strings.substr("abcd", 1, 2) == "bc"
```

`ostream` formats output with C++ stream state. `ostringstream` stores output
in memory; `ofstream` writes it to a file. `cout`, `cerr` and `clog` share the
same state through the compatibility imports and follow Python's current
standard output/error streams. Registered custom formatters apply through both
paths. `strings` supplies the existing C string conversions and translated
`std::string` operations.

`istreams` reads standard numeric types, words and characters; the interpreter
adapter explicitly supplies ROOT's numeric typedef names. `printf` supplies C
formatting and output functions; ROOT's `Printf`, `Form` and diagnostic functions
stay in the interpreter adapter. `objects` supplies arrays, overloads, iteration,
sorting, exceptions and stores; ROOT lifetime, placement and dereference behavior
stays in the adapter. Their old imports resolve to the same generic objects.

`cmath` preserves the existing scalar calculations and NumPy array promotion.
A caller can supply its result constructor for objects carrying C++ vector type
names; the interpreter adapter supplies the original RVec construction. Generic
use returns NumPy arrays. `units.duration_literal` creates the existing timedelta
values for standard chrono suffixes, and returns `None` for other suffixes. ROOT
pad-length literals remain in the interpreter adapter.

This is the subset already implemented by xrdroot. ROOT namespace resolution,
ROOT type aliases, RVec construction and tree readers belong to the adapters.
