# C++ standard-library facilities

`xrdroot.cppstd` holds generic C++ behavior used by engines and translated
programs. It ships with the existing `xrdroot` project; there is no separate
installation or optional extra.

The first modules are `streams` and `strings`. They use Python's standard
library and can be imported without loading ROOT wrappers or the interpreter.
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

This is the subset already implemented by xrdroot. ROOT namespace resolution,
RVec overloads and tree readers belong to the ROOT adapters. Other standard
containers and runtime helpers remain at their existing locations until their
own gated extraction.
