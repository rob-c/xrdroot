# The C++ translator: pycxx

`xrdroot.pycxx` translates C++ into Python that reads like it, and runs it.
It knows the language and its standard library, and no library past them: it
depends only on [cppstd](cppstd.md), NumPy and Python's standard library, and
imports in a fresh process with everything above it blocked. A program written
against a larger library is read in that library's *dialect*, which the
library describes as data; [pycint](pycint.md) is ROOT's.

## What it owns

| Part | Modules |
|---|---|
| Reading C++ | `tokens`, `preprocessor`, `ppexpr`, `cursor`, `parse_types`, `parse_expr`, `parse_stmt`, `parse_decl`, `nodes`, `literals` |
| Understanding it | `ctype` (C types and the standard's typedefs), `symbols`, `program`, `infer` (type inference), `loops`, `typeids`, `nrvo`, `freeops`, `operators` |
| Writing Python | `base`, `emit_names`, `emit_expr`, `emit_calls`, `emit_stmt`, `emit_vars`, `emit_funcs`, `emit_classes`, `emit_module`, `writer` |
| The pipeline | `translation`, `cache` (translations kept on disk), `dialect`, `errors` |
| Running it | `execute`: a program from `main`, its GoogleTest tests or an entry function; a source's declarations; a line |
| What translated code calls | `runtime`: cells, call references, `type_info`, C's arithmetic, `printf`, streams, `<cmath>`, the host namespace, and GoogleTest (`runtime.gtest`, with the macros of `gtest_macros`) |

Every translation starts `from xrdroot.pycxx.runtime import *` and finds every
name it did not declare in the dialect's *host*: `CXX.std.vector['double']` in
plain C++. GoogleTest is C++ rather than any library's, so its macros and
runner are here and every dialect has them.

## The dialect interface

A `Dialect` (`xrdroot.pycxx.dialect`) is a frozen dataclass of data and a few
small callables, empty for plain C++ (`CXX`, the default). One is *active* while
a program is translated; `translation(..., dialect=...)` applies the one given,
and `using(dialect)` makes one active for a block, which is how code that
parses C++ itself reads it in a library's dialect.

| Concern | Fields |
|---|---|
| Types | `typedefs`, `pointer_typedefs`, `class_pattern` and `typedef_pattern` (what an undeclared class or typedef looks like), `type_namespaces`, `templates`, `literal_suffixes` |
| Headers and macros | `predefined` (`#define` text), `preludes` (header → `Prelude` of macros), `out_of_scope` (headers refused by name) |
| Names at run time | `host` (the Python name), `host_namespace` (the C++ namespace that is the host), `host_object` (a `HostNamespace`), `runtime` and `runtime_names` (the module translations import), `constants`, `functions`, `member_pattern` |
| Calls and values | `calls` and `call_classes` (calls written as other Python), `member_statement`, `out_parameters`, `returns`, `numeric_namespaces`, `method_returns`, `field_makers`, `read_only`, `string_classes`, `string_methods`, `guards`, `value_readers`, `array_readers`, `held_results`, `assigned_pattern` |
| Containers | `containers`, `typed_containers`, `vector`, `vector_aliases`, `vector_bool`, `vector_namespaces`, `vector_spellings` |
| Statements | `counted_loop`, `range_loop`: hooks that may write a loop the library's way |
| Running | `spelling`, `shown`, `status` (an entry function's value printed and exited with), `session` (what every run happens inside), `sources` (code hashed into a cached translation's key) |

A hook is called with the emitter itself and returns `None` or `False` to let
the translator write the construct as plain C++ would. A dialect's runtime
module exports everything `xrdroot.pycxx.runtime.LIBRARY` names, its host under
`host`, and its own helpers of the same names where its objects mean more.

A host whose library has exceptions or test support of its own registers them
with `runtime.gtest.exceptions.adopt` (its gaps, its errors, what its exception
classes catch, its refusal), `runtime.gtest.provide` (checks its test macros
call) and `runtime.gtest.runner.AFTER_TEST`.

## Using it without ROOT

```python
from pathlib import Path

from xrdroot.pycxx import run_program, translate

source = """
#include <iostream>
#include <vector>
int main() {
   std::vector<int> v{3, 1, 4};
   int total = 0;
   for (int n : v) total += n;
   std::cout << total / 2 << std::endl;
   return 0;
}
"""
print(translate(source, "sum.cxx"))      # Python, with integer division as idiv
Path("sum.cxx").write_text(source)
assert run_program(["sum.cxx"]) == 0      # prints 4
```

A library describes its dialect in a few lines:

```python
import re

from xrdroot.pycxx import Dialect, translate
from xrdroot.pycxx.runtime.namespace import HostNamespace

LAB = Dialect(
    name="lab",
    typedefs={"Real": "double"},
    class_pattern=re.compile(r"^Lab[A-Z]\w*$"),
    predefined="#define LAB_VERSION 3\n",
    host="LAB",
    runtime="lab.runtime",                 # exports pycxx's runtime and LAB
    host_object=HostNamespace(lambda: __import__("lab")),
)
print(translate("void f() { LabHist h; Real x = LAB_VERSION; }", dialect=LAB))
```

`tests/pycxx/` runs with nothing above this component imported:

```bash
python tools/components.py isolate pycxx --tests pycxx
```
