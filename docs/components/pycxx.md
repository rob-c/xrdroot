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
| Understanding it | `ctype` (C types and the standard's typedefs), `symbols`, `program`, `infer` (type inference), `loops`, `typeids`, `nrvo`, `freeops`, `operators`, `instantiate` and `substitute` (templates), `limits` (`std::numeric_limits`), `sizes` (`sizeof`), `traits` (`<type_traits>`) |
| Writing Python | `base`, `emit_names`, `emit_expr`, `emit_calls`, `emit_stmt`, `emit_vars`, `emit_funcs`, `emit_classes`, `emit_templates`, `emit_module`, `writer` |
| The pipeline | `translation`, `cache` (translations kept on disk), `dialect`, `errors` |
| Running it | `execute`: a program from `main`, its GoogleTest tests or an entry function; a source's declarations; a line |
| What translated code calls | `runtime`: cells, call references, `type_info`, C's arithmetic, `printf`, streams, `<cmath>`, the host namespace, and GoogleTest (`runtime.gtest`, with the macros of `gtest_macros`) |

Every translation starts `from xrdroot.pycxx.runtime import *` and finds every
name it did not declare in the dialect's *host*: `CXX.std.vector['double']` in
plain C++. GoogleTest is C++ rather than any library's, so its macros and
runner are here and every dialect has them.

## Templates

A template is translated once for each set of template arguments it is used
with, as C++ compiles it: an *instance* is the template's syntax tree copied
with every parameter replaced by its argument (`substitute`), and then an
ordinary class or function of the program, so everything that depends on `T`
is C++'s - `T x;` is `0.0` for a `double`, `sizeof(T)` and
`std::numeric_limits<T>::epsilon()` are the numbers they are for that type,
`x / 2` truncates when `T` is `int`, each instance has its own static
members, `std::is_same_v<T, int>` is a constant, and `if constexpr` keeps
only the branch it chooses (the other is never compiled, and need not make
sense for these types).

| Step | Where | What happens |
|---|---|---|
| Reading | `parse_decl`, `parse_types` | Class and function templates keep their parameters, value parameters and defaults; partial (`template <class T> struct Box<T*>`) and explicit (`template <> struct Box<int>`) specialisations keep the arguments they are for; alias templates (`template <class T> using Vec = std::vector<T>`) are put in where they are named; `template <class T> int Box<T>::count = 0;` is a static member of each instance |
| Classes | `instantiate.Templates`, from `Program` | Before any Python is written, every type and name in the program that is a class template with arguments all known (none a parameter of an enclosing template) becomes an instance: the most specialised matching partial specialisation, the explicit one, or the primary template, its arguments' defaults filled in. The type then names the instance's *key* |
| Functions | `emit_templates` | A call of a function template is a call of the instance for its arguments - given in `f<int>(...)`, deduced from the inferred types of the call's arguments (`const T&`, `T*`, `std::vector<T>&`, `T (&)[N]`, `Box<T>`), or defaulted. An explicit specialisation `template <> f<int>(int)` is that instance. A member template's instance is a function at module level handed the object, `Maker_make__int(m, 2)` for `m.make<int>(2)` |
| Members | `instantiate.Templates.instance_member` | A member of an explicit specialisation defined outside it, `Box<int>::Box()`, is that class's; a member specialised for one instance, `template <> int Box<int>::kind()`, replaces the template's in that instance alone |
| Writing | `emit_module`, `emit_templates` | A class instance is written before the first declaration at namespace scope that needs it, after its template, with the instances it needs before it; a function instance is made as its first call is written and put before that declaration, with any class instance it is the first to need |

An instance's key is a Python identifier made of its C++ spelling and no C++
name can have, `Box<int*>` being `Box__int_p` (`instantiate.mangled`); the
spelling itself is kept (`Templates.spellings`) for what names the type to a
library or the user - `_cxx_layout_`, `typeid`, a library template's
arguments. The memo is the spelling: the same arguments, however written, are
one instance, made once.

The template itself is still written, once, for uses whose arguments cannot
be known here - a function template called with an argument of a library's
type the translator does not see. Where its body cannot be written without
its arguments (a `T` variable, `sizeof(T)`), it is written as a stand-in that
raises `Refusal` when it is used (`runtime.uninstantiated`), with the reason.

A GoogleTest typed test is a class template too, instantiated for each type of
its suite's `::testing::Types` list where the test is registered.

Not instantiated, and so written generically as before: variadic templates,
template template parameters and variable templates.

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
