---
title: "Differential-test the cppstd stand-in against a real C++ compiler"
kind: "instruction"
summary: "To check xrdroot's C++ stdlib stand-in for correctness, run the same program through clang++ and through `run_source` and diff stdout — it found missing members, wrong signs and crashed casts that the Python tests did not."
tags: [cppstd, pycxx, testing, char, algorithms]
created: "2026-11-26T22:35:40.426Z"
updated: "2026-11-26T22:35:40.426Z"
---
The stdlib stand-in (`src/xrdroot/cppstd/`) is checked for *shape* by its own tests, but a shape test cannot tell whether `std::string::compare(pos, len, s)` compares the substring or `(int)c` casts a char. A real compiler can, and one is on every machine that builds anything: write a program with `int main()`, compile it with `clang++ -std=c++17 -O0`, run it, then run the same source through `xrdroot.pycxx.run_source(src, "a.cpp", call=False)["main"]()` and compare stdout.

```text
   corpus of small C++ programs
        |                    |
        v                    v
  clang++ -O0           run_source(..., call=False)["main"]()
        |                    |
        +------ diff stdout -+--> identical, or a stand-in bug
```

Print with `printf`/`std::cout` only, avoid anything time- or address-dependent, and give every program the headers it uses: a missing `#include` shows up as a clang error, not a stand-in bug.

## What a first pass found

- `std::vector::erase`, `vector::assign(n, v)`, `vector`/`set`/`map` `lower_bound`/`upper_bound`/`rbegin`/`find`, `std::tuple`, and eleven `<algorithm>`/`<numeric>` functions (count, count_if, rotate, nth_element, partial_sort, min/max_element, inner_product, find, unique, remove, all/any/none_of) were missing, each refusing by name or failing with an AttributeError.
- `(int)char` dropped the cast, so `c - '0'` in a loop raised TypeError; `s.at(i)` never threw `std::out_of_range`; `std::stoi` never threw for an out-of-range number; `s.compare(pos, len, t)` compared the *whole* string.
- `1.0/0.0` (both literals) raised ZeroDivisionError where C++ gives inf, because the emitter only routes a division through `fdiv` when a variable is involved.
- `std::vector<Point> v{{1,2},{3,4}}` kept each brace as a Python list, so `for (auto &p : v) p.x` raised AttributeError.
- `std::string::front()`/`back()`/`at()` were not typed `char`, so they printed as numbers.

Keep the corpus as scratch; the committed tests pin the *answers* - `tests/test_cint_std_algorithms.py`, `tests/cppstd/test_cppstd_pure_members.py` and `tests/cppstd/test_cppstd_pure_ranges.py` - while the corpus is for finding the next gap.
