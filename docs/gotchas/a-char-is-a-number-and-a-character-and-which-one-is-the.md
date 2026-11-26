---
title: "A char is a number and a character; which one is the emitter's business"
kind: "gotcha"
summary: "A translated char is a number in C++'s eyes and text in PyROOT's, so insert `char_code`/`char_text` by static type at casts, arithmetic and streams — and make `iterate` over a std::string yield numbers."
tags: [char, pycxx, cppstd, streams, strings]
created: "2026-11-26T22:35:54.197Z"
updated: "2026-11-26T22:35:54.197Z"
---
A `char` has two faces in this kit: C++ sees it as a number, PyROOT hands it out as a one-character string. Both exist at once, so the translator inserts `char_code(x)` where a value must be a number and `char_text(x)` where it must be printed as a character, and both helpers accept either form. The rule that decides where to insert them is *the static type and the source expression*, not the runtime value.

```text
 char_at(s, i) -> int         s[i], s.at(i), s.front()/back(), a char variable,
                              a char literal, pair<char,..>::first        -> numbers
 vector<char>[i]              PyROOT's one-character string                -> text
        |                                    |
        +-- arithmetic, (int), float --> char_code(x)  (ord if str, else int)
        +-- std::cout / a stream ------> char_text(x)  (x if str, else chr)
```

## Where it bites

- `std::string` is a Python `str` here, so `for (char c : s)` iterates one-character strings. `objects.iterate` therefore yields the characters' **numbers** for a `str`, which is what `s[i]` and `char_at` already give; a range-for whose loop variable is a reference writes the numbers back with `text_from_codes` (`_write_back_string` in `emit_stmt.py`), because a Python `str` cannot be written where it lies.
- `_Vector._shown` still hands a `vector<char>`'s element out as text, because PyROOT does and one of ROOT's own tests asserts it. That is the only source of a *text* char, so `text_char()` in `emit_expr.py` is deliberately narrow: a `char`-typed `Index` into a container whose element type is `char` and which is not a string.
- A cast to `float` needed the same treatment as one to `int` (`_arithmetic_store`); it was missed at first, so `float f = v[0]` produced `f32('7')`.
- C's `<cctype>` classes are ASCII's, not Python's Unicode ones: `isspace(chr(0xA0))` is true in Python and false in C's `C` locale, so `cppstd/strings.py` keeps explicit sets.
