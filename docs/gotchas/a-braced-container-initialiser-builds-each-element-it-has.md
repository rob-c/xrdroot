---
title: "A braced container initialiser builds each element it has"
kind: "gotcha"
summary: "`std::vector<Point> v{{1,2},{3,4}}` must build a Point per brace; the parser's brace elision and the element type decide the split, and a map's brace must stay a plain list."
tags: [pycxx, cppstd, vector, translation]
created: "2026-11-26T22:36:03.678Z"
updated: "2026-11-26T22:36:03.678Z"
---
`std::vector<Point> v{{1, 2}, {3, 4}}` reaches the emitter as one `InitList` per element, or as one `InitList` holding them all, depending on the spelling - and each brace is the **element type's construction**, not a Python list. Left as a list, `for (auto &p : v) p.x` raises `AttributeError: 'list' object has no attribute 'x'`, which is how this was found.

```text
 v{{1,2.0},{3,4.0}}      -> args = [InitList, InitList]        -> P(1,2.0), P(3,4.0)
 v({ {1,2.0},{3,4.0} })  -> args = [InitList[InitList,InitList]] -> one element per brace
 v{{1,2.0}}              -> args = [InitList] whose items are numbers -> ONE P(1,2.0)
 vector<vector<int>> g{{1,2,3}} -> args = [InitList] -> ONE vector<int>([1,2,3])
 map<int,long> m{{1,7}}  -> args = [InitList] of a pair -> left as a list, the runtime makes the pair
```

## Where it lives

`VariableEmitter._braced_items` / `_wrapped_brace` in `pycxx/emit_vars.py` decide, and `ExprEmitter._braced_element` in `pycxx/emit_expr.py` builds one element: a class takes the brace's values as constructor arguments, a container takes them as its list. `_braced_element_call` in `emit_calls.py` does the same for `v.push_back({1, 2})`, where the element type comes from the receiver.

Two traps: a **map's** brace must stay a list (its `pair` is made by the runtime, and building `str('one', ...)` from a multimap's brace is wrong), and a single brace argument is ambiguous - it is the list of elements when its own items are braces or when the element is a number, and one element otherwise. Only `SEQUENCE_CONTAINERS` (vector, deque, list, forward_list, array) take this path at all.
