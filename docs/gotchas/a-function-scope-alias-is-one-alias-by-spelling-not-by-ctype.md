---
title: "A function-scope alias is one alias by spelling, not by CType identity"
kind: "gotcha"
summary: "The translator's program-wide alias table must compare a function-scope using/typedef by its spelling, not by CType identity, or identical declarations in two functions are refused — which hid ten TMVA ConvNet tests."
tags: [pycxx, translator, typedef, aliases, tmva]
created: "2026-11-26T00:14:14.033Z"
updated: "2026-11-26T00:14:14.033Z"
---
The C++ translator keeps **one** alias table for a program's function-scope `using`/`typedef` declarations (`Program._local_alias` in `src/xrdroot/pycxx/program.py`), because a call such as `Net_t(x)` is emitted with no view of the block it stands in. It refused a second function declaring the *same* name for the *same* type, comparing the two `CType` objects — and two structurally equal types may still be distinct objects, since their template arguments are parsed separately.

```text
template <typename A> void f() { using Net_t = TDeepNet<A>; … }   same spelling -> one alias (kept)
template <typename A> void g() { using Net_t = TDeepNet<A>; … }
void h() { using L = duration<double, milli>;   }                 two spellings under one name -> refused
void k() { using L = duration<double, ratio<1>>; }
```

The refusal is now made on the **spelling** (`emit_names.type_text`), so ROOT's own TMVA ConvNet tests — which write `using Net_t = TDeepNet<Architecture>;` in each of their test functions — translate, and only a name given two different spellings is refused.

## Why it matters

Measured: the ten `other/tmva/tmva/DNN/CNN/*` programs moved from this translator refusal to their next wall (`TCpu`, `TCpuTensor`, `TDeepNet`, `Architecture`), so a general refusal was hiding ten tests behind a diagnostic that looked like a user error. `tests/test_cint_typedefs.py` pins both halves: `test_a_name_two_functions_spell_the_same_way_is_one_alias` and `test_a_name_two_functions_give_two_types_is_refused_by_name`.
