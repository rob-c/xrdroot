"""``Draw("y:x", "sel")``: several formulas run in **one** loop, as ``TTreeFormulaManager`` does.

ROOT compiles an axis list and its cut into separate formulas and hands them to one manager,
which matches the loops they each imply and runs a single implicit loop - to the **shortest** of
them. So ``pt`` drawn with a cut on ``eta`` pairs each jet with that same jet, a number per entry
drawn with a cut on the jets is repeated for every jet the cut looks at, and two collections in one
entry are matched element for element as far as the shorter goes. That is the rule
``evaluate_together`` implements, and the reason two loops would not do.
"""

from __future__ import annotations

import numpy as np
import pytest

from xrdroot import Jagged, compile_formula
from xrdroot.formula.together import evaluate_together

#: Two entries, three jets and two jets: a collection whose length differs entry by entry.
JETS = Jagged(np.array([10.0, 20.0, 30.0, 40.0, 50.0]), np.array([0, 3, 5], dtype=np.int64))
#: The same jets' pseudorapidities, so a cut on them selects jets and not entries.
ETE = Jagged(np.array([1.0, 2.5, 3.0, 0.5, 1.5]), np.array([0, 3, 5], dtype=np.int64))
#: A number per entry, which the loop repeats along the jets.
MET = np.array([100.0, 200.0])


def _together(*texts: str, columns: dict[str, object] | None = None, **kwargs: object):
    table = {"pt": JETS, "eta": ETE, "met": MET} if columns is None else columns
    formulas = [compile_formula(text, table) for text in texts]
    return evaluate_together(formulas, table, **kwargs)


def test_two_formulas_share_one_loop_and_pair_each_element_with_its_own():
    """``Draw("pt:eta", "eta < 3")`` runs one loop over the jets, so the pair is one jet's two
    numbers rather than one jet's momentum against every eta. Each formula answers **its own**
    mask and the caller ANDs them, as ``TTreeFormulaManager::EvalSelection`` does."""
    (values, mask), (selection, cut_mask) = _together("pt", "eta < 3")
    assert values.tolist() == [[10.0, 20.0, 30.0], [40.0, 50.0]]
    assert mask.tolist() == [[True, True, True], [True, True]]  # pt has no condition of its own
    assert selection.tolist() == [[True, True, False], [True, True]]  # the cut, as a value
    assert cut_mask.tolist() == [[True, True, True], [True, True]]  # a condition rejects nothing
    # and the caller keeps a jet where every mask holds and the selection's value is true
    kept = np.logical_and(mask.content, selection.content)  # the flat arrays the rows span
    assert kept.tolist() == [True, True, False, True, True]


def test_a_number_per_entry_is_repeated_along_the_loop():
    """``met`` drawn with a cut on the jets is one value per jet the cut looks at - the same number
    three times for the entry with three jets, twice for the entry with two."""
    (met, mask), _ = _together("met", "eta > 0")
    assert met.tolist() == [[100.0, 100.0, 100.0], [200.0, 200.0]]
    assert mask.tolist() == [[True, True, True], [True, True]]


def test_two_collections_in_one_entry_are_matched_to_the_shorter():
    """ROOT's documentation says ``TTreeFormula`` pairs two collections up to the shorter, so a
    three-element ``l`` drawn with a two-element ``s`` yields two pairs, not three."""
    columns = {
        "s": Jagged(np.array([1.0, 2.0]), np.array([0, 2], dtype=np.int64)),
        "l": Jagged(np.array([9.0, 8.0, 7.0]), np.array([0, 3], dtype=np.int64)),
    }
    (short, _), (long, _) = _together("s", "l", columns=columns)
    assert short.tolist() == [[1.0, 2.0]]
    assert long.tolist() == [[9.0, 8.0]]  # the third element is not reached


def test_a_column_of_another_entry_count_is_refused():
    """One loop cannot walk two entry counts: ``a`` has one entry and ``b`` two, and the formula
    says so in its own words rather than silently pairing the first entry with both."""
    columns = {
        "a": Jagged(np.array([1.0, 2.0, 3.0]), np.array([0, 3], dtype=np.int64)),
        "b": Jagged(np.array([9.0, 8.0, 7.0, 6.0]), np.array([0, 2, 4], dtype=np.int64)),
    }
    with pytest.raises(ValueError, match="different lengths"):
        _together("a", "b", columns=columns)


def test_a_chunk_of_the_tree_sees_the_entry_numbers_it_was_read_for():
    """``rows`` and ``entry_numbers`` describe the chunk being read: one entry of two, read alone,
    answers that entry's number - so ``Entry$`` still names where in the tree the row came from."""
    columns = {"met": np.array([200.0])}
    chunk = {"entries": 2, "rows": 1, "entry_numbers": np.array([1])}
    (value, _), = _together("met", columns=columns, **chunk)
    assert value.tolist() == [200.0]
    (number, _), = _together("Entry$", columns=columns, **chunk)
    assert number.tolist() == [1]
