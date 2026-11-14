"""ROOT 7's bin indices, their ranges, the axes, the axes of a histogram and the slice
specifications, as ``ROOT::Experimental`` has them."""

from __future__ import annotations

import copy
import math

import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import expect

E = ROOT.Experimental
I = E.Internal


def test_bin_index_arithmetic_leaves_the_normal_bins_for_the_invalid_index() -> None:
    three, under = E.RBinIndex(3), E.RBinIndex.Underflow()
    expect(((three + -1).GetIndex(), 2), ((three - -2).GetIndex(), 5),
           ((three + (1 << 64) - 3).IsInvalid(), True), ((three - 4).IsInvalid(), True),
           ((under - 1).IsInvalid(), True), (E.RBinIndex(-1).IsInvalid(), True),
           (E.RBinIndex(three), three), (three != E.RBinIndex(3), False),
           (three > E.RBinIndex(2), True), (under > three, False), (three <= three, True),
           (under <= E.RBinIndex.Overflow(), False), (repr(E.RBinIndex()), "RBinIndex(Invalid)"),
           (three == 3, False))  # fmt: skip


def test_a_range_walks_from_the_underflow_through_its_bins_to_the_overflow() -> None:
    full = I.CreateBinIndexRange(E.RBinIndex.Underflow(), E.RBinIndex(), 2)
    walked = list(full)
    it = full.begin()
    first = copy.deepcopy(it)
    it += 2
    stuck = E.RBinIndexRange().begin()
    stuck += 1  # an invalid iterator stays where it is
    expect((walked, [E.RBinIndex.Underflow(), E.RBinIndex(0), E.RBinIndex(1),
                      E.RBinIndex.Overflow()]),
           (it.value, E.RBinIndex(1)), (it.GetIndex(), 1), (first.IsUnderflow(), True),
           (it != full.end(), True), (stuck.value.IsInvalid(), True),
           (full != E.RBinIndexRange(), True), (repr(full).startswith("RBinIndexRange("), True),
           (E.RBinIndexRange().IsInvalid(), True), (full.GetEnd().IsInvalid(), True))
    with pytest.raises(AttributeError):
        it._missing  # noqa: B018
    with pytest.raises(TypeError):
        hash(it)


def test_a_multidimensional_range_steps_its_last_dimension_fastest() -> None:
    rows = I.CreateBinIndexRange(0, 2, 0)
    columns = I.CreateBinIndexRange(E.RBinIndex.Underflow(), E.RBinIndex(), 1)
    both = E.RBinIndexMultiDimRange([rows, columns])
    walked = [[index.GetIndex() for index in each] for each in both]
    it, steps = both.begin(), 0
    copied = copy.copy(it)
    while it != both.end():
        it += 1
        steps += 1
    under, over = E.RBinIndex.Underflow().GetIndex(), E.RBinIndex.Overflow().GetIndex()
    expect((walked, [[0, under], [0, 0], [0, over], [1, under], [1, 0], [1, over]]),
           (steps, 6), (list(copied.value), [E.RBinIndex(0), E.RBinIndex.Underflow()]),
           (copied.size(), 2), (both.GetRanges().size(), 2), (both != E.RBinIndexMultiDimRange(),
           True), (list(E.RBinIndexMultiDimRange()), []),
           (E.RBinIndexMultiDimRange().begin() == E.RBinIndexMultiDimRange().end(), False))
    empty = E.RBinIndexMultiDimRange([rows, I.CreateBinIndexRange(0, 0, 0)])
    assert empty.begin() == empty.end()


def test_a_regular_axis_checks_its_bins_and_slices_as_root_does() -> None:
    axis = E.RRegularAxis(10, (0.0, 10.0))
    sliced = axis.Slice(E.RSliceSpec(axis.GetNormalRange(2, 8), E.RSliceSpec.ROperationRebin(3)))
    expect((axis.ComputeLinearizedIndex(9.99).fIndex, 10),
           (axis.GetLinearizedIndex(E.RBinIndex(10)).fValid, False),
           ((sliced.GetNNormalBins(), sliced.GetLow(), sliced.GetHigh()), (2, 2.0, 8.0)),
           (axis.Slice(E.RSliceSpec()) == axis, True), (axis == E.RRegularAxis(10, (0, 10), False),
           False), (repr(axis), "RRegularAxis(10, (0.0, 10.0), True)"),
           (repr(axis.ComputeLinearizedIndex(1)), "RLinearizedIndex(2, True)"))  # fmt: skip
    for bad, why in ((lambda: axis.ComputeLowEdge(10), "bin must be inside the axis"),
                     (lambda: axis.ComputeHighEdge(10), "bin must be inside the axis"),
                     (lambda: E.RRegularAxis(2, (-math.inf, 0)), "low must be a finite value, "
                      "but is -inf"),
                     (lambda: axis.ComputeLinearizedIndex("a"), "invalid type of argument"),
                     (lambda: axis.GetNormalRange(E.RBinIndex.Underflow(), 2),
                      "begin must be a normal bin"),
                     (lambda: axis.GetNormalRange(10, 10), "begin must be inside the axis"),
                     (lambda: axis.GetNormalRange(0, E.RBinIndex.Overflow()),
                      "end must be a normal bin"),
                     (lambda: axis.GetNormalRange(0, 11), "end must be inside or past the axis")):
        with pytest.raises(ValueError, match=why):
            bad()
    for spec, why in ((E.RSliceSpec.ROperationSum(), "sum operation makes dimension disappear"),
                      (E.RSliceSpec.ROperationRebin(3), "rebin operation does not divide")):
        with pytest.raises(RuntimeError, match=why):
            axis.Slice(spec)
    with pytest.raises(RuntimeError, match="unable to store RRegularAxis"):
        axis.Streamer()


def test_a_variable_axis_refuses_nan_edges_and_slices_every_group_th_edge() -> None:
    axis = E.RVariableBinAxis([0, 1, 2, 4, 8], False)
    full = axis.GetFullRange()
    expect((list(axis.Slice(E.RSliceSpec.ROperationRebin(2)).GetBinEdges()), [0.0, 2.0, 8.0]),
           (axis.edges(), [0.0, 1.0, 2.0, 4.0, 8.0]), (axis.value_index(8.0), (5, False)),
           (axis.value_index(3.0), (2, True)), ((full.GetBegin(), full.GetEnd()),
           (E.RBinIndex(0), E.RBinIndex(4))), (axis == E.RVariableBinAxis([0, 1, 2, 4, 8]), False),
           (repr(axis), "RVariableBinAxis([0.0, 1.0, 2.0, 4.0, 8.0], False)"))  # fmt: skip
    for edges, why in (([], "must have >= 2 bin edges"), ([math.nan, 1], "bin egde 0 is NaN"),
                       ([0, math.nan], "bin egde 1 is NaN")):
        with pytest.raises(ValueError, match=why):
            E.RVariableBinAxis(edges)


def test_a_categorical_axis_takes_names_and_only_its_whole_self_as_a_slice() -> None:
    axis = E.RCategoricalAxis(["a", "b"])
    bare = E.RCategoricalAxis(["a", "b"], False)
    expect((axis.ComputeLinearizedIndex("b").fIndex, 1), (axis.value_index("z"), (2, True)),
           (bare.value_index("z"), (2, False)), (axis.bin_index(E.RBinIndex.Overflow()),
           (2, True)), (axis.bin_index(E.RBinIndex.Underflow()), (0, False)),
           (axis.bin_index(E.RBinIndex(5)), (5, False)), (list(axis.GetCategories()), ["a", "b"]),
           (len(list(axis.GetFullRange())), 3), (len(list(bare.GetFullRange())), 2),
           (bare.Slice(E.RSliceSpec()).HasOverflowBin(), True), (axis == bare, False),
           (repr(bare), "RCategoricalAxis(['a', 'b'], False)"))  # fmt: skip
    with pytest.raises(ValueError, match="invalid type of argument"):
        axis.ComputeLinearizedIndex(1)
    with pytest.raises(ValueError, match="duplicate category 'a' for bin 1"):
        E.RCategoricalAxis(["a", "a"])
    with pytest.raises(ValueError, match="must have at least one category"):
        E.RCategoricalAxis([])
    for spec, why in ((E.RSliceSpec.ROperationSum(), "sum operation"),
                      (axis.GetNormalRange(), "slicing of RCategoricalAxis not implemented"),
                      (E.RSliceSpec.ROperationRebin(1), "cannot rebin RCategoricalAxis")):
        with pytest.raises(RuntimeError, match=why):
            axis.Slice(spec)


def test_a_variant_holds_one_axis_and_answers_for_it() -> None:
    regular = E.RAxisVariant(E.RRegularAxis(4, (0, 1)))
    variable = E.RAxisVariant(E.RAxisVariant(E.RVariableBinAxis([0, 1, 3])))
    categorical = E.RAxisVariant(E.RCategoricalAxis(["a"]))
    expect(([v.GetVariant().index() for v in (regular, variable, categorical)], [0, 1, 2]),
           (variable.GetVariableBinAxis() is not None, True), (regular.GetCategoricalAxis(),
           None), (categorical.GetRegularAxis(), None), (regular.GetTotalNBins(), 6),
           (len(list(regular.GetNormalRange(1, 3))), 2), (len(list(regular.GetFullRange())), 6),
           (regular.Slice(E.RSliceSpec.ROperationRebin(2)).GetNNormalBins(), 2),
           (regular == variable, False), (repr(categorical).startswith("RAxisVariant("), True))
    with pytest.raises(ValueError, match="An RAxisVariant holds"):
        E.RAxisVariant(3)
    with pytest.raises(RuntimeError, match="unable to store RAxisVariant"):
        regular.Streamer()


def test_the_axes_of_a_histogram_number_its_bins_and_refuse_wrong_arguments() -> None:
    axes = I.RAxes([E.RRegularAxis(2, (0, 2), False), E.RCategoricalAxis(["a", "b"])])
    expect((axes.ComputeTotalNBins(), 6), (axes.ComputeGlobalIndex((1.5, "b")).fIndex, 4),
           (axes.ComputeGlobalIndex((5.0, 3)).fValid, False),  # stops before the bad type
           (axes.ComputeGlobalIndex([1, E.RBinIndex.Overflow()]).fIndex, 5),
           (axes.ComputeGlobalIndex([E.RBinIndex.Underflow(), 0]).fValid, False),
           (axes.GetNDimensions(), 2), (axes.Get().size(), 2), (axes != axes, False),
           (len(list(axes.GetFullMultiDimRange())), 6))
    for given, why in (((1.0,), "invalid number of arguments to ComputeGlobalIndex"),
                       ([1], "invalid number of indices passed to ComputeGlobalIndex"),
                       ((1.0, 2), "invalid type of argument")):
        with pytest.raises(ValueError, match=why):
            axes.ComputeGlobalIndex(given)
    with pytest.raises(RuntimeError, match="unable to store RAxes"):
        axes.Streamer()
    with pytest.raises(ValueError, match="must have at least 1 axis object"):
        I.RAxes([])


def test_a_slice_specification_holds_a_range_and_at_most_one_operation() -> None:
    normal = I.CreateBinIndexRange(0, 1, 0)
    spec = E.RSliceSpec(normal, E.RSliceSpec.ROperationSum())
    copied = E.RSliceSpec(spec)
    expect((copied.GetRange(), normal), (copied.GetOperationSum() is not None, True),
           (copied.GetOperationRebin(), None), (copied.HasOperation(), True),
           (E.RSliceSpec.ROperationRebin(4).GetNGroup(), 4))
    with pytest.raises(ValueError, match="nGroup must be > 0"):
        E.RSliceSpec.ROperationRebin(0)
    with pytest.raises(ValueError, match="An RSliceSpec is made of"):
        E.RSliceSpec(3)


def test_the_slice_mapper_moves_bins_outside_a_range_into_the_flow_bins() -> None:
    spec = E.RSliceSpec(I.CreateBinIndexRange(2, 4, 0), E.RSliceSpec.ROperationRebin(2))
    summed = E.RSliceSpec(I.CreateBinIndexRange(1, 2, 0), E.RSliceSpec.ROperationSum())
    mapper = I.RSliceBinIndexMapper([spec, summed])
    over, under = E.RBinIndex.Overflow(), E.RBinIndex.Underflow()
    full = I.RSliceBinIndexMapper([E.RSliceSpec(I.CreateBinIndexRange(under, E.RBinIndex(), 5))])
    mapped = [None]
    expect((mapper.mapped([E.RBinIndex(3), E.RBinIndex(1)]), [E.RBinIndex(0)]),
           (mapper.mapped([E.RBinIndex(1), E.RBinIndex(1)]), [under]),
           (mapper.mapped([E.RBinIndex(5), E.RBinIndex(1)]), [over]),
           (mapper.mapped([under, E.RBinIndex(1)]), [under]),
           (mapper.mapped([over, E.RBinIndex(1)]), [over]),
           (mapper.mapped([E.RBinIndex(3), E.RBinIndex(0)]), None),
           (full.mapped([E.RBinIndex(2)]), [E.RBinIndex(2)]),
           (mapper.Map([3, 1], mapped), True), (mapped, [E.RBinIndex(0)]),
           (mapper.Map([3, 0], mapped), False), (mapper.GetMappedDimensionality(), 1),
           (mapper.GetSliceSpecs().size(), 2))
    for original, given, why in (([3], [None], "invalid number of original indices"),
                                 ([3, 1], [], "invalid size of mapped indices"),
                                 ([None, 1], [None], "invalid bin index")):
        with pytest.raises(ValueError, match=why):
            mapper.Map(original, given)
    with pytest.raises(ValueError, match="must have at least 1 slice specification"):
        I.RSliceBinIndexMapper([])
