"""ROOT's entry order and C++ reduction order survive batches and empty type probes."""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from xrdroot.tree import Jagged


@pytest.mark.parametrize('kind', ['TRandom', 'TRandom1', 'TRandom2', 'TRandom3'])
@pytest.mark.parametrize('step', [1, 3, 100_000])
def test_uniform_columns_share_the_same_entry_order_at_every_batch_size(kind, step):
    ROOT.gRandom = getattr(ROOT, kind)(4357)
    frame = ROOT.RDataFrame(8, step=step).Define('x', 'gRandom->Uniform(-1., 1.)')
    frame = frame.Define('y', 'gRandom->Uniform(-1., 1.)')
    actual = frame.AsNumpy(['x', 'y'])
    next_draw = ROOT.gRandom.Rndm()
    ROOT.gRandom.SetSeed(4357)
    expected = np.asarray([[ROOT.gRandom.Uniform(-1., 1.), ROOT.gRandom.Uniform(-1., 1.)]
                           for _ in range(8)])
    assert np.array_equal(actual['x'].view(np.uint64), expected[:, 0].copy().view(np.uint64))
    assert np.array_equal(actual['y'].view(np.uint64), expected[:, 1].copy().view(np.uint64))
    assert next_draw == ROOT.gRandom.Rndm()
    assert frame.GetColumnType('x') == frame.GetColumnType('y') == 'double'


def test_uniform_overloads_keep_the_generator_state_and_rounding():
    ROOT.gRandom = ROOT.TRandom3(5)
    frame = ROOT.RDataFrame(7).Define('a', 'gRandom->Uniform()')
    frame = frame.Define('b', 'gRandom->Uniform(-2.)').Define('c', 'gRandom->Uniform(10., 20.)')
    actual = frame.AsNumpy(['a', 'b', 'c'])
    following = ROOT.gRandom.Rndm()
    ROOT.gRandom.SetSeed(5)
    expected = np.asarray([[ROOT.gRandom.Uniform(), ROOT.gRandom.Uniform(-2.),
                            ROOT.gRandom.Uniform(10., 20.)] for _ in range(7)])
    for at, name in enumerate(['a', 'b', 'c']):
        assert np.array_equal(actual[name].view(np.uint64), expected[:, at].copy().view(np.uint64))
    assert following == ROOT.gRandom.Rndm()


def test_explicit_positive_uniform_bounds_and_global_constants_keep_their_values():
    ROOT.gRandom = ROOT.TRandom3(5)
    frame = ROOT.RDataFrame(4).Define('a', 'gRandom->Uniform(+2.)')
    actual = frame.AsNumpy(['a'])['a']
    ROOT.gRandom.SetSeed(5)
    assert actual.tolist() == [ROOT.gRandom.Uniform(2.) for _ in range(4)]
    frame = ROOT.RDataFrame(3).Define('a', 'gRandom->Uniform(M_PI)')
    ROOT.gRandom.SetSeed(5)
    actual = frame.AsNumpy(['a'])['a']
    ROOT.gRandom.SetSeed(5)
    assert actual.tolist() == [ROOT.gRandom.Uniform(math.pi) for _ in range(3)]


def test_a_custom_generator_override_is_called_for_each_draw():
    class Observed(ROOT.TRandom3):
        def __init__(self):
            super().__init__(5)
            self.calls = 0

        def Uniform(self):
            self.calls += 1
            return super().Uniform()

    ROOT.gRandom = Observed()
    frame = ROOT.RDataFrame(4).Define('a', 'gRandom->Uniform()').Define('b', 'gRandom->Uniform()')
    actual = frame.AsNumpy(['a', 'b'])
    assert ROOT.gRandom.calls == 8
    expected = ROOT.TRandom3(5)
    pairs = [[expected.Uniform(), expected.Uniform()] for _ in range(4)]
    assert np.array_equal(np.column_stack([actual['a'], actual['b']]), pairs)


def test_a_random_define_and_a_python_callback_keep_their_interleaved_side_effects():
    ROOT.gRandom = ROOT.TRandom3(5)
    frame = ROOT.RDataFrame(4).Define('a', 'gRandom->Uniform()')
    frame = frame.Define('b', lambda: ROOT.gRandom.Uniform())
    actual = frame.AsNumpy(['a', 'b'])
    ROOT.gRandom.SetSeed(5)
    pairs = [[ROOT.gRandom.Uniform(), ROOT.gRandom.Uniform()] for _ in range(4)]
    assert np.array_equal(np.column_stack([actual['a'], actual['b']]), pairs)


def _scoped_draw(generator, arguments):
    from xrdroot.pyroot.rdf.randomexpr import _RandomDraw

    namespace = {'ROOT': SimpleNamespace(gRandom=generator), 'arguments': arguments}
    exec('def draw():\n    return ROOT.gRandom.Uniform(*arguments)\n', namespace)
    return _RandomDraw(namespace['draw'], 'Uniform', arguments)


def test_defines_bound_to_different_namespaces_keep_their_own_random_streams():
    from xrdroot.rdf import RDataFrame

    a, b = ROOT.TRandom3(5), ROOT.TRandom3(9)
    frame = RDataFrame(4).Define('a', _scoped_draw(a, []))
    frame = frame.Define('b', _scoped_draw(b, []))
    actual = frame.AsNumpy(['a', 'b']).GetValue()
    a.SetSeed(5)
    b.SetSeed(9)
    assert actual['a'].tolist() == [a.Uniform() for _ in range(4)]
    assert actual['b'].tolist() == [b.Uniform() for _ in range(4)]


def test_nonfinite_uniform_bounds_keep_the_scalar_path_and_generator_state():
    from xrdroot.rdf import RDataFrame

    generator = ROOT.TRandom3(5)
    frame = RDataFrame(4).Define('a', _scoped_draw(generator, [math.inf]))
    assert frame.AsNumpy(['a'])['a'].tolist() == [math.inf] * 4
    following = generator.Rndm()
    generator.SetSeed(5)
    assert [generator.Uniform(math.inf) for _ in range(4)] == [math.inf] * 4
    assert generator.Rndm() == following


@pytest.mark.parametrize('literal', ['1', '1u', '1l', '1ul', '1ll', '1ull', '1.f', '1.', '1.L'])
def test_numeric_literal_types_are_converted_to_the_declared_array_element_type(literal):
    frame = ROOT.RDataFrame(2).Define('p', f'std::array<double,1> p{{{literal}}}; return p;')
    assert frame.GetColumnType('p') == 'array<double,1>'
    assert [row.tolist() for row in frame.AsNumpy(['p'])['p']] == [[1.], [1.]]


@pytest.mark.parametrize('count', [0, 7])
def test_an_unused_random_define_does_not_advance_the_stream(count):
    ROOT.gRandom = ROOT.TRandom3(5)
    frame = ROOT.RDataFrame(count).Define('unused', 'gRandom->Uniform(-1., 1.)')
    frame = frame.Define('used', 'gRandom->Uniform()')
    actual = frame.AsNumpy(['used'])['used']
    following = ROOT.gRandom.Rndm()
    ROOT.gRandom.SetSeed(5)
    assert actual.tolist() == [ROOT.gRandom.Uniform() for _ in range(count)]
    assert following == ROOT.gRandom.Rndm()


def test_pending_actions_still_request_both_random_defines():
    ROOT.gRandom = ROOT.TRandom3(5)
    frame = ROOT.RDataFrame(5).Define('a', 'gRandom->Uniform()').Define('b', 'gRandom->Uniform()')
    a, b = frame.Take('a'), frame.Take('b')
    actual = list(a.GetValue()), list(b.GetValue())
    ROOT.gRandom.SetSeed(5)
    expected = np.asarray([[ROOT.gRandom.Uniform(), ROOT.gRandom.Uniform()] for _ in range(5)])
    assert np.array_equal(actual, expected.T)


def test_mixed_random_distributions_keep_their_scalar_call_order():
    ROOT.gRandom = ROOT.TRandom3(4357)
    frame = ROOT.RDataFrame(9).Define('a', 'gRandom->Gaus(1., 2.)')
    frame = frame.Define('b', 'gRandom->Uniform()')
    actual = frame.AsNumpy(['a', 'b'])
    following = ROOT.gRandom.Rndm()
    ROOT.gRandom.SetSeed(4357)
    expected = [[ROOT.gRandom.Gaus(1., 2.), ROOT.gRandom.Uniform()] for _ in range(9)]
    assert np.array_equal(np.column_stack([actual['a'], actual['b']]), expected)
    assert following == ROOT.gRandom.Rndm()


@pytest.mark.parametrize('count,width', [(0, 2), (3, 0), (3, 2), (3, 4)])
def test_array_initializers_preserve_type_and_zero_fill_remaining_elements(count, width):
    frame = ROOT.RDataFrame(count).Define('x', 'double(rdfentry_)')
    values = 'x, -x + 1.' if width else ''
    frame = frame.Define('p', f'std::array<double,{width}> p{{{values}}}; return p;')
    assert frame.GetColumnType('p') == f'array<double,{width}>'
    actual = frame.AsNumpy(['p'])['p']
    assert actual.dtype == object and actual.shape == (count,)
    expected = np.zeros((count, width))
    if width:
        expected[:, 0], expected[:, 1] = np.arange(count), 1. - np.arange(count)
    assert [row.tolist() for row in actual] == expected.tolist()


def test_a_constant_array_initializer_is_counted_without_reading_other_columns():
    frame = ROOT.RDataFrame(4).Define('p', 'std::array<double,3> p{2.,3.}; return p;')
    assert frame.GetColumnType('p') == 'array<double,3>'
    assert [row.tolist() for row in frame.AsNumpy(['p'])['p']] == [[2., 3., 0.]] * 4


def test_a_body_requiring_preprocessing_keeps_the_translators_result():
    body = ('#define RDF_BATCH_WIDTH_TEST 2\n'
            'std::array<double,RDF_BATCH_WIDTH_TEST> v{1.,2.};\nreturn v;')
    frame = ROOT.RDataFrame(3).Define('p', body)
    assert [row.tolist() for row in frame.AsNumpy(['p'])['p']] == [[1., 2.]] * 3


@pytest.mark.parametrize('declaration', ['auto', 'ROOT::RDF::RResultPtr<ULong64_t>'])
def test_dereferencing_a_booked_count_materializes_its_numeric_value(declaration):
    from xrdroot.cint.execute import run_source

    source = f'''int counted() {{
        ROOT::RDataFrame frame(8);
        {declaration} count = frame.Filter("rdfentry_ % 2 == 0").Count();
        bool started = count.IsReady();
        auto value = *count;
        return 2 * value + (count.IsReady() ? 100 : 0) + (started ? 1000 : 0);
    }}'''
    namespace = run_source(source, '<booked count>', call=False)
    assert namespace['counted']() == 108


@pytest.mark.parametrize('fixed', [False, True])
def test_a_range_for_reduction_starts_each_row_with_its_initializer_and_adds_in_order(fixed):
    rows = np.asarray([[1e16, 1., 1., 1., -1e16, 1., 1., 1.], [0.] * 8])
    ragged = Jagged(np.concatenate([rows[0], [], rows[1]]), [0, 8, 8, 16])
    data = rows if fixed else ragged

    def values(entries):
        return data[np.asarray(entries, int)] if fixed else data.take(entries)

    frame = ROOT.RDataFrame(len(data), step=1).Define('p', values, ['rdfentry_'])
    frame = frame.Define('s', 'double s=1.; for (auto w:p) { s += w; } return s;')
    actual = frame.AsNumpy(['s'])['s']
    assert actual.tolist() == ([3., 1.] if fixed else [3., 1., 1.])
    assert frame.GetColumnType('s') == 'double'


def test_the_two_coordinate_radius_matches_the_oracle_without_reading_a_shadowed_column():
    ROOT.gRandom = ROOT.TRandom3(4357)
    frame = ROOT.RDataFrame(8).Define('x', 'gRandom->Uniform(-1.,1.)')
    frame = frame.Define('y', 'gRandom->Uniform(-1.,1.)')
    frame = frame.Define('p', 'std::array<double,2> v{x,y}; return v;')
    frame = frame.Define('r', 'double r=0.; for(auto&& x:p) r+=x*x; return sqrt(r);')
    actual = frame.AsNumpy(['x', 'y', 'r'])
    # ROOT 6.40.04, seed 4357: these are the first coordinates in its event order.
    assert actual['x'][:3].tolist() == [
        0.999483497813344, -0.43476438941434026, -0.5366869145072997]
    assert actual['y'][:3].tolist() == [
        -0.6741802492178977, 0.8944021640345454, -0.03005277132615447]
    expected = [math.sqrt(x*x + y*y) for x, y in zip(actual['x'], actual['y'], strict=True)]
    assert np.array_equal(actual['r'], expected)
    assert list(frame._inner._columns['r'].inputs.values()) == [frame._inner._columns['p']]
