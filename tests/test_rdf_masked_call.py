"""Batched condition arguments preserve scalar calls, callbacks and random streams."""
import importlib
import sys

import numpy as np
import pytest

from xrdroot import Jagged
from xrdroot import pyroot as ROOT
from xrdroot.cint.execute import run_source
from xrdroot.pyroot import stl
from xrdroot.pyroot.rdf import masknative, rvec
from xrdroot.pyroot.rdf.batched import _Batched, compiled
from xrdroot.pyroot.rdf.entrywise import entrywise


@pytest.fixture
def prepared():
    ROOT.gInterpreter.Declare('''
        float maskedcall_test_sum(RVecF values) { return ROOT::VecOps::Sum(values); }
    ''')
    source = '''auto maskedcall_test_wrapper(RVecF values, RVecI conditions) {
        return maskedcall_test_sum(values[conditions]);
    }'''
    original = run_source(source, '<masked call contracts>', call=False)['maskedcall_test_wrapper']
    optimized = compiled(source, ['values', 'conditions'], original)
    assert isinstance(optimized, _Batched)
    offsets = np.array([0, 3, 5, 7])
    values = Jagged(np.array([11, 23, 47, 2, 3, 19, 29], dtype='f4'), offsets)
    conditions = Jagged(np.array([1, 0, 1, 0, 1, 1, 0], dtype='i4'), offsets)
    return original, optimized, (values, conditions)


def _same(prepared):
    original, optimized, columns = prepared
    expected = entrywise(original)(*columns)
    actual = optimized(*columns)
    assert actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()
    return actual


def test_declared_root_functions_actually_use_batched_argument_selection(prepared):
    original, optimized, columns = prepared
    assert masknative.callee(original, 'maskedcall_test_sum', True) is not None
    assert masknative.accepts(columns, rvec._made)
    entered = []
    previous = sys.getprofile()
    def observe(frame, event, result):
        if event == 'call' and frame.f_code.co_name == 'invoke':
            if frame.f_code.co_filename.endswith('/rdf/maskedcall.py'):
                entered.append(frame.f_code)
    try:
        sys.setprofile(observe)
        result = optimized(*columns)
    finally:
        sys.setprofile(previous)
    assert result.tolist() == [58., 3., 19.]
    assert len(entered) == 3


def test_prefiltering_keeps_the_scalar_function_and_random_stream_in_entry_order(prepared):
    original, optimized, columns = prepared
    ROOT.gInterpreter.Declare('''
        float maskedcall_test_sum(RVecF values) {
            return ROOT::VecOps::Sum(values) + gRandom->Gaus();
        }
    ''')
    ROOT.gRandom.SetSeed(4357)
    expected = entrywise(original)(*columns)
    after_expected = ROOT.gRandom.Rndm()
    ROOT.gRandom.SetSeed(4357)
    actual = optimized(*columns)
    assert actual.tobytes() == expected.tobytes()
    assert ROOT.gRandom.Rndm() == after_expected


@pytest.mark.parametrize('name', ['__init__', '_construct', 'assign'])
def test_custom_vector_constructors_keep_their_original_calls(prepared, monkeypatch, name):
    original, optimized, columns = prepared
    method = getattr(stl._Vector, name)
    calls = []
    def track(self, *args):
        calls.append(type(self))
        return method(self, *args)
    monkeypatch.setattr(stl._Vector, name, track)
    expected = entrywise(original)(*columns)
    recorded = calls[:]
    calls.clear()
    actual = optimized(*columns)
    assert actual.tobytes() == expected.tobytes()
    assert calls == recorded and calls


def test_custom_mask_indexing_keeps_its_original_calls(prepared, monkeypatch):
    original, optimized, columns = prepared
    method = rvec._RVec.__getitem__
    calls = []
    def track(self, index):
        calls.append(type(index))
        return method(self, index)
    monkeypatch.setattr(rvec._RVec, '__getitem__', track)
    expected = entrywise(original)(*columns)
    recorded = calls[:]
    calls.clear()
    actual = optimized(*columns)
    assert actual.tobytes() == expected.tobytes()
    assert calls == recorded and calls


@pytest.mark.parametrize('name', ['asarray', 'zeros'])
def test_custom_numpy_construction_keeps_its_original_calls(prepared, monkeypatch, name):
    original, optimized, columns = prepared
    function = getattr(np, name)
    calls = []
    def track(*args, **kwargs):
        calls.append(1)
        return function(*args, **kwargs)
    monkeypatch.setattr(np, name, track)
    expected = entrywise(original)(*columns)
    recorded = len(calls)
    calls.clear()
    actual = optimized(*columns)
    observed = len(calls)
    assert actual.tobytes() == expected.tobytes()
    assert observed == recorded and observed


def test_custom_template_factories_keep_their_original_calls(prepared, monkeypatch):
    original, optimized, columns = prepared
    factory = rvec.RVec
    calls = []
    class Factory:
        def __getitem__(self, dtype):
            calls.append(dtype.str)
            return factory[dtype]
    monkeypatch.setattr(rvec, 'RVec', Factory())
    expected = entrywise(original)(*columns)
    recorded = calls[:]
    calls.clear()
    actual = optimized(*columns)
    assert actual.tobytes() == expected.tobytes()
    assert calls == recorded and calls


def test_changed_cpp_type_mappings_keep_their_existing_conversion(prepared, monkeypatch):
    monkeypatch.setitem(stl.CPP_NAMES, 'float', 'double')
    assert _same(prepared).tolist() == [58., 3., 19.]


@pytest.mark.parametrize('name', ['__getattr__', 'namespace', '_used'])
def test_custom_root_lookup_keeps_its_original_callbacks(prepared, monkeypatch, name):
    from xrdroot.cint.runtime.root import RootProxy
    original, optimized, columns = prepared
    method = getattr(RootProxy, name)
    calls = []
    def track(self, *args):
        calls.append(args)
        return method(self, *args)
    monkeypatch.setattr(RootProxy, name, track)
    expected = entrywise(original)(*columns)
    recorded = calls[:]
    calls.clear()
    actual = optimized(*columns)
    assert actual.tobytes() == expected.tobytes()
    assert calls == recorded


def test_replacing_the_declared_function_preserves_python_callbacks(prepared, monkeypatch):
    original, optimized, columns = prepared
    calls = []
    def changed(values):
        calls.append(list(values))
        return sum(values) + 1
    monkeypatch.setattr(ROOT, 'maskedcall_test_sum', changed)
    expected = entrywise(original)(*columns)
    recorded = calls[:]
    calls.clear()
    actual = optimized(*columns)
    assert actual.tobytes() == expected.tobytes()
    assert calls == recorded


def test_replacing_the_outer_function_keeps_its_new_body(prepared):
    original, optimized, columns = prepared
    def changed(values, conditions):
        return 123
    original.__code__ = changed.__code__
    assert optimized(*columns).tolist() == [123, 123, 123]


def test_mismatched_rows_preserve_the_original_exception_and_prefix_of_calls(prepared):
    original, optimized, (values, _) = prepared
    conditions = Jagged(np.ones(8, dtype='i4'), np.array([0, 3, 6, 8]))
    with pytest.raises(TypeError) as expected:
        entrywise(original)(values, conditions)
    with pytest.raises(TypeError) as actual:
        optimized(values, conditions)
    assert str(actual.value) == str(expected.value)


def test_noncompact_rows_keep_the_original_slicing_behavior(prepared):
    original, optimized, (values, conditions) = prepared
    a = Jagged(np.concatenate([[-99], values.content]).astype('f4'), values.offsets + 1)
    b = Jagged(np.concatenate([[0], conditions.content]).astype('i4'), conditions.offsets + 1)
    expected = entrywise(original)(a, b)
    assert optimized(a, b).tobytes() == expected.tobytes()


def test_replaced_masking_kernels_are_not_invoked_by_the_optimization(prepared, monkeypatch):
    kernels = importlib.import_module('xrdroot.rdf.kernels')
    def foreign(*args):
        raise AssertionError('Argument batching invoked a replaced kernel.')
    monkeypatch.setattr(kernels, 'masked', foreign)
    assert _same(prepared).tolist() == [58., 3., 19.]


def test_readonly_vector_parameters_keep_their_existing_results(prepared):
    _, _, columns = prepared
    source = '''auto readonly_masked_call(const RVecF &values, RVecI conditions) {
        return maskedcall_test_sum(values[conditions]);
    }'''
    original = run_source(source, '<readonly masked call>', call=False)['readonly_masked_call']
    function = compiled(source, ['values', 'conditions'], original) or entrywise(original)
    assert function(*columns).tolist() == [58., 3., 19.]


def test_constant_indices_keep_their_scalar_values(prepared):
    _, _, (values, _) = prepared
    ROOT.gInterpreter.Declare('float maskedcall_identity(float value) { return value; }')
    source = '''auto first_value(RVecF values) { return maskedcall_identity(values[0]); }'''
    original = run_source(source, '<constant index call>', call=False)['first_value']
    function = compiled(source, ['values'], original) or entrywise(original)
    assert function(values).tolist() == [11., 2., 19.]


def test_a_function_can_change_a_global_condition_between_entries():
    ROOT.gInterpreter.Declare('''
        RVecI shared_call_condition = {1, 0, 1};
        float changing_mask(RVecF values) {
            shared_call_condition[0] = 0;
            return ROOT::VecOps::Sum(values);
        }
    ''')
    source = '''auto changing_selection(RVecF values) {
        return changing_mask(values[shared_call_condition]);
    }'''
    original = run_source(source, '<mutable global conditions>', call=False)['changing_selection']
    function = compiled(source, ['values'], original) or entrywise(original)
    values = Jagged(np.array([1, 2, 3, 4, 5, 6], dtype='f4'), np.array([0, 3, 6]))
    assert function(values).tolist() == [4., 6.]


def test_an_outer_float_return_keeps_its_own_rounding(prepared):
    _, _, columns = prepared
    ROOT.gInterpreter.Declare('''
        double masked_double_sum(RVecF values) { return ROOT::VecOps::Sum(values) + 1e-9; }
    ''')
    source = '''float rounded_selection(RVecF values, RVecI conditions) {
        return masked_double_sum(values[conditions]);
    }'''
    original = run_source(source, '<outer return rounding>', call=False)['rounded_selection']
    function = compiled(source, ['values', 'conditions'], original) or entrywise(original)
    assert function(*columns).tolist() == [58., 3., 19.]


def test_a_nullary_declared_call_still_runs_once_per_entry():
    ROOT.gInterpreter.Declare('''
        int masked_call_counter = 0;
        int next_masked_call() { return ++masked_call_counter; }
    ''')
    source = 'auto counted_call() { return next_masked_call(); }'
    original = run_source(source, '<nullary declared call>', call=False)['counted_call']
    function = compiled(source, [], original) or entrywise(original, counted=True)
    assert function(np.arange(3)).tolist() == [1, 2, 3]


def test_callable_objects_are_used_without_reading_their_metadata(prepared, monkeypatch):
    class Callback:
        @property
        def __module__(self):
            raise AssertionError('The optimizer read callback metadata.')

        def __call__(self, values):
            return sum(values)
    monkeypatch.setattr(ROOT, 'maskedcall_test_sum', Callback())
    assert _same(prepared).tolist() == [58., 3., 19.]


def test_foreign_function_module_metadata_is_never_compared(prepared, monkeypatch):
    class Foreign:
        def __eq__(self, other):
            raise AssertionError('The optimizer ran foreign metadata equality.')
    monkeypatch.setattr(ROOT.maskedcall_test_sum, '__module__', Foreign())
    assert _same(prepared).tolist() == [58., 3., 19.]


def test_python_decorators_keep_their_original_callback_behavior(prepared, monkeypatch):
    import functools
    original = ROOT.maskedcall_test_sum
    calls = []
    @functools.wraps(original)
    def decorated(values):
        calls.append(list(values))
        return original(values)
    monkeypatch.setattr(ROOT, 'maskedcall_test_sum', decorated)
    assert _same(prepared).tolist() == [58., 3., 19.]
    assert calls == [[11., 47.], [3.], [19.]] * 2


def test_custom_jagged_iteration_keeps_its_original_callbacks(prepared, monkeypatch):
    method = Jagged.__iter__
    calls = []
    def track(self):
        calls.append(1)
        return method(self)
    monkeypatch.setattr(Jagged, '__iter__', track)
    assert _same(prepared).tolist() == [58., 3., 19.]
    assert len(calls) == 4


def test_a_custom_template_cache_keeps_its_original_lookups(prepared, monkeypatch):
    calls = []
    class Cache(dict):
        def __getitem__(self, key):
            calls.append(key)
            return super().__getitem__(key)
    original, optimized, columns = prepared
    monkeypatch.setattr(rvec.RVec, '_made', Cache(rvec.RVec._made))
    expected = entrywise(original)(*columns)
    recorded = calls[:]
    calls.clear()
    assert optimized(*columns).tobytes() == expected.tobytes()
    assert calls == recorded and calls


@pytest.mark.parametrize('name', ['_data', '_size'])
def test_custom_vector_storage_keeps_its_original_accesses(prepared, monkeypatch, name):
    original, optimized, columns = prepared
    descriptor = stl._Vector.__dict__[name]
    calls = []
    class Storage:
        def __get__(self, instance, owner):
            if instance is None:
                return self
            calls.append('get')
            return descriptor.__get__(instance, owner)

        def __set__(self, instance, value):
            calls.append('set')
            descriptor.__set__(instance, value)
    monkeypatch.setattr(stl._Vector, name, Storage())
    expected = entrywise(original)(*columns)
    recorded = calls[:]
    calls.clear()
    assert optimized(*columns).tobytes() == expected.tobytes()
    assert calls == recorded and calls


@pytest.mark.parametrize('name', ['_bound', '_resolved'])
def test_custom_root_storage_keeps_its_original_accesses(prepared, monkeypatch, name):
    from xrdroot.cint.runtime.root import ROOT as proxy
    from xrdroot.cint.runtime.root import RootProxy
    original, optimized, columns = prepared
    value = vars(proxy)[name]
    calls = []
    def read(self):
        calls.append(name)
        return value
    monkeypatch.setattr(RootProxy, name, property(read), raising=False)
    expected = entrywise(original)(*columns)
    recorded = calls[:]
    calls.clear()
    assert optimized(*columns).tobytes() == expected.tobytes()
    assert calls == recorded and calls


def test_extra_root_dictionary_entries_do_not_change_a_call(prepared, monkeypatch):
    from xrdroot.cint.runtime.root import ROOT as proxy
    monkeypatch.setitem(vars(proxy), 42, 'ignored metadata')
    assert _same(prepared).tolist() == [58., 3., 19.]


def test_unknown_root_cache_keys_keep_the_original_lookup(prepared, monkeypatch):
    from xrdroot.cint.runtime.root import ROOT as proxy
    monkeypatch.setitem(proxy._resolved, (42,), 'ignored cache entry')
    assert _same(prepared).tolist() == [58., 3., 19.]


def test_readonly_namespace_stacks_keep_the_original_lookup(prepared, monkeypatch):
    from xrdroot.cint.runtime.root import ROOT as proxy
    monkeypatch.setattr(proxy, '_bound', (ROOT,))
    assert _same(prepared).tolist() == [58., 3., 19.]


def test_a_bound_namespace_does_not_need_the_default_module_name(prepared, monkeypatch):
    runtime = importlib.import_module('xrdroot.cint.runtime.root')
    monkeypatch.setattr(runtime.ROOT, '_bound', [ROOT])
    monkeypatch.setattr(runtime, 'DEFAULT', object())
    assert _same(prepared).tolist() == [58., 3., 19.]


def test_foreign_namespace_objects_keep_the_original_lookup(prepared, monkeypatch):
    import types

    from xrdroot.cint.runtime.root import ROOT as proxy
    namespace = types.SimpleNamespace(
        maskedcall_test_sum=ROOT.maskedcall_test_sum, VecOps=ROOT.VecOps)
    monkeypatch.setattr(proxy, '_bound', [namespace])
    assert _same(prepared).tolist() == [58., 3., 19.]


def test_a_replaced_vector_maker_keeps_all_original_calls(prepared, monkeypatch):
    original, optimized, columns = prepared
    make = rvec._made
    calls = []
    class Maker:
        def __call__(self, values):
            calls.append(1)
            return make(values)
    monkeypatch.setattr(rvec, '_made', Maker())
    expected = entrywise(original)(*columns)
    recorded = calls[:]
    calls.clear()
    assert optimized(*columns).tobytes() == expected.tobytes()
    assert calls == recorded and calls


def test_jagged_subclasses_keep_their_row_iteration(prepared):
    original, optimized, (values, conditions) = prepared
    class Rows(Jagged):
        pass
    given = (Rows(values.content, values.offsets), conditions)
    assert optimized(*given).tobytes() == entrywise(original)(*given).tobytes()


def test_array_subclasses_keep_their_row_accesses(prepared):
    original, optimized, (values, conditions) = prepared
    calls = []
    class Content(np.ndarray):
        def __getitem__(self, index):
            calls.append(index)
            return super().__getitem__(index)
    values.content = values.content.view(Content)
    expected = entrywise(original)(values, conditions)
    recorded = calls[:]
    calls.clear()
    assert optimized(values, conditions).tobytes() == expected.tobytes()
    assert calls == recorded and calls


def test_custom_template_attribute_access_is_preserved(prepared, monkeypatch):
    original, optimized, columns = prepared
    method = stl._Template.__getattribute__
    calls = []
    def read(self, name):
        calls.append(name)
        return method(self, name)
    monkeypatch.setattr(stl._Template, '__getattribute__', read)
    expected = entrywise(original)(*columns)
    recorded = calls[:]
    calls.clear()
    assert optimized(*columns).tobytes() == expected.tobytes()
    assert calls == recorded and calls


def test_noninteger_template_arities_keep_existing_conversion(prepared, monkeypatch):
    monkeypatch.setattr(rvec.RVec, '_arity', True)
    assert _same(prepared).tolist() == [58., 3., 19.]


def test_cached_callable_constructors_keep_their_calls(prepared, monkeypatch):
    original, optimized, columns = prepared
    kind = rvec.RVec['float']
    calls = []
    def construct(*args):
        calls.append(1)
        return kind(*args)
    monkeypatch.setitem(rvec.RVec._made, ('float',), construct)
    expected = entrywise(original)(*columns)
    recorded = calls[:]
    calls.clear()
    assert optimized(*columns).tobytes() == expected.tobytes()
    assert calls == recorded and calls


def test_cached_vector_subclasses_keep_their_behavior(prepared, monkeypatch):
    kind = rvec.RVec['float']
    class Vector(kind):
        pass
    monkeypatch.setitem(rvec.RVec._made, ('float',), Vector)
    assert _same(prepared).tolist() == [58., 3., 19.]


def test_cached_vectors_with_string_dtypes_keep_their_conversion(prepared, monkeypatch):
    monkeypatch.setattr(rvec.RVec['float'], 'dtype', 'float32')
    assert _same(prepared).tolist() == [58., 3., 19.]


def test_functions_in_the_same_translation_keep_their_live_binding(prepared):
    _, _, columns = prepared
    source = '''
        float local_masked_sum(RVecF values) { return ROOT::VecOps::Sum(values); }
        auto local_masked_selection(RVecF values, RVecI conditions) {
            return local_masked_sum(values[conditions]);
        }
    '''
    original = run_source(source, '<same translation calls>', call=False)['local_masked_selection']
    wrapper = source[source.index('auto local_masked_selection'):]
    function = compiled(wrapper, ['values', 'conditions'], original)
    assert isinstance(function, _Batched)
    assert masknative.callee(original, 'local_masked_sum', False) is not None
    assert function(*columns).tolist() == [58., 3., 19.]
    original.__globals__['local_masked_sum'] = lambda values: sum(values) + 2
    assert function(*columns).tolist() == [60., 5., 21.]


def test_a_replaced_root_proxy_keeps_its_original_attributes(prepared, monkeypatch):
    from types import SimpleNamespace
    original, _, _ = prepared
    replacement = SimpleNamespace(maskedcall_test_sum=ROOT.maskedcall_test_sum)
    monkeypatch.setitem(original.__globals__, 'ROOT', replacement)
    assert _same(prepared).tolist() == [58., 3., 19.]


@pytest.mark.parametrize('vector', ['RVec<float>', 'RVec<TString>', 'std::vector<float>'])
def test_templated_collection_arguments_keep_their_original_dispatch(prepared, vector):
    _, _, columns = prepared
    source = f'''auto templated_masked_selection({vector} values, RVecI conditions) {{
        return maskedcall_test_sum(values[conditions]);
    }}'''
    made = run_source(source, '<templated mask calls>', call=False)
    original = made['templated_masked_selection']
    function = compiled(source, ['values', 'conditions'], original) or entrywise(original)
    assert function(*columns).tolist() == [58., 3., 19.]


def test_foreign_metaclasses_are_rejected_without_metadata_callbacks():
    class Metadata(type):
        def __getattribute__(self, name):
            raise AssertionError('A native guard invoked foreign metaclass code.')
    class Foreign(metaclass=Metadata):
        pass
    value = object.__new__(Foreign)
    assert masknative._dispatcher_type(value) is None
    assert not masknative._dictionary(Foreign)


def test_statements_before_a_masked_call_keep_their_per_entry_effects(prepared):
    _, _, columns = prepared
    ROOT.gInterpreter.Declare('int masked_prelude_counter = 0;')
    source = '''auto counted_selection(RVecF values, RVecI conditions) {
        ++masked_prelude_counter;
        return maskedcall_test_sum(values[conditions]);
    }'''
    original = run_source(source, '<masked prelude>', call=False)['counted_selection']
    assert compiled(source, ['values', 'conditions'], original) is None
    assert entrywise(original)(*columns).tolist() == [58., 3., 19.]
    assert ROOT.masked_prelude_counter == 3


def test_unselected_collection_arguments_keep_their_full_values(prepared):
    _, _, (values, _) = prepared
    source = 'auto full_collection(RVecF values) { return maskedcall_test_sum(values); }'
    original = run_source(source, '<full collection argument>', call=False)['full_collection']
    assert compiled(source, ['values'], original) is None
    assert entrywise(original)(values).tolist() == [81., 5., 48.]


def test_a_replaced_entrywise_maker_keeps_its_per_row_calls(prepared, monkeypatch):
    from xrdroot.pyroot.rdf import entrywise as wrappers

    original, optimized, columns = prepared
    make = wrappers._made
    calls = []
    def maker(values):
        calls.append(np.asarray(values).tolist())
        return make(values)
    monkeypatch.setattr(wrappers, '_made', maker)
    expected = entrywise(original)(*columns)
    recorded = calls[:]
    calls.clear()
    assert optimized(*columns).tobytes() == expected.tobytes()
    assert calls == recorded and calls


def test_a_replaced_rvec_array_conversion_keeps_its_calls(prepared, monkeypatch):
    original, optimized, columns = prepared
    previous = rvec._RVec.__array__
    calls = []
    def convert(self, *args, **kwargs):
        calls.append(self.dtype)
        return previous(self, *args, **kwargs)
    monkeypatch.setattr(rvec._RVec, '__array__', convert)
    expected = entrywise(original)(*columns)
    recorded = calls[:]
    calls.clear()
    assert optimized(*columns).tobytes() == expected.tobytes()
    assert calls == recorded and calls


def test_custom_jagged_construction_keeps_its_scalar_calls(prepared, monkeypatch):
    original, optimized, columns = prepared
    previous = Jagged.__init__
    calls = []
    def construct(self, content, offsets):
        calls.append(len(content))
        return previous(self, content, offsets)
    monkeypatch.setattr(Jagged, '__init__', construct)
    expected = entrywise(original)(*columns)
    recorded = calls[:]
    calls.clear()
    assert optimized(*columns).tobytes() == expected.tobytes()
    assert calls == recorded and calls
