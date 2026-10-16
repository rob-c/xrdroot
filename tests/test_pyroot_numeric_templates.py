"""Numeric template shortcuts preserve the normalization and live specialization cache."""
import numpy as np
import pytest

from xrdroot.pyroot import stl


@pytest.mark.parametrize('name', list(stl.CPP_NAMES))
def test_numeric_templates_build_once_from_the_original_dtype(name):
    given = np.dtype(name, metadata={'description': 'native column'}).newbyteorder('>')
    calls = []

    def build(kind):
        calls.append(kind)
        return type('Item', (), {})

    template = stl._Template('numeric', 1, build)
    first = template[given]
    assert template[given] is first
    assert template[np.dtype(name)] is first
    assert len(calls) == 1
    assert calls[0] is given


def test_numeric_template_keys_follow_changes_to_the_cpp_spelling_table(monkeypatch):
    template = stl._Template('numeric', 1, lambda kind: type('Item', (), {}))
    given = np.dtype('float64')
    first = template[given]
    monkeypatch.setitem(stl.CPP_NAMES, 'float64', 'changed double')
    second = template[given]
    assert second is not first
    monkeypatch.setitem(stl.CPP_NAMES, 'float64', 'double')
    assert template[given] is first


def test_numeric_templates_keep_the_existing_specialization_cache_live():
    template = stl._Template('numeric', 1, lambda kind: type('Item', (), {}))
    given = np.dtype('float64')
    template[given]
    template._made[('double',)] = int
    assert template[given] is int


def test_numeric_templates_still_call_the_current_builder_on_a_cache_miss():
    template = stl._Template('numeric', 1, lambda kind: int)
    assert template[np.dtype('float64')] is int
    template._build = lambda kind: str
    assert template[np.dtype('float32')] is str


def test_numeric_arguments_to_multi_argument_templates_keep_the_arity_error():
    template = stl._Template('numeric', 2, lambda *kinds: int)
    with pytest.raises(TypeError, match='takes 2 template arguments'):
        template[np.dtype('float64')]


def test_unknown_numeric_dtypes_keep_the_original_key_error():
    template = stl._Template('numeric', 1, lambda kind: int)
    with pytest.raises(KeyError) as error:
        template[np.dtype('complex64')]
    assert error.value.args == ('complex64',)


def test_foreign_class_properties_are_only_read_by_the_original_conversion():
    calls = []

    class Probe:
        @property
        def __class__(self):
            calls.append('class')
            return int

    template = stl._Template('numeric', 1, lambda kind: int)
    with pytest.raises(TypeError, match='is not a type an STL container can hold'):
        template[Probe()]
    assert calls == ['class'] * 4


def test_foreign_metaclass_equality_is_not_entered_by_dtype_eligibility():
    class Meta(type):
        def __eq__(self, other):
            raise AssertionError('Dtype eligibility compared an arbitrary metaclass.')

    class Probe(metaclass=Meta):
        pass

    template = stl._Template('numeric', 1, lambda kind: int)
    with pytest.raises(TypeError, match='is not a type an STL container can hold'):
        template[Probe()]
