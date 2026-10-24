"""Restoration must keep cached classes and their provenance coherent together."""

from __future__ import annotations

import numpy as np

from xrdroot.cppstd import containers, streams, vectorvalue


def test_state_restoration_pairs_template_caches_with_native_vector_registries(
    cppstd_state_context,
):
    original = containers.std.vector["int"]
    factory = containers.TEMPLATES["vector"]
    with cppstd_state_context():
        factory._made.pop(("int",))
        temporary = containers.std.vector["int"]
        assert temporary is not original and vectorvalue.native(temporary)
    assert containers.std.vector["int"] is original and vectorvalue.native(original)
    assert temporary not in vectorvalue._KINDS and not vectorvalue.native(temporary)


def test_state_restoration_preserves_exact_stream_fields_and_numpy_rng_state(cppstd_state_context):
    old = dict(vars(streams.cout))
    state = np.random.get_state()
    local = np.random.RandomState()
    local.set_state(state)
    expected = local.randint(0, 2**20)
    with cppstd_state_context():
        streams.cout.alpha = not old["alpha"]
        streams.cout.width = 17
        np.random.seed(9)
        np.random.random(4)
    assert vars(streams.cout) == old
    assert np.random.randint(0, 2**20) == expected
