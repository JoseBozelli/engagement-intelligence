import numpy as np
import pytest

from engagement_intelligence.core.seeding import rng_for, stream_key

def _draw(seed, stream, index):
    return rng_for(seed, stream, index).random(5)

def test_same_inputs_give_identical_draws():
    np.testing.assert_array_equal(_draw(7, "latents", 3), _draw(7, "latents", 3))

def test_draws_are_order_invariant():
    forward = {i: _draw(7, "latents", i) for i in range(10)}
    backward = {i: _draw(7, "latents", i) for i in reversed(range(10))}
    for i in range(10):
        np.testing.assert_array_equal(forward[i], backward[i])


@pytest.mark.parametrize("other", [(8, "latents", 3), (7, "noise", 3), (7, "latents", 4)])
def test_any_input_change_changes_draws(other):
    assert not np.array_equal(_draw(*other), _draw(7, "latents", 3))

def test_stream_key_is_stable_and_rejects_empty():
    assert stream_key("latents") == stream_key("latents")
    with pytest.raises(ValueError):
        stream_key("")

def test_negative_inputs_rejected():
    with pytest.raises(ValueError):
        rng_for(-1, "latents", 0)
    with pytest.raises(ValueError):
        rng_for(1, "latents", -1)