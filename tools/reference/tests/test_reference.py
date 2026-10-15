from pathlib import Path

import numpy as np
import pytest

from qwen35_ref import safetensors
from qwen35_ref.cli import DEFAULT_CHECKPOINT

needs_weights = pytest.mark.skipif(
    not (DEFAULT_CHECKPOINT / "config.json").exists(), reason="run scripts/download_weights.sh first"
)

IDS = [760, 6511, 314, 9338, 369, 11751, 13, 198]  # "The capital of France is Paris.\n"


@pytest.fixture(scope="module")
def model():
    from qwen35_ref.model import Qwen35

    return Qwen35(DEFAULT_CHECKPOINT)


def test_safetensors_round_trip(tmp_path: Path):
    tensors = {"a": np.arange(6, dtype=np.float32).reshape(2, 3), "ids": np.array([1, 2, 3], np.int32)}
    safetensors.save(tmp_path / "x.safetensors", tensors, {"k": "v"})
    f = safetensors.SafetensorsFile(tmp_path / "x.safetensors")
    assert f.metadata == {"k": "v"}
    np.testing.assert_array_equal(f.get("a"), tensors["a"])
    np.testing.assert_array_equal(f.get("ids"), tensors["ids"])


@needs_weights
def test_decode_matches_prefill(model):
    """Token-by-token decode must reproduce the full-sequence logits (KV cache, conv and recurrent state)."""
    full = model.forward(IDS, model.new_state())
    state = model.new_state()
    steps = np.concatenate([model.forward([t], state) for t in IDS])
    np.testing.assert_allclose(steps, full, rtol=1e-3, atol=1e-3)


@needs_weights
def test_chunked_prefill_matches_prefill(model):
    full = model.forward(IDS, model.new_state())
    state = model.new_state()
    parts = np.concatenate([model.forward(IDS[:3], state), model.forward(IDS[3:], state)])
    np.testing.assert_allclose(parts, full, rtol=1e-3, atol=1e-3)


@needs_weights
def test_knows_the_capital_of_france(model):
    logits = model.forward(IDS[:5], model.new_state())
    assert int(np.argmax(logits[-1])) == 11751  # " Paris"
