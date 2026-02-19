"""Shared fixtures for hodoscope tests."""

import json
from pathlib import Path

import numpy as np
import pytest

SAMPLE_EVALS_DIR = Path(__file__).parent / "sample_evals"


@pytest.fixture
def sample_eval_path():
    """Path to sample1.eval test file."""
    return SAMPLE_EVALS_DIR / "sample1.eval"


@pytest.fixture
def sample_evals_dir():
    """Path to sample_evals directory containing .eval files."""
    return SAMPLE_EVALS_DIR


@pytest.fixture
def fake_analysis_json(tmp_path):
    """Create a fake .hodoscope.json file with fake embeddings for viz tests.

    Returns (path, doc) tuple.
    """
    from hodoscope.io import encode_embedding

    rng = np.random.RandomState(42)
    dim = 64  # Small dims for testing

    summaries = []
    for i in range(20):
        score = 1.0 if i < 10 else 0.0
        entry = {
            "trajectory_id": f"sample_{i}",
            "turn_id": 1,
            "summary": f"Test action summary {i}",
            "action_text": f"TOOL_CALL: test_function arg_{i}",
            "embedding": encode_embedding(rng.randn(dim).astype(np.float32)),
            "metadata": {
                "score": score,
                "epoch": 1,
                "instance_id": f"test_{i}",
            },
        }
        if i in (0, 5, 15):
            entry["task_context"] = f"System prompt for test {i}\n\n---\n\nUser task for test {i}"
        summaries.append(entry)

    doc = {
        "version": 1,
        "created_at": "2026-01-01T00:00:00+00:00",
        "source": "test",
        "fields": {"model": "test-model"},
        "embedding_model": "test-model",
        "embedding_dimensionality": dim,
        "summaries": summaries,
    }

    path = tmp_path / "test_output.hodoscope.json"
    with open(path, "w") as f:
        json.dump(doc, f)

    return path, doc


@pytest.fixture
def fake_analysis_json_pair(tmp_path):
    """Create two fake .hodoscope.json files with different models.

    Returns (path_a, path_b) tuple.
    """
    from hodoscope.io import encode_embedding

    rng = np.random.RandomState(42)
    dim = 64

    paths = []
    for model_name in ["model-alpha", "model-beta"]:
        summaries = []
        for i in range(15):
            summaries.append({
                "trajectory_id": f"{model_name}_sample_{i}",
                "turn_id": 1,
                "summary": f"Action by {model_name} #{i}",
                "action_text": f"TOOL_CALL: func arg_{i}",
                "embedding": encode_embedding(rng.randn(dim).astype(np.float32)),
                "metadata": {
                    "score": float(i % 2),
                    "epoch": 1,
                },
            })

        doc = {
            "version": 1,
            "created_at": "2026-01-01T00:00:00+00:00",
            "source": f"test_{model_name}",
            "fields": {"model": model_name},
            "embedding_model": "test-model",
            "embedding_dimensionality": dim,
            "summaries": summaries,
        }

        path = tmp_path / f"{model_name}.hodoscope.json"
        with open(path, "w") as f:
            json.dump(doc, f)
        paths.append(path)

    return tuple(paths)
