"""Tests for the public Python API (no API keys needed)."""

import json

import numpy as np
import pytest

from tests.conftest import SAMPLE_EVALS_DIR
from hodoscope.io import group_summaries_from_list
from hodoscope.pipeline import extract_actions, load_eval, load_trajectory_dir, HodoscopeError


# ---------------------------------------------------------------------------
# extract_actions
# ---------------------------------------------------------------------------

class TestExtractActions:
    def test_extracts_assistant_tool_calls(self):
        """Assistant turns with tool_calls should become actions."""
        messages = [
            {"role": "user", "content": "Fix the bug"},
            {"role": "assistant", "tool_calls": [
                {"function": "edit_file", "arguments": {"path": "foo.py"}},
            ]},
            {"role": "tool", "content": "File edited"},
            {"role": "assistant", "tool_calls": [
                {"function": "run_tests", "arguments": {}},
            ]},
            {"role": "tool", "content": "Tests passed"},
        ]
        actions = extract_actions(messages)
        assert len(actions) == 2
        assert actions[0]["role"] == "assistant"
        assert "edit_file" in actions[0]["action_text"]
        assert "run_tests" in actions[1]["action_text"]

    def test_returns_empty_for_no_assistant_turns(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "World"},
        ]
        actions = extract_actions(messages)
        assert actions == []

    def test_returns_empty_for_empty_messages(self):
        assert extract_actions([]) == []

    def test_feedback_merged_into_action(self):
        """Tool responses should be merged into the preceding assistant turn."""
        messages = [
            {"role": "assistant", "tool_calls": [
                {"function": "read_file", "arguments": {"path": "x.py"}},
            ]},
            {"role": "tool", "content": "file contents here"},
        ]
        actions = extract_actions(messages)
        assert len(actions) == 1
        assert "file contents here" in actions[0]["action_text"]


# ---------------------------------------------------------------------------
# group_summaries_from_list
# ---------------------------------------------------------------------------

class TestGroupSummariesFromList:
    def _make_summary(self, score=1.0, embedding=True):
        rng = np.random.RandomState(42)
        return {
            "trajectory_id": "t1",
            "turn_id": 1,
            "summary": "test",
            "action_text": "test",
            "embedding": rng.randn(4).astype(np.float32).tolist() if embedding else None,
            "metadata": {"score": score, "epoch": 1},
        }

    def test_group_by_metadata(self):
        summaries = [
            self._make_summary(score=1.0),
            self._make_summary(score=1.0),
            self._make_summary(score=0.0),
        ]
        grouped = group_summaries_from_list(summaries, group_by="score")
        assert "1.0" in grouped
        assert "0.0" in grouped
        assert len(grouped["1.0"]) == 2
        assert len(grouped["0.0"]) == 1

    def test_fallback_to_default_fields(self):
        summaries = [self._make_summary()]
        grouped = group_summaries_from_list(
            summaries, group_by="model", default_fields={"model": "gpt-5"}
        )
        assert "gpt-5" in grouped
        assert len(grouped["gpt-5"]) == 1

    def test_fallback_to_unknown(self):
        summaries = [self._make_summary()]
        grouped = group_summaries_from_list(summaries, group_by="nonexistent")
        assert "unknown" in grouped

    def test_skips_none_embeddings(self):
        summaries = [
            self._make_summary(embedding=True),
            self._make_summary(embedding=False),
        ]
        grouped = group_summaries_from_list(summaries, group_by="score")
        total = sum(len(v) for v in grouped.values())
        assert total == 1

    def test_empty_input(self):
        assert group_summaries_from_list([], group_by="score") == {}


# ---------------------------------------------------------------------------
# load_eval
# ---------------------------------------------------------------------------

class TestLoadEval:
    @pytest.fixture
    def eval_path(self):
        p = SAMPLE_EVALS_DIR / "sample1.eval"
        if not p.exists():
            pytest.skip("sample1.eval not found")
        return p

    def test_load_returns_trajectories_and_fields(self, eval_path):
        trajectories, fields = load_eval(eval_path)
        assert isinstance(trajectories, list)
        assert len(trajectories) > 0
        assert "model" in fields
        assert "category" not in fields

    def test_accepts_str_path(self, eval_path):
        trajectories, fields = load_eval(str(eval_path))
        assert len(trajectories) > 0

    def test_limit(self, eval_path):
        trajectories, _ = load_eval(eval_path, limit=3)
        assert len(trajectories) == 3

    def test_trajectories_have_messages(self, eval_path):
        trajectories, _ = load_eval(eval_path, limit=1)
        assert "messages" in trajectories[0]
        assert len(trajectories[0]["messages"]) > 0

    def test_fields_contain_eval_metadata(self, eval_path):
        """Verify rich eval-level fields: task, dataset_name, solver, accuracy, etc."""
        _, fields = load_eval(eval_path)
        assert "task" in fields
        assert "dataset_name" in fields
        assert "solver" in fields
        assert "accuracy" in fields
        assert fields["task"] == "popularity"
        assert fields["dataset_name"] == "popularity"
        assert fields["accuracy"] == 0.8

    def test_sample_metadata_has_target(self, eval_path):
        """Verify target is extracted into per-trajectory metadata."""
        trajectories, _ = load_eval(eval_path, limit=1)
        meta = trajectories[0]["metadata"]
        assert "target" in meta

    def test_sample_metadata_has_token_usage(self, eval_path):
        """Verify input_tokens, output_tokens, total_tokens in metadata."""
        trajectories, _ = load_eval(eval_path, limit=1)
        meta = trajectories[0]["metadata"]
        assert "input_tokens" in meta
        assert "output_tokens" in meta
        assert "total_tokens" in meta

    def test_sample_metadata_passthrough(self, eval_path):
        """Verify arbitrary sample.metadata keys are passed through."""
        trajectories, _ = load_eval(eval_path, limit=1)
        meta = trajectories[0]["metadata"]
        # sample_evals/sample1.eval has label_confidence in sample metadata
        assert "label_confidence" in meta

    def test_invalid_path_raises(self, tmp_path):
        with pytest.raises(HodoscopeError, match="Not a valid .eval file"):
            load_eval(tmp_path / "nonexistent.eval")


# ---------------------------------------------------------------------------
# load_trajectory_dir
# ---------------------------------------------------------------------------

class TestLoadTrajectoryDir:
    def test_invalid_path_raises(self, tmp_path):
        with pytest.raises(HodoscopeError, match="Not a directory"):
            load_trajectory_dir(tmp_path / "nonexistent")

    def test_loads_from_directory(self, tmp_path):
        """Create a minimal trajectory dir and load it."""
        traj = {
            "id": "test_1",
            "model": "test-model",
            "metadata": {"score": 1.0, "epoch": 1},
            "messages": [
                {"role": "user", "content": "do something"},
                {"role": "assistant", "tool_calls": [
                    {"function": "test_fn", "arguments": {}},
                ]},
            ],
        }
        with open(tmp_path / "traj1.json", "w") as f:
            json.dump(traj, f)

        trajectories, fields = load_trajectory_dir(tmp_path)
        assert len(trajectories) == 1
        assert fields["model"] == "test-model"
        assert "category" not in fields

    def test_accepts_str_path(self, tmp_path):
        traj = {"id": "t", "messages": [], "metadata": {}}
        with open(tmp_path / "t.json", "w") as f:
            json.dump(traj, f)
        trajectories, _ = load_trajectory_dir(str(tmp_path))
        assert len(trajectories) == 1
