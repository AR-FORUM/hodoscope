"""Tests for hodoscope I/O: JSON round-trip, encoding, grouping."""

import json

import numpy as np
import pytest

from hodoscope.io import (
    decode_embedding,
    encode_embedding,
    filter_summaries,
    group_summaries,
    group_summaries_from_list,
    read_analysis_json,
    write_analysis_json,
)


class TestEncoding:
    def test_encode_decode_roundtrip(self):
        """Encode float32 array to base85 and back, verify exact match."""
        original = np.array([1.0, -2.5, 3.14, 0.0, -1e-6], dtype=np.float32)
        encoded = encode_embedding(original)
        decoded = decode_embedding(encoded)
        np.testing.assert_array_equal(original, decoded)

    def test_encode_decode_large(self):
        """Round-trip a 3072-dim embedding."""
        rng = np.random.RandomState(123)
        original = rng.randn(3072).astype(np.float32)
        encoded = encode_embedding(original)
        decoded = decode_embedding(encoded)
        np.testing.assert_array_equal(original, decoded)

    def test_encode_returns_string(self):
        arr = np.array([1.0, 2.0], dtype=np.float32)
        result = encode_embedding(arr)
        assert isinstance(result, str)

    def test_decode_returns_float32(self):
        arr = np.array([1.0, 2.0], dtype=np.float32)
        result = decode_embedding(encode_embedding(arr))
        assert result.dtype == np.float32


class TestWriteRead:
    def test_write_read_roundtrip(self, tmp_path):
        """Write a full analysis JSON, read it back, verify all fields."""
        rng = np.random.RandomState(42)
        summaries = [
            {
                "trajectory_id": "traj_1",
                "turn_id": 3,
                "summary": "Update assertion to match output",
                "action_text": "TOOL_CALL: edit file.py",
                "embedding": rng.randn(64).astype(np.float32),
                "metadata": {"score": 1.0, "epoch": 1},
            },
            {
                "trajectory_id": "traj_2",
                "turn_id": 5,
                "summary": "Search for method definition",
                "action_text": "TOOL_CALL: grep pattern",
                "embedding": rng.randn(64).astype(np.float32),
                "metadata": {"score": 0.0, "epoch": 2},
            },
        ]
        fields = {"model": "gpt-5", "category": "oneoff", "env": "test"}
        path = tmp_path / "output.hodoscope.json"

        write_analysis_json(
            path=path,
            summaries=summaries,
            fields=fields,
            source="test_source.eval",
            embedding_dimensionality=64,
        )

        assert path.exists()

        doc = read_analysis_json(path)
        assert doc["version"] == 1
        assert doc["source"] == "test_source.eval"
        assert doc["fields"] == fields
        assert doc["embedding_dimensionality"] == 64
        assert len(doc["summaries"]) == 2

        # Verify embeddings round-tripped
        for i, s in enumerate(doc["summaries"]):
            np.testing.assert_array_almost_equal(
                s["embedding"], summaries[i]["embedding"], decimal=6
            )
            assert s["trajectory_id"] == summaries[i]["trajectory_id"]
            assert s["turn_id"] == summaries[i]["turn_id"]
            assert s["summary"] == summaries[i]["summary"]
            assert s["metadata"] == summaries[i]["metadata"]

    def test_write_with_list_embedding(self, tmp_path):
        """Embeddings passed as plain lists should be handled."""
        summaries = [{
            "trajectory_id": "t1",
            "turn_id": 1,
            "summary": "test",
            "action_text": "test",
            "embedding": [1.0, 2.0, 3.0],
            "metadata": {},
        }]
        path = tmp_path / "list_emb.hodoscope.json"
        write_analysis_json(path, summaries, {}, "test")

        doc = read_analysis_json(path)
        np.testing.assert_array_almost_equal(
            doc["summaries"][0]["embedding"], [1.0, 2.0, 3.0]
        )

    def test_write_with_none_embedding(self, tmp_path):
        """Summaries with None embeddings should be preserved."""
        summaries = [{
            "trajectory_id": "t1",
            "turn_id": 1,
            "summary": "[error: timeout]",
            "action_text": "test",
            "embedding": None,
            "metadata": {},
        }]
        path = tmp_path / "none_emb.hodoscope.json"
        write_analysis_json(path, summaries, {}, "test")

        doc = read_analysis_json(path)
        assert doc["summaries"][0]["embedding"] is None

    def test_json_is_human_readable(self, tmp_path):
        """The JSON file should be indented and readable."""
        summaries = [{
            "trajectory_id": "t1",
            "turn_id": 1,
            "summary": "test",
            "action_text": "test",
            "embedding": np.zeros(4, dtype=np.float32),
            "metadata": {"score": 1.0},
        }]
        path = tmp_path / "readable.hodoscope.json"
        write_analysis_json(path, summaries, {"model": "test"}, "test")

        text = path.read_text()
        # Should have indentation (not a single line)
        assert text.count("\n") > 5
        # Fields should be readable
        assert '"model": "test"' in text


class TestGroupSummaries:
    def test_group_by_file_field(self, fake_analysis_json):
        """Group by a file-level field (model)."""
        path, doc = fake_analysis_json
        from hodoscope.io import read_analysis_json
        loaded = read_analysis_json(path)
        grouped = group_summaries([loaded], group_by="model")

        assert "test-model" in grouped
        assert len(grouped["test-model"]) == 20

    def test_group_by_metadata_field(self, fake_analysis_json):
        """Group by a per-summary metadata field (score)."""
        path, _ = fake_analysis_json
        from hodoscope.io import read_analysis_json
        loaded = read_analysis_json(path)
        grouped = group_summaries([loaded], group_by="score")

        assert "1.0" in grouped
        assert "0.0" in grouped
        assert len(grouped["1.0"]) == 10
        assert len(grouped["0.0"]) == 10

    def test_group_by_default_is_model(self, fake_analysis_json):
        """Default group_by should be 'model'."""
        path, _ = fake_analysis_json
        from hodoscope.io import read_analysis_json
        loaded = read_analysis_json(path)
        grouped = group_summaries([loaded])

        assert "test-model" in grouped

    def test_group_across_files(self, fake_analysis_json_pair):
        """Group summaries across multiple files by model."""
        path_a, path_b = fake_analysis_json_pair
        from hodoscope.io import read_analysis_json
        doc_a = read_analysis_json(path_a)
        doc_b = read_analysis_json(path_b)

        grouped = group_summaries([doc_a, doc_b], group_by="model")
        assert "model-alpha" in grouped
        assert "model-beta" in grouped
        assert len(grouped["model-alpha"]) == 15
        assert len(grouped["model-beta"]) == 15

    def test_group_unknown_field(self, fake_analysis_json):
        """Grouping by a nonexistent field should fall back to 'unknown'."""
        path, _ = fake_analysis_json
        from hodoscope.io import read_analysis_json
        loaded = read_analysis_json(path)
        grouped = group_summaries([loaded], group_by="nonexistent")

        assert "unknown" in grouped
        assert len(grouped["unknown"]) == 20

    def test_skips_no_embedding(self, tmp_path):
        """Summaries without embeddings should be skipped."""
        doc = {
            "version": 1,
            "created_at": "2026-01-01",
            "source": "test",
            "fields": {"model": "m"},
            "embedding_model": "test",
            "embedding_dimensionality": 4,
            "summaries": [
                {
                    "trajectory_id": "t1",
                    "turn_id": 1,
                    "summary": "ok",
                    "action_text": "x",
                    "embedding": None,
                    "metadata": {},
                },
            ],
        }
        grouped = group_summaries([doc], group_by="model")
        assert len(grouped) == 0


class TestFilterSummaries:
    def test_filter_summaries_basic(self):
        """filter_summaries with a lambda keeps matching items."""
        summaries = [
            {"metadata": {"score": 1.0}, "summary": "a"},
            {"metadata": {"score": 0.0}, "summary": "b"},
            {"metadata": {"score": 1.0}, "summary": "c"},
        ]
        result = filter_summaries(summaries, lambda s: s["metadata"]["score"] == 1.0)
        assert len(result) == 2
        assert all(s["metadata"]["score"] == 1.0 for s in result)

    def test_filter_summaries_empty_result(self):
        """filter_summaries returns empty list when nothing matches."""
        summaries = [{"metadata": {"score": 0.0}}]
        result = filter_summaries(summaries, lambda s: s["metadata"]["score"] == 1.0)
        assert result == []

    def test_group_summaries_from_list_with_filter(self):
        """group_summaries_from_list with filter applies before grouping."""
        rng = np.random.RandomState(99)
        summaries = [
            {
                "metadata": {"score": 1.0, "model": "a"},
                "embedding": rng.randn(4).astype(np.float32),
            },
            {
                "metadata": {"score": 0.0, "model": "a"},
                "embedding": rng.randn(4).astype(np.float32),
            },
            {
                "metadata": {"score": 1.0, "model": "b"},
                "embedding": rng.randn(4).astype(np.float32),
            },
        ]
        grouped = group_summaries_from_list(
            summaries,
            group_by="model",
            filter=lambda s: s["metadata"]["score"] == 1.0,
        )
        assert "a" in grouped
        assert "b" in grouped
        assert len(grouped["a"]) == 1
        assert len(grouped["b"]) == 1

    def test_group_summaries_with_filter(self, fake_analysis_json):
        """group_summaries with filter callable reduces results."""
        path, _ = fake_analysis_json
        from hodoscope.io import read_analysis_json
        loaded = read_analysis_json(path)
        grouped = group_summaries(
            [loaded],
            group_by="score",
            filter=lambda s: s["metadata"]["score"] == 1.0,
        )
        assert "1.0" in grouped
        assert "0.0" not in grouped
        assert len(grouped["1.0"]) == 10
