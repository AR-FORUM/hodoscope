"""Tests for the analyze pipeline (end-to-end, requires API keys)."""

import json

import pytest

from hodoscope.io import read_analysis_json

# Mark all tests in this module as slow (they call LLM + embedding APIs)
pytestmark = pytest.mark.slow


class TestAnalyzeSingleEval:
    def test_analyze_produces_output(self, sample_eval_path, tmp_path):
        """Run analyze on a single .eval file, verify output structure."""
        from hodoscope.pipeline import analyze

        output_path = tmp_path / "output.hodoscope.json"
        results = analyze(
            sources=(str(sample_eval_path),),
            output=str(output_path),
            limit=2,  # Only process 2 trajectories to keep test fast
        )

        assert len(results) == 1
        assert output_path.exists()

        doc = read_analysis_json(output_path)
        assert doc["version"] == 1
        assert doc["fields"]["model"] == "gpt-4o-mini"
        assert "summaries" in doc
        assert len(doc["summaries"]) > 0

        # Each summary should have required keys
        for s in doc["summaries"]:
            assert "trajectory_id" in s
            assert "turn_id" in s
            assert "summary" in s
            assert "action_text" in s
            assert "embedding" in s
            assert "metadata" in s

    def test_default_output_name(self, sample_eval_path):
        """Without -o, output should be {source}.hodoscope.json."""
        from hodoscope.pipeline import analyze

        expected_output = sample_eval_path.with_suffix(".hodoscope.json")
        try:
            results = analyze(
                sources=(str(sample_eval_path),),
                limit=2,
            )
            assert len(results) == 1
            assert results[0] == expected_output
            assert expected_output.exists()
        finally:
            expected_output.unlink(missing_ok=True)


class TestAnalyzeBatchDir:
    def test_batch_from_directory(self, sample_evals_dir, tmp_path):
        """Run analyze on a directory of .eval files, verify multiple outputs."""
        from hodoscope.pipeline import analyze

        # Use save_samples so output defaults next to source
        results = analyze(
            sources=(str(sample_evals_dir),),
            limit=2,
        )

        # Should produce one output per .eval file
        assert len(results) == 2

        for path in results:
            assert path.exists()
            doc = read_analysis_json(path)
            assert doc["version"] == 1
            assert len(doc["summaries"]) > 0

        # Clean up
        for path in results:
            path.unlink(missing_ok=True)


class TestAnalyzeCustomFields:
    def test_custom_fields_in_output(self, sample_eval_path, tmp_path):
        """Custom --field values should appear in output JSON fields."""
        from hodoscope.pipeline import analyze

        output_path = tmp_path / "custom.hodoscope.json"
        analyze(
            sources=(str(sample_eval_path),),
            output=str(output_path),
            fields=["env=test", "run_id=42"],
            limit=2,
        )

        doc = read_analysis_json(output_path)
        assert doc["fields"]["env"] == "test"
        assert doc["fields"]["run_id"] == "42"
        # Auto-detected fields should still be present
        assert "model" in doc["fields"]


class TestAnalyzeSaveSamples:
    def test_save_samples(self, sample_eval_path, tmp_path):
        """--save-samples should save extracted trajectory JSONs."""
        from hodoscope.pipeline import analyze

        samples_dir = tmp_path / "saved_samples"
        output_path = tmp_path / "out.hodoscope.json"

        analyze(
            sources=(str(sample_eval_path),),
            output=str(output_path),
            save_samples=str(samples_dir),
            limit=3,
        )

        # Should have created sample files
        assert samples_dir.exists()
        json_files = list(samples_dir.rglob("*.json"))
        assert len(json_files) > 0

        # Each should be valid trajectory JSON
        for jf in json_files:
            with open(jf) as f:
                traj = json.load(f)
            assert "id" in traj
            assert "messages" in traj
