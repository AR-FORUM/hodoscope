"""Tests for visualization from analysis JSONs (uses fake embeddings, no API calls)."""

import pytest

from hodoscope.io import group_summaries, read_analysis_json


class TestVizSingleFile:
    def test_viz_single_file_produces_html(self, fake_analysis_json, tmp_path):
        """Load one analysis JSON, group by model, verify HTML output."""
        from hodoscope.visualization import visualize_action_summaries

        path, _ = fake_analysis_json
        doc = read_analysis_json(path)
        grouped = group_summaries([doc], group_by="model")

        output_file = tmp_path / "viz_output.html"
        visualize_action_summaries(grouped, output_file, methods=["pca"])

        assert output_file.exists()


class TestVizMultiFile:
    def test_multi_file_group_by_model(self, fake_analysis_json_pair, tmp_path):
        """Load two JSONs, group by model, verify visualization."""
        from hodoscope.visualization import visualize_action_summaries

        path_a, path_b = fake_analysis_json_pair
        doc_a = read_analysis_json(path_a)
        doc_b = read_analysis_json(path_b)

        grouped = group_summaries([doc_a, doc_b], group_by="model")
        assert len(grouped) == 2

        output_file = tmp_path / "viz_multi.html"
        visualize_action_summaries(grouped, output_file, methods=["pca"])

        assert output_file.exists()

    def test_multi_method(self, fake_analysis_json_pair, tmp_path):
        """Multiple methods should produce a single HTML with method switcher."""
        from hodoscope.visualization import visualize_action_summaries

        path_a, path_b = fake_analysis_json_pair
        doc_a = read_analysis_json(path_a)
        doc_b = read_analysis_json(path_b)

        grouped = group_summaries([doc_a, doc_b], group_by="model")

        output_file = tmp_path / "viz_multi_method.html"
        visualize_action_summaries(grouped, output_file, methods=["pca", "tsne"])

        assert output_file.exists()

        # Verify HTML contains both method names
        content = output_file.read_text()
        assert "PCA" in content
        assert "t-SNE" in content


class TestGroupByScore:
    def test_group_by_score(self, fake_analysis_json):
        """Group by score field, verify dict keys match score values."""
        path, _ = fake_analysis_json
        doc = read_analysis_json(path)
        grouped = group_summaries([doc], group_by="score")

        assert "1.0" in grouped
        assert "0.0" in grouped
        # Score=1.0 for first 10, score=0.0 for last 10
        assert len(grouped["1.0"]) == 10
        assert len(grouped["0.0"]) == 10


class TestVizPipeline:
    def test_viz_function(self, fake_analysis_json, tmp_path):
        """Test the viz() pipeline function end-to-end."""
        from hodoscope.pipeline import viz

        path, _ = fake_analysis_json
        output_file = tmp_path / "viz_pipeline.html"

        viz(
            sources=(str(path),),
            group_by="model",
            proj=["pca"],
            output_file=str(output_file),
        )

        assert output_file.exists()

    def test_viz_with_filter(self, fake_analysis_json, tmp_path):
        """Test viz() with a filter callable narrows results."""
        from hodoscope.pipeline import viz

        path, _ = fake_analysis_json
        output_file = tmp_path / "viz_filtered.html"

        viz(
            sources=(str(path),),
            group_by="score",
            proj=["pca"],
            output_file=str(output_file),
            filter=lambda s: s.get("metadata", {}).get("score") == 1.0,
        )

        assert output_file.exists()
