"""Tests for FPS-based sampling (uses synthetic clustered data, no API calls)."""

import json

import numpy as np
import pytest

from hodoscope.io import encode_embedding
from hodoscope.sampling import (
    ALL_PLOT_METHODS,
    UNRANKED_SENTINEL,
    PlotData,
    collect_plot_data,
    compute_projection,
    compute_fps_ranks,
    rank_summaries,
)


# ---------------------------------------------------------------------------
# Fixtures: clustered synthetic data
# ---------------------------------------------------------------------------

def _make_clustered_summaries(
    n_per_cluster: int = 20,
    cluster_centers: list[tuple[float, ...]] | None = None,
    cluster_spread: float = 0.3,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Build summaries with embeddings that form well-separated clusters.

    Each cluster center is a point in 16-D space. Points are Gaussian-
    distributed around these centers. Two groups ("alpha", "beta") each
    get half of every cluster so FPS can be tested within a single group.

    Returns:
        {group_label: [summary_dicts]} with numpy array embeddings.
    """
    rng = np.random.RandomState(seed)
    dim = 16

    if cluster_centers is None:
        # 4 well-separated clusters at corners of a hypercube
        cluster_centers = [
            tuple([5.0] * (dim // 2) + [0.0] * (dim // 2)),
            tuple([0.0] * (dim // 2) + [5.0] * (dim // 2)),
            tuple([-5.0] * (dim // 2) + [0.0] * (dim // 2)),
            tuple([0.0] * (dim // 2) + [-5.0] * (dim // 2)),
        ]

    summaries_alpha = []
    summaries_beta = []

    for c_idx, center in enumerate(cluster_centers):
        center_arr = np.array(center, dtype=np.float32)
        points = rng.randn(n_per_cluster, dim).astype(np.float32) * cluster_spread + center_arr

        for p_idx, emb in enumerate(points):
            s = {
                "trajectory_id": f"traj_c{c_idx}_p{p_idx}",
                "turn_id": 1,
                "summary": f"Cluster {c_idx} point {p_idx}",
                "action_text": f"action_c{c_idx}_p{p_idx}",
                "embedding": emb,
                "metadata": {"score": float(c_idx % 2), "cluster": c_idx},
            }
            if p_idx < n_per_cluster // 2:
                summaries_alpha.append(s)
            else:
                summaries_beta.append(s)

    return {"alpha": summaries_alpha, "beta": summaries_beta}


@pytest.fixture
def clustered_summaries():
    """Grouped summaries with 4 well-separated clusters."""
    return _make_clustered_summaries()


@pytest.fixture
def clustered_analysis_json(tmp_path):
    """Write a .hodoscope.json with clustered embeddings. Returns (path, doc)."""
    grouped = _make_clustered_summaries()
    all_summaries = []
    for sums in grouped.values():
        for s in sums:
            s_enc = dict(s)
            s_enc["embedding"] = encode_embedding(s["embedding"])
            all_summaries.append(s_enc)

    doc = {
        "version": 1,
        "created_at": "2026-01-01T00:00:00+00:00",
        "source": "test_clustered",
        "fields": {"model": "test-model"},
        "embedding_model": "test-model",
        "embedding_dimensionality": 16,
        "summaries": all_summaries,
    }
    path = tmp_path / "clustered.hodoscope.json"
    with open(path, "w") as f:
        json.dump(doc, f)
    return path, doc


# ---------------------------------------------------------------------------
# compute_projection
# ---------------------------------------------------------------------------

class TestComputeProjection:
    def test_pca_returns_correct_shape(self, clustered_summaries):
        data = collect_plot_data(clustered_summaries)
        X_2d = compute_projection(data.X, 'pca')
        assert X_2d.shape == (len(data.X), 2)

    def test_unknown_method_raises(self, clustered_summaries):
        data = collect_plot_data(clustered_summaries)
        with pytest.raises(ValueError, match="Unknown projection method"):
            compute_projection(data.X, 'bogus')


# ---------------------------------------------------------------------------
# compute_fps_ranks
# ---------------------------------------------------------------------------

class TestComputeFpsRanks:
    def test_returns_correct_length(self, clustered_summaries):
        data = collect_plot_data(clustered_summaries)
        X_2d = compute_projection(data.X, 'pca')
        ranks = compute_fps_ranks(X_2d, data.labels, len(data.type_names))
        assert len(ranks) == len(data.X)

    def test_each_category_has_rank_zero(self, clustered_summaries):
        data = collect_plot_data(clustered_summaries)
        X_2d = compute_projection(data.X, 'pca')
        ranks = compute_fps_ranks(X_2d, data.labels, len(data.type_names))

        for cat_idx in range(len(data.type_names)):
            cat_ranks = [
                ranks[i] for i in range(len(ranks)) if data.labels[i] == cat_idx
            ]
            assert 0 in cat_ranks, f"Category {cat_idx} has no rank-0 point"

    def test_fps_samples_from_different_clusters_early(self, clustered_summaries):
        """The first few FPS picks per group should come from different clusters.

        With 4 well-separated clusters and ~10 points per cluster per group,
        FPS should visit all 4 clusters within the first 4 picks.
        """
        data = collect_plot_data(clustered_summaries)
        X_2d = compute_projection(data.X, 'pca')
        ranks = compute_fps_ranks(X_2d, data.labels, len(data.type_names))

        for cat_idx, group_name in enumerate(data.type_names):
            # Collect (rank, cluster_id) for this group
            ranked_points = []
            for i in range(len(ranks)):
                if data.labels[i] == cat_idx and ranks[i] < UNRANKED_SENTINEL:
                    # Parse cluster from trajectory_id: "traj_c{cluster}_p{idx}"
                    traj_id = data.trajectory_ids[i]
                    cluster_id = int(traj_id.split("_c")[1].split("_p")[0])
                    ranked_points.append((ranks[i], cluster_id))

            ranked_points.sort()

            # The first 4 picks should cover all 4 clusters
            first_n = 4
            early_clusters = {cluster for _, cluster in ranked_points[:first_n]}
            assert len(early_clusters) == 4, (
                f"Group '{group_name}': expected 4 clusters in first {first_n} picks, "
                f"got {len(early_clusters)}: {early_clusters}. "
                f"First picks: {ranked_points[:first_n]}"
            )


# ---------------------------------------------------------------------------
# rank_summaries
# ---------------------------------------------------------------------------

class TestBalancedProjection:
    """Group-balanced projection gives small groups equal spatial spread."""

    def test_small_group_not_collapsed(self):
        """With 200 vs 10 points, the 10-point group should still have
        reasonable spatial spread when labels are passed (balanced projection).
        """
        rng = np.random.RandomState(123)
        dim = 16

        # Two well-separated clusters
        center_a = np.zeros(dim, dtype=np.float32)
        center_b = np.full(dim, 5.0, dtype=np.float32)

        n_large = 200
        n_small = 10
        X_large = rng.randn(n_large, dim).astype(np.float32) * 0.5 + center_a
        X_small = rng.randn(n_small, dim).astype(np.float32) * 0.5 + center_b

        X = np.vstack([X_large, X_small])
        labels = np.array([0] * n_large + [1] * n_small)

        X_2d_balanced = compute_projection(X, 'pca', labels=labels)
        X_2d_unbalanced = compute_projection(X, 'pca')

        def _spread(X_2d, mask):
            pts = X_2d[mask]
            return np.std(pts, axis=0).mean()

        large_mask = labels == 0
        small_mask = labels == 1

        # With balanced projection, the small group's spread relative to
        # the large group should be higher than without balancing
        ratio_balanced = _spread(X_2d_balanced, small_mask) / _spread(X_2d_balanced, large_mask)
        ratio_unbalanced = _spread(X_2d_unbalanced, small_mask) / _spread(X_2d_unbalanced, large_mask)

        assert ratio_balanced >= ratio_unbalanced * 0.9, (
            f"Balanced ratio {ratio_balanced:.3f} should be >= unbalanced ratio "
            f"{ratio_unbalanced:.3f} (small group should not be more collapsed)"
        )

    def test_no_labels_unchanged(self):
        """Without labels, compute_projection behaves exactly as before."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 8).astype(np.float32)
        X_2d_a = compute_projection(X, 'pca')
        X_2d_b = compute_projection(X, 'pca', labels=None)
        np.testing.assert_array_equal(X_2d_a, X_2d_b)

    def test_equal_groups_unchanged(self):
        """With equal-sized groups, balancing is a no-op — same output shape."""
        rng = np.random.RandomState(42)
        X = rng.randn(40, 8).astype(np.float32)
        labels = np.array([0] * 20 + [1] * 20)
        X_2d = compute_projection(X, 'pca', labels=labels)
        assert X_2d.shape == (40, 2)


class TestRankSummaries:
    def test_output_sorted_by_fps_rank(self, clustered_summaries):
        ranked = rank_summaries(clustered_summaries, method='pca')
        for group, sums in ranked.items():
            ranks = [s['fps_rank'] for s in sums]
            assert ranks == sorted(ranks), f"Group '{group}' not sorted by fps_rank"

    def test_n_truncates(self, clustered_summaries):
        ranked = rank_summaries(clustered_summaries, method='pca', n=3)
        for group, sums in ranked.items():
            assert len(sums) <= 3

    def test_adds_fps_rank_key(self, clustered_summaries):
        ranked = rank_summaries(clustered_summaries, method='pca')
        for group, sums in ranked.items():
            for s in sums:
                assert 'fps_rank' in s
                assert isinstance(s['fps_rank'], int)

    def test_preserves_original_keys(self, clustered_summaries):
        ranked = rank_summaries(clustered_summaries, method='pca', n=2)
        for group, sums in ranked.items():
            for s in sums:
                assert 'trajectory_id' in s
                assert 'summary' in s
                assert 'action_text' in s
                assert 'embedding' in s
                assert 'metadata' in s

    def test_does_not_mutate_input(self, clustered_summaries):
        """rank_summaries should not modify the input dicts."""
        original_keys = {
            group: [set(s.keys()) for s in sums]
            for group, sums in clustered_summaries.items()
        }
        rank_summaries(clustered_summaries, method='pca')
        for group, sums in clustered_summaries.items():
            for i, s in enumerate(sums):
                assert set(s.keys()) == original_keys[group][i]


# ---------------------------------------------------------------------------
# sample() pipeline
# ---------------------------------------------------------------------------

class TestSamplePipeline:
    def test_json_output(self, clustered_analysis_json, tmp_path):
        from hodoscope.pipeline import sample

        path, _ = clustered_analysis_json
        out_path = tmp_path / "sampled.json"

        result = sample(
            sources=(str(path),),
            group_by="model",
            n=5,
            method="pca",
            output=str(out_path),
        )

        assert out_path.exists()
        with open(out_path) as f:
            data = json.load(f)

        assert data["method"] == "pca"
        assert data["n_per_group"] == 5
        assert "test-model" in data["groups"]
        samples = data["groups"]["test-model"]["samples"]
        assert len(samples) <= 5
        # JSON output should not contain embeddings
        for s in samples:
            assert "embedding" not in s
            assert "rank" in s
            assert "summary" in s

    def test_terminal_output_grouped(self, clustered_analysis_json):
        """Default terminal output groups by label."""
        from unittest.mock import patch
        from hodoscope.pipeline import sample

        path, _ = clustered_analysis_json

        with patch("click.echo_via_pager") as mock_pager:
            result = sample(
                sources=(str(path),),
                group_by="model",
                n=3,
                method="pca",
            )

        mock_pager.assert_called_once()
        pager_text = mock_pager.call_args[0][0]
        assert "test-model" in pager_text
        assert "===" in pager_text
        assert isinstance(result, dict)

    def test_terminal_output_interleaved(self, clustered_analysis_json):
        """--interleave flag interleaves groups by rank."""
        from unittest.mock import patch
        from hodoscope.pipeline import sample

        path, _ = clustered_analysis_json

        with patch("click.echo_via_pager") as mock_pager:
            result = sample(
                sources=(str(path),),
                group_by="model",
                n=3,
                method="pca",
                interleave=True,
            )

        mock_pager.assert_called_once()
        pager_text = mock_pager.call_args[0][0]
        assert "--- #1 ---" in pager_text
        assert "--- #2 ---" in pager_text
        assert isinstance(result, dict)
