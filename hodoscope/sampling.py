"""
Standalone FPS-based sampling for representative action selection.

Extracts the most "interesting" (diverse + density-weighted) summaries per group
using Farthest Point Sampling on 2D projections. Used by both the visualization
module (for interactive flagging) and the `hodoscope sample` CLI command.

Public API:
  - compute_projection(X, method, labels=None) — dim-reduce embeddings to 2D (group-balanced when labels given)
  - compute_bandwidth(X_2d) — Scott's rule bandwidth for KDE
  - compute_kde_densities(X_2d, labels, n_categories, bandwidth) — per-category KDE at all points
  - compute_fps_ranks(X_2d, labels, n_categories, ...) — density-weighted FPS ranking
  - rank_summaries(summaries_by_group, ...) — high-level: group dict in, ranked group dict out
"""

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KernelDensity


ALL_PLOT_METHODS = ['pca', 'tsne', 'umap', 'trimap', 'pacmap']

UNRANKED_SENTINEL = 10**9

SAMPLING_METHOD_DISPLAY_NAMES = {
    'pca': 'PCA',
    'tsne': 't-SNE',
    'umap': 'UMAP',
    'trimap': 'TriMAP',
    'pacmap': 'PaCMAP',
}


@dataclass
class PlotData:
    """Collected and indexed data from summaries_by_type for plotting."""
    X: np.ndarray               # (N, embed_dim) embeddings
    labels: np.ndarray           # (N,) int label indices
    type_names: list             # group names in dict order
    trajectory_ids: list
    trajectory_uuids: list      # disambiguated IDs for navigation (source___id)
    turn_ids: list
    summaries: list
    action_texts: list
    task_contexts: list


def collect_plot_data(summaries_by_type: dict) -> PlotData:
    """Extract plotting arrays from a summaries_by_type dict.

    Iterates groups in dict order, assigns integer labels, and collects
    embeddings + metadata into parallel arrays.
    """
    type_names = list(summaries_by_type.keys())
    label_map = {name: idx for idx, name in enumerate(type_names)}

    all_embeddings = []
    all_labels = []
    all_trajectory_ids = []
    all_trajectory_uuids = []
    all_turn_ids = []
    all_summaries = []
    all_action_texts = []
    all_task_contexts = []

    for type_name, summaries in summaries_by_type.items():
        if summaries is None or len(summaries) == 0:
            continue
        label = label_map[type_name]

        for s in summaries:
            if s.get('embedding') is not None:
                all_embeddings.append(s['embedding'])
                all_labels.append(label)
                traj_id = s['trajectory_id']
                all_trajectory_ids.append(traj_id)
                source_file = s.get('_source_file', '')
                uuid = f"{source_file}___{traj_id}" if source_file else traj_id
                all_trajectory_uuids.append(uuid)
                all_turn_ids.append(s['turn_id'])
                all_summaries.append(s['summary'])
                all_action_texts.append(s['action_text'])
                all_task_contexts.append(s.get('task_context', ''))

    return PlotData(
        X=np.array(all_embeddings),
        labels=np.array(all_labels),
        type_names=type_names,
        trajectory_ids=all_trajectory_ids,
        trajectory_uuids=all_trajectory_uuids,
        turn_ids=all_turn_ids,
        summaries=all_summaries,
        action_texts=all_action_texts,
        task_contexts=all_task_contexts,
    )


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def _balance_groups(X: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, int]:
    """Oversample smaller groups so every group has the same point count.

    Duplicate points get tiny jitter (1e-6 * std) to avoid exact duplicates
    which can break t-SNE perplexity calculations.

    Args:
        X: (N, D) embedding matrix.
        labels: (N,) integer group labels.

    Returns:
        (X_balanced, n_original) where X_balanced has oversampled rows appended
        and n_original == len(X).
    """
    rng = np.random.RandomState(42)
    unique_labels = np.unique(labels)
    counts = {lab: int((labels == lab).sum()) for lab in unique_labels}
    max_count = max(counts.values())

    # Nothing to do if all groups are the same size
    if min(counts.values()) == max_count:
        return X, len(X)

    noise_scale = 1e-6 * np.std(X)
    extra_rows = []
    for lab in unique_labels:
        n = counts[lab]
        deficit = max_count - n
        if deficit == 0:
            continue
        mask = np.where(labels == lab)[0]
        sampled_idx = rng.choice(mask, size=deficit, replace=True)
        noise = rng.randn(deficit, X.shape[1]).astype(X.dtype) * noise_scale
        extra_rows.append(X[sampled_idx] + noise)

    X_balanced = np.vstack([X] + extra_rows)
    return X_balanced, len(X)


def compute_projection(X: np.ndarray, method: str, labels: np.ndarray | None = None) -> np.ndarray:
    """Run a single dim-reduction method. Returns (N, 2) array.

    When ``labels`` is provided, smaller groups are oversampled (with tiny
    noise) to match the largest group before fitting the projection. This
    ensures all groups have equal influence on the layout. Only the original
    N point positions are returned.

    Args:
        X: (N, D) embedding matrix.
        method: One of 'pca', 'tsne', 'umap', 'trimap', 'pacmap'.
        labels: Optional (N,) integer group labels. When given, enables
            group-balanced projection via oversampling.

    Returns:
        (N, 2) numpy array of 2D coordinates.

    Raises:
        ValueError: If method is unknown.
    """
    n_original = len(X)
    if labels is not None:
        X, _ = _balance_groups(X, labels)

    if method == 'pca':
        X_2d = PCA(n_components=2).fit_transform(X)
    elif method == 'tsne':
        perplexity = min(30, len(X) - 1)
        X_2d = TSNE(
            n_components=2, random_state=42,
            perplexity=perplexity, max_iter=1000,
            n_jobs=-1,
        ).fit_transform(X)
    elif method == 'umap':
        import umap
        X_2d = umap.UMAP(
            n_components=2, random_state=42,
            n_neighbors=15, min_dist=0.1,
        ).fit_transform(X)
    elif method == 'trimap':
        import trimap
        X_2d = trimap.TRIMAP(n_dims=2).fit_transform(X)
    elif method == 'pacmap':
        import pacmap
        X_2d = pacmap.PaCMAP(n_components=2, random_state=42).fit_transform(X)
    else:
        raise ValueError(f"Unknown projection method: {method}")

    return X_2d[:n_original]


# ---------------------------------------------------------------------------
# KDE helpers
# ---------------------------------------------------------------------------

def compute_bandwidth(X_2d: np.ndarray) -> float:
    """Compute KDE bandwidth using Scott's rule with a floor of 0.5.

    Args:
        X_2d: (N, 2) array of 2D coordinates.

    Returns:
        Bandwidth value (float).
    """
    n = len(X_2d)
    std_x = np.std(X_2d[:, 0])
    std_y = np.std(X_2d[:, 1])
    bandwidth = 1.06 * min(std_x, std_y) * (n ** (-1 / 5))
    return max(bandwidth, 0.001)


def compute_kde_densities(
    X_2d: np.ndarray,
    labels: np.ndarray,
    n_categories: int,
    bandwidth: float,
) -> list[np.ndarray]:
    """Compute per-category KDE density at every point in X_2d.

    Args:
        X_2d: (N, 2) projected coordinates.
        labels: (N,) integer category labels.
        n_categories: Total number of categories.
        bandwidth: KDE bandwidth.

    Returns:
        List of length n_categories, each a (N,) array of density values.
    """
    point_densities = []
    for label_idx in range(n_categories):
        mask = labels == label_idx
        X_cat = X_2d[mask]
        if len(X_cat) > 0:
            kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
            kde.fit(X_cat)
            density_points = np.exp(kde.score_samples(X_2d))
        else:
            density_points = np.zeros(len(X_2d))
        point_densities.append(density_points)
    return point_densities


# ---------------------------------------------------------------------------
# FPS ranking
# ---------------------------------------------------------------------------

def compute_fps_ranks(
    X_2d: np.ndarray,
    labels: np.ndarray,
    n_categories: int,
    point_densities: list[np.ndarray] | None = None,
    alpha: float = 1.0,
    beta: float = 0.1,
    max_per_group: int = 500,
    bandwidth: float | None = None,
) -> list[int]:
    """Density-weighted Farthest Point Sampling ranking.

    For each category, iteratively selects the point that maximizes
    density_gap * min_distance^alpha, where density_gap measures how
    much more the point belongs to its own category vs others.
    Distances are computed in axis-normalized (unit-variance) 2D space.
    Density gaps are piecewise-linearly scaled with zero mapped to beta:
    negative gaps → [0, beta], positive gaps → [beta, 1].

    Args:
        X_2d: (N, 2) projected coordinates.
        labels: (N,) integer category labels.
        n_categories: Total number of categories.
        point_densities: Optional precomputed per-category KDE densities
            (list of n_categories arrays, each length N). If None, computed
            internally using bandwidth.
        alpha: Distance exponent (higher = more spatial spread). Default 2.0.
        beta: Density gap midpoint — the weight assigned to zero-gap points.
            Negative gaps scale to [0, beta], positive to [beta, 1]. Default 0.2.
        max_per_group: Max points to rank per category. Default 500.
        bandwidth: KDE bandwidth (used only if point_densities is None).

    Returns:
        List of length N with per-point FPS rank (0 = most important).
        Points beyond max_per_group get rank UNRANKED_SENTINEL (10**9).
    """
    N = len(X_2d)
    if point_densities is None:
        if bandwidth is None:
            bandwidth = compute_bandwidth(X_2d)
        point_densities = compute_kde_densities(X_2d, labels, n_categories, bandwidth)

    # Normalize axes by std for isotropic distance computation
    std = np.std(X_2d, axis=0)
    std[std == 0] = 1.0
    X_2d_norm = X_2d / std

    fps_orders = {}
    for cat_idx in range(n_categories):
        mask = labels == cat_idx
        cat_indices = np.where(mask)[0]
        n_cat = len(cat_indices)
        if n_cat == 0:
            continue

        X_cat = X_2d_norm[cat_indices]
        n_others = n_categories - 1

        # Density gap: own density - avg of other categories' density
        density_gaps = np.zeros(n_cat)
        for i, global_idx in enumerate(cat_indices):
            sum_other = sum(
                point_densities[j][global_idx]
                for j in range(n_categories) if j != cat_idx
            )
            avg_other = sum_other / n_others if n_others > 0 else 0
            own_density = point_densities[cat_idx][global_idx]
            density_gaps[i] = own_density - avg_other

        gap_min, gap_max = density_gaps.min(), density_gaps.max()
        if gap_max > gap_min:
            # Piecewise linear: negative gaps → [0, beta], positive → [beta, 1]
            density_gaps_norm = np.empty(n_cat)
            pos = density_gaps >= 0
            if gap_max > 0:
                density_gaps_norm[pos] = beta + (1 - beta) * (density_gaps[pos] / gap_max)
            else:
                density_gaps_norm[pos] = beta
            neg = ~pos
            if gap_min < 0:
                density_gaps_norm[neg] = beta + beta * (density_gaps[neg] / (-gap_min))
            else:
                density_gaps_norm[neg] = beta
        else:
            density_gaps_norm = np.ones(n_cat)

        selected = []
        min_dists = np.full(n_cat, np.inf)

        for rank in range(min(max_per_group, n_cat)):
            if rank == 0:
                scores = density_gaps_norm.copy()
            else:
                scores = density_gaps_norm * (min_dists ** alpha)
            scores[selected] = -np.inf
            best_local = np.argmax(scores)
            selected.append(best_local)
            fps_orders[int(cat_indices[best_local])] = rank

            new_point = X_cat[best_local]
            dists = np.sqrt(np.sum((X_cat - new_point) ** 2, axis=1))
            min_dists = np.minimum(min_dists, dists)

    return [fps_orders.get(i, UNRANKED_SENTINEL) for i in range(N)]


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def rank_summaries(
    summaries_by_group: dict[str, list[dict]],
    method: str = 'tsne',
    n: int | None = None,
    alpha: float = 1.0,
    beta: float = 0.1,
    bandwidth: float | None = None,
) -> dict[str, list[dict]]:
    """Rank summaries by importance within each group using FPS.

    Takes the same ``{label: [summary_dicts]}`` dict that
    ``group_summaries`` / ``group_summaries_from_list`` produce,
    projects embeddings to 2D, runs density-weighted FPS, and returns
    each group sorted by rank (most important first) with an ``fps_rank``
    key added to every dict.

    Args:
        summaries_by_group: Dict mapping group labels to lists of summary
            dicts (must have ``embedding`` key with numpy arrays).
        method: Projection method ('pca', 'tsne', 'umap', 'trimap', 'pacmap').
        n: If given, truncate each group to the top n summaries.
        alpha: FPS distance exponent (higher = more spatial spread).
        beta: Density gap midpoint (weight for zero-gap points). Default 0.2.
        bandwidth: KDE bandwidth. If None, auto-computed.

    Returns:
        Dict with same keys, values sorted by fps_rank ascending.
        Each summary dict gets an ``fps_rank`` int key added.
    """
    data = collect_plot_data(summaries_by_group)
    if len(data.X) == 0:
        return {k: [] for k in summaries_by_group}

    X_2d = compute_projection(data.X, method, labels=data.labels)
    fps_ranks = compute_fps_ranks(
        X_2d, data.labels, len(data.type_names),
        alpha=alpha, beta=beta, bandwidth=bandwidth,
    )

    # Build a mapping from (trajectory_id, turn_id, label_idx) → global index
    # so we can map fps_ranks back to the original summary dicts.
    # We iterate summaries_by_group in the same order as collect_plot_data
    # to maintain alignment.
    result = {}
    global_idx = 0
    for group_label in data.type_names:
        group_summaries_list = summaries_by_group[group_label]
        ranked = []
        for s in group_summaries_list:
            if s.get('embedding') is None:
                continue
            rank = fps_ranks[global_idx]
            s_copy = dict(s)
            s_copy['fps_rank'] = rank
            ranked.append(s_copy)
            global_idx += 1
        ranked.sort(key=lambda x: x['fps_rank'])
        if n is not None:
            ranked = ranked[:n]
        result[group_label] = ranked

    return result
