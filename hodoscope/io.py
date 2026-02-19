"""
I/O for hodoscope analysis JSON files.

Handles reading/writing the single-file output format with base85-encoded
embeddings, and grouping summaries by arbitrary metadata fields for visualization.
"""

import base64
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

from .config import DEFAULT_EMBED_DIM, DEFAULT_EMBEDDING_MODEL, DEFAULT_GROUP_BY

# Current schema version
FORMAT_VERSION = 1


def encode_embedding(arr: np.ndarray) -> str:
    """Encode a float32 numpy array to an RFC 1924 base85 string.

    Uses b85encode which avoids ", ', \\, and , — safe for JSON embedding.
    ~25% more compact than base64.
    """
    return base64.b85encode(arr.astype(np.float32).tobytes()).decode("ascii")


def decode_embedding(b85: str) -> np.ndarray:
    """Decode an RFC 1924 base85 string back to a float32 numpy array."""
    raw = base64.b85decode(b85)
    return np.frombuffer(raw, dtype=np.float32).copy()


def write_analysis_json(
    path: Path,
    summaries: list[dict],
    fields: dict[str, Any],
    source: str,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_dimensionality: int | None = DEFAULT_EMBED_DIM,
) -> Path:
    """Write analysis results to a .hodoscope.json file.

    Args:
        path: Output file path.
        summaries: List of summary dicts. Each must have at least:
            trajectory_id, turn_id, summary, action_text, embedding (np.ndarray or None),
            metadata (dict).
        fields: File-level metadata (model, category, custom fields).
        source: Original source path/identifier.
        embedding_model: Name of the embedding model used.
        embedding_dimensionality: Dimensionality of embeddings.

    Returns:
        The path written to.
    """
    path = Path(path)

    serialized_summaries = []
    for s in summaries:
        entry = {
            "trajectory_id": s["trajectory_id"],
            "turn_id": s["turn_id"],
            "summary": s["summary"],
            "action_text": s["action_text"],
            "task_context": s.get("task_context", ""),
            "embedding": None,
            "metadata": s.get("metadata", {}),
        }
        emb = s.get("embedding")
        if emb is not None:
            if isinstance(emb, np.ndarray):
                entry["embedding"] = encode_embedding(emb)
            elif isinstance(emb, list):
                entry["embedding"] = encode_embedding(np.array(emb, dtype=np.float32))
            elif isinstance(emb, str):
                # Already encoded
                entry["embedding"] = emb
        serialized_summaries.append(entry)

    doc = {
        "version": FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "fields": fields,
        "embedding_model": embedding_model,
        "embedding_dimensionality": embedding_dimensionality,
        "summaries": serialized_summaries,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)

    return path


def read_analysis_json(path: Path) -> dict:
    """Read a .hodoscope.json file and decode embeddings to numpy arrays.

    Returns:
        Dict with keys: version, created_at, source, fields,
        embedding_model, embedding_dimensionality, summaries.
        Each summary's 'embedding' is a np.ndarray (or None).
    """
    path = Path(path)
    with open(path) as f:
        doc = json.load(f)

    for s in doc.get("summaries", []):
        emb = s.get("embedding")
        if emb is not None and isinstance(emb, str):
            s["embedding"] = decode_embedding(emb)
        elif emb is None:
            s["embedding"] = None

    return doc


def filter_summaries(
    summaries: list[dict],
    predicate: Callable[[dict], bool],
) -> list[dict]:
    """Filter summaries using a predicate. Returns matching subset."""
    return [s for s in summaries if predicate(s)]


def group_summaries_from_list(
    summaries: list[dict],
    group_by: str = DEFAULT_GROUP_BY,
    fallbacks: list[dict] | None = None,
    # Deprecated alias kept for backwards compat
    default_fields: dict[str, Any] | None = None,
    filter: Callable[[dict], bool] | None = None,
) -> dict[str, list[dict]]:
    """Group a flat list of summaries by a metadata field.

    Resolution order for group_by value:
    1. Per-summary ``metadata[group_by]``
    2. Each dict in *fallbacks*, in order
    3. ``"unknown"``

    Args:
        summaries: List of summary dicts.
        group_by: Field name to group by.
        fallbacks: Ordered list of fallback dicts to check after per-summary metadata.
        default_fields: Deprecated — equivalent to ``fallbacks=[default_fields]``.

    Returns:
        Dict mapping group labels to lists of summary dicts.
        Summaries without embeddings are skipped.
    """
    if fallbacks is None:
        fallbacks = [default_fields] if default_fields else []

    if filter is not None:
        summaries = filter_summaries(summaries, filter)

    groups: dict[str, list[dict]] = {}
    n_dropped = 0

    for s in summaries:
        if s.get("embedding") is None:
            n_dropped += 1
            continue

        meta = s.get("metadata", {})
        value = meta.get(group_by)
        if value is None:
            for fb in fallbacks:
                value = fb.get(group_by)
                if value is not None:
                    break
        if value is None:
            value = "unknown"

        label = str(value)
        groups.setdefault(label, []).append(s)

    if n_dropped:
        logger.warning(
            "Dropped %d/%d summaries with missing embeddings during grouping",
            n_dropped, n_dropped + sum(len(v) for v in groups.values()),
        )

    return groups


def group_summaries(
    analysis_docs: list[dict],
    group_by: str | None = None,
    filter: Callable[[dict], bool] | None = None,
) -> dict[str, list[dict]]:
    """Group summaries from one or more analysis docs by a metadata field.

    Resolution order for group_by field:
    1. Per-summary metadata (e.g., metadata.score)
    2. File-level fields (e.g., fields.model)
    3. Top-level doc keys (e.g., source, embedding_model)

    If group_by is None, defaults to "model".

    Args:
        analysis_docs: List of loaded analysis JSON dicts (from read_analysis_json).
        group_by: Field name to group by.

    Returns:
        Dict mapping group labels to lists of summary dicts.
        Each summary dict has: trajectory_id, turn_id, summary, action_text,
        embedding (np.ndarray), metadata.
    """
    if group_by is None:
        group_by = DEFAULT_GROUP_BY

    groups: dict[str, list[dict]] = {}

    for doc in analysis_docs:
        file_fields = doc.get("fields", {})
        doc_fallback = {group_by: doc.get(group_by)}

        doc_groups = group_summaries_from_list(
            doc.get("summaries", []),
            group_by=group_by,
            fallbacks=[file_fields, doc_fallback],
            filter=filter,
        )
        for label, sums in doc_groups.items():
            groups.setdefault(label, []).extend(sums)

    return groups
