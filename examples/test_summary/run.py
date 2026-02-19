"""
Test summarization quality on hand-crafted action/feedback examples.

Loads examples from examples.json, pipes them through the real parsing pipeline
(turns_from_messages -> merge_consecutive_turns -> extract_actions_from_turns)
to build action_text, then runs summarize_action on each. Embeds summaries
and generates a pairwise cosine similarity heatmap.

Usage:
    python examples/test_summary/run.py
    python examples/test_summary/run.py --model gemini/gemini-2.0-flash
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from hodoscope.config import Config
from hodoscope.parsers import turns_from_messages, merge_consecutive_turns
from hodoscope.core import extract_actions_from_turns, embed_text
from hodoscope.actions import summarize_action


def main():
    # Load env + config first so we get the right default model
    config = Config.from_env()

    parser = argparse.ArgumentParser(description="Test summarization on examples")
    parser.add_argument(
        "--model", default=config.summarize_model,
        help=f"Summarization model (default from env/config: {config.summarize_model})",
    )
    parser.add_argument(
        "--examples", default=Path(__file__).parent / "examples.json",
        help="Path to examples JSON file",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output JSON path (default: results_{model_slug}.json next to examples)",
    )
    args = parser.parse_args()

    examples = json.loads(Path(args.examples).read_text())

    print(f"Model: {args.model}")
    print(f"Embedding model: {config.embedding_model}")
    print(f"Examples: {len(examples)}")
    print("=" * 80)

    results = []
    for ex in examples:
        # Use real pipeline to build action_text
        turns = turns_from_messages(ex["messages"])
        merged = merge_consecutive_turns(turns)
        actions = extract_actions_from_turns(merged)

        if not actions:
            print(f"\n[{ex['name']}] -- no actions extracted, skipping")
            continue

        action_text = actions[0]["action_text"]

        # Summarize
        if args.model == "no_summary":
            summary = action_text
        else:
            summary = summarize_action(
                action_text,
                model=args.model,
                reasoning_effort=config.reasoning_effort,
            )

        # Embed
        embedding = embed_text(
            summary,
            model=config.embedding_model,
            output_dimensionality=config.embed_dim,
            normalize=config.normalize_embeddings,
        )

        results.append({
            "name": ex["name"],
            "action_text": action_text,
            "summary": summary,
            "embedding": embedding,
        })

        print(f"\n### {ex['name']}")
        print(f"Action text ({len(action_text)} chars):")
        preview = action_text[:200] + ("..." if len(action_text) > 200 else "")
        print(f"  {preview}")
        print(f"Summary:")
        print(f"  {summary}")
        embed_status = f"{len(embedding)}d" if embedding else "FAILED"
        print(f"Embedding: {embed_status}")
        print("-" * 80)

    # Filter to results with valid embeddings for the heatmap
    valid = [r for r in results if r["embedding"] is not None]
    names = [r["name"] for r in valid]
    X = np.array([r["embedding"] for r in valid])

    # Pairwise cosine similarity
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    X_normed = X / norms
    sim = np.clip(X_normed @ X_normed.T, 0, 1)

    # Heatmap — scale figure size with number of items
    n = len(names)
    cell_size = 0.8  # inches per cell
    margin = 3.0     # inches for labels / colorbar / title
    side = n * cell_size + margin
    fig, ax = plt.subplots(figsize=(side, side))
    cmap = plt.cm.viridis  # high = bright/yellow, low = dark/purple
    im = ax.imshow(sim, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(names, fontsize=9)
    for i in range(n):
        for j in range(n):
            # white text on dark cells, black on bright
            color = "w" if sim[i, j] < 0.65 else "k"
            ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color=color)
    fig.colorbar(im, ax=ax, label="Cosine Similarity", shrink=0.8)
    ax.set_title(f"Pairwise Cosine Similarity of Summary Embeddings\n({config.embedding_model})",
                 fontsize=12, pad=12)
    fig.tight_layout()

    heatmap_path = Path(args.examples).parent / "cosine_similarity.png"
    fig.savefig(heatmap_path, dpi=150)
    print(f"\nHeatmap saved to {heatmap_path}")

    # Write JSON output (embeddings as lists for JSON serialization)
    if args.output:
        out_path = Path(args.output)
    else:
        slug = args.model.replace("/", "_")
        out_path = Path(args.examples).parent / f"results_{slug}.json"
    output = {
        "model": args.model,
        "embedding_model": config.embedding_model,
        "results": [
            {k: v for k, v in r.items() if k != "embedding"}
            | ({"embedding_dim": len(r["embedding"])} if r["embedding"] else {})
            for r in results
        ],
        "cosine_similarity": sim.tolist(),
        "labels": names,
    }
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Wrote {len(results)} results to {out_path}")


if __name__ == "__main__":
    main()
