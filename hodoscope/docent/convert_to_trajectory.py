"""
Convert Docent transcript data to trajectory format for analysis pipeline.

Outputs the universal trajectory format defined in hodoscope.schema.
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Any


def convert_transcript(transcript: dict) -> dict:
    """
    Convert a single Docent transcript to universal trajectory format.

    Passes through all ``agent_run_metadata`` keys into per-trajectory
    metadata.  Special cases:
    - ``score`` ← ``scores.resolved`` (for grouping / filtering)
    - ``model`` ← ``model_name_or_path`` (set at top-level for field detection)
    - ``created_at`` ← transcript-level ``created_at``

    Args:
        transcript: Docent transcript dictionary

    Returns:
        Trajectory-format dictionary compatible with the analysis pipeline
    """
    metadata = transcript.get("metadata_json") or {}
    run_metadata = transcript.get("agent_run_metadata") or {}

    instance_id = run_metadata.get("instance_id") or metadata.get("instance_id", "")
    run_id = transcript.get("agent_run_id") or transcript.get("id", "")
    if instance_id:
        traj_id = f"{instance_id}_{run_id}"
    else:
        traj_id = transcript.get("name") or run_id

    input_text = metadata.get("input", transcript.get("description", ""))

    # Derive score from scores.resolved
    scores = run_metadata.get("scores") or {}
    score = scores.get("resolved")

    # Build per-trajectory metadata: start with passthrough of all
    # agent_run_metadata keys, then layer on structured fields.
    traj_metadata = dict(run_metadata)
    traj_metadata.update({
        "epoch": 1,
        "sandbox": metadata.get("sandbox", {}),
        "instance_id": instance_id or None,
        "score": score,
        "docent_id": transcript.get("id"),
        "docent_collection_id": transcript.get("collection_id"),
        "docent_agent_run_id": transcript.get("agent_run_id"),
    })

    # Add created_at from transcript if available
    if transcript.get("created_at"):
        traj_metadata["created_at"] = transcript["created_at"]

    result = {
        "id": traj_id,
        "source": "docent",
        "input": input_text,
        "messages": transcript.get("messages") or [],
        "metadata": traj_metadata,
    }

    # Set model at top level so load_docent / load_trajectory_dir can pick
    # it up for file-level fields.
    model = run_metadata.get("model_name_or_path")
    if model:
        result["model"] = model

    return result


def extract_docent_fields(transcripts: list[dict]) -> dict:
    """Extract file-level fields from raw Docent transcripts.

    Analogous to ``extract_eval_fields`` for .eval files.  Inspects the
    first transcript's ``agent_run_metadata`` to pull out fields shared
    across the collection.

    Args:
        transcripts: Raw transcript dicts (before conversion).

    Returns:
        Dict of file-level metadata (e.g. ``model``, ``trajectory_format``).
    """
    if not transcripts:
        return {}

    run_metadata = transcripts[0].get("agent_run_metadata") or {}
    fields: dict[str, Any] = {}

    model = run_metadata.get("model_name_or_path")
    if model:
        fields["model"] = model

    fmt = run_metadata.get("trajectory_format")
    if fmt:
        fields["trajectory_format"] = fmt

    return fields


def write_trajectory_files(
    transcripts: list[dict],
    output_dir: Path,
    model_name: str,
    category: str,
) -> int:
    """
    Write trajectory JSON files in expected directory structure.

    Creates: output_dir/{model_name}/{category}/samples/{id}_epoch_1.json
    """
    samples_dir = output_dir / model_name / category / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for transcript in transcripts:
        trajectory = convert_transcript(transcript)

        safe_id = str(trajectory["id"]).replace("/", "_").replace("\\", "_")
        output_file = samples_dir / f"{safe_id}_epoch_1.json"

        with open(output_file, "w") as f:
            json.dump(trajectory, f, indent=2, default=str)

        written += 1

    return written


def load_transcripts(input_path: Path) -> list[dict]:
    """Load transcripts from pickle file."""
    with open(input_path, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Convert Docent transcripts to trajectory format"
    )
    parser.add_argument("--input", "-i", type=Path, required=True,
                        help="Input pickle file from export_transcripts.py")
    parser.add_argument("--output", "-o", type=Path, default=Path("extracted_trajectories_docent"),
                        help="Output directory (default: extracted_trajectories_docent)")
    parser.add_argument("--model-name", "-m", required=True,
                        help="Model name for directory structure")
    parser.add_argument("--category", "-c", default="original",
                        help="Category (default: original)")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1

    print(f"Loading transcripts from: {args.input}")
    transcripts = load_transcripts(args.input)
    print(f"Loaded {len(transcripts)} transcripts")

    if not transcripts:
        print("No transcripts to convert")
        return 0

    print(f"Writing to: {args.output}/{args.model_name}/{args.category}/samples/")
    written = write_trajectory_files(
        transcripts=transcripts,
        output_dir=args.output,
        model_name=args.model_name,
        category=args.category,
    )

    print(f"Converted {written} transcripts to trajectory format")
    print(f"\nTo analyze, run:")
    print(f"  hodoscope analyze --base-dir {args.output}")
    return 0


if __name__ == "__main__":
    exit(main())
