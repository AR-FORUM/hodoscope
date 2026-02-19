"""
Export raw transcript data from a Docent collection.

Usage:
    hodoscope import docent --collection-id <id>
    hodoscope import docent --collection-id <id> --format trajectory --model-name "my-model"

Requirements:
    pip install docent-python python-dotenv

Environment:
    DOCENT_API_KEY - Your Docent API key
"""

import argparse
import json
import os
import pickle
from pathlib import Path

from docent import Docent
from dotenv import load_dotenv

from .convert_to_trajectory import write_trajectory_files

# Load .env from project root
load_dotenv(Path(__file__).parent.parent.parent / ".env")


def get_client() -> Docent:
    """Initialize Docent client with API key from environment."""
    api_key = os.getenv("DOCENT_API_KEY")
    if not api_key:
        raise ValueError(
            "DOCENT_API_KEY not found. Set it in .env or environment.\n"
            "Get your key at: https://docent.transluce.org/settings/api-keys"
        )
    return Docent(api_key=api_key)


def _decode_json_field(data: dict, field: str, label: str = "") -> None:
    """Decode a binary/string JSON field in place."""
    if data.get(field):
        try:
            if isinstance(data[field], bytes):
                data[field] = json.loads(data[field].decode("utf-8"))
            elif isinstance(data[field], str):
                data[field] = json.loads(data[field])
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"Warning: Could not decode {field} for {label}: {e}")


def export_transcripts(
    collection_id: str,
    output_path: Path,
    limit: int | None = None,
) -> list[dict]:
    """
    Export all transcripts from a collection with agent_run metadata.

    Args:
        collection_id: The Docent collection ID
        output_path: Path to save the pickle file
        limit: Optional limit on number of transcripts

    Returns:
        List of transcript dictionaries with agent_run metadata merged in
    """
    client = get_client()

    transcript_columns = [
        "id", "collection_id", "agent_run_id", "name", "description",
        "transcript_group_id", "messages", "metadata_json", "dict_key", "created_at",
    ]

    query = f"""
    SELECT {', '.join(transcript_columns)}
    FROM transcripts
    ORDER BY created_at DESC
    """
    if limit:
        query += f" LIMIT {limit}"

    print(f"Executing DQL query on collection: {collection_id}")
    result = client.execute_dql(collection_id, query)
    transcripts = client.dql_result_to_dicts(result)
    print(f"Retrieved {len(transcripts)} transcripts")

    for t in transcripts:
        _decode_json_field(t, "messages", f"transcript {t.get('id')}")
        _decode_json_field(t, "metadata_json", f"transcript {t.get('id')}")

    agent_run_ids = list(set(t["agent_run_id"] for t in transcripts if t.get("agent_run_id")))
    if agent_run_ids:
        runs_query = """
        SELECT id, name, description, metadata_json
        FROM agent_runs
        """
        print("Fetching agent_run metadata...")
        runs_result = client.execute_dql(collection_id, runs_query)
        runs = client.dql_result_to_dicts(runs_result)

        runs_by_id = {}
        for r in runs:
            _decode_json_field(r, "metadata_json", f"agent_run {r.get('id')}")
            runs_by_id[r["id"]] = r

        for t in transcripts:
            run_id = t.get("agent_run_id")
            if run_id and run_id in runs_by_id:
                run = runs_by_id[run_id]
                t["agent_run_metadata"] = run.get("metadata_json") or {}
                t["agent_run_name"] = run.get("name")
                t["agent_run_description"] = run.get("description")

        print(f"Merged metadata from {len(runs_by_id)} agent_runs")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(transcripts, f)
    print(f"Saved transcripts to: {output_path}")

    return transcripts


def print_schema(collection_id: str):
    """Print the DQL schema for a collection."""
    client = get_client()
    schema = client.get_dql_schema(collection_id)
    print("\nAvailable tables and columns:")
    print(json.dumps(schema, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Export raw transcript data from a Docent collection"
    )
    parser.add_argument("--collection-id", "-c", required=True, help="Docent collection ID")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output path")
    parser.add_argument("--limit", "-l", type=int, default=None, help="Limit transcripts")
    parser.add_argument("--schema", action="store_true", help="Print schema and exit")
    parser.add_argument("--format", "-f", choices=["pickle", "trajectory"], default="pickle",
                        help="Output format")
    parser.add_argument("--model-name", "-m", help="Model name (required for trajectory format)")
    parser.add_argument("--category", default="original",
                        help="Category (default: original)")

    args = parser.parse_args()

    if args.schema:
        print_schema(args.collection_id)
        return

    if args.format == "trajectory" and not args.model_name:
        parser.error("--model-name is required when using --format trajectory")

    if args.output is None:
        args.output = Path("transcripts.pkl") if args.format == "pickle" else Path("extracted_trajectories_docent")

    if args.format == "trajectory":
        pickle_path = args.output / ".transcripts_temp.pkl"
    else:
        pickle_path = args.output

    transcripts = export_transcripts(
        collection_id=args.collection_id,
        output_path=pickle_path,
        limit=args.limit,
    )

    if args.format == "trajectory" and transcripts:
        print("\nConverting to trajectory format...")
        written = write_trajectory_files(
            transcripts=transcripts,
            output_dir=args.output,
            model_name=args.model_name,
            category=args.category,
        )
        print(f"Wrote {written} trajectory files to: {args.output}/{args.model_name}/{args.category}/samples/")
        pickle_path.unlink()
        print(f"\nTo analyze, run:")
        print(f"  hodoscope run --base-dir {args.output}")
    elif transcripts:
        print(f"\nSample keys: {list(transcripts[0].keys())}")


if __name__ == "__main__":
    main()
