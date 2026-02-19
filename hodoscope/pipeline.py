"""
Pipeline orchestration for hodoscope v2.

Public API (composable building blocks):
  - load_eval() — load trajectories from .eval file
  - load_trajectory_dir() — load trajectories from directory
  - load_docent() — load trajectories from Docent collection
  - process_trajectories() — summarize + embed trajectories
  - extract_actions() — messages → actions (no LLM calls)

High-level orchestrators (used by CLI):
  - analyze() — end-to-end: sources → .hodoscope.json files
  - viz() — load analysis JSONs and generate visualizations
  - show_info() — print metadata for analysis files
"""

import json
import os
import random
import tempfile
import zipfile
from pathlib import Path

from tqdm import tqdm


class HodoscopeError(Exception):
    """Raised for pipeline-level errors."""


# ---------------------------------------------------------------------------
# Source detection helpers
# ---------------------------------------------------------------------------

def _is_eval_file(path: Path) -> bool:
    """Check if a path is an .eval file."""
    return path.is_file() and path.suffix == ".eval"


def _is_trajectory_dir(path: Path) -> bool:
    """Check if a directory contains trajectory JSON files (has samples/ subdir)."""
    if not path.is_dir():
        return False
    samples = path / "samples"
    if samples.is_dir() and list(samples.glob("*.json")):
        return True
    if list(path.glob("*.json")):
        return True
    return False


def _is_openhands_jsonl(path: Path) -> bool:
    """Check if a JSONL file looks like OpenHands evaluation output.

    Peeks at the first non-empty line and checks for V1 SDK markers:
    ``instance_id`` and ``history`` keys.
    """
    if not path.is_file() or path.suffix != ".jsonl":
        return False
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                first = json.loads(line)
                return "instance_id" in first and "history" in first
    except (json.JSONDecodeError, OSError):
        pass
    return False



def _resolve_sources(sources: tuple[str, ...], docent_id: str | None = None) -> list[dict]:
    """Resolve CLI source arguments into a list of source descriptors.

    Each descriptor is a dict: {"type": "eval"|"openhands"|"dir"|"docent", "path": Path, ...}
    """
    resolved = []

    if docent_id:
        resolved.append({"type": "docent", "collection_id": docent_id})

    for src in sources:
        p = Path(src)
        if _is_eval_file(p):
            resolved.append({"type": "eval", "path": p})
        elif _is_openhands_jsonl(p):
            resolved.append({"type": "openhands", "path": p})
        elif p.is_dir():
            eval_files = sorted(p.glob("*.eval"))
            oh_jsonl_files = [f for f in sorted(p.glob("*.jsonl")) if _is_openhands_jsonl(f)]
            if eval_files:
                for ef in eval_files:
                    resolved.append({"type": "eval", "path": ef})
            if oh_jsonl_files:
                for jf in oh_jsonl_files:
                    resolved.append({"type": "openhands", "path": jf})
            if not eval_files and not oh_jsonl_files:
                if _is_trajectory_dir(p):
                    resolved.append({"type": "dir", "path": p})
                else:
                    # Check subdirs for trajectory structure (model/category/samples/)
                    has_trajs = False
                    for subdir in sorted(p.iterdir()):
                        if subdir.is_dir():
                            for catdir in sorted(subdir.iterdir()):
                                if catdir.is_dir() and (catdir / "samples").is_dir():
                                    resolved.append({"type": "dir", "path": catdir})
                                    has_trajs = True
                    if not has_trajs:
                        print(f"WARNING: No .eval files or trajectory data found in {p}, skipping")
        else:
            print(f"WARNING: Source not found: {src}, skipping")

    return resolved


def _resolve_analysis_sources(sources: tuple[str, ...]) -> list[Path]:
    """Resolve source arguments to .hodoscope.json file paths.

    Accepts individual file paths and directories (which are scanned for
    *.hodoscope.json files).
    """
    json_paths = []
    for src in sources:
        p = Path(src)
        if p.is_file() and p.name.endswith(".hodoscope.json"):
            json_paths.append(p)
        elif p.is_dir():
            found = sorted(p.glob("*.hodoscope.json"))
            json_paths.extend(found)
        else:
            print(f"WARNING: Not a .hodoscope.json file or directory: {src}, skipping")
    return json_paths


# ---------------------------------------------------------------------------
# Loading: .eval files
# ---------------------------------------------------------------------------

def load_eval(
    path: str | Path,
    limit: int | None = None,
    sample: bool = True,
    seed: int | None = None,
    save_samples: str | Path | None = None,
) -> tuple[list[dict], dict]:
    """Load trajectories from an .eval file.

    Args:
        path: Path to an .eval file (str or Path).
        limit: Max number of trajectories to load.
        sample: Randomly sample trajectories (default: True; pass False for first N).
        seed: Random seed for reproducible sampling.
        save_samples: Optional directory to save extracted trajectory JSONs.

    Returns:
        (trajectories, fields) — list of trajectory dicts and file-level metadata.

    Raises:
        HodoscopeError: If path is not a valid .eval file.
    """
    path = Path(path)
    if not _is_eval_file(path):
        raise HodoscopeError(f"Not a valid .eval file: {path}")
    save_samples = Path(save_samples) if save_samples else None

    from .eval.convert_to_trajectory import (
        extract_eval_fields,
        _load_scores,
        convert_eval_sample,
    )

    with zipfile.ZipFile(path, "r") as zf:
        header = json.loads(zf.read("header.json"))
        scores = _load_scores(zf)

        sample_names = sorted(
            n for n in zf.namelist()
            if n.startswith("samples/") and n.endswith(".json")
        )
        first_sample = json.loads(zf.read(sample_names[0])) if sample_names else None
        fields = extract_eval_fields(header, first_sample)

        if limit:
            if sample:
                random.Random(seed).shuffle(sample_names)
            sample_names = sample_names[:limit]

        trajectories = []
        for name in sample_names:
            sample = json.loads(zf.read(name))
            sample_id = sample.get("id", "")
            score_info = scores.get(sample_id)
            traj = convert_eval_sample(sample, score_info=score_info)
            trajectories.append(traj)

        if save_samples:
            model_name = fields.get("model", "unknown")
            samples_dir = save_samples / model_name / "samples"
            samples_dir.mkdir(parents=True, exist_ok=True)
            for traj in trajectories:
                safe_id = str(traj["id"]).replace("/", "_").replace("\\", "_")
                out_file = samples_dir / f"{safe_id}_epoch_{traj['metadata']['epoch']}.json"
                with open(out_file, "w") as f:
                    json.dump(traj, f, indent=2, default=str)
            print(f"  Saved {len(trajectories)} trajectory samples to {samples_dir}")

    return trajectories, fields


def load_trajectory_dir(
    path: str | Path,
    limit: int | None = None,
    sample: bool = True,
    seed: int | None = None,
) -> tuple[list[dict], dict]:
    """Load trajectories from a directory of JSON files.

    Args:
        path: Path to a directory containing trajectory JSONs (str or Path).
        limit: Max number of trajectories to load.
        sample: Randomly sample trajectories (default: True; pass False for first N).
        seed: Random seed for reproducible sampling.

    Returns:
        (trajectories, fields) — list of trajectory dicts and file-level metadata.

    Raises:
        HodoscopeError: If path is not a directory.
    """
    path = Path(path)
    if not path.is_dir():
        raise HodoscopeError(f"Not a directory: {path}")

    samples_dir = path / "samples"
    if samples_dir.is_dir():
        json_files = sorted(samples_dir.glob("*.json"))
    else:
        json_files = sorted(path.glob("*.json"))

    if sample:
        random.Random(seed).shuffle(json_files)
    if limit:
        json_files = json_files[:limit]

    trajectories = []
    for jf in json_files:
        with open(jf) as f:
            traj = json.load(f)
        trajectories.append(traj)

    fields = {}
    if trajectories:
        first = trajectories[0]
        if first.get("model"):
            fields["model"] = first["model"]

    return trajectories, fields


# ---------------------------------------------------------------------------
# Loading: OpenHands results
# ---------------------------------------------------------------------------

def load_openhands(
    path: str | Path,
    limit: int | None = None,
    sample: bool = True,
    seed: int | None = None,
    save_samples: str | Path | None = None,
) -> tuple[list[dict], dict]:
    """Load trajectories from an OpenHands JSONL file.

    Accepts a ``.jsonl`` file containing OpenHands evaluation output (one JSON
    object per line with ``instance_id`` and ``history``).  Optionally reads a
    sibling ``report.json`` for aggregate stats.

    Args:
        path: Path to an OpenHands ``.jsonl`` file. Accepts ``str`` or ``Path``.
        limit: Max number of trajectories to load.
        sample: Randomly sample trajectories (default: True; pass False for first N).
        seed: Random seed for reproducible sampling.
        save_samples: Optional directory to save extracted trajectory JSONs.

    Returns:
        (trajectories, fields) — list of trajectory dicts and file-level metadata.

    Raises:
        HodoscopeError: If path is not a valid OpenHands JSONL file.
    """
    path = Path(path)

    from .openhands.convert_to_trajectory import (
        convert_openhands_instance,
        extract_openhands_fields,
    )

    if not path.is_file():
        raise HodoscopeError(f"Not a file: {path}")

    # Read all lines
    instances = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                instances.append(json.loads(line))

    if not instances:
        return [], {}

    # Sampling
    if limit:
        if sample:
            random.Random(seed).shuffle(instances)
        instances = instances[:limit]

    # Load sibling JSON files if available
    report = None
    report_path = path.parent / "report.json"
    if report_path.exists():
        try:
            with open(report_path) as f:
                report = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    file_metadata = None
    metadata_path = path.parent / "metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path) as f:
                file_metadata = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    fields = extract_openhands_fields(
        instances[0], report=report, file_metadata=file_metadata,
    )

    trajectories = [convert_openhands_instance(inst) for inst in instances]

    if save_samples:
        save_samples = Path(save_samples)
        model_name = fields.get("model", "unknown")
        samples_dir = save_samples / model_name / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        for traj in trajectories:
            safe_id = str(traj["id"]).replace("/", "_").replace("\\", "_")
            attempt = traj["metadata"].get("attempt", 1)
            out_file = samples_dir / f"{safe_id}_attempt_{attempt}.json"
            with open(out_file, "w") as f:
                json.dump(traj, f, indent=2, default=str)
        print(f"  Saved {len(trajectories)} trajectory samples to {samples_dir}")

    return trajectories, fields


# ---------------------------------------------------------------------------
# Core analysis: trajectories -> summaries
# ---------------------------------------------------------------------------

def process_trajectories(
    trajectories: list[dict],
    config=None,
    skip_set: set | None = None,
    save_callback: callable = None,
    save_interval: int = 100,
) -> list[dict]:
    """Extract actions from trajectories, summarize each with an LLM, and embed.

    Args:
        trajectories: List of trajectory dicts (each must have a 'messages' key).
        config: Config instance. Defaults to Config() (hardcoded defaults, no env).
        skip_set: Set of (trajectory_id, turn_id) tuples to skip (for resume).
        save_callback: Called periodically with current results list for incremental saves.
        save_interval: How often (in completed tasks) to call save_callback.

    Returns:
        List of summary dicts with keys: trajectory_id, turn_id, summary,
        action_text, embedding (list[float] or None), metadata.
    """
    from .actions import process_single_action
    from .config import Config
    from .core import extract_actions_from_turns, run_parallel
    from .parsers import turns_from_messages, merge_consecutive_turns, extract_task_context

    if config is None:
        config = Config()

    all_tasks = []
    global_idx = 0
    skipped = 0

    for traj in tqdm(trajectories, desc="Extracting actions"):
        traj_id = traj.get("id", "unknown")
        epoch = traj.get("metadata", {}).get("epoch", 1)
        full_traj_id = f"{traj_id}_epoch_{epoch}" if epoch != 1 else str(traj_id)

        meta = traj.get("metadata", {})
        traj_metadata = {k: v for k, v in meta.items() if v is not None}

        messages = traj.get("messages", [])
        if not messages:
            continue

        task_context = extract_task_context(messages)

        turns = turns_from_messages(messages)
        merged = merge_consecutive_turns(turns)
        actions = extract_actions_from_turns(merged)

        for action in actions:
            action['task_context'] = task_context
            if skip_set and (full_traj_id, action['turn_id']) in skip_set:
                skipped += 1
                continue
            task = (action, config, global_idx, full_traj_id, traj_metadata)
            all_tasks.append(task)
            global_idx += 1

    total_found = len(all_tasks) + skipped
    print(f"Found {total_found} actions across {len(trajectories)} trajectories")
    if skipped:
        print(f"  Skipping {skipped} already-processed actions (resume)")
        print(f"  Processing {len(all_tasks)} remaining actions")

    if not all_tasks:
        return []

    results = run_parallel(
        func=process_single_action,
        tasks=all_tasks,
        desc="Summarizing & embedding",
        max_workers=config.max_workers,
        save_callback=save_callback,
        save_interval=save_interval,
    )

    return results


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------

def analyze(
    sources: tuple[str, ...] = (),
    docent_id: str | None = None,
    output: str | None = None,
    fields: list[str] | None = None,
    limit: int | None = None,
    save_samples: str | None = None,
    model_name: str | None = None,
    sample: bool = True,
    seed: int | None = None,
    resume: bool = True,
    config=None,
    reembed: bool = False,
) -> list[Path]:
    """Analyze source files and produce .hodoscope.json output(s).

    Args:
        sources: Paths to .eval files, directories, etc.
        docent_id: Docent collection ID (alternative source).
        output: Output path (only for single source).
        fields: List of "KEY=VALUE" custom metadata fields.
        limit: Max trajectories per source.
        save_samples: Directory to save extracted trajectory JSON files.
        model_name: Override auto-detected model name (metadata, not LLM).
        sample: Randomly sample trajectories (default: True; pass False for first N).
        seed: Random seed for sample reproducibility.
        resume: Resume from existing output file (skip already-processed actions).
        config: Config instance. Defaults to Config.from_env() (loads .env).
        reembed: Re-embed existing summaries using current config (e.g. after
            changing embedding model or dimensions). Implies resume.

    Returns:
        List of paths to written analysis JSON files.

    Raises:
        HodoscopeError: If no valid sources found.
    """
    from .config import Config
    from .io import write_analysis_json, read_analysis_json

    if config is None:
        config = Config.from_env()

    resolved = _resolve_sources(sources, docent_id)
    if not resolved:
        raise HodoscopeError("No valid sources found.")

    if output and len(resolved) > 1:
        print("WARNING: -o/--output ignored for batch mode (multiple sources)")
        output = None

    custom_fields = {}
    for f in (fields or []):
        if "=" not in f:
            print(f"WARNING: Ignoring malformed --field '{f}' (expected KEY=VALUE)")
            continue
        k, v = f.split("=", 1)
        custom_fields[k.strip()] = v.strip()

    save_samples_path = Path(save_samples) if save_samples else None
    written_files = []

    for src in resolved:
        src_type = src["type"]

        print(f"\n{'=' * 60}")
        if src_type == "eval":
            print(f"Processing .eval file: {src['path']}")
            trajectories, auto_fields = load_eval(
                src["path"], limit=limit, sample=sample, seed=seed,
                save_samples=save_samples_path,
            )
            source_label = str(src["path"])
            default_output = src["path"].with_suffix(".hodoscope.json")
        elif src_type == "dir":
            print(f"Processing directory: {src['path']}")
            trajectories, auto_fields = load_trajectory_dir(
                src["path"], limit=limit, sample=sample, seed=seed,
            )
            source_label = str(src["path"])
            default_output = src["path"].parent / f"{src['path'].name}.hodoscope.json"
        elif src_type == "openhands":
            print(f"Processing OpenHands JSONL: {src['path']}")
            trajectories, auto_fields = load_openhands(
                src["path"], limit=limit, sample=sample, seed=seed,
                save_samples=save_samples_path,
            )
            source_label = str(src["path"])
            default_output = src["path"].with_suffix(".hodoscope.json")
        elif src_type == "docent":
            print(f"Processing Docent collection: {src['collection_id']}")
            trajectories, auto_fields = load_docent(
                src["collection_id"], limit=limit, sample=sample, seed=seed,
                save_samples=save_samples_path,
            )
            source_label = f"docent:{src['collection_id']}"
            default_output = Path(f"docent_{src['collection_id']}.hodoscope.json")
        else:
            continue

        if not trajectories:
            print("  No trajectories found, skipping.")
            continue

        file_fields = {**auto_fields, **custom_fields}
        if model_name:
            file_fields["model"] = model_name

        out_path = Path(output) if output else default_output
        dim = config.embed_dim

        # Resume: load existing summaries and build skip set
        existing_summaries = []
        skip_set = None
        if resume and out_path.exists():
            try:
                existing_doc = read_analysis_json(out_path)
                existing_summaries = existing_doc.get("summaries", [])
                skip_set = {
                    (s["trajectory_id"], s["turn_id"])
                    for s in existing_summaries
                }
                print(f"  Resuming: found {len(existing_summaries)} existing summaries in {out_path}")
            except Exception as e:
                print(f"  WARNING: Could not load existing file for resume: {e}")
                existing_summaries = []
                skip_set = None

        # Re-embed existing summaries if requested
        if reembed and existing_summaries:
            from .actions import embed_summary
            from .core import run_parallel

            print(f"  Re-embedding {len(existing_summaries)} existing summaries...")

            def _reembed_one(args):
                s, cfg = args
                return embed_summary(s, cfg)

            existing_summaries = run_parallel(
                func=_reembed_one,
                tasks=[(s, config) for s in existing_summaries],
                desc="Re-embedding",
                max_workers=config.max_workers,
            )

        print(f"  Trajectories: {len(trajectories)}")
        print(f"  Fields: {file_fields}")
        print("=" * 60)

        # Build save callback for periodic writes (always, not just resume)
        def _make_save_callback(out_path, existing_summaries, file_fields,
                                source_label, embedding_model, dim):
            def _save(new_results):
                all_summaries = existing_summaries + new_results
                write_analysis_json(
                    path=out_path,
                    summaries=all_summaries,
                    fields=file_fields,
                    source=source_label,
                    embedding_model=embedding_model,
                    embedding_dimensionality=dim,
                )
                tqdm.write(f"  [checkpoint] Saved {len(all_summaries)} summaries to {out_path}")
            return _save

        save_callback = _make_save_callback(
            out_path, existing_summaries, file_fields,
            source_label, config.embedding_model, dim,
        )

        summaries = process_trajectories(
            trajectories,
            config=config,
            skip_set=skip_set,
            save_callback=save_callback,
            save_interval=100,
        )

        # Merge existing + new summaries
        all_summaries = existing_summaries + summaries

        if not all_summaries:
            print("  No actions extracted, skipping.")
            continue

        write_analysis_json(
            path=out_path,
            summaries=all_summaries,
            fields=file_fields,
            source=source_label,
            embedding_model=config.embedding_model,
            embedding_dimensionality=dim,
        )

        print(f"\n  Wrote {len(all_summaries)} summaries to {out_path}")
        if existing_summaries:
            print(f"    ({len(existing_summaries)} resumed + {len(summaries)} new)")
        written_files.append(out_path)

    print(f"\nDone! Wrote {len(written_files)} analysis file(s).")
    return written_files


# ---------------------------------------------------------------------------
# Loading: Docent collections
# ---------------------------------------------------------------------------

def load_docent(
    collection_id: str,
    limit: int | None = None,
    sample: bool = True,
    seed: int | None = None,
    save_samples: str | Path | None = None,
) -> tuple[list[dict], dict]:
    """Load trajectories from a Docent collection.

    Args:
        collection_id: Docent collection ID.
        limit: Max number of trajectories to load.
        sample: Randomly sample trajectories (default: True; pass False for first N).
        seed: Random seed for reproducible sampling.
        save_samples: Optional directory to save extracted trajectory JSONs.

    Returns:
        (trajectories, fields) — list of trajectory dicts and file-level metadata.

    Raises:
        HodoscopeError: If docent is not installed.
    """
    try:
        from .docent.export_transcripts import export_transcripts
        from .docent.convert_to_trajectory import convert_transcript, extract_docent_fields
    except ImportError:
        raise HodoscopeError(
            "docent package not installed. Install with: pip install -e '.[docent]'"
        )

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        # When sampling, fetch all transcripts and sample locally
        fetch_limit = None if sample else limit
        transcripts = export_transcripts(
            collection_id=collection_id,
            output_path=tmp_path,
            limit=fetch_limit,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    if not transcripts:
        return [], {}

    # Extract file-level fields from raw transcripts before conversion
    fields = extract_docent_fields(transcripts)

    if sample and limit:
        random.Random(seed).shuffle(transcripts)
        transcripts = transcripts[:limit]

    trajectories = [convert_transcript(t) for t in transcripts]

    if save_samples:
        save_samples = Path(save_samples)
        model_name = fields.get("model", "unknown")
        samples_dir = save_samples / model_name / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        for traj in trajectories:
            safe_id = str(traj["id"]).replace("/", "_").replace("\\", "_")
            epoch = traj["metadata"].get("epoch", 1)
            out_file = samples_dir / f"{safe_id}_epoch_{epoch}.json"
            with open(out_file, "w") as f:
                json.dump(traj, f, indent=2, default=str)
        print(f"  Saved {len(trajectories)} trajectory samples to {samples_dir}")

    return trajectories, fields


# ---------------------------------------------------------------------------
# viz
# ---------------------------------------------------------------------------

def viz(
    sources: tuple[str, ...],
    group_by: str | None = None,
    proj: list[str] | None = None,
    output_file: str | None = None,
    alpha: float | None = None,
    beta: float | None = None,
    filter=None,
) -> None:
    """Visualize analysis JSON files.

    Args:
        sources: Paths to .hodoscope.json files or directories containing them.
        group_by: Field to group summaries by (default: model).
        proj: List of projection methods (pca, tsne, umap, trimap, pacmap).
        output_file: Path for the output HTML file. If None, auto-generates
                     a timestamped filename in CWD.
        alpha: FPS distance exponent. If None, uses FPS_ALPHA env var or default.
        beta: FPS density gap floor. If None, uses FPS_BETA env var or default.
        filter: Callable predicate for filtering summaries.

    Raises:
        HodoscopeError: If no analysis files or summaries found.
    """
    from .config import DEFAULT_GROUP_BY, Config
    from .io import read_analysis_json, group_summaries
    from .visualization import visualize_action_summaries

    # Resolve alpha/beta: explicit arg > env var > default
    cfg = Config.from_env()
    if alpha is None:
        alpha = cfg.fps_alpha
    if beta is None:
        beta = cfg.fps_beta

    json_paths = _resolve_analysis_sources(sources)
    if not json_paths:
        raise HodoscopeError("No .hodoscope.json files found.")

    print(f"Loading {len(json_paths)} analysis file(s)...")

    docs = []
    for jp in json_paths:
        print(f"  {jp}")
        doc = read_analysis_json(jp)
        # Stamp each summary with source filename for trajectory disambiguation
        fname = jp.name if hasattr(jp, 'name') else str(jp).rsplit('/', 1)[-1]
        for s in doc.get("summaries", []):
            s["_source_file"] = fname
        docs.append(doc)

    grouped = group_summaries(docs, group_by=group_by, filter=filter)

    if not grouped:
        raise HodoscopeError("No summaries with embeddings found.")

    total = sum(len(v) for v in grouped.values())
    group_field = group_by or DEFAULT_GROUP_BY
    print(f"\nGrouped by '{group_field}': {len(grouped)} groups, {total} total summaries")
    for label, sums in grouped.items():
        print(f"  {label}: {len(sums)}")

    methods = proj if proj else ["tsne"]

    print(f"\nGenerating unified plot with methods: {', '.join(methods)}...")
    visualize_action_summaries(grouped, output_file=output_file, methods=methods, alpha=alpha, beta=beta)


# ---------------------------------------------------------------------------
# Convenience: extract actions (no LLM calls)
# ---------------------------------------------------------------------------

def extract_actions(messages: list[dict]) -> list[dict]:
    """Extract actions from a message list (no LLM calls).

    Composes: turns_from_messages → merge_consecutive_turns → extract_actions_from_turns.

    Args:
        messages: List of message dicts (as found in trajectory['messages']).

    Returns:
        List of action dicts with keys: turn_id, action_text, role, source.
    """
    from .parsers import turns_from_messages, merge_consecutive_turns
    from .core import extract_actions_from_turns

    turns = turns_from_messages(messages)
    merged = merge_consecutive_turns(turns)
    return extract_actions_from_turns(merged)


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------

def show_info(sources: tuple[str, ...] = ()) -> None:
    """Show metadata and summary counts for analysis JSON files."""
    from .config import _load_env
    from .io import read_analysis_json

    _load_env()

    if not sources:
        sources = tuple(str(p) for p in sorted(Path.cwd().glob("*.hodoscope.json")))
        if not sources:
            print("No .hodoscope.json files found. Pass file paths as arguments.")
            return

    for src in sources:
        p = Path(src)
        if p.is_dir():
            files = sorted(p.glob("*.hodoscope.json"))
        else:
            files = [p]

        for fp in files:
            if not fp.exists():
                print(f"File not found: {fp}")
                continue

            doc = read_analysis_json(fp)
            summaries = doc.get("summaries", [])
            with_emb = sum(1 for s in summaries if s.get("embedding") is not None)

            print(f"\n{fp}")
            print("=" * 50)
            print(f"  Version:      {doc.get('version', '?')}")
            print(f"  Created:      {doc.get('created_at', '?')}")
            print(f"  Source:        {doc.get('source', '?')}")
            print(f"  Embed model:   {doc.get('embedding_model', '?')}")
            print(f"  Embed dims:    {doc.get('embedding_dimensionality', '?')}")
            print(f"  Fields:")
            for k, v in doc.get("fields", {}).items():
                print(f"    {k}: {v}")
            print(f"  Summaries:     {len(summaries)} ({with_emb} with embeddings)")

    print("\nAPI Keys:")
    for key_name in ["OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "DOCENT_API_KEY"]:
        val = os.environ.get(key_name)
        status = f"set ({val[:4]}...)" if val else "not set"
        print(f"  {key_name}: {status}")


# ---------------------------------------------------------------------------
# sample
# ---------------------------------------------------------------------------

def sample(
    sources: tuple[str, ...],
    group_by: str | None = None,
    n: int = 10,
    method: str = 'tsne',
    alpha: float = 1.0,
    beta: float = 0.1,
    output: str | None = None,
    interleave: bool = False,
    filter=None,
) -> dict[str, list[dict]]:
    """Sample representative summaries from analysis JSON files using FPS.

    Loads analysis JSONs, groups summaries, ranks them by importance using
    density-weighted Farthest Point Sampling on 2D projections, and either
    writes a lightweight JSON file or displays paginated terminal output.

    Args:
        sources: Paths to .hodoscope.json files or directories containing them.
        group_by: Field to group summaries by (default: model).
        n: Number of summaries per group to return.
        method: Projection method for FPS ('pca', 'tsne', 'umap', 'trimap', 'pacmap').
        alpha: FPS distance exponent (higher = more spatial spread).
        beta: FPS density gap floor (negative gaps mapped to [0, beta],
            positive gaps to [beta, 1]).
        output: If given, write results as JSON to this path.
            If None, display paginated terminal output.
        interleave: If True, interleave groups by rank (#1 from each group,
            then #2, etc.). Default: group-by-group.
        filter: Callable predicate for filtering summaries.

    Returns:
        Dict mapping group labels to lists of ranked summary dicts.

    Raises:
        HodoscopeError: If no analysis files or summaries found.
    """
    import click
    from .config import DEFAULT_GROUP_BY
    from .io import read_analysis_json, group_summaries
    from .sampling import rank_summaries, SAMPLING_METHOD_DISPLAY_NAMES

    json_paths = _resolve_analysis_sources(sources)
    if not json_paths:
        raise HodoscopeError("No .hodoscope.json files found.")

    print(f"Loading {len(json_paths)} analysis file(s)...")

    docs = []
    for jp in json_paths:
        print(f"  {jp}")
        doc = read_analysis_json(jp)
        docs.append(doc)

    grouped = group_summaries(docs, group_by=group_by, filter=filter)

    if not grouped:
        raise HodoscopeError("No summaries with embeddings found.")

    group_field = group_by or DEFAULT_GROUP_BY
    total = sum(len(v) for v in grouped.values())
    print(f"\nGrouped by '{group_field}': {len(grouped)} groups, {total} total summaries")

    display = SAMPLING_METHOD_DISPLAY_NAMES.get(method, method.upper())
    print(f"Computing {display} + FPS ranking (n={n} per group, alpha={alpha}, beta={beta})...")

    ranked = rank_summaries(grouped, method=method, n=n, alpha=alpha, beta=beta)

    if output:
        # Write lightweight JSON (no embeddings)
        out_data = {
            "group_by": group_field,
            "method": method,
            "n_per_group": n,
            "groups": {},
        }
        for label, sums in ranked.items():
            out_data["groups"][label] = {
                "total": len(grouped[label]),
                "samples": [
                    {
                        "rank": s["fps_rank"],
                        "trajectory_id": s["trajectory_id"],
                        "turn_id": s["turn_id"],
                        "summary": s["summary"],
                        "action_text": s.get("action_text", ""),
                        "metadata": s.get("metadata", {}),
                    }
                    for s in sums
                ],
            }

        out_path = Path(output)
        with open(out_path, "w") as f:
            json.dump(out_data, f, indent=2, default=str)
        print(f"\nWrote {sum(len(v) for v in ranked.values())} samples to {out_path}")
    else:
        # Paginated terminal output
        lines = []

        def _format_entry(s, label_prefix=""):
            meta = s.get("metadata", {})
            meta_parts = []
            for k in ("score", "epoch", "instance_id"):
                if k in meta and meta[k] is not None:
                    meta_parts.append(f"{k}: {meta[k]}")
            meta_str = "  " + "  ".join(meta_parts) if meta_parts else ""
            prefix = f"[{label_prefix}]  " if label_prefix else ""
            return (
                f"  {prefix}trajectory: {s['trajectory_id']}  turn: {s['turn_id']}"
                f"{meta_str}\n"
                f"    {s['summary']}"
            )

        if interleave:
            labels = list(ranked.keys())
            for label in labels:
                total_in_group = len(grouped[label])
                lines.append(f"  {label}: {len(ranked[label])} of {total_in_group} summaries")
            lines.append("")

            max_rank = max((len(sums) for sums in ranked.values()), default=0)
            for rank_idx in range(max_rank):
                lines.append(f"--- #{rank_idx + 1} ---")
                for label in labels:
                    sums = ranked[label]
                    if rank_idx >= len(sums):
                        continue
                    lines.append(_format_entry(sums[rank_idx], label_prefix=label))
                lines.append("")
        else:
            for label, sums in ranked.items():
                total_in_group = len(grouped[label])
                lines.append(f"=== {label} ({len(sums)} of {total_in_group} summaries) ===\n")
                for i, s in enumerate(sums):
                    lines.append(f"#{i + 1}")
                    lines.append(_format_entry(s))
                    lines.append("")
                lines.append("")

        click.echo_via_pager("\n".join(lines))

    return ranked
