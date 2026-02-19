"""
Convert .eval archive samples to trajectory format.

.eval files are ZIP archives (from Inspect AI) containing:
  - header.json           — eval metadata (task, model, config)
  - samples/*.json        — individual trajectory samples
  - reductions.json       — per-sample scores
"""

import json
import zipfile


def _detect_model(header: dict, sample: dict | None = None) -> str:
    """Extract model name from header stats or a sample's model_usage."""
    stats = header.get("stats", {})
    model_usage = stats.get("model_usage", {})
    if model_usage:
        raw = next(iter(model_usage))
        return raw.split("/", 1)[-1] if "/" in raw else raw

    if sample:
        model = (sample.get("output") or {}).get("model", "")
        if model:
            return model

    return "unknown"


def extract_eval_fields(header: dict, sample: dict | None = None) -> dict:
    """Extract comprehensive file-level fields from .eval header.

    Replaces separate _detect_model() + _detect_category() calls.
    Returns flat dict of metadata fields.

    Args:
        header: Parsed header.json from the .eval archive.
        sample: Optional first sample for model fallback detection.

    Returns:
        Dict of file-level metadata fields.
    """
    fields = {}

    # Model
    fields["model"] = _detect_model(header, sample)

    # Eval-level info
    eval_info = header.get("eval", {})
    task = eval_info.get("task", "")
    if task:
        fields["task"] = task

    dataset = eval_info.get("dataset", {})
    if dataset.get("name"):
        fields["dataset_name"] = dataset["name"]
    if dataset.get("location"):
        fields["dataset_location"] = dataset["location"]
    if dataset.get("samples") is not None:
        fields["dataset_samples"] = dataset["samples"]
    if dataset.get("shuffled") is not None:
        fields["dataset_shuffled"] = dataset["shuffled"]

    # Plan / solver chain
    plan = header.get("plan", {})
    steps = plan.get("steps", [])
    if steps:
        solver_names = [s.get("solver", "") for s in steps if s.get("solver")]
        if solver_names:
            fields["solver"] = ", ".join(solver_names)

    # Run metadata
    run_id = header.get("run_id", "")
    if run_id:
        fields["run_id"] = run_id
    status = header.get("status", "")
    if status:
        fields["run_status"] = status
    created = header.get("created", "")
    if created:
        fields["run_created"] = created

    # Top-level results / accuracy
    results = header.get("results", {})
    scores_list = results.get("scores", [])
    if scores_list:
        metrics = scores_list[0].get("metrics", {})
        accuracy = metrics.get("accuracy", {})
        if isinstance(accuracy, dict) and accuracy.get("value") is not None:
            fields["accuracy"] = accuracy["value"]

    return fields


def _load_scores(zf: zipfile.ZipFile) -> dict[str, dict]:
    """Load per-sample scores from reductions.json inside the archive.

    Returns:
        Dict mapping sample_id to score info dict with keys:
        value (float), answer (str|None), explanation (str|None).
    """
    try:
        data = json.loads(zf.read("reductions.json"))
    except (KeyError, json.JSONDecodeError):
        return {}

    scores: dict[str, dict] = {}
    if isinstance(data, list) and data:
        for sample in data[0].get("samples", []):
            sid = sample.get("sample_id", "")
            if sid:
                scores[sid] = {
                    "value": sample.get("value", 0.0),
                    "answer": sample.get("answer"),
                    "explanation": sample.get("explanation"),
                }
    return scores


def convert_eval_sample(sample: dict, score_info: dict | None = None) -> dict:
    """Convert a single .eval sample to universal trajectory format.

    Args:
        sample: Parsed JSON of a sample file from the .eval archive.
        score_info: Optional score dict from _load_scores() with keys:
            value, answer, explanation.

    Returns:
        Trajectory-format dictionary.
    """
    sample_id = sample.get("id", "")
    epoch = sample.get("epoch", 1)
    input_text = sample.get("input", "")
    messages = sample.get("messages", [])

    model_usage = sample.get("model_usage", {})
    model = ""
    if model_usage:
        raw = next(iter(model_usage))
        model = raw.split("/", 1)[-1] if "/" in raw else raw
    if not model:
        model = (sample.get("output") or {}).get("model", "")

    # Score: prefer reductions.json score_info, fall back to inline scores
    score = None
    if score_info is not None:
        score = score_info.get("value")
    else:
        scores_dict = sample.get("scores", {})
        if scores_dict:
            first_scorer = next(iter(scores_dict.values()))
            if isinstance(first_scorer, dict):
                score = first_scorer.get("value")

    # Build metadata — start with all sample.metadata keys (passthrough)
    metadata_raw = sample.get("metadata", {}) or {}
    metadata = dict(metadata_raw)

    # Core fields (layered on top of passthrough)
    metadata["epoch"] = epoch
    metadata["instance_id"] = sample_id or None
    metadata["score"] = score

    # Score answer/explanation from reductions.json
    if score_info:
        if score_info.get("answer") is not None:
            metadata["score_answer"] = score_info["answer"]
        if score_info.get("explanation") is not None:
            metadata["score_explanation"] = score_info["explanation"]

    # Target
    target = sample.get("target")
    if target is not None:
        metadata["target"] = target

    # Token usage from output
    output = sample.get("output") or {}
    usage = output.get("usage", {})
    if usage.get("input_tokens") is not None:
        metadata["input_tokens"] = usage["input_tokens"]
    if usage.get("output_tokens") is not None:
        metadata["output_tokens"] = usage["output_tokens"]
    if usage.get("total_tokens") is not None:
        metadata["total_tokens"] = usage["total_tokens"]

    # Response time
    if output.get("time") is not None:
        metadata["response_time"] = output["time"]

    # Sandbox, timing, UUID
    sandbox = sample.get("sandbox", {})
    if sandbox:
        metadata["sandbox"] = sandbox
    if sample.get("uuid") is not None:
        metadata["eval_uuid"] = sample["uuid"]
    if sample.get("total_time") is not None:
        metadata["total_time"] = sample["total_time"]
    if sample.get("working_time") is not None:
        metadata["working_time"] = sample["working_time"]

    return {
        "id": sample_id,
        "source": "eval",
        "model": model,
        "input": input_text,
        "messages": messages,
        "metadata": metadata,
    }
