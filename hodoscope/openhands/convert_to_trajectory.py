"""
Convert OpenHands evaluation results to trajectory format.

Supports the **V1 SDK format** (OpenHands/benchmarks) where each event in the
``history`` array has a ``kind`` discriminator (Pydantic ``DiscriminatedUnionMixin``):

  - SystemPromptEvent  → system message
  - MessageEvent       → user or assistant message
  - ActionEvent        → agent tool call (terminal, file_editor, browser_use, think, finish)
  - ObservationEvent   → tool result
  - ConversationStateEvent → state snapshots (skipped)

The **V0 runtime format** (All-Hands-AI/OpenHands ``event_to_dict()``) uses a
different layout — ``{"action": "run", "args": {...}}`` with no ``kind`` field.
V0 events are detected and a warning is emitted; they are not converted.

The converter maps V1 events into the standard message list expected by
``turns_from_messages`` (role / content / tool_calls).
"""

import json
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_text(content) -> str:
    """Extract plain text from an OpenHands content value.

    Content can be a string, a single dict with a ``text`` key, or a list of
    such dicts.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("text", "")
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text", "")
                if text:
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return ""


def _extract_model(metadata: dict) -> str:
    """Extract a clean model name from metadata.

    Checks ``model`` first, then ``llm.model``.
    Strips the ``litellm_proxy/`` prefix if present.
    """
    metadata = metadata or {}
    raw = metadata.get("model", "") or ((metadata.get("llm") or {}).get("model", ""))
    if not raw:
        return "unknown"
    if raw.startswith("litellm_proxy/"):
        raw = raw[len("litellm_proxy/"):]
    return raw


def _action_text(evt: dict) -> str:
    """Build a human-readable text for an ActionEvent."""
    action = evt.get("action") or {}
    tool = evt.get("tool_name", "unknown")
    kind = action.get("kind", "")

    # Gather thought / reasoning text
    thought_parts = []
    rc = evt.get("reasoning_content")
    if rc:
        thought_parts.append(rc)
    thought_items = evt.get("thought", [])
    if thought_items:
        t = _extract_text(thought_items)
        if t:
            thought_parts.append(t)
    # Action-level thought (think tool stores its thought here)
    action_thought = action.get("thought")
    if action_thought and isinstance(action_thought, str):
        thought_parts.append(action_thought)
    thought_text = "\n".join(thought_parts).strip()

    # Build tool call description
    if kind == "TerminalAction":
        call_text = f"TOOL_CALL: {tool} {json.dumps({'command': action.get('command', '')})}"
    elif kind == "FileEditorAction":
        args = {}
        for k in ("command", "path", "view_range", "old_str", "new_str", "file_text", "insert_line"):
            v = action.get(k)
            if v is not None and v != "":
                args[k] = v
        call_text = f"TOOL_CALL: {tool} {json.dumps(args)}"
    elif kind == "BrowserUseAction":
        call_text = f"TOOL_CALL: {tool} {json.dumps({'command': action.get('command', '')})}"
    else:
        # think, finish, or other tools
        args = {k: v for k, v in action.items() if k != "kind" and v is not None and v != ""}
        call_text = f"TOOL_CALL: {tool} {json.dumps(args)}" if args else f"TOOL_CALL: {tool}"

    if thought_text:
        return f"{thought_text}\n{call_text}"
    return call_text


def _observation_text(evt: dict) -> str:
    """Build text for an ObservationEvent."""
    obs = evt.get("observation", {})
    content = obs.get("content", "")
    text = _extract_text(content)
    # Include error info if present
    if obs.get("is_error"):
        exit_code = obs.get("exit_code")
        prefix = f"[ERROR exit_code={exit_code}] " if exit_code is not None else "[ERROR] "
        return prefix + text
    return text


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert_openhands_instance(instance: dict) -> dict:
    """Convert a single OpenHands JSONL instance to universal trajectory format.

    Args:
        instance: Parsed JSON object from output.jsonl (one line).

    Returns:
        Trajectory-format dictionary compatible with the analysis pipeline.
    """
    instance_id = instance.get("instance_id", "")
    attempt = instance.get("attempt", 1)
    oh_metadata = instance.get("metadata") or {}

    # Build messages from history events
    messages = []
    v0_warned = False
    for evt in instance.get("history", []):
        kind = evt.get("kind", "")

        # Detect V0 runtime format: has "action" as a string (e.g. "run")
        # instead of "kind" discriminator.  V1 SDK format has "kind" and
        # "action" is a dict (the Action object).
        if not kind and ("action" in evt or "observation" in evt):
            if not v0_warned:
                logger.warning(
                    "Instance %s appears to use V0 OpenHands event format "
                    "(no 'kind' field, found 'action'/'observation' string "
                    "keys). V0 events are not supported and will be skipped. "
                    "Convert with OpenHands V1 SDK or re-run the evaluation.",
                    instance_id,
                )
                v0_warned = True
            continue

        if kind == "SystemPromptEvent":
            system_prompt = evt.get("system_prompt", {})
            text = _extract_text(system_prompt)
            if text:
                messages.append({"role": "system", "content": text})

        elif kind == "MessageEvent":
            llm_msg = evt.get("llm_message", {})
            role = llm_msg.get("role", "user")
            content = llm_msg.get("content", "")
            text = _extract_text(content)
            if text:
                messages.append({"role": role, "content": text})

        elif kind == "ActionEvent":
            text = _action_text(evt)
            tool_name = evt.get("tool_name", "unknown")
            action = evt.get("action", {})
            messages.append({
                "role": "assistant",
                "content": text,
                "tool_calls": [{
                    "function": tool_name,
                    "arguments": action,
                }],
            })

        elif kind == "ObservationEvent":
            text = _observation_text(evt)
            messages.append({"role": "tool", "content": text})

        # Skip ConversationStateEvent and unknown kinds

    # Build per-trajectory metadata — absorb all sub-objects
    traj_metadata = {
        "instance_id": instance_id or None,
        "attempt": attempt,
    }

    # Absorb test_result (score, ground_truth, model_answer, git_patch, ...)
    test_result = instance.get("test_result") or {}
    for k, v in test_result.items():
        if v is not None:
            traj_metadata[k] = v

    # Absorb metrics (accumulated_cost, accumulated_token_usage, ...)
    metrics = instance.get("metrics") or {}
    for k, v in metrics.items():
        if v is not None:
            traj_metadata[k] = v

    # Absorb instance-level info (Level, Question, etc.)
    inst_info = instance.get("instance") or {}
    for k, v in inst_info.items():
        if v is not None:
            traj_metadata[k] = v

    if instance.get("error"):
        traj_metadata["error"] = instance["error"]

    model = _extract_model(oh_metadata)

    return {
        "id": instance_id,
        "source": "openhands",
        "model": model,
        "input": instance.get("instruction", ""),
        "messages": messages,
        "metadata": traj_metadata,
    }


def extract_openhands_fields(
    first_instance: dict,
    report: dict | None = None,
    file_metadata: dict | None = None,
) -> dict:
    """Extract file-level fields from an OpenHands instance + optional report.

    Args:
        first_instance: First parsed JSONL instance (for model detection).
        report: Optional parsed report.json with aggregate stats.
        file_metadata: Optional parsed metadata.json (sibling of output.jsonl).
            All keys are absorbed into fields (lowest priority).

    Returns:
        Dict of file-level metadata fields.
    """
    fields = {}

    # Start with file-level metadata.json (lowest priority — everything absorbed)
    fm = file_metadata or {}
    for k, v in fm.items():
        if v is not None:
            fields[k] = v

    # Instance-level metadata (from first JSONL line) overrides file metadata
    oh_metadata = first_instance.get("metadata") or {}
    for k, v in oh_metadata.items():
        if v is not None:
            fields[k] = v

    # Extract clean model name from merged fields (instance llm overrides file llm)
    model = _extract_model(fields)
    if model and model != "unknown":
        fields["model"] = model

    # Report stats
    if report:
        total = report.get("total_instances")
        resolved = report.get("resolved_instances")
        if total is not None and resolved is not None and total > 0:
            fields["accuracy"] = resolved / total
        if total is not None:
            fields["dataset_samples"] = total

    fields["trajectory_format"] = "openhands"

    return fields
