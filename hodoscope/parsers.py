"""
Trajectory parsing functions.
Extract and process trajectory data from message lists.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def turns_from_messages(messages: list[dict]) -> list[dict]:
    """Extract conversation turns from a list of message dicts.

    Each turn has: turn_id, role, content, source.
    """
    turns = []
    for i, msg in enumerate(messages):
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')

        if role == 'assistant':
            tool_calls = msg.get('tool_calls', [])
            if tool_calls:
                content_parts = []
                for tc in tool_calls:
                    func = tc.get('function', 'unknown')
                    args = tc.get('arguments', {})
                    args_str = json.dumps(args) if args else ''
                    content_parts.append(f"TOOL_CALL: {func} {args_str}")
                content_str = ' | '.join(content_parts)
            else:
                if isinstance(content, list):
                    content_parts = []
                    for item in content:
                        if isinstance(item, dict):
                            reasoning = item.get('reasoning', item.get('text', ''))
                            if reasoning:
                                content_parts.append(reasoning)
                    content_str = ' '.join(content_parts) if content_parts else ''
                else:
                    content_str = str(content)
        else:
            if msg.get('error', ''):
                content_str = 'ERROR: ' + msg['error'].get('message', '')
            elif isinstance(content, list):
                content_parts = []
                for item in content:
                    if isinstance(item, dict):
                        text = item.get('text', item.get('reasoning', ''))
                        if text:
                            content_parts.append(text)
                    else:
                        content_parts.append(str(item))
                content_str = ' '.join(content_parts) if content_parts else ''
            else:
                content_str = str(content) if content else ''

        if not content_str:
            logger.warning("Empty content in message %d (role=%s): %s", i, role, msg)

        turns.append({
            'turn_id': i,
            'role': role,
            'content': content_str,
            'source': msg.get('source', 'unknown'),
        })

    return turns


def turns_from_trajectory_file(traj_file: Path) -> list[dict]:
    """Extract conversation turns from a trajectory JSON file."""
    with open(traj_file) as f:
        data = json.load(f)
    return turns_from_messages(data.get('messages', []))


def extract_task_context(messages: list[dict]) -> str:
    """Extract task context from a message list: first system + first user content.

    Iterates through messages, collects the first system-role content and the
    first user-role content, then joins them with a separator. Returns empty
    string if neither is found.
    """
    system_text = None
    user_text = None

    for msg in messages:
        role = msg.get('role', '')
        content = msg.get('content', '')

        if role == 'system' and system_text is None:
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        text = item.get('text', item.get('reasoning', ''))
                        if text:
                            parts.append(text)
                    else:
                        parts.append(str(item))
                system_text = ' '.join(parts) if parts else ''
            else:
                system_text = str(content) if content else ''

        elif role == 'user' and user_text is None:
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        text = item.get('text', item.get('reasoning', ''))
                        if text:
                            parts.append(text)
                    else:
                        parts.append(str(item))
                user_text = ' '.join(parts) if parts else ''
            else:
                user_text = str(content) if content else ''

        if system_text is not None and user_text is not None:
            break

    parts = []
    if system_text:
        parts.append(system_text)
    if user_text:
        parts.append(user_text)
    return '\n\n---\n\n'.join(parts)


def merge_consecutive_turns(turns: list[dict]) -> list[dict]:
    """Merge consecutive tool/user turns into previous turn."""
    turns_merged = []
    for turn in turns:
        if turn['role'] in ['tool', 'user'] and len(turns_merged) > 0:
            prev_turn = turns_merged[-1]
            prev_turn['content'] += "\n" + "-" * 40 + ' FEEDBACK ' + "-" * 40 + "\n" + turn['content']
            turns_merged[-1] = prev_turn
        else:
            turns_merged.append(turn)
    return turns_merged
