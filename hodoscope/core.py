"""
Shared utility functions for trajectory processing.
"""

import time
import random
import concurrent.futures
from typing import Callable, Any
from tqdm import tqdm

import litellm

from .config import DEFAULT_EMBEDDING_MODEL, DEFAULT_EMBEDDING_TASK_TYPE


def extract_actions_from_turns(turns: list[dict]) -> list[dict[str, Any]]:
    """Extract actions from merged turns (assistant tool calls)."""
    actions = []
    for turn in turns:
        if turn['role'] == 'assistant':
            actions.append({
                'turn_id': turn['turn_id'],
                'action_text': turn['content'],
                'role': turn['role'],
                'source': turn['source']
            })
    return actions


def embed_text(
    text: str,
    model: str = DEFAULT_EMBEDDING_MODEL,
    max_retries: int = 5,
    initial_delay: float = 3.0,
    max_delay: float = 60.0,
    output_dimensionality: int | None = None,
    normalize: bool = True,
) -> list[float] | None:
    """Get embedding for text via LiteLLM with exponential backoff."""
    delay = initial_delay

    for attempt in range(max_retries):
        try:
            kwargs = dict(
                model=model,
                input=[text],
            )
            if output_dimensionality:
                kwargs['dimensions'] = output_dimensionality
            response = litellm.embedding(**kwargs)
            embedding = response.data[0]['embedding']
            # Sanity check: verify dimensionality matches if requested
            if output_dimensionality and len(embedding) != output_dimensionality:
                tqdm.write(
                    f"WARNING: Requested {output_dimensionality} dims but got "
                    f"{len(embedding)} — model may not support `dimensions` param"
                )
                return None
            # Normalize reduced-dimension embeddings
            if normalize:
                import numpy as np
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = (np.array(embedding) / norm).tolist()
            return embedding
        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = 'rate' in error_str or '429' in error_str or 'quota' in error_str

            if attempt < max_retries - 1 and is_rate_limit:
                jitter = random.uniform(0, delay * 0.1)
                sleep_time = delay + jitter
                tqdm.write(f"Rate limited, retrying in {sleep_time:.1f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(sleep_time)
                delay = min(delay * 2, max_delay)
            else:
                if attempt == max_retries - 1:
                    tqdm.write(f"Failed after {max_retries} attempts: {e}")
                return None

    return None


def run_parallel(
    func: Callable,
    tasks: list[Any],
    desc: str = "Processing",
    max_workers: int = 10,
    save_callback: Callable | None = None,
    save_interval: int = 100,
) -> list[Any]:
    """Run tasks in parallel with progress bar and periodic saves."""
    results = []
    processed_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(func, task): i for i, task in enumerate(tasks)}

        with tqdm(total=len(tasks), desc=desc) as pbar:
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                    processed_count += 1

                    if save_callback and processed_count % save_interval == 0:
                        save_callback(results)

                except Exception as e:
                    tqdm.write(f"Error: {e}")
                pbar.update(1)

    if save_callback and results:
        save_callback(results)

    return results
