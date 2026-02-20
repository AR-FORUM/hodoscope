# Hodoscope

Analyze AI agent trajectories: extract actions, summarize them with LLMs, embed the summaries, and create interactive visualizations to identify behavioral patterns.

## Installation

```bash
pip install hodoscope
```

For development (editable install with tests):

```bash
pip install -e ".[dev]"
```

## Configuration

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your-openai-key
GEMINI_API_KEY=your-gemini-key

# Optional: override defaults
# SUMMARIZE_MODEL=openai/gpt-5.2
# EMBEDDING_MODEL=gemini/gemini-embedding-001
# EMBED_DIM=768
# MAX_WORKERS=10
```

## Quick Start

### CLI

```bash
# Analyze a single .eval file
hodoscope analyze run.eval

# Analyze all .eval files in a directory
hodoscope analyze evals/

# Visualize results
hodoscope viz run.hodoscope.json

# Compare models
hodoscope viz model_a.hodoscope.json model_b.hodoscope.json --group-by model

# Show metadata
hodoscope info run.hodoscope.json
```

### Python API

```python
import hodoscope

# Load trajectories from .eval file
trajectories, fields = hodoscope.load_eval("run.eval", limit=5, sample=True)

# Or from a directory of trajectory JSONs
trajectories, fields = hodoscope.load_trajectory_dir("path/to/samples/")

# Summarize + embed (requires API keys)
summaries = hodoscope.process_trajectories(trajectories, summarize_model="openai/gpt-4o")

# Extract actions only (no LLM calls)
actions = hodoscope.extract_actions(trajectories[0]["messages"])

# Group and visualize in-memory summaries
grouped = hodoscope.group_summaries_from_list(summaries, group_by="score")
hodoscope.visualize_action_summaries(grouped, "plots/", methods=["tsne"])

# Or save to disk and use the file-based workflow
hodoscope.write_analysis_json("output.hodoscope.json", summaries, fields, source="run.eval")
```

## CLI Reference

### `hodoscope analyze`

Process source files (.eval, directories, Docent collections) into `.hodoscope.json` analysis files.

```bash
hodoscope analyze SOURCES [OPTIONS]

Options:
  --docent-id TEXT                Docent collection ID as source
  -o, --output TEXT               Output JSON path (single source only)
  --field TEXT                    KEY=VALUE metadata (repeatable)
  -l, --limit INTEGER            Limit trajectories per source
  --save-samples PATH            Save extracted trajectory JSONs to directory
  --embed-dim INTEGER             Embedding dimensionality (default: 768)
  -m, --model-name TEXT           Override auto-detected model name
  --summarize-model TEXT          LiteLLM model for summarization (default: openai/gpt-5.2)
  --embedding-model TEXT          LiteLLM model for embeddings (default: gemini/gemini-embedding-001)
  --sample / --no-sample          Randomly sample trajectories (use with --limit)
  --seed INTEGER                  Random seed for --sample reproducibility
  --resume / --no-resume          Resume from existing output (default: on)
  --reasoning-effort [low|medium|high]  Reasoning effort for summarization model
  --max-workers INTEGER           Max parallel workers for LLM calls (default: 10)
```

Examples:

```bash
hodoscope analyze run.eval                             # .eval → analysis JSON
hodoscope analyze *.eval                               # batch: all .eval files
hodoscope analyze evals/                               # batch: dir of .eval files
hodoscope analyze run.eval -o my_output.json           # custom output path
hodoscope analyze run.eval --field env=prod            # add custom metadata
hodoscope analyze run.eval --save-samples ./samples/   # save extracted trajectories
hodoscope analyze --docent-id COLLECTION_ID            # docent source
hodoscope analyze path/to/samples/                     # directory of trajectory JSONs
hodoscope analyze run.eval --summarize-model gemini/gemini-2.0-flash
hodoscope analyze run.eval --limit 5 --sample --seed 42
hodoscope analyze run.eval --no-resume                 # overwrite existing output
```

### `hodoscope viz`

Visualize analysis JSON files with interactive plots. Groups summaries by any metadata field.

```bash
hodoscope viz SOURCES [OPTIONS]

Options:
  --group-by TEXT     Field to group by (default: model)
  --plots TEXT        Plot types: pca, tsne, umap, trimap, pacmap, dynamic, density
  --output-dir TEXT   Directory for HTML output files
  --open              Open the generated HTML in the default browser
```

Examples:

```bash
hodoscope viz output.json                              # visualize (groups by model)
hodoscope viz output.json --group-by task              # group by task
hodoscope viz output.json --group-by score             # group by score field
hodoscope viz *.json                                   # batch: all JSONs
hodoscope viz a.json b.json --group-by model           # cross-file comparison
hodoscope viz output.json --plots tsne umap            # specific plot types
hodoscope viz output.json --open                       # open in default browser
```

### `hodoscope info`

Show metadata, summary counts, and API key status for analysis JSON files.

```bash
hodoscope info output.json
hodoscope info results/
```

## Python API Reference

The library exposes composable building blocks as first-class public functions. The CLI is a thin wrapper on top.

### Loading trajectories

```python
import hodoscope

# From .eval file (Inspect AI format)
trajectories, fields = hodoscope.load_eval("run.eval", limit=10, sample=True, seed=42)

# From directory of trajectory JSONs
trajectories, fields = hodoscope.load_trajectory_dir("path/to/samples/")

# From Docent collection
trajectories, fields = hodoscope.load_docent("COLLECTION_ID")
```

All loaders return `(trajectories, fields)` where `trajectories` is a list of trajectory dicts (each with a `messages` key) and `fields` is auto-detected file-level metadata. For `.eval` files, fields include `model`, `task`, `dataset_name`, `solver`, `run_id`, `accuracy`, and more.

### Processing

```python
# Full pipeline: extract actions → summarize with LLM → embed (requires API keys)
summaries = hodoscope.process_trajectories(
    trajectories,
    summarize_model="openai/gpt-4o",       # optional, defaults from env/config
    embedding_model="gemini/gemini-embedding-001",
    embed_dim=768,
)

# Extract actions only (no LLM calls, pure data transform)
actions = hodoscope.extract_actions(trajectories[0]["messages"])
```

### Grouping and visualization

```python
# Group in-memory summaries by any metadata field
grouped = hodoscope.group_summaries_from_list(summaries, group_by="score")

# Or group from saved analysis files
doc = hodoscope.read_analysis_json("output.hodoscope.json")
grouped = hodoscope.group_summaries([doc], group_by="model")

# Visualize
hodoscope.visualize_action_summaries(grouped, "plots/", methods=["tsne", "pca"])
```

### Saving results

```python
hodoscope.write_analysis_json(
    "output.hodoscope.json",
    summaries=summaries,
    fields=fields,
    source="run.eval",
    embedding_model="gemini/gemini-embedding-001",
    embedding_dimensionality=768,
)
```

## Output Format

Each `hodoscope analyze` run produces a `.hodoscope.json` file:

```json
{
  "version": 1,
  "created_at": "2026-02-10T12:00:00Z",
  "source": "path/to/run.eval",
  "fields": {
    "model": "gpt-5",
    "task": "swe_bench",
    "dataset_name": "swe_bench_verified",
    "solver": "system_message, generate",
    "accuracy": 0.8
  },
  "embedding_model": "gemini-embedding-001",
  "embedding_dimensionality": 3072,
  "summaries": [
    {
      "trajectory_id": "django__django-12345_epoch_1",
      "turn_id": 3,
      "summary": "Update assertion to match expected output",
      "action_text": "...",
      "embedding": "<base85-encoded float32 array>",
      "metadata": {
        "score": 1.0,
        "epoch": 1,
        "instance_id": "django__django-12345",
        "target": "expected output",
        "input_tokens": 620,
        "output_tokens": 20,
        "total_tokens": 640
      }
    }
  ]
}
```

Key concepts:
- **`fields`**: File-level metadata auto-detected from .eval header (model, task, dataset_name, solver, run_id, accuracy, etc.) plus custom `--field` values. Same for all summaries.
- **`metadata`**: Per-trajectory metadata. All `sample.metadata` keys from .eval files are passed through, plus extracted keys (score, epoch, target, token usage, etc.). Varies per summary.
- **`--group-by` resolution**: Checks per-summary `metadata` first, then file-level `fields`.
- **`embedding`**: RFC 1924 base85-encoded `float32` numpy array.

## Universal Trajectory Format

All trajectory sources are normalized to a canonical JSON schema before processing:

```json
{
  "id": "unique-trajectory-id",
  "source": "eval",
  "model": "gpt-5",
  "input": "Task description...",
  "messages": [{"role": "user", "content": "..."}],
  "metadata": {
    "epoch": 1,
    "score": 1.0,
    "instance_id": "django__django-12345",
    "target": "expected output",
    "input_tokens": 620,
    "output_tokens": 20,
    "total_tokens": 640,
    "response_time": 1.23,
    "label_confidence": 0.89
  }
}
```

## Testing

```bash
# Unit tests (no API keys needed)
pytest tests/test_io.py tests/test_viz.py tests/test_api.py

# End-to-end tests (requires API keys)
pytest tests/test_analyze.py
```
