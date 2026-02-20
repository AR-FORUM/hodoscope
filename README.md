# Hodoscope


[![PyPI version](https://img.shields.io/pypi/v/hodoscope)](https://pypi.org/project/hodoscope/)
[![Python versions](https://img.shields.io/pypi/pyversions/hodoscope)](https://pypi.org/project/hodoscope/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Unsupervised, human-in-the-loop trajectory analysis for AI agents. Summarize, embed, and visualize thousands of agent actions to find patterns across models and configurations. Supports common evaluation formats and any LiteLLM-compatible model (e.g. OpenAI, Gemini, Google, Anthropic, etc.) for summarization and embedding.

[Homepage](https://hodoscope.dev) · [Announcement blog](https://hodoscope.dev/blog/announcement.html)

## Why Hodoscope?

Running evals across multiple models and configurations produces a mountain of raw logs, but reading them one-by-one doesn't scale. Hodoscope gives you a bird's-eye view: it extracts every agent action from your eval trajectories, summarizes each one with an LLM, embeds the summaries into a shared vector space, and then projects them into interactive 2D plots. The result is a visual map where you can spot behavioral clusters, group by any metadata field, and use density overlays to see exactly where two groups of trajectories diverge or converge. No labels or pre-defined taxonomies required.

## Features

- **Multiple supported formats** — [Inspect AI](https://inspect.ai-safety-institute.org.uk/) `.eval` files, [OpenHands](https://github.com/All-Hands-AI/OpenHands) JSONL trajectories, [Docent](https://github.com/docent-ai/docent) collections, and raw trajectory JSONs
- **Summarization & embedding** — distill raw agent actions into concise natural-language summaries and embed them via any LLM supported by [LiteLLM](https://docs.litellm.ai/)
- **Dimensionality reduction** — project embedded summaries into interactive 2D scatter plots with t-SNE (recommended), PCA, UMAP, TriMap, or PaCMAP
- **Density diffing and overlay** — overlay difference in kernel density estimates to visualize where trajectory distributions differ
- **Flexible grouping** — group summaries by any metadata field (`--group-by model`, `--group-by score`, `--group-by task`, etc.) to compare
- **Resumable processing** — interrupt and resume long analysis runs with `--resume`; already-processed trajectories are skipped
- **Python API** — every CLI command maps to a public function you can call directly in notebooks or scripts

## How It Works

```
source file ─→ actions ─→ summarize ─→ embed ─→ distribution diffing ─→ visualize
```

Hodoscope takes an unsupervised approach with three main steps:

1. **Summarization** — Condense each agent action into a high-level summary that captures the behavior rather than setup-specific details (e.g., "edit test file to fix assertion" instead of "modify /home/user/repo/django/tests/utils.py line 42 to assert error_code == 403").

2. **Embedding** — Embed summaries into a shared vector space where similar behaviors (e.g., "run test suite" or "edit source file") end up close together, then project into 2D via t-SNE for visualization.

3. **Distribution diffing** — Compute and compare kernel density estimates (KDE) of different agent setups where abnormal behaviors are expected to manifest only in some setups. For example, we may compare agents using different LLMs on the same task. Overlaying these density differences on the embedding visualization helps identify behaviors unique to one setup.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Trajectory Format](#trajectory-format)
- [Output Format](#output-format)
- [Testing](#testing)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## Prerequisites

- Python 3.11+
- By default: An **OpenAI** and a **Gemini** API key for summarization and embedding
- It's also possible to use other LLM API keys. For example, a single **[OpenRouter](https://openrouter.ai/)** API key

## Installation

```bash
pip install hodoscope
```

For development (editable install with tests):

```bash
pip install -e ".[dev]"
```

## Configuration

Create a `.env` file in the project root. Hodoscope loads it automatically at startup.

```
OPENAI_API_KEY=your-openai-key
GEMINI_API_KEY=your-gemini-key

# Optional: override defaults
# ⚠️ Default summarization model (gpt-5.2) could be expensive!
# SUMMARIZE_MODEL=openai/gpt-5.2
# EMBEDDING_MODEL=gemini/gemini-embedding-001
# MAX_WORKERS=10
```

You can also export these variables directly in your shell instead of using a `.env` file.

**Using OpenRouter (single API key):** If you prefer to use an OpenRouter key for both summarization and embedding, set `OPENROUTER_API_KEY` and prefix your model names with `openrouter/`:

```
OPENROUTER_API_KEY=your-openrouter-key
SUMMARIZE_MODEL=openrouter/openai/gpt-5.2
EMBEDDING_MODEL=openrouter/gemini/gemini-embedding-001
```

## Quick Start

```bash
# Analyze a single .eval file
hodoscope analyze run.eval

# Analyze all trajectory files in a directory
hodoscope analyze evals/

# Compare models
hodoscope viz model_*.hodoscope.json --group-by model --open

# Visualize a single result
hodoscope viz run.hodoscope.json --open
```

## CLI Reference

### `hodoscope analyze`

Process source files (.eval, directories, Docent collections) into `.hodoscope.json` analysis files.

```bash
hodoscope analyze SOURCES [OPTIONS]

Options:
  --docent-id TEXT          Docent collection ID as source
  -o, --output TEXT         Output JSON path (single source only)
  --field TEXT              KEY=VALUE metadata (repeatable)
  -l, --limit INTEGER       Limit trajectories per source
  --save-samples PATH       Save extracted trajectory JSONs to directory
  --embed-dim INTEGER       Embedding dimensionality (default: follow API default)
  -m, --model-name TEXT     Override auto-detected model name
  --summarize-model TEXT    LiteLLM model for summarization (default: openai/gpt-5.2)
  --embedding-model TEXT    LiteLLM model for embeddings (default: gemini/gemini-embedding-001)
  --sample / --no-sample    Randomly sample trajectories (use with --limit)
  --seed INTEGER            Random seed for --sample reproducibility
  --resume / --no-resume    Resume from existing output (default: on)
  --reasoning-effort [low|medium|high]
                            Reasoning effort for summarization model
  --max-workers INTEGER     Max parallel workers for LLM calls (default: 10)
  --reembed                 Re-embed existing summaries (e.g. after changing embedding model/dim)
```

Examples:

```bash
hodoscope analyze run.eval                             # .eval → analysis JSON
hodoscope analyze *.eval                               # batch: all .eval files
hodoscope analyze evals/                               # batch: dir of .eval files
hodoscope analyze run.eval -o my_output.hodoscope.json  # custom output path
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
  --proj TEXT         Projection methods: pca, tsne, umap, trimap, pacmap
                      (comma-separated or repeated; * or all for all; default: tsne)
  -o, --output TEXT   Output HTML file path (default: auto-generated timestamped name)
  --filter TEXT       KEY=VALUE metadata filter (repeatable, AND logic)
  --open              Open the generated HTML in the default browser
```

Examples:

```bash
hodoscope viz output.hodoscope.json                    # visualize a single analysis file (grouped by model)
hodoscope viz *.hodoscope.json --group-by score        # group by score field
hodoscope viz *.hodoscope.json --proj tsne,umap        # specific projection methods
hodoscope viz *.hodoscope.json --proj '*'              # all methods (will be slow!)
hodoscope viz *.hodoscope.json --filter score=1.0      # only score=1.0 ones
hodoscope viz *.hodoscope.json --open                  # open in default browser
```

### `hodoscope sample`

Sample representative summaries using density-weighted Farthest Point Sampling on 2D projections.

> **Note:** While this command could be useful for scripting and automated pipelines, we find the interactive visualization (`hodoscope viz`) to be more intuitive and effective for human-in-the-loop explorations.

```bash
hodoscope sample SOURCES [OPTIONS]

Options:
  --group-by TEXT       Field to group by (default: model)
  -n, --samples-per-group INTEGER
                        Number of representative samples per group (default: 10)
  --proj TEXT           Projection method for FPS ranking (pca, tsne, umap, trimap, pacmap; default: tsne)
  -o, --output TEXT     JSON output file (default: paginated terminal display)
  --interleave          Interleave groups by rank (#1 from each group, then #2, etc.)
  --filter TEXT         KEY=VALUE metadata filter (repeatable, AND logic)
```

Examples:

```bash
hodoscope sample *.hodoscope.json                         # suggest 10 per group
hodoscope sample *.hodoscope.json --group-by score -n 5   # suggest 5 per score group
hodoscope sample *.hodoscope.json --proj pca              # use PCA projection
hodoscope sample *.hodoscope.json -o sampled.json         # write JSON output
hodoscope sample *.hodoscope.json --interleave            # interleave groups by rank for easier comparison
hodoscope sample *.hodoscope.json --filter score=1.0      # only score=1.0 summaries
```

### `hodoscope info`

Show metadata, summary counts, and API key status for analysis JSON files.

```bash
hodoscope info output.hodoscope.json
hodoscope info results/
```

## Trajectory Format

Hodoscope first converts other trajectory sources (`.eval` files, Docent collections, etc.) to the canonical JSON format before processing. You can also pass trajectories directly in this format:

```json
{
  "id": "unique-trajectory-id",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    ...
  ],
  "metadata": {...}
}
```

## Output Format

`hodoscope analyze` produces `.hodoscope.json` files:

```json
{
  "version": 1,
  "created_at": "...",
  "source": "path/to/run.eval",
  "fields": {"model": "gpt-5", "task": "swe_bench", "accuracy": 0.8, "...": "..."},
  "embedding_model": "gemini/gemini-embedding-001",
  "embedding_dimensionality": 768,
  "summaries": [
    {
      "trajectory_id": "django__django-12345_epoch_1",
      "turn_id": 3,
      "summary": "Update assertion to match expected output",
      "action_text": "...",
      "task_context": "...",
      "embedding": "<base85-encoded float32 array>",
      "metadata": {"score": 1.0, "instance_id": "django__django-12345", "...": "..."}
    },
    "..."
  ]
}
```

Key concepts:
- **`fields`**: File-level metadata auto-detected from .eval header (model, task, dataset_name, solver, run_id, accuracy, etc.) plus custom `--field` values. Same for all summaries.
- **`metadata`**: Per-trajectory metadata. All `sample.metadata` keys from .eval files are passed through, plus extracted keys (score, epoch, target, token usage, etc.). Varies per summary.
- **`--group-by` resolution**: Checks per-summary `metadata` first, then file-level `fields`.
- **`embedding`**: RFC 1924 base85-encoded `float32` numpy array.

## Testing

```bash
# Run the full test suite
pytest

# Unit tests only (no API keys needed)
pytest tests/test_io.py tests/test_viz.py tests/test_api.py tests/test_sampling.py

# End-to-end tests (requires API keys)
pytest tests/test_analyze.py
```

## Contributing

Contributions are welcome! We recommend opening an issue to discuss what you'd like to change before submitting a pull request.

## Citation

```bibtex
@article{zhong2026hodoscope,
  title={Hodoscope: Unsupervised Behavior Discovery in AI Agents},
  author={Zhong, Ziqian and Saxena, Shashwat and Raghunathan, Aditi},
  year={2026},
  url={https://hodoscope.dev/blog/announcement.html}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
