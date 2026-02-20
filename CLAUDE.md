**NOTE:** You should **always** update CLAUDE.md whenever you make changes to the codebase.

# Hodoscope

## Project Status

Installable Python library + CLI (`hodoscope`) for analyzing AI agent trajectories. Extracts actions, summarizes with LLMs, embeds, and creates interactive visualizations. v2 uses single-file JSON output with arbitrary grouping. Uses LiteLLM for provider-agnostic LLM/embedding calls.

## Installation & CLI

```bash
pip install -e .           # Install package + CLI
pip install -e ".[dev]"    # With test dependencies
hodoscope --help           # Show all commands
hodoscope analyze          # Process sources → .hodoscope.json
hodoscope viz              # Visualize analysis JSONs
hodoscope sample           # FPS-based representative sampling
hodoscope info             # Show metadata
```

## File Structure

```
pyproject.toml                        # Package config, entry point, deps
README.md                             # Human-facing docs

hodoscope/                  # Main library package
├── __init__.py                       # Public API exports
├── config.py                         # Config dataclass, defaults, env loading
├── cli.py                            # Click-based CLI entry point (4 commands)
├── pipeline.py                       # Orchestration logic (analyze, viz, sample, info)
├── sampling.py                       # FPS-based sampling & projection utilities
├── io.py                             # JSON I/O, base85 encoding, grouping, filtering
├── core.py                           # Shared utilities (embed_text, run_parallel)
├── parsers.py                        # Trajectory parsing
├── actions.py                        # Action processing & summarization
├── visualization.py                  # Interactive visualizations
├── openhands/                        # OpenHands evaluation result integration
│   ├── __init__.py
│   └── convert_to_trajectory.py      # Convert OpenHands JSONL → universal format
├── docent/                           # Docent integration
│   ├── __init__.py
│   ├── export_transcripts.py         # Export from Docent collections
│   └── convert_to_trajectory.py      # Convert Docent → universal format
└── eval/                             # .eval file integration (Inspect AI)
    ├── __init__.py
    └── convert_to_trajectory.py      # Convert .eval → universal format

tests/                                # Test suite
├── __init__.py
├── conftest.py                       # Shared fixtures
├── test_io.py                        # I/O round-trip tests (no API)
├── test_api.py                       # Public Python API tests (no API)
├── test_analyze.py                   # End-to-end pipeline tests (needs API)
├── test_viz.py                       # Visualization tests (no API)
├── test_sampling.py                  # FPS sampling tests (no API, clustered mock data)
└── sample_evals/                     # Test data
    ├── sample1.eval                  # 10 samples, gpt-4o-mini, popularity
    └── sample2.eval                  # 10 samples, gpt-4o-mini, popularity
```

## Module Overview

### `hodoscope/config.py`
**Purpose:** Centralized configuration — all defaults and env loading.

**Constants (single source of truth):**
- `DEFAULT_SUMMARIZE_MODEL` — `"openai/gpt-5.2"`
- `DEFAULT_EMBEDDING_MODEL` — `"gemini/gemini-embedding-001"`
- `DEFAULT_EMBEDDING_TASK_TYPE` — `"RETRIEVAL_DOCUMENT"`
- `DEFAULT_EMBED_DIM` — `None`
- `DEFAULT_MAX_WORKERS` — `10`
- `DEFAULT_GROUP_BY` — `"model"`
- `DEFAULT_FPS_ALPHA` — `1.0`
- `DEFAULT_FPS_BETA` — `0.1`

**Helpers:**
- `_load_env()` — Load `.env` from project root or cwd

**Config dataclass:**
- `Config(summarize_model, embedding_model, embed_dim, max_workers, reasoning_effort, normalize_embeddings, summarize_prompt, fps_alpha, fps_beta)` — Processing configuration with hardcoded defaults. `embed_dim=None` means "respect the API return dimensionality." `summarize_prompt=None` means "use the default SWE-oriented prompt" (`DEFAULT_SUMMARIZE_PROMPT` from `actions.py`). `fps_alpha` and `fps_beta` control FPS density-weighted ranking behavior.
- `Config.from_env(**overrides)` — Resolve from `.env` + env vars, apply explicit overrides (None values skipped). This is the **only** place that does env-loading side effects. Does **not** validate API keys (LiteLLM will report errors at call time).

---

### `hodoscope/cli.py`
**Purpose:** Click-based CLI entry point (`hodoscope` command)

4 commands: `analyze`, `viz`, `sample`, `info`. The `analyze` command builds a `Config.from_env()` from CLI args and passes it to `pipeline.analyze()`. CLI options no longer use `envvar=` — env var resolution is handled entirely by `Config.from_env()`.

**Helpers:**
- `_build_filter(filter_strings)` — Parse `--filter KEY=VALUE` tuples into a callable predicate (returns `None` if empty). Checks `summary['metadata'][key]` with string and numeric comparison. All filters AND'd. Used by `viz` and `sample` commands.

---

### `hodoscope/pipeline.py`
**Purpose:** Public Python API and CLI orchestration.

**Public API (composable building blocks):**
- `load_eval(path, limit, sample, seed, save_samples)` — Load trajectories from .eval file. Returns `(trajectories, fields)`. Accepts `str | Path`.
- `load_trajectory_dir(path, limit, sample, seed)` — Load trajectories from directory of JSONs. Returns `(trajectories, fields)`. Accepts `str | Path`.
- `load_openhands(path, limit, sample, seed, save_samples)` — Load trajectories from an OpenHands `.jsonl` file. Reads JSONL lines, converts V1 SDK events to messages. Reads sibling `report.json` for accuracy stats and sibling `metadata.json` as fallback for model/dataset when instance-level metadata is null. Returns `(trajectories, fields)`. Accepts `str | Path`.
- `load_docent(collection_id, limit, sample, seed, save_samples)` — Load trajectories from Docent collection. Returns `(trajectories, fields)`. Uses `extract_docent_fields` on raw transcripts for file-level fields.
- `process_trajectories(trajectories, config, skip_set, save_callback, save_interval)` — Extract actions → summarize → embed. `config` defaults to `Config()` (hardcoded defaults, no env magic). All trajectory `metadata` keys with non-None values are passed through to summaries. Returns `list[dict]` of summaries.
- `extract_actions(messages)` — Messages → actions (no LLM calls). Composes `turns_from_messages` → `merge_consecutive_turns` → `extract_actions_from_turns`.
- `HodoscopeError` — Exception class for pipeline errors (caught by CLI layer)

**High-level orchestrators (used by CLI):**
- `analyze(sources, docent_id, output, fields, limit, save_samples, model_name, sample, seed, resume, config, reembed)` — End-to-end: sources → .hodoscope.json files. `config` defaults to `Config.from_env()` (convenient CLI-like behavior). `reembed=True` re-embeds existing summaries using current config (e.g. after changing embedding model/dim). Calls `load_eval`/`load_trajectory_dir`/`load_openhands`/`load_docent` → `process_trajectories` → `write_analysis_json`.
- `viz(sources, group_by, proj, output_file, alpha, beta, filter)` — Visualize analysis JSONs (single unified HTML). Returns the output `Path`. `alpha`/`beta` default to `None` (resolved from `FPS_ALPHA`/`FPS_BETA` env vars or hardcoded defaults). `filter` accepts a `Callable[[dict], bool]` to filter summaries before grouping.
- `sample(sources, group_by, n, method, alpha, beta, output, interleave, filter)` — FPS-based representative sampling from analysis JSONs. Paginated terminal output (grouped or interleaved by rank) or lightweight JSON (no embeddings). `filter` accepts a `Callable[[dict], bool]` to filter summaries before grouping.
- `show_info(sources)` — Print metadata and summary counts

**Internal Helpers:**
- `_resolve_sources(sources, docent_id)` — Resolve CLI args into source descriptors (for analyze)
- `_resolve_analysis_sources(sources)` — Resolve source args to `.hodoscope.json` paths (shared by viz/sample)

---

### `hodoscope/io.py`
**Purpose:** JSON I/O for .hodoscope.json files, base85 encoding, grouping, filtering

**Key Functions:**
- `write_analysis_json(path, summaries, fields, source, ...)` — Serialize summaries to JSON with base85 embeddings. `embedding_dimensionality` defaults to `DEFAULT_EMBED_DIM` (`None`), accepts `None`.
- `read_analysis_json(path)` → dict with decoded numpy embeddings
- `encode_embedding(np_array)` → RFC 1924 base85 string
- `decode_embedding(b85_string)` → np.ndarray (float32)
- `filter_summaries(summaries, predicate)` → filtered list of summaries. Exported for direct Python use.
- `group_summaries_from_list(summaries, group_by, fallbacks, filter)` → `{label: [summaries]}` dict (single implementation). Resolution: per-summary `metadata`, then each dict in `fallbacks` in order, then `"unknown"`. Logs a warning when summaries without embeddings are dropped. Accepts deprecated `default_fields` kwarg (equivalent to `fallbacks=[default_fields]`). If `filter` (callable) is provided, applies it before grouping.
- `group_summaries(analysis_docs, group_by, filter)` → `{label: [summaries]}` dict for viz (from analysis docs). Delegates to `group_summaries_from_list` per doc with `fallbacks=[file_fields, {group_by: doc-level value}]`. Passes `filter` through.

---

### `hodoscope/core.py`
**Purpose:** Shared utility functions used across all modules

**Key Functions:**
- `extract_actions_from_turns()` - Extract tool calls from conversation turns
- `embed_text(text, model, ..., output_dimensionality, normalize)` - Single source of truth for embeddings (uses LiteLLM, exponential backoff, supports `output_dimensionality`). Preprocesses text (normalizes line endings, strips blank lines). `normalize` param controls whether embeddings are L2-normalized (default: True).
- `run_parallel()` - Parallel execution with progress bars and automatic final save

Constants are imported from `config.py` (`DEFAULT_EMBEDDING_MODEL`, `DEFAULT_EMBEDDING_TASK_TYPE`).

---

### `hodoscope/parsers.py`
**Purpose:** Parse and extract data from trajectory messages

**Key Functions:**
- `turns_from_messages(messages)` - Parse message list into turns (canonical implementation)
- `turns_from_trajectory_file(traj_file)` - Parse trajectory JSON file into turns
- `extract_task_context(messages)` - Extract task context (first system + first user content, joined with `\n\n---\n\n`). Handles both `str` and `list` content formats. Returns empty string if neither is found.
- `merge_consecutive_turns(turns)` - Merge consecutive tool/user turns

---

### `hodoscope/actions.py`
**Purpose:** Action summarization and embedding pipeline

**Constants:**
- `DEFAULT_SUMMARIZE_PROMPT` — The default SWE-oriented system prompt for summarization. Exported from `__init__.py`. `SUMMARIZE_SYS_PROMPT` is a backwards-compatible alias.

**Key Functions:**
- `summarize_action(action_text, model, reasoning_effort, system_prompt)` - Generate action summary via LiteLLM. Passes action_text directly as user message. `system_prompt` defaults to `DEFAULT_SUMMARIZE_PROMPT`.
- `summarize_action_only(args)` - Summarize without embedding. Args tuple: `(action, config, idx, traj_id, traj_metadata)`. Uses `config.summarize_prompt` if set, otherwise the default prompt. Passes through `action.get('task_context', '')` into the result dict.
- `embed_summary(summary_dict, config)` - Add embedding to summary dict using config settings
- `process_single_action(args)` - Combined summarize + embed. Tuple: `(action, config, idx, traj_id, traj_metadata)`

---

### `hodoscope/sampling.py`
**Purpose:** Standalone FPS-based sampling for representative action selection. Shared by both visualization (interactive flagging) and `hodoscope sample` CLI.

**Constants:**
- `ALL_PLOT_METHODS` — `['pca', 'tsne', 'umap', 'trimap', 'pacmap']`
- `SAMPLING_METHOD_DISPLAY_NAMES` — Human-readable names for methods (e.g., `'tsne'` → `'t-SNE'`)
- `UNRANKED_SENTINEL` — `10**9`, rank value for unranked points in FPS output

**Dataclasses:**
- `PlotData` — Parallel arrays (X, labels, type_names, trajectory_ids, trajectory_uuids, turn_ids, summaries, action_texts, task_contexts)

**Public Functions:**
- `collect_plot_data(summaries_by_type)` → `PlotData` — Extract plotting arrays from grouped summaries
- `compute_projection(X, method, labels=None)` → `np.ndarray` (N, 2) — Run a single dim-reduction. When `labels` is provided, smaller groups are oversampled (with tiny noise) to match the largest group before fitting, so all groups have equal influence on the layout. Only the original N point positions are returned. With equal-sized groups this is a no-op. t-SNE uses sklearn with `n_jobs=-1` for parallelism
- `compute_bandwidth(X_2d)` → `float` — Scott's rule with floor of 0.5
- `compute_kde_densities(X_2d, labels, n_categories, bandwidth)` → `list[np.ndarray]` — Per-category KDE at all points
- `compute_fps_ranks(X_2d, labels, n_categories, point_densities=None, alpha=1.0, beta=0.1, max_per_group=500, bandwidth=None)` → `list[int]` — Density-weighted FPS ranking. Distances computed in axis-normalized (unit-variance) 2D space; density gaps piecewise-scaled with zero mapped to `beta` (negative → [0, beta], positive → [beta, 1]). Accepts optional precomputed `point_densities` to avoid recomputing KDE.
- `rank_summaries(summaries_by_group, method='tsne', n=None, alpha=1.0, beta=0.1, bandwidth=None)` → `{label: [dicts sorted by fps_rank]}` — High-level API: projects, ranks, adds `fps_rank` key, sorts. If `n` given, truncates per group.

---

### `hodoscope/visualization.py`
**Purpose:** Create a unified interactive visualization of action summary embeddings. Imports projection/FPS logic from `sampling.py`.

Produces a single HTML file (default: `trajectory_explorer_{timestamp}.html`) with:
- **Method switcher dropdown** ("Projection:") — switch between dim-reduction projections client-side; switching re-computes heatmap and FPS inline
- **Density heatmap overlay** ("Density overlay:") — RdBu palette at 40% opacity; defaults to first category selected ("None" at end of dropdown); works with single groups too (shows density vs zero)
- **Suggest samples slider** — FPS-based filtering; □/■ indicators in widget titles auto-switch between search and suggest modes
- **Search** — text search across summaries and raw actions ("All text", "Summaries", or "Actions" mode). Supports regex queries via `re:pattern` or `/pattern/flags` (case-insensitive by default, `g` ignored). Enter key always triggers search mode, updates the □/■ mode indicators, and refreshes results even when query text is unchanged (via explicit submit callback). Changing the "Within" dropdown also switches to search mode.
- **Click-to-inspect** — resizable detail panel with structured metadata, summary, and full action text. Selected point highlighted as inverted triangle with category color and black outline. Metadata includes FPS order and density gap (`own_density - mean(other_density)`) shown as `(+)`/`(-)` with magnitude.
- **Click-to-inspect** — resizable detail panel with structured metadata, summary, and full action text. Selected point highlighted as inverted triangle with category color and black outline. Metadata includes FPS order and density gap (`own_density - mean(other_density)`) shown as `(+)`/`(-)` with magnitude. The `Turn` row includes `Prev turn` / `Next turn` buttons that navigate within the same trajectory using `trajectory_uuid` (disambiguated by source file) across points. Display still shows clean `trajectory_id`. After the "Original Action" section, a collapsible "Task Context" section shows the system prompt + first user message (when non-empty). Collapse state persists across point clicks via `window._hodoContextCollapsed` (defaults to collapsed).
- **Navigation** — eye icon in the status area (top controls) is non-nearest: in suggest/rank mode it randomly selects among global lowest-FPS visible point(s) across all visible classes, preferring a different point than the current one when possible. In search mode, status-eye jumps to a random visible active point. The eye icon in the detail panel (shown when a selected point is not active) uses nearest-visible-point logic. Detail panel Prev/Next keeps the current class first (ordered by FPS rank within that class), and only switches class at class boundaries. Active set = visible points in either suggest or search mode.
- **Selection state** — clicking empty plot space clears selection highlight and closes the detail panel.
- **Legend-aware counts** — status counts ("N shown"/"N matches"/"N total") respect legend visibility and update when legend items are toggled.

All controls in a single row matching plot width. Status area shows "N total", "N matches", or "N shown" depending on mode, with top-eye random navigation, and the eye is visible at initial page load before any interaction. The status area stretches to fill the remaining right-side control space so longer text is not cramped. Detail panel metadata includes FPS order as `#N`, or `#N+` for unranked points beyond the ranked cap. Mode auto-switches when interacting with slider (suggest) or search input. Page title: "Trajectory Explorer - Hodoscope".

**Constants:**
- `DEFAULT_COLORS` — 12-color palette used by all plots
- `SIDEBAR_HTML` — HTML/CSS/JS for the click-to-inspect detail panel

**Dataclasses:**
- `MethodProjection` — Pre-computed projection: name, display_name, X_2d, x/y ranges, density_grids, fps_ranks, grid_size

**Internal Helpers:**
- `_compute_method_projection(data, method, grid_size, bandwidth)` → `MethodProjection` — Projection + KDE grids + FPS ranks (delegates to `sampling` for projection and FPS)
- `_make_scatter_sources(p, X_2d, data, ...)` → `(sources, renderers, source_masks)` — Per-category scatter loop + HoverTool
- `_add_tap_callback(sources)` — Wire tap-to-detail-panel JS
- `_make_search_widgets()` → `(search_input, search_mode, match_div)`
- `_build_unified_bokeh(method_projections, data, html_path)` — Build the single unified HTML
- `_save_with_sidebar(layout, html_path, title, external_data)` — Save HTML + append sidebar + inject external data as `<script>` tags

**Public Function:**
- `visualize_action_summaries(summaries_by_type, output_file=None, methods=None, grid_size=80, bandwidth=None, alpha=1.0, beta=0.1)` → `Path` — Unified visualization. Returns the output HTML path.
  - Dict order = render order: last key on top
  - `methods`: `'pca'`, `'tsne'`, `'umap'`, `'trimap'`, `'pacmap'` (default: `['tsne']`)
  - All methods pre-computed in Python; switching is instant in JS
  - `output_file`: path for output HTML; `None` → `trajectory_explorer_{YYYYMMDD_HHMMSS}.html` in CWD

---

### `hodoscope/openhands/`
**Purpose:** Import trajectory data from OpenHands evaluation results (used by `load_openhands`)

Supports the **V1 SDK format** (OpenHands/benchmarks) where events in the `history` array use Pydantic `DiscriminatedUnionMixin` with `kind` = class name: `SystemPromptEvent`, `MessageEvent`, `ActionEvent`, `ObservationEvent`, `ConversationStateEvent`. Action objects also have `kind` (e.g. `TerminalAction`, `FileEditorAction`, `BrowserUseAction`, `ThinkAction`, `FinishAction`). The **V0 runtime format** (`{"action": "run", "args": {...}}`, no `kind` field) is detected and warned about but not converted.

OpenHands JSONL files are first-class sources: pass a `.jsonl` file directly or a directory containing them. Auto-detected by peeking at the first line for `instance_id` + `history` keys (via `_is_openhands_jsonl()`).

**Key Functions (`convert_to_trajectory.py`):**
- `convert_openhands_instance(instance)` — Convert a single JSONL instance to trajectory format. Maps V1 events to messages: `SystemPromptEvent` → system, `MessageEvent` → user/assistant, `ActionEvent` → assistant with tool_calls, `ObservationEvent` → tool. Detects V0 format and logs a warning. Absorbs all fields from `test_result`, `metrics`, and `instance` sub-objects into per-trajectory metadata generically.
- `extract_openhands_fields(first_instance, report, file_metadata)` — Extract file-level fields. Absorbs all keys from `file_metadata` (lowest priority), then instance metadata (overrides). Extracts clean model name via `_extract_model` (checks `model` key first, then `llm.model`, strips `litellm_proxy/` prefix). Uses `report.json` for accuracy/sample count.

**Source detection in pipeline:**
- `_is_openhands_jsonl(path)` — Peek first line of a `.jsonl` file, check for `instance_id` + `history` keys

---

### `hodoscope/docent/`
**Purpose:** Import transcript data from Docent collections (used by `load_docent`)

**Key Functions (`convert_to_trajectory.py`):**
- `convert_transcript(transcript)` — Convert a single Docent transcript to trajectory format. Passes through all `agent_run_metadata` keys into per-trajectory metadata. Special cases: `score` ← `scores.resolved`, `model` ← `model_name_or_path` (top-level), `created_at` from transcript.
- `extract_docent_fields(transcripts)` — Extract file-level fields from raw transcripts (before conversion). Analogous to `extract_eval_fields`. Returns `model` and `trajectory_format` from first transcript's `agent_run_metadata`.

### `hodoscope/eval/`
**Purpose:** Import trajectory data from .eval files (used by `load_eval`)

**Key internal functions (reused by pipeline):**
- `_detect_model(header, sample)` — Extract model name from .eval header
- `extract_eval_fields(header, sample)` — Extract comprehensive file-level fields (model, task, dataset_name, dataset_location, dataset_samples, dataset_shuffled, solver, run_id, run_status, run_created, accuracy)
- `_load_scores(zf)` — Load per-sample scores from reductions.json (returns `{sample_id: {"value": float, "answer": str, "explanation": str}}`)
- `convert_eval_sample(sample, score_info)` — Convert single sample to trajectory format with rich metadata (target, token usage, response_time, score_answer, score_explanation, full sample.metadata passthrough)

## Model Configuration

All processing config is centralized in the `Config` dataclass. Resolution order:

```
CLI flag  >  .env variable  >  hardcoded default
```

This resolution happens in `Config.from_env(**overrides)` — the single place that performs env-loading side effects. `Config()` uses hardcoded defaults only (no env magic).

| Setting | CLI flag | Env var | Default |
|---------|----------|---------|---------|
| Summarization model | `--summarize-model` | `SUMMARIZE_MODEL` | `openai/gpt-5.2` |
| Embedding model | `--embedding-model` | `EMBEDDING_MODEL` | `gemini/gemini-embedding-001` |
| Embedding dimensions | `--embed-dim` | `EMBED_DIM` | `None` (API default) |
| Max parallel workers | `--max-workers` | `MAX_WORKERS` | `10` |
| Reasoning effort | `--reasoning-effort` | `REASONING_EFFORT` | `None` |
| Normalize embeddings | — | `NORMALIZE_EMBEDDINGS` | `false` |
| FPS distance exponent | — | `FPS_ALPHA` | `1.0` |
| FPS density gap floor | — | `FPS_BETA` | `0.1` |

Example `.env`:
```
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...

# Optional: override default models
SUMMARIZE_MODEL=anthropic/claude-sonnet-4-5-20250929
EMBEDDING_MODEL=gemini/gemini-embedding-001
EMBED_DIM=768
MAX_WORKERS=10

# Optional: FPS tuning
FPS_ALPHA=1.0
FPS_BETA=0.1
```

## CLI Commands

### `hodoscope analyze`
```bash
hodoscope analyze run.eval                             # .eval → analysis JSON
hodoscope analyze *.eval                               # batch: all .eval files
hodoscope analyze evals/                               # batch: dir of .eval files
hodoscope analyze run.eval -o my_output.json           # custom output path
hodoscope analyze run.eval --field env=prod            # add custom metadata
hodoscope analyze run.eval --save-samples ./samples/   # save extracted trajectories
hodoscope analyze --docent-id COLLECTION_ID            # docent source
hodoscope analyze path/to/samples/                     # directory of trajectory JSONs
hodoscope analyze run.eval --summarize-model gemini/gemini-2.0-flash  # custom summarizer
hodoscope analyze run.eval --embedding-model gemini/gemini-embedding-001 --embed-dim 256
hodoscope analyze run.eval --limit 5                    # random 5 trajectories (sample is default)
hodoscope analyze run.eval --limit 5 --seed 42           # reproducible random sample
hodoscope analyze run.eval --limit 5 --no-sample         # first 5 trajectories (no randomization)
hodoscope analyze run.eval --no-resume                 # overwrite existing output (default: resume)
hodoscope analyze run.eval --reasoning-effort low      # low reasoning effort for summarization
hodoscope analyze run.eval --max-workers 20            # increase parallelism
hodoscope analyze run.eval --reembed --embedding-model openai/text-embedding-3-small --embed-dim 256  # re-embed with new model
hodoscope analyze output.jsonl                         # OpenHands JSONL file (auto-detected)
hodoscope analyze results/                             # batch: all .jsonl files in dir
```

### `hodoscope viz`
```bash
hodoscope viz output.hodoscope.json                              # visualize (groups by model, default: tsne)
hodoscope viz output.hodoscope.json --group-by task              # group by task
hodoscope viz output.hodoscope.json --group-by score             # group by score field
hodoscope viz results/                                           # batch: all JSONs in dir
hodoscope viz a.hodoscope.json b.hodoscope.json --group-by model # cross-file comparison
hodoscope viz output.hodoscope.json --proj tsne pca umap          # multiple methods in one plot
hodoscope viz output.hodoscope.json --proj '*'                    # all methods (pca,tsne,umap,trimap,pacmap)
hodoscope viz output.hodoscope.json -o my_plot.html               # custom output path
hodoscope viz output.hodoscope.json --filter score=1.0            # only score=1.0 summaries
hodoscope viz output.hodoscope.json --filter score=1.0 --filter epoch=1  # AND logic
hodoscope viz output.hodoscope.json --open                               # open in default browser
```

### `hodoscope sample`
```bash
hodoscope sample output.hodoscope.json                               # top 10 per group (paginated terminal)
hodoscope sample output.hodoscope.json --group-by score -n 5         # top 5 per score group
hodoscope sample output.hodoscope.json --proj pca                    # use PCA projection
hodoscope sample output.hodoscope.json -o sampled.json               # write lightweight JSON (no embeddings)
hodoscope sample a.hodoscope.json b.hodoscope.json --group-by model  # cross-file sampling
hodoscope sample a.hodoscope.json b.hodoscope.json --interleave      # interleave groups by rank
hodoscope sample output.hodoscope.json --filter score=1.0            # only score=1.0 summaries
```

### `hodoscope info`
```bash
hodoscope info output.json                             # show metadata
hodoscope info results/                                # scan directory
```

## Output JSON Format

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
    "run_id": "2wRuCQ9h...",
    "accuracy": 0.8
  },
  "embedding_model": "gemini/gemini-embedding-001",
  "embedding_dimensionality": 768,
  "summaries": [
    {
      "trajectory_id": "django__django-12345_epoch_1",
      "turn_id": 3,
      "summary": "Update assertion to match expected output",
      "action_text": "...",
      "task_context": "You are a coding assistant...\n\n---\n\nFix the failing test in test_utils.py",
      "embedding": "<base85-encoded float32 array>",
      "metadata": {
        "score": 1.0,
        "epoch": 1,
        "instance_id": "django__django-12345",
        "target": "expected output",
        "input_tokens": 620,
        "output_tokens": 20,
        "total_tokens": 640,
        "response_time": 1.23
      }
    }
  ]
}
```

- **`fields`**: File-level metadata. Set via `--field` + auto-detected from .eval header (model, task, dataset_name, dataset_location, dataset_samples, dataset_shuffled, solver, run_id, run_status, run_created, accuracy).
- **`metadata`**: Per-trajectory metadata. For .eval sources: all `sample.metadata` keys are passed through, plus extracted keys: score, epoch, instance_id, target, input_tokens, output_tokens, total_tokens, response_time, score_answer, score_explanation. For Docent sources: all `agent_run_metadata` keys are passed through, plus: score (from `scores.resolved`), epoch, instance_id, sandbox, docent_id, docent_collection_id, docent_agent_run_id, created_at.
- **`--group-by` resolution**: Per-summary `metadata` first, then file-level `fields`, then top-level doc keys (e.g. `source`).
- **`embedding`**: RFC 1924 base85-encoded `float32` numpy array.

## Python API

The library exposes composable building blocks for use in Python data pipelines:

```python
import hodoscope

# Load from .eval (no disk output)
trajectories, fields = hodoscope.load_eval("run.eval", limit=5)  # sample=True by default

# Or load from directory of trajectory JSONs
trajectories, fields = hodoscope.load_trajectory_dir("path/to/samples/")

# Summarize + embed with explicit config (no env side effects)
config = hodoscope.Config(summarize_model="openai/gpt-4o", embed_dim=256)
summaries = hodoscope.process_trajectories(trajectories, config=config)

# Custom summarization prompt (e.g. for non-SWE domains)
config = hodoscope.Config(summarize_prompt="Summarize this customer support action in ~10 words.")
summaries = hodoscope.process_trajectories(trajectories, config=config)

# Or use Config.from_env() for CLI-like behavior (loads .env)
config = hodoscope.Config.from_env(summarize_model="openai/gpt-4o")
summaries = hodoscope.process_trajectories(trajectories, config=config)

# Extract actions only (no LLM calls)
actions = hodoscope.extract_actions(trajectories[0]["messages"])

# Filter summaries (Python API)
filtered = hodoscope.filter_summaries(summaries, lambda s: s["metadata"]["score"] == 1.0)

# Group + visualize in-memory summaries
grouped = hodoscope.group_summaries_from_list(summaries, group_by="score")
hodoscope.visualize_action_summaries(grouped, "plots/explorer.html", methods=["tsne"])

# FPS-based sampling: rank by importance, take top 10
ranked = hodoscope.rank_summaries(grouped, method='tsne', n=10)
# ranked["1.0"][0]["fps_rank"] == 0  (most important)

# Lower-level building blocks
X_2d = hodoscope.compute_projection(embeddings, method='tsne')  # labels=... for balanced projection
ranks = hodoscope.compute_fps_ranks(X_2d, labels, n_groups)
```

## Data Flow

```
Raw Sources (.eval files, Docent, trajectory JSONs, OpenHands .jsonl files)
    ↓
[pipeline] load_eval() / load_trajectory_dir() / load_openhands() / load_docent()
    ↓
[config] Config() or Config.from_env()          # build processing config
    ↓
[pipeline] process_trajectories(trajectories, config=config)
    ├── [parsers] extract_task_context(messages) → task_context string
    ├── [parsers] turns_from_messages() → merge_consecutive_turns()
    ├── [core] extract_actions_from_turns() → action['task_context'] = task_context
    └── [actions] process_single_action((action, config, idx, id, meta))
         ├── summarize_action_only() → summarize via LiteLLM
         └── embed_summary() → embed via LiteLLM
    ↓
[io] write_analysis_json() → .hodoscope.json        # if using analyze()
    ↓
[io] read_analysis_json() + group_summaries()        # from disk
— or —
[io] group_summaries_from_list()                     # from memory
    ↓
[sampling] rank_summaries() → {label: [ranked dicts]}   # FPS-based sampling
— or —
[visualization] visualize_action_summaries() → Single unified HTML
```

## Development Notes

- All file operations are thread-safe (using `SHARED_LOCK` in core.py)
- Parallel processing with progress bars via `run_parallel()`, with periodic checkpoint saves via `save_callback`
- Resume support: `--resume` (default on) loads existing output, builds skip set of `(trajectory_id, turn_id)` pairs, and only processes new actions
- Embeddings encoded as RFC 1924 base85 in JSON (JSON-safe, ~25% smaller than base64)
- `process_single_action` takes a 5-element tuple: `(action, config, idx, traj_id, traj_metadata)` — Config carries all model/dim/normalize settings
- Pipeline functions raise `HodoscopeError` on errors; CLI catches and exits cleanly
- Visualization functions take generic `{label: [summaries]}` dicts — grouping is decoupled
- Visualization scatter sources initialize an `alpha` column at creation time, so legend-aware counts and navigation work correctly before any filter interaction
- Visualization JS tracks active mode in a global (`window._hodoMode`) so status-eye behavior stays strict (rank mode = lowest-FPS only; search mode = random visible), while detail-panel eye keeps nearest-visible behavior
- Visualization navigation forces detail-pane visibility and re-emits target selection on eye navigation to avoid intermittent "pane did not open" behavior
- Sidebar script initialization must preserve existing `window._hodo*` globals (set by Bokeh `DocumentReady`) instead of resetting them, otherwise top-eye can fail before first manual pane open
- Status/legend callbacks also refresh `window._hodoSources`/`window._hodoRenderers`, so top-eye has live nav state even if `DocumentReady` init order is inconsistent
- Eye-navigation debug logging is emitted to browser console (`[hodo eye] ...`) for top/panel eye clicks, active candidate sets, and chosen targets
- Initial status-eye HTML must use direct quoted `onclick="_hodoNav('status')"` (not escaped `\\x27`) because it is injected from Python, not via JS string unescaping
- **HTML size optimization**: `task_context` and `action_text` are stored outside Bokeh's `ColumnDataSource` to reduce file size. `task_context` is deduplicated into `window._hodoTaskContexts` (keyed by `trajectory_uuid`), since all turns in a trajectory share the same context. `action_text` is stored as `window._hodoActionTexts` (flat array indexed by `_global_idx`). Both are gzip-compressed and base64-encoded by `_compress_data()`, then injected as a single `<script>` tag with an async `_hodoDecode()` function that decompresses via `DecompressionStream` on page load. Sources store a `_global_idx` column (integer) for lookup into these arrays. This yields ~60% total file size reduction for large visualizations (e.g. 30 MB → 12 MB).
- Scatter `ColumnDataSource` includes both `trajectory_id` (for display) and `trajectory_uuid` (for prev/next turn navigation). `trajectory_uuid` is `{source_filename}___{trajectory_id}` when `_source_file` is stamped by `viz()`, otherwise falls back to plain `trajectory_id`. This disambiguates same-ID trajectories from different `.hodoscope.json` files.
- Scatter `ColumnDataSource` and `GlyphRenderer` models are named `hodo_source_{idx}` / `hodo_renderer_{idx}` so sidebar JS can lazily recover nav state from `Bokeh.index` before first interaction
- Docent is a required dependency (installed automatically)
- LiteLLM handles provider routing via model string prefix (e.g. `openai/`, `gemini/`, `anthropic/`)
- API keys are read from env vars automatically by LiteLLM (no early validation — errors surface at call time)
- Config resolution: `Config()` = hardcoded defaults only (pure, no side effects); `Config.from_env()` = loads .env + env vars (used by CLI and `analyze()` default)
- `embed_dim=None` in Config means "respect the API return dimensionality" — no `output_dimensionality` is passed to the embedding call
- Default group-by field for viz/sample/grouping is `DEFAULT_GROUP_BY` (`"model"`), defined once in `config.py`
- `fps_alpha` / `fps_beta` in Config control density-weighted FPS ranking. Exposed via `FPS_ALPHA` / `FPS_BETA` env vars. `pipeline.viz()` resolves from env when not explicitly passed; `pipeline.sample()` accepts them as direct args.

## CI/CD

GitHub Actions workflows in `.github/workflows/`:
- **`test.yml`** — Runs tests on push/PR (also supports `workflow_dispatch`)
- **`publish.yml`** — Builds and publishes to PyPI via trusted publishing on version tags (`v*`). To release: `git tag v0.X.Y && git push origin v0.X.Y`

## Testing

```bash
pytest tests/test_io.py tests/test_viz.py tests/test_api.py tests/test_sampling.py  # Unit tests (no API keys)
pytest tests/test_analyze.py                                                         # E2E tests (needs API keys)
```

## Import Examples

```python
from hodoscope import (
    # Config
    Config,
    DEFAULT_SUMMARIZE_MODEL,
    DEFAULT_SUMMARIZE_PROMPT,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBED_DIM,
    DEFAULT_MAX_WORKERS,
    DEFAULT_GROUP_BY,
    DEFAULT_FPS_ALPHA,
    DEFAULT_FPS_BETA,
    # Building blocks (first-class Python API)
    load_eval,
    load_trajectory_dir,
    load_docent,
    process_trajectories,
    extract_actions,
    group_summaries_from_list,
    filter_summaries,
    # Sampling
    compute_projection,
    compute_fps_ranks,
    rank_summaries,
    # High-level orchestrators
    analyze,
    viz,
    show_info,
    sample,
    # I/O
    write_analysis_json,
    read_analysis_json,
    group_summaries,
    # Visualization
    visualize_action_summaries,
    DEFAULT_COLORS,
)
```
