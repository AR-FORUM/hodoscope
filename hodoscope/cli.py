"""CLI entry point for hodoscope v2."""

import sys
import webbrowser

import click


def _handle_error(func):
    """Decorator to catch HodoscopeError and exit cleanly."""
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        from .pipeline import HodoscopeError
        try:
            return func(*args, **kwargs)
        except HodoscopeError as e:
            print(f"ERROR: {e}")
            sys.exit(1)

    return wrapper


def _build_filter(filter_strings: tuple[str, ...]):
    """Parse KEY=VALUE strings into a summary filter predicate.

    Returns None if no filters given. All filters are AND'd.
    Checks summary['metadata'][key]; tries numeric comparison as fallback.
    """
    if not filter_strings:
        return None

    kvs = []
    for f in filter_strings:
        if "=" not in f:
            print(f"WARNING: Ignoring malformed --filter '{f}' (expected KEY=VALUE)")
            continue
        k, v = f.split("=", 1)
        kvs.append((k.strip(), v.strip()))

    if not kvs:
        return None

    def _match(actual, expected_str):
        if actual is None:
            return False
        if str(actual) == expected_str:
            return True
        try:
            return float(actual) == float(expected_str)
        except (ValueError, TypeError):
            return False

    def predicate(summary):
        meta = summary.get("metadata", {})
        return all(_match(meta.get(k), v) for k, v in kvs)

    return predicate


@click.group()
@click.version_option(package_name="hodoscope")
def main():
    """Hodoscope -- analyze AI agent trajectories."""
    from .config import _load_env
    _load_env()


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------

@main.command()
@click.argument("sources", nargs=-1)
@click.option("--docent-id", default=None,
              help="Docent collection ID as source")
@click.option("-o", "--output", default=None,
              help="Output JSON path (only for single source)")
@click.option("--field", multiple=True,
              help="KEY=VALUE metadata fields (repeatable)")
@click.option("--limit", "-l", type=int, default=None,
              help="Limit trajectories per source")
@click.option("--save-samples", type=click.Path(), default=None,
              help="Save extracted trajectory samples to this directory")
@click.option("--embed-dim", type=int, default=None,
              help="Embedding dimensionality (env: EMBED_DIM, default: 768)")
@click.option("--model-name", "-m", default=None,
              help="Override auto-detected model name (metadata, not LLM)")
@click.option("--summarize-model", default=None,
              help="LiteLLM model for summarization (env: SUMMARIZE_MODEL)")
@click.option("--embedding-model", default=None,
              help="LiteLLM model for embeddings (env: EMBEDDING_MODEL)")
@click.option("--sample/--no-sample", default=True,
              help="Randomly sample trajectories when using --limit (default: on; --no-sample for first N)")
@click.option("--seed", type=int, default=None,
              help="Random seed for --sample reproducibility")
@click.option("--resume/--no-resume", default=True,
              help="Resume from existing output file (default: on; --no-resume to overwrite)")
@click.option("--reasoning-effort",
              type=click.Choice(["low", "medium", "high"], case_sensitive=False),
              default=None,
              help="Reasoning effort for summarization model (env: REASONING_EFFORT)")
@click.option("--max-workers", type=int, default=None,
              help="Max parallel workers for LLM calls (env: MAX_WORKERS, default: 10)")
@click.option("--reembed", is_flag=True, default=False,
              help="Re-embed existing summaries (e.g. after changing embedding model/dim)")
@_handle_error
def analyze(sources, docent_id, output, field, limit, save_samples,
            embed_dim, model_name, summarize_model, embedding_model,
            sample, seed, resume, reasoning_effort, max_workers, reembed):
    """Analyze source files and produce .hodoscope.json output(s).

    SOURCES can be .eval files, directories containing .eval files,
    or directories of trajectory JSON files.

    \b
    Examples:
      hodoscope analyze run.eval                    # single .eval file
      hodoscope analyze *.eval                      # batch: all .eval files
      hodoscope analyze evals/                      # batch: dir of .eval files
      hodoscope analyze run.eval -o my_output.json  # custom output path
      hodoscope analyze run.eval --field env=prod   # add custom metadata
      hodoscope analyze --docent-id COLLECTION_ID   # docent source
    """
    from .config import Config
    from .pipeline import analyze as analyze_fn

    config = Config.from_env(
        summarize_model=summarize_model,
        embedding_model=embedding_model,
        embed_dim=embed_dim,
        max_workers=max_workers,
        reasoning_effort=reasoning_effort,
    )

    analyze_fn(
        sources=sources,
        docent_id=docent_id,
        output=output,
        fields=list(field),
        limit=limit,
        save_samples=save_samples,
        model_name=model_name,
        sample=sample,
        seed=seed,
        resume=resume,
        config=config,
        reembed=reembed,
    )


# ---------------------------------------------------------------------------
# viz
# ---------------------------------------------------------------------------

@main.command()
@click.argument("sources", nargs=-1, required=True)
@click.option("--group-by", default=None,
              help="Field to group summaries by (default: model)")
@click.option("--proj", multiple=True,
              help="Projection methods, comma-separated or repeated "
                   "(pca,tsne,umap,trimap,pacmap; * or all for all; default: tsne)")
@click.option("-o", "--output", "output_file", default=None,
              help="Output HTML file path (default: auto-generated timestamped name)")
@click.option("--filter", "filters", multiple=True,
              help="KEY=VALUE metadata filter (repeatable, AND logic)")
@click.option("--open", "open_browser", is_flag=True, default=False,
              help="Open the generated HTML in the default browser")
@_handle_error
def viz(sources, group_by, proj, output_file, filters, open_browser):
    """Visualize analysis JSON files with a unified interactive plot.

    Produces a single HTML file with a method switcher dropdown, density
    heatmap overlay, FPS-based flagging, and search/filter controls.

    SOURCES can be .hodoscope.json files or directories containing them.

    \b
    Examples:
      hodoscope viz output.hodoscope.json                   # single file
      hodoscope viz output.hodoscope.json --group-by score  # group by score
      hodoscope viz results/                                # all JSONs in dir
      hodoscope viz a.hodoscope.json b.hodoscope.json       # cross-file comparison
      hodoscope viz output.hodoscope.json --proj tsne,pca,umap  # multiple methods
      hodoscope viz output.hodoscope.json --proj '*'             # all methods
    """
    from .pipeline import viz as viz_fn
    from .sampling import ALL_PLOT_METHODS

    # Flatten comma-separated values: --proj tsne,pca --proj umap → [tsne, pca, umap]
    # --proj * or --proj all → all available methods
    flat = []
    for p in proj:
        flat.extend(part.strip().lower() for part in p.split(",") if part.strip())

    if flat == ["*"] or flat == ["all"]:
        flat = list(ALL_PLOT_METHODS)

    invalid = [m for m in flat if m not in ALL_PLOT_METHODS]
    if invalid:
        raise click.BadParameter(
            f"Invalid projection method(s): {', '.join(invalid)}. "
            f"Choose from: {', '.join(ALL_PLOT_METHODS)}",
            param_hint="'--proj'",
        )

    html_path = viz_fn(
        sources=sources,
        group_by=group_by,
        proj=flat or None,
        output_file=output_file,
        filter=_build_filter(filters),
    )
    if open_browser and html_path:
        webbrowser.open(html_path.resolve().as_uri())


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------

@main.command()
@click.argument("sources", nargs=-1)
@_handle_error
def info(sources):
    """Show metadata and summary counts for analysis JSON files.

    SOURCES can be .hodoscope.json files or directories. If omitted,
    looks for .hodoscope.json files in the current directory.
    """
    from .pipeline import show_info

    show_info(sources=sources)


# ---------------------------------------------------------------------------
# sample
# ---------------------------------------------------------------------------

@main.command()
@click.argument("sources", nargs=-1, required=True)
@click.option("--group-by", default=None,
              help="Field to group summaries by (default: model)")
@click.option("-n", "--samples-per-group", type=int, default=10,
              help="Number of representative samples per group (default: 10)")
@click.option("--proj", default="tsne",
              help="Projection method for FPS ranking (pca,tsne,umap,trimap,pacmap; default: tsne)")
@click.option("-o", "--output", default=None,
              help="JSON output file (default: paginated terminal display)")
@click.option("--interleave", is_flag=True, default=False,
              help="Interleave groups by rank (#1 from each group, then #2, etc.)")
@click.option("--filter", "filters", multiple=True,
              help="KEY=VALUE metadata filter (repeatable, AND logic)")
@_handle_error
def sample(sources, group_by, samples_per_group, proj, output, interleave, filters):
    """Sample representative summaries using FPS-based ranking.

    Selects the most diverse and informative summaries per group using
    density-weighted Farthest Point Sampling on 2D projections.

    SOURCES can be .hodoscope.json files or directories containing them.

    \b
    Examples:
      hodoscope sample output.hodoscope.json                        # top 10 per group
      hodoscope sample output.hodoscope.json --group-by score -n 5  # top 5 per score
      hodoscope sample output.hodoscope.json --proj pca             # use PCA projection
      hodoscope sample output.hodoscope.json -o sampled.json        # write JSON output
    """
    from .sampling import ALL_PLOT_METHODS
    from .pipeline import sample as sample_fn

    proj = proj.lower()
    if proj not in ALL_PLOT_METHODS:
        raise click.BadParameter(
            f"Invalid projection method: {proj}. "
            f"Choose from: {', '.join(ALL_PLOT_METHODS)}",
            param_hint="'--proj'",
        )

    sample_fn(
        sources=sources,
        group_by=group_by,
        n=samples_per_group,
        method=proj,
        output=output,
        interleave=interleave,
        filter=_build_filter(filters),
    )
