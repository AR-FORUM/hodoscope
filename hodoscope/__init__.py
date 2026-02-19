"""Trajectory analysis library for processing AI agent trajectories."""

__version__ = "0.2.0"

# Config
from .config import (
    Config,
    DEFAULT_SUMMARIZE_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_TASK_TYPE,
    DEFAULT_EMBED_DIM,
    DEFAULT_MAX_WORKERS,
    DEFAULT_GROUP_BY,
    DEFAULT_FPS_ALPHA,
    DEFAULT_FPS_BETA,
)
from .actions import DEFAULT_SUMMARIZE_PROMPT

# I/O
from .io import (
    write_analysis_json,
    read_analysis_json,
    group_summaries,
    group_summaries_from_list,
    filter_summaries,
)

# Sampling
from .sampling import (
    ALL_PLOT_METHODS,
    compute_projection,
    compute_fps_ranks,
    rank_summaries,
)

# Visualization
from .visualization import (
    visualize_action_summaries,
    DEFAULT_COLORS,
)

# Pipeline: building blocks
from .pipeline import (
    load_eval,
    load_trajectory_dir,
    load_docent,
    load_openhands,
    process_trajectories,
    extract_actions,
    HodoscopeError,
)

# Pipeline: high-level orchestrators (used by CLI)
from .pipeline import (
    analyze,
    viz,
    show_info,
    sample,
)

__all__ = [
    # Config
    'Config',
    'DEFAULT_SUMMARIZE_MODEL',
    'DEFAULT_SUMMARIZE_PROMPT',
    'DEFAULT_EMBEDDING_MODEL',
    'DEFAULT_EMBEDDING_TASK_TYPE',
    'DEFAULT_EMBED_DIM',
    'DEFAULT_MAX_WORKERS',
    'DEFAULT_GROUP_BY',
    'DEFAULT_FPS_ALPHA',
    'DEFAULT_FPS_BETA',
    # I/O
    'write_analysis_json',
    'read_analysis_json',
    'group_summaries',
    'group_summaries_from_list',
    'filter_summaries',
    # Sampling
    'ALL_PLOT_METHODS',
    'compute_projection',
    'compute_fps_ranks',
    'rank_summaries',
    # Visualization
    'visualize_action_summaries',
    'DEFAULT_COLORS',
    # Pipeline: building blocks
    'load_eval',
    'load_trajectory_dir',
    'load_docent',
    'load_openhands',
    'process_trajectories',
    'extract_actions',
    'HodoscopeError',
    # Pipeline: orchestrators
    'analyze',
    'viz',
    'show_info',
    'sample',
]
