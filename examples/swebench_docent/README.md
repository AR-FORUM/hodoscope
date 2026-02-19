# SWE-Bench + Docent Workflow

This example shows how to use Hodoscope to analyze agent trajectories stored in a Docent collection (e.g., SWE-Bench evaluation runs).

## Prerequisites

- A Docent collection ID with agent trajectory data
- API keys configured in `.env` (see main README)

## CLI Workflow

```bash
# 1. Analyze trajectories from a Docent collection
hodoscope analyze --docent-id YOUR_COLLECTION_ID -o swebench.hodoscope.json

# 2. Visualize results (groups by model by default)
hodoscope viz swebench.hodoscope.json

# 3. Group by score to see pass/fail patterns
hodoscope viz swebench.hodoscope.json --group-by score

# 4. Sample representative actions
hodoscope sample swebench.hodoscope.json --group-by score -n 10
```

## Python API Workflow

```python
import hodoscope as ta

# Load from Docent
trajectories, fields = ta.load_docent("YOUR_COLLECTION_ID", limit=50)

# Process (summarize + embed)
config = ta.Config.from_env()
summaries = ta.process_trajectories(trajectories, config=config)

# Group by score and visualize
grouped = ta.group_summaries_from_list(summaries, group_by="score")
ta.visualize_action_summaries(grouped, "swebench_explorer.html", methods=["tsne"])

# Sample top actions per group
ranked = ta.rank_summaries(grouped, method="tsne", n=10)
for label, items in ranked.items():
    print(f"\n--- Score: {label} ---")
    for item in items[:3]:
        print(f"  #{item['fps_rank']}: {item['summary']}")
```

## Notes

- Docent metadata (model, score, instance_id, etc.) is automatically extracted and available for grouping
- Use `--group-by` with any metadata field: `model`, `score`, `instance_id`, `sandbox`, etc.
- The `--filter` flag supports numeric and string comparisons: `--filter score=1.0`
