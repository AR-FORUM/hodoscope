# Replicating the SWE-bench Finding

This walkthrough reproduces the analysis that re-discovered the iQuest time-traveling issue using Hodoscope. We compare five models on SWE-bench Verified and show how behavioral outliers surface visually in minutes.

## Prerequisites

- Hodoscope installed (`pip install hodoscope`)
- API keys configured in `.env` (see main README) — needed for summarization and embedding

## Step 1: Fetch traces

We compare five models. Four are Docent collections from the SWE-bench Leaderboard (o3, gpt-4.1, qwen3-coder, deepseek-v3.2-reasoner). The fifth, iQuest-Coder-v1, requires manual preprocessing.

> **Note:** The SWE-bench Leaderboard has since been updated to newer traces. We use the older versions that were available when we ran these experiments.

### iQuest preprocessing

Download and split the single trajectory JSON into individual files:

```bash
wget https://github.com/IQuestLab/IQuest-Coder-V1/raw/044c57a/IQuest-Coder-Eval/SWE-Verified/traj.zip
unzip traj.zip && rm traj.zip && mv traj.json iquest_traj.json

# Split into individual trajectory files
python -c "
import json, pathlib
p = pathlib.Path('iquest_samples')
p.mkdir(exist_ok=True)
for i, t in enumerate(json.load(open('iquest_traj.json'))):
    (p / f'traj_{i:04d}.json').write_text(
        json.dumps({'id': f'traj_{i:04d}', 'messages': t['messages']})
    )
"
rm iquest_traj.json
```

## Step 2: Analyze

Run `hodoscope analyze` on each source. We sample 50 trajectories per model with a fixed seed for reproducibility, and annotate the model names with `--field`:

> **⚠️ Cost note:** This demo uses the default summarization model (`gpt-5.2`) and costs ~$35 in OpenAI API calls total. You can reduce cost by switching summarization model (e.g. `--summarize-model gemini/gemini-3-flash-preview`) or further subsampling.

```bash
# Docent sources
hodoscope analyze --docent-id 565e5680-b913-4031-b537-00721a7a619a -l 50 --seed 42 --field model=o3
hodoscope analyze --docent-id cd7a23c5-a2b1-4cab-b851-6e2c42aaf0f3 -l 50 --seed 42 --field model=gpt-4.1
hodoscope analyze --docent-id f39d3041-d9d7-4f1b-b75e-8a13addb9e6e -l 50 --seed 42 --field model=qwen3-coder
hodoscope analyze --docent-id 7fde5552-6b17-4cb7-ab9c-15fd9fb5b845 -l 50 --seed 42 --field model=deepseek

# Raw trajectory directory
hodoscope analyze iquest_samples/ -l 50 --seed 42 --field model=iquest-coder-v1
```

Each command produces a `.hodoscope.json` file containing action summaries and embeddings.

## Step 3: Visualize

Combine everything into a single interactive explorer:

```bash
hodoscope viz *.hodoscope.json --proj tsne --open
```

This generates a self-contained HTML file and opens it in your browser. Each point corresponds to an agent action, colored by model.

## Step 4: Discover

With all models plotted together, you are ready to discover behavioral patterns! Here are some tips for replicating the iQuest time-traveling finding:

1. **Density overlay** -- Switch the overlay dropdown to iQuest. Clusters unique to iQuest stand out in red.
2. **Click to inspect** -- Click on dots with the darkest colors in the iQuest-only cluster and examine corresponding actions.
3. **Search** -- After you find suspicious actions, you could search for them using keyword search in the search box.