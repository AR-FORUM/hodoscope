#!/usr/bin/env bash
set -euo pipefail

# Step 1: Fetch and preprocess iQuest traces
wget https://github.com/IQuestLab/IQuest-Coder-V1/raw/044c57a/IQuest-Coder-Eval/SWE-Verified/traj.zip
unzip traj.zip && rm traj.zip && mv traj.json iquest_traj.json

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

# Step 2: Analyze (50 trajectories per model, seed=42)
hodoscope analyze --docent-id 565e5680-b913-4031-b537-00721a7a619a -l 50 --seed 42 --field model=o3
hodoscope analyze --docent-id cd7a23c5-a2b1-4cab-b851-6e2c42aaf0f3 -l 50 --seed 42 --field model=gpt-4.1
hodoscope analyze --docent-id f39d3041-d9d7-4f1b-b75e-8a13addb9e6e -l 50 --seed 42 --field model=qwen3-coder
hodoscope analyze --docent-id 7fde5552-6b17-4cb7-ab9c-15fd9fb5b845 -l 50 --seed 42 --field model=deepseek
hodoscope analyze iquest_samples/ -l 50 --seed 42 --field model=iquest-coder-v1

# Step 3: Visualize
hodoscope viz *.hodoscope.json --proj tsne --open
