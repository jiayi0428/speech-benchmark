# %% [markdown]
# # Deep Case Studies: Failure Analysis
#
# This notebook picks audio samples where cascade and direct diverge significantly,
# and provides annotated analysis of what went wrong.

# %% Setup
import sys
sys.path.insert(0, "..")
import json
import numpy as np
from pathlib import Path
from src.config import RESULTS_DIR

# %% [markdown]
# ## Select Divergent Cases
#
# Load results and find samples with the largest output differences between cascade and direct.

# %%
def load_results(name):
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)

cascade_clean = load_results("cascade_clean")
direct_clean = load_results("direct_clean")

if cascade_clean and direct_clean:
    print(f"Ready to analyze {len(cascade_clean)} sample pairs")
    # In a real analysis, compare outputs per task and rank by divergence
else:
    print("No results found. Run 02_run_experiments.py first with real audio.")

# %% [markdown]
# ## Case Study Template
#
# For each selected case, the analysis covers:
#
# 1. **Audio Overview**: Speaker, topic, duration, acoustic quality
# 2. **Cascade Analysis**:
#    - ASR transcript (with errors highlighted)
#    - Downstream task outputs
#    - Error propagation analysis
# 3. **Direct Analysis**:
#    - Task outputs
#    - What paralinguistic cues might have helped?
# 4. **Comparison**:
#    - Where do they agree/disagree?
#    - Why does one perform better?
#    - What does this tell us about the architecture trade-off?

# %% [markdown]
# ## Failure Modes Catalog
#
# | Mode | Description | Cascade Behavior | Direct Behavior |
# |------|-------------|-----------------|-----------------|
# | **Background Noise** | Music, crowd sounds | ASR errors cascade to wrong summary | May filter noise, focus on speech |
# | **Fast Speech** | >180 wpm | Deletions in ASR → incomplete understanding | May struggle or use prosody |
# | **Accented Speech** | Non-native accent | Higher WER → downstream degradation | Possibly more robust |
# | **Overlapping Speakers** | Multiple people talking | ASR produces garbled text | May isolate primary speaker |
# | **Emotional Speech** | Sarcasm, excitement | Loses tone, misinterprets intent | Captures prosodic cues |

print("\nCase study analysis ready.")
print("Fill in real examples after running experiments.")
