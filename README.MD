# StackMatch

**Does a duplicate GitHub issue look the same to a model whether it's written in prose or in code?**

StackMatch is an ML system that detects duplicate GitHub issues. Existing tools (Simili, GitHub's built-in duplicate detection) embed the entire issue text with a single generic embedding model. StackMatch's core hypothesis: an issue actually contains two different kinds of signal — natural language description and code/stack traces — and collapsing both into one embedding space loses information that's useful for duplicate detection.

## The idea

A typical GitHub issue looks like this:

```
Prose:  "My app crashes when I open settings"
Code:   NullPointerException at MainActivity.java:42
```

A generic sentence embedding treats both as plain text. StackMatch segments the two, embeds them with models suited to each (CodeBERT for stack traces, a sentence-transformer for prose), and fuses the two signals — tested against single-embedding baselines on precision, recall, and MRR.

## Results

Each duplicate issue is queried against all 467 candidate originals (467-way retrieval, not a small negative-sampling proxy). Fusion falls back to prose-only similarity when either side of a pair has no code, so prose-heavy pairs aren't penalized by an empty code vector.

| Approach | MRR (overall) | MRR (code-present pairs) | P@1 (code-present) |
|---|---|---|---|
| MiniLM (prose only) | 0.682 | 0.806 | 74.6% |
| CodeBERT (code only) | 0.785* | 0.375 | 32.8% |
| **Dual embedding (ours)** | **0.692** | **0.879** | **83.6%** |

*CodeBERT's high overall MRR is an artifact of trivial zero-vector matches on the 86% of pairs with no code at all — the code-present column is the real signal to look at.

**Key finding:** dual embedding beats the MiniLM baseline everywhere — modestly overall (+1.0% MRR) and substantially on code-present issues (+7.2% MRR, +9.0% P@1), which is exactly where the hypothesis predicted an advantage. Random baseline MRR is ≈0.002, so both models are clearly learning real signal, not noise.

## How the ground-truth dataset was built

Manually labeling duplicate issue pairs would take weeks. Instead, this project exploits something GitHub maintainers already do for free: when they close an issue as a duplicate, they link the original (`Duplicate of #1234`, `/duplicate <url>`, `same as #1234`, etc.). The scraper parses these existing labels rather than generating new ones.

```
GitHub API
    ↓
Fetch closed issues labeled "duplicate" (label name varies per repo)
    ↓
Regex over comments → extract the original issue number
    ↓
Fetch full content of the original issue
    ↓
Save {duplicate, original} pair to JSON
```

### Problems hit along the way

| Problem | Cause | Fix |
|---|---|---|
| 0 pairs found | Regex only matched `#1234` | Added patterns for `/duplicate <url>`, `same as <url>`, `covered by #N` |
| 9 repos returned 0 issues | Not all repos use a "duplicate" label (e.g. NumPy uses `"50 - Duplicate"`) | Wrote a label-discovery script to find each repo's actual label naming |
| Telemetry noise counted as "code" | VSCode issues embed A/B-test IDs inside fenced code blocks | Heuristic filter: blocks that are >60% `key:bignumber` lines are discarded as telemetry |
| First evaluation gave near-random MRR | Ranking true pairs against *other true pairs'* scores instead of doing real retrieval | Rebuilt evaluator to do real 467-way retrieval: each duplicate ranked against every candidate original |
| Dual embedding underperformed MiniLM overall | Fusion averaged prose score with a zero-vector code score on prose-only pairs, halving the score | Smart fusion: only blend prose+code when *both* sides of a pair actually contain code, else fall back to prose-only |

The recurring lesson: real-world data — and your own evaluation pipeline — rarely match your first assumption about them. Explore, measure, and verify before trusting a result.

## Project structure

```
stackmatch/
├── .env                          # GitHub token (not committed)
├── .gitignore
├── requirements.txt
├── scraper/
│   └── github_scraper.py         # mines duplicate pairs from GitHub
├── segmenter/
│   └── segmenter.py              # splits prose vs. code, filters noise
├── embedder/
│   └── embedder.py               # MiniLM + CodeBERT embeddings, saved as vectors
├── evaluator/
│   └── evaluator.py              # real retrieval eval: MRR, Precision@K, Recall@K
└── data/
    ├── ground_truth_pairs.json
    ├── segmented_pairs.json
    ├── embeddings.npz
    ├── metadata.json
    └── evaluation_results.json
```

## Setup

```bash
git clone https://github.com/<your-username>/stackmatch.git
cd stackmatch
python -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file with a GitHub personal access token:
```
GITHUB_TOKEN=your_token_here
```

Run the pipeline in order:
```bash
python scraper/github_scraper.py      # mine ground-truth pairs
python segmenter/segmenter.py         # split prose/code, filter noise
python embedder/embedder.py           # generate embeddings
python evaluator/evaluator.py         # run retrieval evaluation
```

## Why this is different

Most duplicate-detection tools treat an issue as one blob of text. StackMatch's bet is that the *type* of content — natural language vs. code — carries different duplicate signal, and that separating them before embedding outperforms a single-embedding baseline. The evaluation confirms this: the advantage is small on prose-only issues (as expected — there's no code signal to add) and substantial on code-present issues, where dual embedding beats a strong sentence-transformer baseline by 7-9 points.