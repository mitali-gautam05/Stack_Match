# StackMatch

**Does a duplicate GitHub issue look the same to a model whether it's written in prose or in code?**

StackMatch is an ML system that detects duplicate GitHub issues. Existing tools (Simili, GitHub's built-in duplicate detection) embed the entire issue text with a single generic embedding model. StackMatch's core hypothesis: an issue actually contains two different kinds of signal — natural language description and code/stack traces — and collapsing both into one embedding space loses information that's useful for duplicate detection.

## The idea

A typical GitHub issue looks like this:

```
Prose:  "My app crashes when I open settings"
Code:   NullPointerException at MainActivity.java:42
```

A generic sentence embedding treats both as plain text. StackMatch segments the two, embeds them with models suited to each (a code-aware embedding model for stack traces, a sentence-transformer for prose), and fuses the two signals — testing whether this beats a single-embedding baseline on precision, recall, and MRR.

## Status: Week 1 of 4

- [x] **Scraping** — built a ground-truth dataset from real GitHub duplicate labels
- [X] **Segmentation** — separating prose from code within issues
- [] **Dual embedding** — CodeBERT + sentence-transformer (Week 2)
- [ ] **Evaluation** — comparing single-embedding vs. dual-embedding approaches on Precision, Recall, MRR (Week 3)
- [ ] **Deployment** — GitHub bot via webhook + FastAPI (Week 4)

## How the ground-truth dataset was built

Manually labeling duplicate issue pairs would take weeks. Instead, this project exploits something GitHub maintainers already do for free: when they close an issue as a duplicate, they link the original (`Duplicate of #1234`, `/duplicate <url>`, etc.). The scraper parses these existing labels rather than generating new ones.

```
GitHub API
    ↓
Fetch closed issues labeled "duplicate"
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
| 0 pairs found | Regex only matched `#1234` | Added patterns for `/duplicate <url>` and inline sentence-URL formats |
| 9 repos returned 0 issues | Not all repos use a "duplicate" label | Wrote a label-discovery script to find each repo's actual label naming |
| File not found | Output path assumption was wrong | Fixed to write relative to project root |

The recurring lesson: real-world data doesn't match your assumptions about it. Explore the data first, then write the parsing logic.

## Project structure

```
stackmatch/
├── .env                          # GitHub token (not committed)
├── .gitignore
├── scraper/
│   └── github_scraper.py
└── data/
    └── ground_truth_pairs.json
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

Run the scraper:
```bash
python scraper/github_scraper.py
```

## Why this is different

Most duplicate-detection tools treat an issue as one blob of text. StackMatch's bet is that the *type* of content — natural language vs. code — carries different duplicate signal, and that separating them before embedding should outperform a single-embedding baseline. Week 3's evaluation will show whether that bet pays off.