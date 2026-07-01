"""
StackMatch — Segmenter
=======================
Goal: Split a GitHub issue body into two parts:
  1. prose_text  → natural language description
  2. code_text   → code blocks + stack traces combined

Why separate them?
  Prose tells us CONTEXT  → sentence-transformer handles this well
  Code tells us EXACT BUG → CodeBERT handles this well
  
Mixing them loses signal. Separating them preserves it.
"""

import re
import json


# ── PATTERN 1: Fenced Code Blocks ─────────────────────────────────────────────
# Matches content between triple backticks
# 
# How it works:
#   ```          → literal triple backtick (opening)
#   (?:\w+)?     → optional language hint like "python", "java" (non-capturing group)
#   \n           → newline after opening fence
#   (.*?)        → actual code content (captured) — the ? makes it non-greedy
#   \n```        → closing fence on its own line
#
# NON-GREEDY (.*?) is critical here:
#   Greedy  (.*) → matches from FIRST ``` to LAST ``` in entire document (wrong!)
#   Non-greedy (.*?) → matches from opening ``` to the NEAREST closing ``` (correct)
#
# re.DOTALL → makes . match newlines too (code blocks span multiple lines)

FENCED_CODE_PATTERN = re.compile(
    r"```(?:\w+)?\n(.*?)\n```",
    re.DOTALL
)

# ── PATTERN 2: Stack Traces ────────────────────────────────────────────────────
# Stack traces appear WITHOUT backticks but have very recognizable patterns.
# We detect them by looking for "Traceback" keyword or stack frame lines.
#
# Stack frame patterns across languages:
#   Python:  "  File \"app.py\", line 42, in load_profile"
#   Java:    "  at com.example.MyClass.method(MyClass.java:42)"
#   Node.js: "  at Object.<anonymous> (/app/index.js:10:15)"
#   .NET:    "  at MyApp.Program.Main() in Program.cs:line 42"
#
# Strategy: Find a "Traceback" or "Exception" line, then grab everything
# until we hit a blank line (stack traces end with blank line usually)

TRACEBACK_START = re.compile(
    r"^(Traceback \(most recent call last\)|"  # Python
    r"Exception in thread|"                     # Java
    r"Error:|"                                  # Generic
    r"[\w.]+(?:Error|Exception|Warning):)",     # AnyError: / AnyException:
    re.MULTILINE
)

STACK_FRAME = re.compile(
    r"^\s+(?:"
    r"File \".*?\", line \d+|"      # Python frame
    r"at [\w.$<>]+\(.*?:\d+\)|"    # Java/Node frame  
    r"at .*? in .*?:\d+"            # .NET frame
    r")",
    re.MULTILINE
)


def extract_fenced_blocks(text):
    """
    Extract all fenced code blocks (``` ... ```) from text.
    Returns: (code_blocks_combined, text_with_blocks_removed)
    
    We REMOVE the blocks from prose and COLLECT them separately.
    This is called 'extraction' — pull out what you want, leave the rest.
    """
    code_parts = []
    
    def replace_with_placeholder(match):
        # match.group(1) = the code content inside the fences
        code_parts.append(match.group(1).strip())
        return ""  # Remove from prose (replace with empty string)
    
    # re.sub with a function → for each match, call the function
    # The function both COLLECTS the code AND returns "" to remove it from prose
    cleaned_prose = FENCED_CODE_PATTERN.sub(replace_with_placeholder, text)
    
    return code_parts, cleaned_prose


def extract_stack_traces(text):
    """
    Extract stack traces that appear WITHOUT backticks.
    
    Strategy:
    1. Find lines that look like traceback starts ("Traceback...", "NullPointerException:")
    2. Grab that line + all following indented lines (stack frames)
    3. Remove them from prose
    
    Why this is harder than fenced blocks:
    Fenced blocks have explicit markers (```). 
    Stack traces are just... text that happens to look like code.
    We use STRUCTURE (indentation + known patterns) to identify them.
    """
    lines = text.split('\n')
    code_lines = []
    prose_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if this line starts a traceback
        if TRACEBACK_START.match(line.strip()):
            # Grab this line + all following lines that look like stack frames
            # or are indented (part of the trace)
            trace_block = [line]
            i += 1
            
            while i < len(lines):
                next_line = lines[i]
                
                # Stack trace continues if:
                # - Line is indented (starts with spaces)
                # - Line matches a stack frame pattern
                # - Line is the error message (e.g., "AttributeError: ...")
                # - Line is not blank
                is_indented = next_line.startswith('  ') or next_line.startswith('\t')
                is_frame = bool(STACK_FRAME.match(next_line))
                is_error_line = bool(re.match(r'[\w.]+(?:Error|Exception):', next_line))
                is_blank = next_line.strip() == ''
                
                if is_indented or is_frame or is_error_line:
                    trace_block.append(next_line)
                    i += 1
                elif is_blank and i + 1 < len(lines) and lines[i+1].startswith('  '):
                    # Blank line followed by more indented content = still in trace
                    trace_block.append(next_line)
                    i += 1
                else:
                    break  # Trace ended
            
            code_lines.extend(trace_block)
        else:
            prose_lines.append(line)
            i += 1
    
    return '\n'.join(code_lines), '\n'.join(prose_lines)


def clean_prose(text):
    """
    Clean up prose after code extraction:
    - Remove multiple blank lines (extraction leaves gaps)
    - Remove HTML tags (GitHub issues sometimes have <img>, <details> etc.)
    - Strip leading/trailing whitespace
    - Remove markdown image syntax ![alt](url) — not useful for NLP
    """
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove markdown images: ![alt text](url)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    
    # Collapse multiple blank lines into one
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Clean up lines that are just dashes or equals (markdown headers/dividers)
    text = re.sub(r'^[-=]{3,}$', '', text, flags=re.MULTILINE)
    
    return text.strip()


def segment_issue(title, body):
    """
    MAIN FUNCTION: Takes an issue title + body, returns prose and code separately.
    
    Why include title in prose?
    Title is always natural language — it's the human's one-line summary.
    It's the most important prose signal for duplicate detection.
    
    Pipeline:
    body → extract fenced blocks → extract stack traces → clean remaining prose
    """
    if not body:
        return title, ""
    
    # Step 1: Pull out fenced code blocks
    fenced_blocks, text_after_fenced = extract_fenced_blocks(body)
    
    # Step 2: Pull out stack traces from remaining text
    stack_traces, text_after_traces = extract_stack_traces(text_after_fenced)
    
    # Step 3: Clean what's left → this is our prose
    prose = clean_prose(text_after_traces)
    
    # Step 4: Add title to prose (it's always natural language)
    # Prepend title so it gets full weight in prose embedding
    if title:
        prose = title + "\n\n" + prose if prose else title
    
    # Step 5: Combine all code signals
    all_code_parts = fenced_blocks + ([stack_traces] if stack_traces.strip() else [])
    code = "\n\n---\n\n".join(all_code_parts)  # separator between different blocks
    
    return prose.strip(), code.strip()


def classify_issue(prose, code):
    """
    Classify whether an issue is prose-heavy or code-heavy.
    
    WHY classify? Our evaluation needs to show that the hybrid model
    improves MOST on code-heavy issues. We need this label to run that analysis.
    
    Simple heuristic:
    - Count prose characters vs code characters
    - If code > 30% of total content → code-heavy
    - Otherwise → prose-heavy
    
    30% threshold is a reasonable starting point. Can tune later.
    """
    prose_len = len(prose)
    code_len = len(code)
    total = prose_len + code_len
    
    if total == 0:
        return "empty"
    
    code_ratio = code_len / total
    
    if code_ratio > 0.5:
        return "code_heavy"
    elif code_ratio > 0.3:
        return "mixed"
    else:
        return "prose_heavy"


def process_dataset(input_path="data/ground_truth_pairs.json",
                    output_path="data/segmented_pairs.json"):
    """
    Run segmentation on every pair in our ground truth dataset.
    Adds 'prose' and 'code' fields to each issue in every pair.
    """
    print(f"Loading dataset from {input_path}...")
    
    with open(input_path, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    
    print(f"Segmenting {len(pairs)} pairs...")
    
    segmented = []
    
    for i, pair in enumerate(pairs):
        # Segment both the duplicate and original issue
        dup_prose, dup_code = segment_issue(
            pair["duplicate"]["title"],
            pair["duplicate"]["body"]
        )
        orig_prose, orig_code = segment_issue(
            pair["original"]["title"],
            pair["original"]["body"]
        )
        
        # Classify each issue
        dup_type = classify_issue(dup_prose, dup_code)
        orig_type = classify_issue(orig_prose, orig_code)
        
        segmented.append({
            "repo": pair["repo"],
            "duplicate": {
                **pair["duplicate"],     # keep original fields
                "prose": dup_prose,
                "code": dup_code,
                "issue_type": dup_type
            },
            "original": {
                **pair["original"],
                "prose": orig_prose,
                "code": orig_code,
                "issue_type": orig_type
            }
        })
        
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(pairs)}] done")
    
    # Save
    import os
    os.makedirs("data", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(segmented, f, indent=2, ensure_ascii=False)
    
    # Stats
    print(f"\n✅ Saved to {output_path}")
    print_segmentation_stats(segmented)
    
    return segmented


def print_segmentation_stats(pairs):
    from collections import Counter
    
    dup_types = Counter(p["duplicate"]["issue_type"] for p in pairs)
    
    print("\n── Segmentation Stats ───────────────────────────")
    print(f"Total pairs: {len(pairs)}")
    print(f"\nDuplicate issue types:")
    for t, count in dup_types.most_common():
        pct = 100 * count // len(pairs)
        print(f"  {t}: {count} ({pct}%)")
    
    # How many had actual code content?
    has_code = sum(1 for p in pairs if p["duplicate"]["code"].strip())
    print(f"\nPairs with extracted code: {has_code} ({100*has_code//len(pairs)}%)")
    print("─────────────────────────────────────────────────")


def demo_single_issue():
    """
    Test segmenter on one example issue so you can SEE what it does.
    Run this first to verify segmentation looks correct before processing all 520 pairs.
    """
    sample_title = "App crashes when loading user profile"
    sample_body = """
I'm getting a crash every time I try to load the profile page.
This started after upgrading to v2.1.0. Has anyone seen this before?

Steps to reproduce:
1. Log in to the app
2. Click on any user profile
3. App immediately crashes

```python
def load_profile(user_id):
    profile = db.query(User).filter_by(id=user_id).first()
    return profile.serialize()
```

Traceback (most recent call last):
  File "app.py", line 42, in load_profile
    return profile.serialize()
AttributeError: 'NoneType' object has no attribute 'serialize'

Environment: Python 3.10, SQLAlchemy 1.4, Ubuntu 22.04
"""

    prose, code = segment_issue(sample_title, sample_body)
    issue_type = classify_issue(prose, code)
    
    print("=" * 60)
    print("DEMO: Single Issue Segmentation")
    print("=" * 60)
    
    print("\n── PROSE ──────────────────────────────────────────────")
    print(prose)
    
    print("\n── CODE ───────────────────────────────────────────────")
    print(code)
    
    print("\n── CLASSIFICATION ─────────────────────────────────────")
    print(f"Issue type: {issue_type}")
    print("─────────────────────────────────────────────────────────")


if __name__ == "__main__":
    # Step 1: Demo on one example first
    demo_single_issue()
    
    # Step 2: Process full dataset
    print("\n\nProcessing full dataset...")
    process_dataset()