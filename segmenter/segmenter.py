"""
StackMatch — Segmenter v2 (with Noise Filtering)
==================================================
What's new:
  - Telemetry/A-B experiment blocks filtered out
  - System info tables filtered from prose
  - Low quality pairs removed (too short, no real content)
  - Better code detection
"""

import re
import json
import os


# ── PATTERNS ───────────────────────────────────────────────────────────────────

FENCED_CODE_PATTERN = re.compile(r"```(?:\w+)?\n(.*?)\n```", re.DOTALL)

TRACEBACK_START = re.compile(
    r"^(Traceback \(most recent call last\)|"
    r"Exception in thread|"
    r"[\w.]+(?:Error|Exception|Warning):)",
    re.MULTILINE
)

STACK_FRAME = re.compile(
    r"^\s+(?:File \".*?\", line \d+|at [\w.$<>]+\(.*?:\d+\)|at .*? in .*?:\d+)",
    re.MULTILINE
)

# ── NOISE DETECTION ────────────────────────────────────────────────────────────

def is_telemetry_block(text):
    """
    Detect VSCode A/B experiment blocks and similar telemetry.
    
    Pattern: lines like "vsliv368:30146709" — short key:number pairs
    If MORE than 60% of lines match this pattern → it's telemetry, not code.
    
    WHY 60%? Some code files have constants like PORT=8080, but rarely
    do 60%+ of lines look like random_string:big_number.
    """
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    if not lines:
        return False
    
    # Pattern: word characters, then colon, then 6+ digit number
    telemetry_pattern = re.compile(r'^[\w_]+:\d{6,}$')
    telemetry_lines = sum(1 for l in lines if telemetry_pattern.match(l))
    
    return (telemetry_lines / len(lines)) > 0.6


def is_real_code(text):
    """
    Is this actually code worth embedding with CodeBERT?
    
    Real code has: function definitions, imports, operators, brackets
    Noise has: key:value pairs, random IDs, pure data dumps
    
    WHY this check? CodeBERT was trained on real code. Feeding it
    telemetry IDs wastes computation and adds noise to embeddings.
    """
    if not text or len(text.strip()) < 20:
        return False
    
    if is_telemetry_block(text):
        return False
    
    # Real code signals
    code_signals = [
        r'\bdef \w+\(',           # Python function
        r'\bclass \w+',           # Class definition
        r'\bimport \w+',          # Import statement
        r'\bfunction\s+\w+\(',    # JS function
        r'\bpublic\s+\w+\s+\w+', # Java method
        r'Traceback',             # Python traceback
        r'Exception',             # Any exception
        r'Error:',                # Error line
        r'\bat \w+\.\w+\(',       # Java stack frame
        r'File ".*", line \d+',   # Python stack frame
        r'[{}[\]();]',            # Code brackets/semicolons
        r'\w+\s*=\s*\w+\(',       # Assignment with function call
        r'#include',              # C/C++
        r'package \w+',           # Java/Go package
    ]
    
    for signal in code_signals:
        if re.search(signal, text):
            return True
    
    return False


def clean_prose(text):
    """
    Remove noise from prose:
    1. HTML tags
    2. Markdown images
    3. System info tables (|Item|Value| style markdown tables with tech data)
    4. Multiple blank lines
    5. HTML comments
    """
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove HTML comments
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    
    # Remove markdown images
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    
    # Remove system info markdown tables
    # These look like: |Item|Value| / |---|---| / |CPUs|Intel...|
    # WHY remove? System info (CPU, GPU, memory) is not useful for
    # semantic similarity — two issues with same bug but different hardware
    # should still be detected as duplicates.
    text = re.sub(r'\|[^\n]+\|[^\n]*\n', '', text)
    
    # Remove "A/B Experiments" section heading and surrounding content
    text = re.sub(r'A/B Experiments.*$', '', text, flags=re.DOTALL)
    
    # Remove VS Code version lines (noise — not useful for similarity)
    text = re.sub(r'VS Code version:.*\n?', '', text)
    text = re.sub(r'OS version:.*\n?', '', text)
    
    # Remove markdown dividers
    text = re.sub(r'^[-=]{3,}$', '', text, flags=re.MULTILINE)
    
    # Collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


# ── SEGMENTATION ───────────────────────────────────────────────────────────────

def extract_fenced_blocks(text):
    """Extract fenced code blocks. Filter out telemetry/noise blocks."""
    real_code_parts = []
    
    def handle_match(match):
        content = match.group(1).strip()
        if is_real_code(content):
            real_code_parts.append(content)
        # Either way, remove from prose
        return ""
    
    cleaned_prose = FENCED_CODE_PATTERN.sub(handle_match, text)
    return real_code_parts, cleaned_prose


def extract_stack_traces(text):
    """Extract stack traces that appear without backticks."""
    lines = text.split('\n')
    code_lines = []
    prose_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        if TRACEBACK_START.match(line.strip()):
            trace_block = [line]
            i += 1
            
            while i < len(lines):
                next_line = lines[i]
                is_indented = next_line.startswith('  ') or next_line.startswith('\t')
                is_frame = bool(STACK_FRAME.match(next_line))
                is_error_line = bool(re.match(r'[\w.]+(?:Error|Exception):', next_line))
                is_blank = next_line.strip() == ''
                
                if is_indented or is_frame or is_error_line:
                    trace_block.append(next_line)
                    i += 1
                elif is_blank and i + 1 < len(lines) and lines[i+1].startswith('  '):
                    trace_block.append(next_line)
                    i += 1
                else:
                    break
            
            code_lines.extend(trace_block)
        else:
            prose_lines.append(line)
            i += 1
    
    return '\n'.join(code_lines), '\n'.join(prose_lines)


def segment_issue(title, body):
    """Main segmentation function."""
    if not body:
        return title or "", ""
    
    fenced_blocks, text_after_fenced = extract_fenced_blocks(body)
    stack_traces, text_after_traces = extract_stack_traces(text_after_fenced)
    prose = clean_prose(text_after_traces)
    
    if title:
        prose = title + "\n\n" + prose if prose else title
    
    # Only add stack trace if it's real
    all_code = fenced_blocks
    if stack_traces.strip() and is_real_code(stack_traces):
        all_code.append(stack_traces)
    
    code = "\n\n---\n\n".join(all_code)
    return prose.strip(), code.strip()


def classify_issue(prose, code):
    """Classify issue by content type."""
    prose_len = len(prose)
    code_len = len(code)
    total = prose_len + code_len
    
    if total == 0:
        return "empty"
    
    code_ratio = code_len / total
    
    if code_ratio > 0.5:
        return "code_heavy"
    elif code_ratio > 0.2:   # Lowered from 0.3 → catches more mixed cases
        return "mixed"
    else:
        return "prose_heavy"


# ── QUALITY FILTER ─────────────────────────────────────────────────────────────

def is_quality_pair(pair):
    """
    Filter out low quality pairs that would hurt ML training.
    
    WHY filter? ML models learn from patterns. Bad examples teach bad patterns.
    Better to have 400 clean pairs than 520 noisy ones.
    
    Filters:
    1. Both issues must have meaningful prose (min 50 chars)
    2. Title must be non-trivial (not just "bug" or "the github")
    3. Issues must not be identical (same person filing twice)
    """
    dup = pair["duplicate"]
    orig = pair["original"]
    
    # Both must have enough prose
    if len(dup["prose"]) < 50 or len(orig["prose"]) < 50:
        return False
    
    # Title must be meaningful (more than 3 words)
    dup_title_words = len(dup["title"].split())
    orig_title_words = len(orig["title"].split())
    if dup_title_words < 3 or orig_title_words < 3:
        return False
    
    # Issues shouldn't be character-for-character identical
    # (same person filing same issue twice with copy-paste)
    if dup["prose"][:200] == orig["prose"][:200]:
        return False
    
    return True


# ── MAIN PIPELINE ──────────────────────────────────────────────────────────────

def process_dataset(input_path="data/ground_truth_pairs.json",
                    output_path="data/segmented_pairs.json"):
    
    print(f"Loading {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    
    print(f"Processing {len(pairs)} pairs...")
    
    segmented = []
    filtered_count = 0
    
    for i, pair in enumerate(pairs):
        dup_prose, dup_code = segment_issue(
            pair["duplicate"]["title"],
            pair["duplicate"]["body"]
        )
        orig_prose, orig_code = segment_issue(
            pair["original"]["title"],
            pair["original"]["body"]
        )
        
        candidate = {
            "repo": pair["repo"],
            "duplicate": {
                **pair["duplicate"],
                "prose": dup_prose,
                "code": dup_code,
                "issue_type": classify_issue(dup_prose, dup_code)
            },
            "original": {
                **pair["original"],
                "prose": orig_prose,
                "code": orig_code,
                "issue_type": classify_issue(orig_prose, orig_code)
            }
        }
        
        if is_quality_pair(candidate):
            segmented.append(candidate)
        else:
            filtered_count += 1
        
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(pairs)}] kept: {len(segmented)}, filtered: {filtered_count}")
    
    os.makedirs("data", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(segmented, f, indent=2, ensure_ascii=False)
    
    print_stats(segmented, filtered_count, len(pairs))
    return segmented


def print_stats(pairs, filtered, total):
    from collections import Counter
    
    print(f"\n✅ Saved to data/segmented_pairs.json")
    print(f"\n── Dataset Quality Stats ─────────────────────────")
    print(f"Total raw pairs:     {total}")
    print(f"Filtered (noise):    {filtered}")
    print(f"Clean pairs kept:    {len(pairs)}")
    print(f"Retention rate:      {100*len(pairs)//total}%")
    
    types = Counter(p["duplicate"]["issue_type"] for p in pairs)
    print(f"\nIssue types:")
    for t, count in types.most_common():
        print(f"  {t}: {count} ({100*count//len(pairs)}%)")
    
    has_code = sum(1 for p in pairs if p["duplicate"]["code"].strip())
    print(f"\nWith real code:      {has_code} ({100*has_code//len(pairs)}%)")
    print("──────────────────────────────────────────────────")


if __name__ == "__main__":
    process_dataset()