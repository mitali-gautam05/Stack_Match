"""
Pattern Discovery Script
=========================
Goal: Find ALL the ways maintainers reference duplicate issues in comments.
Instead of guessing patterns, we READ actual comments and categorize them.

This is called "corpus analysis" — study your data before writing rules.
"""

import os
import re
import json
import requests
from collections import Counter
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {
    "Authorization": f"token {TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# Repos we already know have duplicates
REPOS = {
    "microsoft/vscode":     ["*duplicate"],
    "pandas-dev/pandas":    ["Duplicate Report", "duplicated"],
    "numpy/numpy":          ["50 - Duplicate"],
}

def get_issues(repo, label, count=50):
    url = f"https://api.github.com/repos/{repo}/issues"
    params = {"state": "closed", "labels": label, "per_page": count}
    r = requests.get(url, headers=HEADERS, params=params)
    return [i for i in r.json() if "pull_request" not in i]

def get_comments(repo, issue_number):
    url = f"https://api.github.com/repos/{repo}/issues/{issue_number}/comments"
    r = requests.get(url, headers=HEADERS)
    return r.json() if r.status_code == 200 else []

def extract_relevant_line(comment_body):
    """
    From a comment, extract ONLY the line that mentions duplicate/dup.
    We don't want to print entire comments — just the relevant part.
    """
    lines = comment_body.split('\n')
    for line in lines:
        line_lower = line.lower()
        if any(word in line_lower for word in [
            'duplicate', 'dup', 'same as', 'same issue', 'same bug',
            'already reported', 'already exists', 'tracked in',
            'see #', 'see https', 'refer', 'closed by', 'fixed by',
            'related to', 'superseded'
        ]):
            return line.strip()
    return None

def categorize_pattern(line):
    """
    Look at a line and figure out what PATTERN it uses.
    Returns a category label.
    """
    line_lower = line.lower()
    
    # Pattern 1: Slash command
    if line_lower.startswith('/duplicate'):
        return "SLASH_COMMAND"
    
    # Pattern 2: Full URL with keyword
    if 'https://github.com' in line and any(w in line_lower for w in ['duplicate', 'dup']):
        return "KEYWORD_PLUS_URL"
    
    # Pattern 3: Just URL, no keyword (e.g. "See https://github.com/...")
    if 'https://github.com' in line and 'issues/' in line:
        return "URL_ONLY"
    
    # Pattern 4: Hash reference with keyword (#1234)
    if re.search(r'#\d+', line) and any(w in line_lower for w in ['duplicate', 'dup', 'same']):
        return "HASH_WITH_KEYWORD"
    
    # Pattern 5: Just hash, no keyword ("See #1234", "^^ #1234")
    if re.search(r'#\d+', line):
        return "HASH_ONLY"
    
    # Pattern 6: "Same as" / "Same issue as"
    if 'same as' in line_lower or 'same issue' in line_lower:
        return "SAME_AS"
    
    # Pattern 7: "Already reported/tracked"
    if any(w in line_lower for w in ['already reported', 'already exists', 'already tracked']):
        return "ALREADY_REPORTED"
    
    # Pattern 8: "Tracked in" / "Fixed in"
    if any(w in line_lower for w in ['tracked in', 'fixed in', 'closed by']):
        return "TRACKED_IN"
    
    return "OTHER"

def main():
    all_patterns = []
    pattern_examples = {}  # category → list of example lines
    
    print("Scanning comments for duplicate patterns...\n")
    
    for repo, labels in REPOS.items():
        print(f"[{repo}]")
        
        for label in labels:
            issues = get_issues(repo, label, count=80)
            print(f"  Label '{label}': {len(issues)} issues")
            
            for issue in issues:
                comments = get_comments(repo, issue['number'])
                
                for comment in comments:
                    body = comment.get('body', '') or ''
                    relevant_line = extract_relevant_line(body)
                    
                    if relevant_line:
                        category = categorize_pattern(relevant_line)
                        all_patterns.append(category)
                        
                        # Store up to 3 examples per category
                        if category not in pattern_examples:
                            pattern_examples[category] = []
                        if len(pattern_examples[category]) < 3:
                            pattern_examples[category].append({
                                'repo': repo,
                                'issue': issue['number'],
                                'line': relevant_line[:200]  # truncate long lines
                            })
    
    # ── Results ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("PATTERN ANALYSIS RESULTS")
    print("="*60)
    
    counter = Counter(all_patterns)
    total = sum(counter.values())
    
    print(f"\nTotal duplicate-referencing comments found: {total}")
    print(f"\nPattern breakdown:\n")
    
    for pattern, count in counter.most_common():
        pct = 100 * count // total
        print(f"  {pattern:<25} {count:>4} occurrences ({pct}%)")
        
        # Show examples
        if pattern in pattern_examples:
            for ex in pattern_examples[pattern]:
                print(f"    Example (#{ex['issue']}):")
                print(f"    >>> {ex['line']}")
        print()
    
    # ── Which patterns our current regex MISSES ────────────────────────────────
    print("="*60)
    print("PATTERNS OUR CURRENT REGEX HANDLES vs MISSES")
    print("="*60)
    
    currently_handled = {"SLASH_COMMAND", "KEYWORD_PLUS_URL", "HASH_WITH_KEYWORD"}
    
    for pattern, count in counter.most_common():
        pct = 100 * count // total
        status = "✅ HANDLED" if pattern in currently_handled else "❌ MISSED"
        print(f"  {status}  {pattern:<25} {count:>4} ({pct}%)")
    
    missed = sum(count for p, count in counter.items() if p not in currently_handled)
    print(f"\nTotal missed: {missed}/{total} ({100*missed//total}%)")
    
    # Save for reference
    with open("data/pattern_analysis.json", "w") as f:
        json.dump({
            "counts": dict(counter),
            "examples": pattern_examples
        }, f, indent=2)
    print(f"\n✅ Full analysis saved to data/pattern_analysis.json")

if __name__ == "__main__":
    main()