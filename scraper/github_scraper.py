"""
StackMatch — GitHub Scraper (Final)
=====================================
Fix: Each repo now uses its actual duplicate label (found via check_labels.py)
"""

import os
import re
import json
import time
import requests
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("GITHUB_TOKEN")

HEADERS = {
    "Authorization": f"token {TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# ── REPO → EXACT LABEL MAPPING ─────────────────────────────────────────────────
# WHY a dict instead of one label for all?
# Because every team has their own conventions.
# We discovered this by running check_labels.py first — always explore before coding.
REPOS = {
    "microsoft/vscode":        ["*duplicate"],
    "huggingface/transformers": ["duplicate"],
    "django/django":           ["duplicate"],
    "numpy/numpy":             ["50 - Duplicate"],
    "pandas-dev/pandas":       ["Duplicate Report", "duplicated"],  # two labels!
    "keras-team/keras":        ["Duplicate"],
}

# ── REGEX PATTERNS ─────────────────────────────────────────────────────────────
# Three real-world formats we found in debug_comments.py:
# 1. /duplicate https://github.com/owner/repo/issues/1234  (vscode slash command)
# 2. duplicate issue of https://github.com/.../issues/1234 (sentence with URL)
# 3. Duplicate of #1234                                     (old style hash ref)

SLASH_PATTERN = re.compile(
    r"\/duplicate\s+https://github\.com/[^/\s]+/[^/\s]+/issues/(\d+)",
    re.IGNORECASE
)

URL_PATTERN = re.compile(
    r"(?:duplicate(?:\s+issue)?(?:\s+of)?|dup(?:\s+of)?|close\s+as\s+dup|same\s+as|see|duplicates)\s+"
    r"https://github\.com/[^/\s]+/[^/\s]+/issues/(\d+)",
    re.IGNORECASE
)

HASH_PATTERN = re.compile(
    r"(?:duplicate\s+of|dup\s+of|duplicates|same\s+as|see)\s*#(\d+)",
    re.IGNORECASE
)


def extract_original_number(text):
    """
    Try all 3 patterns in order of specificity.
    Return the first match found, or None.
    
    WHY order matters: More specific patterns first avoids false positives.
    e.g. "/duplicate URL" is unambiguous; "see #123" could be coincidental.
    """
    for pattern in [SLASH_PATTERN, URL_PATTERN, HASH_PATTERN]:
        m = pattern.search(text)
        if m:
            return int(m.group(1))
    return None


def make_request(url, params=None):
    """Rate-limit aware GET request. Waits and retries on 403/429."""
    while True:
        response = requests.get(url, headers=HEADERS, params=params)

        if response.status_code in (403, 429):
            reset_time = int(response.headers.get("X-RateLimit-Reset", time.time() + 60))
            wait = max(reset_time - time.time(), 0) + 5
            print(f"  Rate limited. Waiting {int(wait)}s...")
            time.sleep(wait)
            continue

        if response.status_code == 200:
            return response.json()

        print(f"  ERROR {response.status_code} → {url}")
        return None


def get_duplicate_issues(repo, label, max_issues=500):
    """Fetch closed issues with a specific duplicate label."""
    print(f"  Fetching label: '{label}'")
    issues = []
    page = 1

    while len(issues) < max_issues:
        data = make_request(
            f"https://api.github.com/repos/{repo}/issues",
            params={
                "state": "closed",
                "labels": label,
                "per_page": 100,
                "page": page
            }
        )

        if not data:
            break

        # Filter out pull requests — GitHub API mixes them with issues
        real_issues = [i for i in data if "pull_request" not in i]
        issues.extend(real_issues)
        print(f"    Page {page}: {len(real_issues)} issues (total: {len(issues)})")

        if len(data) < 100:
            break

        page += 1
        time.sleep(0.5)

    return issues


def find_original_number(repo, issue_number):
    """
    Search comments + issue body for duplicate reference.
    WHY both? Some maintainers edit the issue body; others leave a comment.
    """
    # Check all comments
    comments = make_request(
        f"https://api.github.com/repos/{repo}/issues/{issue_number}/comments"
    )
    if comments:
        for comment in comments:
            result = extract_original_number(comment.get("body", "") or "")
            if result:
                return result

    # Check issue body itself
    issue = make_request(f"https://api.github.com/repos/{repo}/issues/{issue_number}")
    if issue:
        result = extract_original_number(issue.get("body", "") or "")
        if result:
            return result

    return None


def fetch_issue(repo, number):
    """Fetch a single issue by number."""
    return make_request(f"https://api.github.com/repos/{repo}/issues/{number}")


def scrape_repo(repo, labels, max_issues=500):
    """
    Main scraping logic for one repo.
    Handles repos with MULTIPLE duplicate labels (e.g. pandas has two).
    """
    print(f"\n[{repo}]")

    # Collect issues across ALL duplicate labels for this repo
    all_issues = []
    seen_numbers = set()  # avoid processing same issue twice

    for label in labels:
        issues = get_duplicate_issues(repo, label, max_issues)
        for issue in issues:
            if issue["number"] not in seen_numbers:
                all_issues.append(issue)
                seen_numbers.add(issue["number"])

    print(f"  Total unique duplicate issues: {len(all_issues)}")
    print(f"  Finding original issues...")

    pairs = []

    for idx, dup_issue in enumerate(all_issues):
        dup_number = dup_issue["number"]

        original_number = find_original_number(repo, dup_number)

        if original_number is None or original_number == dup_number:
            continue

        original_issue = fetch_issue(repo, original_number)
        if original_issue is None:
            continue

        # Skip if original is ALSO marked as duplicate (chained duplicates — messy data)
        original_labels = [l["name"].lower() for l in original_issue.get("labels", [])]
        if any("dup" in l for l in original_labels):
            continue

        pairs.append({
            "repo": repo,
            "duplicate": {
                "number": dup_issue["number"],
                "title": dup_issue["title"],
                "body": dup_issue.get("body", "") or "",
                "url": dup_issue["html_url"]
            },
            "original": {
                "number": original_issue["number"],
                "title": original_issue["title"],
                "body": original_issue.get("body", "") or "",
                "url": original_issue["html_url"]
            }
        })

        if (idx + 1) % 25 == 0:
            print(f"  [{idx+1}/{len(all_issues)}] {len(pairs)} pairs found")

        time.sleep(0.3)

    return pairs


def save_dataset(all_pairs, path="data/ground_truth_pairs.json"):
    os.makedirs("data", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_pairs, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved {len(all_pairs)} pairs → {path}")


def print_stats(all_pairs):
    print("\n── Dataset Stats ────────────────────────────────")
    print(f"Total pairs: {len(all_pairs)}")

    from collections import Counter
    for repo, count in Counter(p["repo"] for p in all_pairs).most_common():
        print(f"  {repo}: {count}")

    code_heavy = sum(
        1 for p in all_pairs
        if "```" in p["duplicate"]["body"] or "```" in p["original"]["body"]
    )
    pct = 100 * code_heavy // max(len(all_pairs), 1)
    print(f"\nCode-block pairs: {code_heavy} ({pct}%)")
    print("─────────────────────────────────────────────────")


if __name__ == "__main__":
    all_pairs = []

    for repo, labels in REPOS.items():
        try:
            pairs = scrape_repo(repo, labels, max_issues=300)
            all_pairs.extend(pairs)
            print(f"  ✓ Done: {len(pairs)} pairs")
        except Exception as e:
            print(f"  ✗ Failed: {e}")

    save_dataset(all_pairs)
    print_stats(all_pairs)