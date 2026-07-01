"""
Debug Script — Dekho actual comments kaisi dikhti hain
Isse pata chalega ki regex kyun match nahi ho raha
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {
    "Authorization": f"token {TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

def get_comments(repo, issue_number):
    url = f"https://api.github.com/repos/{repo}/issues/{issue_number}/comments"
    r = requests.get(url, headers=HEADERS)
    return r.json()

def get_duplicate_issues(repo, count=5):
    url = f"https://api.github.com/repos/{repo}/issues"
    params = {"state": "closed", "labels": "duplicate", "per_page": count}
    r = requests.get(url, headers=HEADERS, params=params)
    return r.json()

# vscode ke 5 duplicate issues ke comments print karo
repo = "microsoft/vscode"
issues = get_duplicate_issues(repo, count=5)

# Filter out PRs
issues = [i for i in issues if "pull_request" not in i]

print(f"Found {len(issues)} duplicate issues\n")

for issue in issues[:5]:
    print(f"{'='*60}")
    print(f"Issue #{issue['number']}: {issue['title']}")
    print(f"URL: {issue['html_url']}")
    
    comments = get_comments(repo, issue['number'])
    print(f"Total comments: {len(comments)}")
    
    for i, c in enumerate(comments):
        body = c.get('body', '') or ''
        print(f"\n  Comment {i+1} (by {c['user']['login']}):")
        print(f"  >>> {body[:300]}")  # First 300 chars only
    print()