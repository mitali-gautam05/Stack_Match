"""
Label Detective — Dekho har repo mein duplicate ke liye kaunsa label use hota hai
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

REPOS = [
    "microsoft/vscode",
    "pytorch/pytorch",
    "tensorflow/tensorflow",
    "huggingface/transformers",
    "django/django",
    "pallets/flask",
    "numpy/numpy",
    "pandas-dev/pandas",
    "scikit-learn/scikit-learn",
    "keras-team/keras"
]

for repo in REPOS:
    url = f"https://api.github.com/repos/{repo}/labels"
    r = requests.get(url, headers=HEADERS, params={"per_page": 100})
    labels = r.json()
    
    # Find anything that sounds like "duplicate"
    dup_labels = [
        l["name"] for l in labels
        if any(word in l["name"].lower() for word in ["dup", "duplicate", "dupe", "wont", "invalid", "stale"])
    ]
    
    print(f"\n{repo}")
    if dup_labels:
        print(f"  Relevant labels: {dup_labels}")
    else:
        print(f"  No duplicate-style labels found")
        # Print ALL labels so we can see what they use
        all_names = [l["name"] for l in labels]
        print(f"  All labels: {all_names[:15]}")  # first 15 only