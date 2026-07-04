"""
Updated regex patterns for github_scraper.py
Based on pattern_discovery.py analysis of 399 real comments.

Coverage after fix: ~58% (was 56%, OTHER 25% is unrecoverable)

Replace the regex section in scraper/github_scraper.py with this.
"""

import re

# ── PATTERN 1: Slash command (/duplicate #123 or /duplicate URL) ──────────────
# Matches: "/duplicate of #323798"
#          "/duplicate https://github.com/.../issues/319821"
# Note: vscode bot uses "/duplicate of #N" (with "of") — discovered from data!
SLASH_PATTERN = re.compile(
    r"\/duplicate(?:\s+of)?\s+(?:https://github\.com/[^/\s]+/[^/\s]+/issues/(\d+)|#(\d+))",
    re.IGNORECASE
)

# ── PATTERN 2: Keyword + URL ───────────────────────────────────────────────────
# Matches: "duplicate of https://github.com/.../issues/1234"
#          "same as https://github.com/.../issues/1234"
#          "same issue as https://github.com/.../issues/1234"
#          "linked to https://github.com/.../issues/1234"
URL_PATTERN = re.compile(
    r"(?:duplicate(?:\s+issue)?(?:\s+of)?|dup(?:\s+of)?|same\s+(?:as|issue\s+as)|"
    r"same\s+bug\s+as|linked\s+to|tracked\s+in|covered\s+by|see|fixed\s+by|closed\s+by)\s+"
    r"https://github\.com/[^/\s]+/[^/\s]+/issues/(\d+)",
    re.IGNORECASE
)

# ── PATTERN 3: Keyword + Hash (#1234) ─────────────────────────────────────────
# Matches: "Duplicate of #317948"
#          "same as #319834"
#          "This is covered by #316354"
#          "fixed by #57957"
#          "closing in favor of #1234"
HASH_PATTERN = re.compile(
    r"(?:duplicate(?:\s+issue)?(?:\s+of)?|dup(?:\s+of)?|same\s+(?:as|issue)|"
    r"covered\s+by|fixed\s+(?:in|by)|closed\s+by|closing\s+(?:in\s+favor\s+of|as)|"
    r"see|tracks?)\s*#(\d+)",
    re.IGNORECASE
)

# ── PATTERN 4: Bare URL (no keyword needed) ────────────────────────────────────
# Matches: "https://github.com/microsoft/vscode/issues/320551"
# WHY: Some comments just paste a URL with no keyword — e.g.
#      "Same issue as  https://github.com/microsoft/vscode/issues/320551 btw"
#      URL_PATTERN already handles "same issue as URL" but sometimes
#      there's just a URL in a clearly duplicate-closing comment.
# CAUTION: This is risky — any URL in any comment would match.
# So we ONLY use this pattern when the issue IS labeled duplicate.
# (The scraper already filters by label, so this is safe here.)
BARE_URL_PATTERN = re.compile(
    r"https://github\.com/[^/\s]+/[^/\s]+/issues/(\d+)",
    re.IGNORECASE
)


def extract_original_number(comment_body, use_bare_url=False):
    """
    Try all patterns in order of specificity (most specific first).
    
    WHY order matters:
    More specific patterns = less chance of false positive.
    SLASH_PATTERN is most explicit (bot command).
    BARE_URL_PATTERN is least explicit (any URL could match).
    
    use_bare_url=False by default — only enable when you're sure
    the comment is a duplicate-closing comment (e.g., by a bot).
    """
    # Try slash command first — most explicit
    m = SLASH_PATTERN.search(comment_body)
    if m:
        # Group 1 = URL match, Group 2 = hash match
        return int(m.group(1) or m.group(2))
    
    # Try keyword + URL
    m = URL_PATTERN.search(comment_body)
    if m:
        return int(m.group(1))
    
    # Try keyword + hash
    m = HASH_PATTERN.search(comment_body)
    if m:
        return int(m.group(1))
    
    # Try bare URL only if caller says it's safe
    if use_bare_url:
        m = BARE_URL_PATTERN.search(comment_body)
        if m:
            return int(m.group(1))
    
    return None


# ── TEST: Verify all patterns work on real examples from our data ──────────────
if __name__ == "__main__":
    test_cases = [
        # SLASH_COMMAND (vscode bot style)
        ("/duplicate of #323798", 323798),
        ("/duplicate https://github.com/microsoft/vscode/issues/319821", 319821),
        
        # KEYWORD_PLUS_URL
        ("same as https://github.com/microsoft/vscode/issues/320551 btw", 320551),
        ("I think it is linked to https://github.com/microsoft/vscode/issues/321007", 321007),
        ("This is covered by #316354 which tracks the Agents window", 316354),
        
        # HASH_WITH_KEYWORD
        ("same as #319834", 319834),
        ("Duplicate of #317948", 317948),
        ("fixed by #57957 and will be released as part of 3.0", 57957),
        ("closing in favor of #1234", 1234),
        
        # Should NOT match (false positives to avoid)
        ("This could be related to GPU acceleration", None),
        ("This appears to be a regression introduced in 1.122.0", None),
        ("I met this issue when I use Remote SSH. Refer the following figure", None),
    ]
    
    print("Pattern Test Results:")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for text, expected in test_cases:
        result = extract_original_number(text)
        status = "✅" if result == expected else "❌"
        
        if result == expected:
            passed += 1
        else:
            failed += 1
        
        print(f"{status} Expected: {str(expected):>8} | Got: {str(result):>8}")
        print(f"   Text: {text[:70]}")
        print()
    
    print(f"Results: {passed}/{len(test_cases)} passed")
    
    if failed == 0:
        print("\n🎉 All patterns working correctly!")
        print("Now update scraper/github_scraper.py with these patterns.")
    else:
        print(f"\n⚠️  {failed} patterns failing — fix before updating scraper.")