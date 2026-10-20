"""Opt-in same-turn review. A reasoning aid, not a calibrated truth classifier."""

REVIEW_TOOLS = frozenset({"read", "ls", "glob", "grep"})

REVIEW = """meditate: review the draft once before finalizing.
Compare what the user asked with what the draft actually answers. Break material
factual claims into small chunks. Seek disjoint evidence: a source not already
used to form the draft, only where it can test an important claim. Read-only
filesystem tools are available; do not mutate anything, spawn work, or repeat
completed actions. Existing permissions and budgets still apply.
Reassemble against the original request and correct the answer once. If the truth
requires a live service probe or evidence you cannot access, state that limit;
do not treat absence of verification as a contradiction. Return the useful final
answer, preserving verified artifacts. Do not claim this review proves correctness.
Do not repeat the review or append an internal checklist to the user's answer.
"""
