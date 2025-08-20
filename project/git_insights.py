"""
Git Insights utilities.
Parses recent git history to summarize authors, changed files, and commit message patterns.
Designed for CI environments (no external deps).
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from collections import Counter
from typing import List, Dict, Any
import subprocess
import json
import re


def _run(cmd: List[str]) -> str:
    """Run a shell command and return stdout or raise RuntimeError on failure."""
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{r.stderr}")
    return r.stdout.strip()


def _safe_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


@dataclass
class GitSummary:
    commit_count: int
    top_authors: List[Dict[str, Any]]
    top_files: List[Dict[str, Any]]
    top_words: List[Dict[str, Any]]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def get_commit_messages(limit: int = 50) -> List[str]:
    """Return the last `limit` commit subject lines."""
    out = _run(["git", "log", f"-n{limit}", "--pretty=%s"])
    if not out:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def get_authors(limit: int = 50) -> List[str]:
    """Return the last `limit` commit author names."""
    out = _run(["git", "log", f"-n{limit}", "--pretty=%an"])
    if not out:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def get_files_changed(limit: int = 50) -> List[str]:
    """
    Return a list of files changed across the last `limit` commits.
    Uses --name-only; ignores merge separators and blank lines.
    """
    out = _run(["git", "log", f"-n{limit}", "--name-only", "--pretty=format:"])
    files = []
    for line in out.splitlines():
        line = line.strip()
        if not line or line.startswith("commit "):
            continue
        files.append(line)
    return files


_STOPWORDS = {
    "the", "a", "an", "to", "and", "or", "of", "in", "on", "for", "by",
    "with", "add", "fix", "feat", "chore", "docs", "refactor", "merge",
    "update", "improve", "use"
}


def tokenize(text: str) -> List[str]:
    # Split on non-alphabetic; keep lowercase words of length >= 3.
    words = re.split(r"[^a-zA-Z]+", text.lower())
    return [w for w in words if len(w) >= 3 and w not in _STOPWORDS]


def collect_summary(limit: int = 50, top_k: int = 5) -> GitSummary:
    authors = get_authors(limit)
    files = get_files_changed(limit)
    subjects = get_commit_messages(limit)

    author_counts = Counter(authors)
    file_counts = Counter(files)

    word_counts = Counter()
    for subj in subjects:
        word_counts.update(tokenize(subj))

    top_auth = [{"name": name, "count": cnt} for name, cnt in author_counts.most_common(top_k)]
    top_files = [{"path": path, "count": cnt} for path, cnt in file_counts.most_common(top_k)]
    top_words = [{"word": w, "count": cnt} for w, cnt in word_counts.most_common(top_k)]

    return GitSummary(
        commit_count=len(subjects),
        top_authors=top_auth,
        top_files=top_files,
        top_words=top_words
    )
# TODO: improve tokenizer 
