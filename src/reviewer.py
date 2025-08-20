import os
import subprocess
import argparse
import textwrap
import json
import glob
import time
import random


def run(cmd: str) -> str:
    """Run a shell command and return stdout (raise on failure)."""
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"Command failed: {cmd}\n{r.stderr}")
    return r.stdout.strip()


def git_diff(base: str, head: str) -> str:
    return run(f"git diff --unified=2 {base} {head}")


def git_commit_message(ref: str) -> str:
    return run(f"git log -1 --pretty=%B {ref}")


def git_changed_files(base: str, head: str) -> str:
    return run(f"git diff --name-only {base} {head}")


def read_docs_snippets(doc_dir: str = "docs", max_chars: int = 2000) -> str:
    """Concatenate repo docs (RAG-lite context) with simple size cap."""
    buf = []
    for path in glob.glob(f"{doc_dir}/**/*", recursive=True):
        if any(path.endswith(ext) for ext in [".md", ".txt", ".rst"]):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    buf.append(f"\n\n---\n# {path}\n" + f.read())
            except Exception:
                pass
    out = ("\n".join(buf))[:max_chars]
    return out or "(no docs found)"


def build_prompt(commit_msg: str, diff_text: str, docs_context: str, files_block: str):
    sys_prompt = (
        "You are a precise senior code reviewer. "
        "Return actionable, concise findings with sections: Summary, Risks, "
        "Style/Docs, Correctness, Security, Suggested Patches. "
        "Reference the repository style guide if relevant."
    )
    user_prompt = textwrap.dedent(f"""
    ## Repository Context (docs)
    {docs_context}

    ## Files Changed
    {files_block}

    ## Commit Message
    {commit_msg}

    ## Diff
    {diff_text}

    ## Review Goals
    - Identify logic errors, edge cases, and unsafe patterns.
    - Enforce style guide (docstrings, type hints, PEP8).
    - Suggest concrete fixes with small code blocks.
    """)
    return sys_prompt, user_prompt


def call_llm(system_msg: str, user_msg: str) -> str:
    """
    Call OpenAI-compatible APIs with resilience:
      - small pre-call jitter (to avoid hotspots)
      - retries with exponential backoff on 429/5xx
      - prefer /chat/completions; fallback to /responses (two shapes)
      - reads OPENAI_TEMPERATURE and OPENAI_MAX_TOKENS
    """
    import requests

    # ---- Config (trim whitespace) -------------------------------------------
    ok = (os.getenv("OPENAI_API_KEY") or "").strip()
    ob = (os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").strip()
    om = (os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip()
    org = (os.getenv("OPENAI_ORG") or "").strip()
    proj = (os.getenv("OPENAI_PROJECT") or "").strip()

    gk = (os.getenv("LLM_API_KEY") or "").strip()
    gb = (os.getenv("LLM_API_BASE") or "").strip()

    # Tuning knobs (env-overridable)
    temperature = float((os.getenv("OPENAI_TEMPERATURE") or "0.2").strip())
    max_tokens = int((os.getenv("OPENAI_MAX_TOKENS") or "600").strip())
    pre_delay = os.getenv("OPENAI_PRE_DELAY_SEC")
    try:
        pre_delay = float(pre_delay) if pre_delay is not None else None
    except ValueError:
        pre_delay = None
    # Retry controls
    max_attempts = int((os.getenv("OPENAI_MAX_ATTEMPTS") or "5").strip())
    backoff_cap = int((os.getenv("OPENAI_BACKOFF_CAP_SEC") or "25").strip())

    # Small jitter before first call to dodge bursts
    if pre_delay is None:
        time.sleep(random.uniform(0.8, 1.8))
    elif pre_delay > 0:
        time.sleep(pre_delay)

    def post_with_retries(url, headers, payload, timeout=60):
        last_exc = None
        for attempt in range
::contentReference[oaicite:0]{index=0}
