import os
import subprocess
import argparse
import textwrap
import json
import glob
import time
import random
import hashlib
import pathlib
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

# =========================
# Config constants (must be defined before functions that use them)
# =========================
CHARS_PER_CHUNK = int(os.getenv("CHARS_PER_CHUNK", "8000"))
MAX_DIFF_CHARS  = int(os.getenv("MAX_DIFF_CHARS", "120000"))

CACHE_DIR = pathlib.Path(os.getenv("REVIEW_CACHE_DIR", ".ai-review-cache"))
CACHE_DIR.mkdir(exist_ok=True)
CACHE_TTL_SEC = int(os.getenv("REVIEW_CACHE_TTL_SEC", "0"))
CACHE_BUSTER  = os.getenv("REVIEW_CACHE_BUSTER", "").strip()

LLM_BATCH_SIZE = int(os.getenv("LLM_BATCH_SIZE", "4"))

# =========================
# Shell & Git helpers
# =========================

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


# =========================
# Docs (RAG-lite) context
# =========================

def read_docs_snippets(doc_dir: str = "docs", max_chars: int = 1000) -> str:
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


# =========================
# Prompt builders
# =========================

def build_batch_prompt(
    commit_msg: str,
    docs_context: str,
    items: List[Dict[str, str]],
) -> Tuple[str, str]:
    sys_prompt = (
        "You are a precise senior code reviewer. "
        "For each section, return actionable findings grouped by the EXACT heading shown. "
        "Use sections per heading with: Summary, Risks, Correctness, Security, Style/Docs, Suggested Patches. "
        "Keep it concise but concrete; include code blocks for fixes when relevant."
    )

    header = textwrap.dedent(f"""
    ## Repository Context (docs)
    {docs_context}

    ## Commit Message
    {commit_msg}

    ## Review Goals
    - Identify logic errors, edge cases, and unsafe patterns.
    - Enforce style guide (docstrings, type hints, PEP8).
    - Suggest concrete fixes with small code blocks.
    - IMPORTANT: Echo each heading exactly as provided so results align with files/chunks.
    """)

    parts = [header, "## Diffs (review each section)"]
    for it in items:
        parts.append(
            f"\n\n### HEADING\n## {it['path']} (chunk {it['idx']}/{it['total']})\n"
            f"```diff\n{it['diff']}\n```"
        )

    user_prompt = "\n".join(parts)
    return sys_prompt, user_prompt


# =========================
# Robust LLM caller
# =========================

def call_llm(system_msg: str, user_msg: str) -> str:
    """
    GPT-5–friendly LLM caller:
      - Skips /chat/completions for gpt-5* models.
      - Uses Responses API with plain-string role content first.
      - Falls back to single-string input.
      - Spaces out calls (OPENAI_INTER_CALL_DELAY_SEC) and surfaces 4xx bodies.
    """
    try:
        import requests
    except Exception:
        return ("**Error:** Python package 'requests' is not installed. "
                "Add it to requirements.txt or install during the workflow.")

    ok   = (os.getenv("OPENAI_API_KEY") or "").strip()
    ob   = (os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").strip()
    om   = (os.getenv("OPENAI_MODEL") or "gpt-5-mini").strip()
    org  = (os.getenv("OPENAI_ORG") or "").strip()
    proj = (os.getenv("OPENAI_PROJECT") or "").strip()

    # Optional Perplexity fallback
    prefer_pplx = (os.getenv("PREFER_PPLX") or "0").strip().lower() in ("1","true","yes")
    pplx_key    = (os.getenv("PPLX_API_KEY") or os.getenv("PERPLEXITY_API_KEY") or "").strip()
    pplx_base   = (os.getenv("PPLX_BASE_URL") or "https://api.perplexity.ai").strip()
    pplx_model  = (os.getenv("PPLX_MODEL") or "llama-3.1-sonar-small-128k-chat").strip()
    if prefer_pplx and pplx_key:
        ok, ob, om = pplx_key, pplx_base, pplx_model
    if "perplexity.ai" in ob:
        ob = "https://api.perplexity.ai"

    temperature   = float((os.getenv("OPENAI_TEMPERATURE") or "0.2").strip())
    max_tokens    = int((os.getenv("OPENAI_MAX_TOKENS")  or "1200").strip())
    pre_delay_raw = os.getenv("OPENAI_PRE_DELAY_SEC")
    inter_delay   = float((os.getenv("OPENAI_INTER_CALL_DELAY_SEC") or "0.0").strip())
    max_attempts  = int((os.getenv("OPENAI_MAX_ATTEMPTS") or "6").strip())
    backoff_cap   = float((os.getenv("OPENAI_BACKOFF_CAP_SEC") or "45").strip())

    try:
        pre_delay = float(pre_delay_raw) if pre_delay_raw is not None else None
    except ValueError:
        pre_delay = None
    if pre_delay is None:
        time.sleep(random.uniform(0.8, 1.8))
    elif pre_delay > 0:
        time.sleep(pre_delay)

    print(f"[provider] base={ob} model={om} prefer_pplx={prefer_pplx}", flush=True)

    if not ok:
        return ("**Error:** No LLM credentials found. "
                "Set OPENAI_* (or PPLX_*) or LLM_API_* in GitHub Secrets.")

    headers = {"Authorization": f"Bearer {ok}", "Content-Type": "application/json"}
    if org:
        headers["OpenAI-Organization"] = org
    if proj:
        headers["OpenAI-Project"] = proj

    def _post_with_retries(url: str, payload: dict, timeout: int = 90):
        import requests
        last_exc = None
        for attempt in range(1, max_attempts + 1):
            try:
                if inter_delay > 0:
                    time.sleep(inter_delay)
                print(f"[http] POST {url} (attempt {attempt}/{max_attempts})", flush=True)
                resp = requests.post(url, headers=headers, json=payload, timeout=timeout)

                if resp.status_code in (429, 500, 502, 503, 504):
                    ra = resp.headers.get("retry-after")
                    try:
                        wait = float(ra) if ra else 0.0
                    except Exception:
                        wait = 0.0
                    if wait <= 0:
                        wait = min(2 ** attempt + random.random(), backoff_cap)
                    print(f"[rate-limit {resp.status_code}] wait {wait:.1f}s", flush=True)
                    time.sleep(wait)
                    last_exc = requests.HTTPError(f"{resp.status_code} {resp.reason}", response=resp)
                    continue

                if 400 <= resp.status_code < 500:
                    body = resp.text[:800] if resp.text else "<no body>"
                    raise requests.HTTPError(f"{resp.status_code} {resp.reason}: {body}", response=resp)

                resp.raise_for_status()
                return resp

            except requests.RequestException as e:
                wait = min(2 ** attempt + random.random(), backoff_cap)
                print(f"[network] {type(e).__name__}: {e}. wait {wait:.1f}s", flush=True)
                last_exc = e
                time.sleep(wait)

        if last_exc:
            raise last_exc
        raise RuntimeError("unexpected retry loop exit")

    # -------- Responses API helpers --------

    def _call_responses_plain_roles():
        url = ob.rstrip("/") + "/responses"
        payload = {
            "model": om,
            "input": [
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            "temperature":       temperature,
            "max_output_tokens": max_tokens,
        }
        r = _post_with_retries(url, payload)
        data = r.json()
        if "output_text" in data:
            return str(data["output_text"]).strip()
        if isinstance(data.get("content"), list):
            parts = []
            for c in data["content"]:
                if isinstance(c, dict):
                    if "text" in c:
                        parts.append(str(c["text"]))
                    elif c.get("type") in ("output_text", "text"):
                        parts.append(str(c.get("text", c.get("output_text", ""))))
            if parts:
                return "\n".join(parts).strip()
        return json.dumps(data)[:4000]

    def _call_responses_single_string():
        url = ob.rstrip("/") + "/responses"
        payload = {
            "model": om,
            "input": f"{system_msg}\n\n{user_msg}",
            "temperature":       temperature,
            "max_output_tokens": max_tokens,
        }
        r = _post_with_retries(url, payload)
        data = r.json()
        if "output_text" in data:
            return str(data["output_text"]).strip()
        return json.dumps(data)[:4000]

    def _call_chat_completions():
        url = ob.rstrip("/") + "/chat/completions"
        payload = {
            "model": om,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            "temperature": temperature,
            "max_tokens":  max_tokens,
        }
        r = _post_with_retries(url, payload)
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()

    if om.startswith("gpt-5"):
        try:
            return _call_responses_plain_roles()
        except Exception:
            return _call_responses_single_string()
    else:
        try:
            return _call_chat_completions()
        except Exception:
            return _call_responses_plain_roles()


# =========================
# Utility (cache, diff, chunking)
# =========================

def _now() -> datetime:
    return datetime.utcnow()

def _is_fresh(path: pathlib.Path) -> bool:
    if not path.exists():
        return False
    if CACHE_TTL_SEC <= 0:
        return True
    age = _now() - datetime.utcfromtimestamp(path.stat().st_mtime)
    return age.total_seconds() <= CACHE_TTL_SEC

def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)

def chunk_text(s: str, max_chars: int = CHARS_PER_CHUNK) -> List[str]:
    if not s:
        return ["(empty)"]
    return [s[i:i + max_chars] for i in range(0, len(s), max_chars)]

def file_diff(base: str, head: str, path: str) -> str:
    try:
        return run(f'git diff --unified=2 -- "{path}" {base} {head}')
    except Exception as e:
        return f"[diff unavailable for {path}: {e}]"

def cache_key(*parts: str) -> str:
    m = hashlib.sha256()
    for p in parts:
        m.update(p.encode("utf-8", errors="ignore"))
        m.update(b"\x00")
    return m.hexdigest()

def read_cache(key: str) -> str | None:
    p = CACHE_DIR / f"{key}.md"
    return p.read_text(encoding="utf-8") if _is_fresh(p) else None

def write_cache(key: str, text: str) -> None:
    (CACHE_DIR / f"{key}.md").write_text(text, encoding="utf-8")


# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--head", required=True)
    args = ap.parse_args()

    changed = git_changed_files(args.base, args.head).splitlines()
    changed = [p.strip() for p in changed if p.strip()]

    CODE_EXTS = {
        '.py', '.js', '.ts', '.tsx', '.java', '.go', '.rs', '.rb', '.php',
        '.c', '.cpp', '.cs', '.swift', '.kt', '.kts', '.scala', '.sh',
        '.ps1', '.yaml', '.yml', '.json', '.toml'
    }
    changed = [p for p in changed if any(p.lower().endswith(ext) for ext in CODE_EXTS)]

    if not changed:
        with open("ai_review_report.md", "w", encoding="utf-8") as f:
            f.write(
                "# 🤖 AI Code Review Report\n\n"
                f"No code changes detected between {args.base} → {args.head}.\n"
            )
        print("No code changes; wrote ai_review_report.md")
        return

    commit_msg = git_commit_message(args.head)
    docs = read_docs_snippets("docs", max_chars=400)

    banner = (
        "# 🤖 AI Code Review Report\n\n"
        f"**Base:** `{args.base}`  \n**Head:** `{args.head}`\n\n"
        f"**Files changed ({len(changed)}):**\n"
        + "\n".join(f"- {p}" for p in changed) + "\n\n---\n"
    )

    items: List[Dict[str, str]] = []
    for path in changed:
        diff_text = file_diff(args.base, args.head, path)
        if not diff_text.strip():
            diff_text = f"(no textual diff available for {path})"

        if len(diff_text) > MAX_DIFF_CHARS:
            diff_text = diff_text[:MAX_DIFF_CHARS] + "\n... [truncated]"

        chunks = chunk_text(diff_text, CHARS_PER_CHUNK)
        total = len(chunks)
        for idx, chunk in enumerate(chunks, 1):
            items.append({"path": path, "idx": str(idx), "total": str(total), "diff": chunk})

    model = (os.getenv("OPENAI_MODEL") or "gpt-5-mini").strip()
    temp  = (os.getenv("OPENAI_TEMPERATURE") or "0.2").strip()
    maxt  = (os.getenv("OPENAI_MAX_TOKENS") or "1200").strip()

    report_blocks: List[str] = []
    any_errors = False

    for i in range(0, len(items), LLM_BATCH_SIZE):
        batch = items[i:i + LLM_BATCH_SIZE]
        labels = [f"{b['path']}#{b['idx']}/{b['total']}" for b in batch]
        sys_msg, usr_msg = build_batch_prompt(commit_msg, docs, batch)

        key = cache_key(
            "v3-batch",
            model, temp, maxt,
            CACHE_BUSTER,
            *labels,
            usr_msg[:1200],
        )

        cached = read_cache(key)
        if cached:
            print(f"[cache] hit: batch {i // LLM_BATCH_SIZE + 1}")
            report_blocks.append(f"## Batch {i // LLM_BATCH_SIZE + 1}\n\n" + cached)
            continue

        approx_tok = sum(estimate_tokens(b["diff"]) for b in batch) + estimate_tokens(docs) + 300
        print(f"[review] batch {i // LLM_BATCH_SIZE + 1} "
              f"({len(batch)} sections, ≈{approx_tok} tok input)")

        try:
            review = call_llm(sys_msg, usr_msg)
            if not isinstance(review, str):
                review = json.dumps(review, indent=2)[:20000]
        except Exception as e:
            any_errors = True
            review = f"**Note:** LLM call failed: {e}\n(This entire batch was skipped.)"

        write_cache(key, review)
        report_blocks.append(f"## Batch {i // LLM_BATCH_SIZE + 1}\n\n" + review)

    with open("ai_review_report.md", "w", encoding="utf-8") as f:
        f.write(banner + "\n\n---\n\n".join(report_blocks) + "\n")
        if any_errors:
            f.write("\n\n---\n### Notice\nSome batches were rate-limited or failed. "
                    "This report includes retries, backoff, and caching. "
                    "Consider lowering LLM_BATCH_SIZE or tokens if this persists.\n")

    print("Wrote ai_review_report.md")


if __name__ == "__main__":
    main()
