import os
import subprocess
import argparse
import textwrap
import json
import glob


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


def read_docs_snippets(doc_dir: str = "docs", max_chars: int = 4000) -> str:
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
    import requests, json, os, time, random

    def post_with_retries(url, headers, payload, timeout=60, max_attempts=4):
        for attempt in range(1, max_attempts + 1):
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            # if not a rate-limit, raise or return immediately
            if resp.status_code != 429:
                resp.raise_for_status()
                return resp
            # backoff on 429
            ra = resp.headers.get("retry-after")
            try:
                wait = int(ra) if ra else 0
            except Exception:
                wait = 0
            if wait <= 0:
                wait = min(2 ** attempt + random.random(), 15)  # jitter
            time.sleep(wait)
        # last try result:
        resp.raise_for_status()
        return resp

    ok = (os.getenv("OPENAI_API_KEY") or "").strip()
    ob = (os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").strip()
    om = (os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip()
    org = (os.getenv("OPENAI_ORG") or "").strip()
    proj = (os.getenv("OPENAI_PROJECT") or "").strip()

    gk = (os.getenv("LLM_API_KEY") or "").strip()
    gb = (os.getenv("LLM_API_BASE") or "").strip()

    temperature = float((os.getenv("OPENAI_TEMPERATURE") or "0.2").strip())
    max_tokens  = int((os.getenv("OPENAI_MAX_TOKENS")  or "800").strip())  # slightly lower to reduce usage

    if ok:
        headers = {"Authorization": f"Bearer {ok}", "Content-Type": "application/json"}
        if org:  headers["OpenAI-Organization"] = org
        if proj: headers["OpenAI-Project"] = proj

        # 1) Try Chat Completions with retry-on-429
        try:
            url = ob.rstrip("/") + "/chat/completions"
            payload = {
                "model": om,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user",   "content": user_msg},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            r = post_with_retries(url, headers, payload)
            data = r.json()
            return data["choices"][0]["message"]["content"].strip()
        except requests.HTTPError as e:
            # fall back to /responses only if chat endpoint itself is unsupported
            txt = getattr(e.response, "text", "") or ""
            if "Unrecognized request URL" not in txt and e.response.status_code != 404:
                raise
        except Exception:
            pass  # try /responses below

        # 2) Responses API attempt A (structured)
        try:
            url = ob.rstrip("/") + "/responses"
            payload = {
                "model": om,
                "input": [
                    {"role": "system", "content": [{"type": "text", "text": system_msg}]},
                    {"role": "user",   "content": [{"type": "text", "text": user_msg}]},
                ],
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            r = post_with_retries(url, headers, payload)
            data = r.json()
            if "output_text" in data:
                return str(data["output_text"]).strip()
            if isinstance(data.get("content"), list):
                parts = []
                for c in data["content"]:
                    if isinstance(c, dict) and c.get("type") in ("output_text", "text"):
                        parts.append(str(c.get("text", c.get("output_text", ""))))
                if parts:
                    return "\n".join(parts).strip()
        except Exception:
            pass

        # 3) Responses API attempt B (simple string)
        url = ob.rstrip("/") + "/responses"
        payload = {
            "model": om,
            "input": f"{system_msg}\n\n{user_msg}",
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        r = post_with_retries(url, headers, payload)
        data = r.json()
        if "output_text" in data:
            return str(data["output_text"]).strip()
        return json.dumps(data)[:4000]

    # Generic non-OpenAI provider
    if gk and gb:
        headers = {"Authorization": f"Bearer {gk}", "Content-Type": "application/json"}
        payload = {"prompt": f"{system_msg}\n\n{user_msg}", "max_tokens": max_tokens, "temperature": temperature}
        r = post_with_retries(gb.rstrip("/"), headers, payload)
        data = r.json()
        for key in ("text", "output", "completion", "result"):
            if key in data:
                return str(data[key]).strip()
        return json.dumps(data)[:4000]

    raise RuntimeError("No LLM credentials found. Set OPENAI_* or LLM_API_* in GitHub Secrets.")
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--head", required=True)
    args = ap.parse_args()

    diff = git_diff(args.base, args.head)[:20000]
    if not diff.strip():
        with open("ai_review_report.md", "w", encoding="utf-8") as f:
            f.write(
                f"# 🤖 AI Code Review Report\n\n"
                f"No code changes detected between {args.base} → {args.head}.\n"
            )
        print("No code changes; wrote ai_review_report.md")
        return

    commit_msg = git_commit_message(args.head)
    docs = read_docs_snippets("docs")
    files = git_changed_files(args.base, args.head)
    files_block = "\n".join(f"- {p}" for p in files.splitlines() if p.strip())

    sys_msg, user_msg = build_prompt(commit_msg, diff, docs, files_block)

    # Keep CI green even if the model call fails
    try:
        review = call_llm(sys_msg, user_msg)
    except Exception as e:
        review = (
            f"**Note:** LLM call failed in CI: {e}\n\n"
            "Proceeding with a static stub so the pipeline stays green."
        )

    banner = (
        "# 🤖 AI Code Review Report\n\n"
        f"**Base:** `{args.base}`  \n**Head:** `{args.head}`\n\n---\n\n"
    )
    with open("ai_review_report.md", "w", encoding="utf-8") as f:
        f.write(banner + review + "\n")
    print("Wrote ai_review_report.md")


if __name__ == "__main__":
    main()
