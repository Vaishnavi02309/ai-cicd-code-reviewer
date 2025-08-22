import os, subprocess, argparse, json, textwrap, requests

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  # your pick
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"

PROMPT = """You are a concise senior code reviewer.
Review the provided unified diff for correctness, edge cases, security, readability, and tests.
Return short bullet points and, if useful, a tiny patch snippet.
Keep it focused; no fluff.
"""

def run(cmd: str) -> str:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr.strip() or f"failed: {cmd}")
    return r.stdout

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--path", default="src/calculator.py")
    args = ap.parse_args()

    # Only diff the single file
    diff = run(f'git diff --unified=2 {args.base} {args.head} -- "{args.path}"').strip()
    if not diff:
        open("ai_review_report.md","w",encoding="utf-8").write(
            "# AI Code Review\n\nNo changes found in the selected file.\n"
        )
        print("No diff; wrote ai_review_report.md")
        return

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is not set")

    prompt = f"{PROMPT}\n\nFILE: {args.path}\n\n```diff\n{diff}\n```"

    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 800
        }
    }

    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
    r = requests.post(GEMINI_URL, headers=headers, data=json.dumps(payload), timeout=120)
    r.raise_for_status()
    data = r.json()

    # Extract the text
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        text = json.dumps(data, indent=2)[:4000]

    report = textwrap.dedent(f"""\
    # AI Code Review

    **Model:** {MODEL}  
    **File:** {args.path}

    ---
    {text}
    """)

    open("ai_review_report.md","w",encoding="utf-8").write(report)
    print("Wrote ai_review_report.md")

if __name__ == "__main__":
    main()
