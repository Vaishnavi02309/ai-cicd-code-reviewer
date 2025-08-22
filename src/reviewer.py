# src/reviewer.py
import os, subprocess, argparse, textwrap
from google import genai
from google.genai import types

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

PROMPT = """You are a concise senior code reviewer.
Review the provided unified git diff for correctness, edge cases, security, readability, and tests.
Return short bullet points and, if useful, a tiny patch snippet. Keep it focused.
"""

def run(cmd: str) -> str:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(r.stderr.strip() or f"failed: {cmd}")
    return r.stdout

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--path", default="src/calculator.py")
    args = ap.parse_args()

    # Only diff the selected file
    diff = run(f'git diff --unified=2 {args.base} {args.head} -- "{args.path}"').strip()
    if not diff:
        open("ai_review_report.md","w",encoding="utf-8").write(
            "# AI Code Review\n\nNo changes found in the selected file.\n"
        )
        print("No diff; wrote ai_review_report.md")
        return

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set GOOGLE_API_KEY (or GEMINI_API_KEY) for Gemini access.")

    client = genai.Client(api_key=api_key)

    prompt = f"{PROMPT}\n\nFILE: {args.path}\n\n```diff\n{diff}\n```"
    resp = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=800,
            thinking_config=types.ThinkingConfig(thinking_budget=0)  # disable “thinking”
        ),
    )

    text = (getattr(resp, "text", None) or "").strip() or str(resp)
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
