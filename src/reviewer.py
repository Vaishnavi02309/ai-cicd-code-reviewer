import os, subprocess, argparse, textwrap, json, glob

def run(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode: raise RuntimeError(f'Command failed: {cmd}\n{r.stderr}')
    return r.stdout.strip()

def git_diff(base, head): return run(f'git diff --unified=2 {base} {head}')
def git_commit_message(ref): return run(f'git log -1 --pretty=%B {ref}')
def git_changed_files(base, head): return run(f'git diff --name-only {base} {head}')

def read_docs_snippets(doc_dir='docs', max_chars=4000):
    buf=[]
    for path in glob.glob(f'{doc_dir}/**/*', recursive=True):
        if any(path.endswith(ext) for ext in ['.md','.txt','.rst']):
            try:
                with open(path,'r',encoding='utf-8',errors='ignore') as f:
                    buf.append(f"\n\n---\n# {path}\n"+f.read())
            except: pass
    return ('\n'.join(buf))[:max_chars] or '(no docs found)'

def build_prompt(msg, diff, docs, files_block):
    sys_p = ("You are a precise senior code reviewer. Return actionable, concise findings with sections: "
             "Summary, Risks, Style/Docs, Correctness, Security, Suggested Patches. Reference the style guide.")
    user_p = textwrap.dedent(f'''
    ## Repository Context (docs)
    {docs}

    ## Files Changed
    {files_block}

    ## Commit Message
    {msg}

    ## Diff
    {diff}

    ## Review Goals
    - Find logic errors, edge cases, unsafe patterns.
    - Enforce style guide (docstrings, type hints, PEP8).
    - Suggest concrete fixes with small code blocks.
    ''')
    return sys_p, user_p

def call_llm(system_msg, user_msg):
    import requests
    # OpenAI-compatible
    ok = os.getenv('OPENAI_API_KEY')
    ob = os.getenv('OPENAI_BASE_URL','https://api.openai.com/v1')
    om = os.getenv('OPENAI_MODEL','gpt-4o-mini')
    # Generic fallback
    gk = os.getenv('LLM_API_KEY')
    gb = os.getenv('LLM_API_BASE')

    temperature = float(os.getenv('OPENAI_TEMPERATURE', '0.2'))
    max_tokens = int(os.getenv('OPENAI_MAX_TOKENS', '1000'))

    if ok:
        url = ob.rstrip('/')+'/chat/completions'
        headers = {'Authorization': f'Bearer {ok}','Content-Type':'application/json'}
        payload = {'model': om,'messages':[{'role':'system','content':system_msg},
                                           {'role':'user','content':user_msg}],
                   'temperature':temperature,'max_tokens':max_tokens}
        r = requests.post(url, headers=headers, json=payload, timeout=60); r.raise_for_status()
        return r.json()['choices'][0]['message']['content'].strip()

    if gk and gb:
        headers = {'Authorization': f'Bearer {gk}','Content-Type':'application/json'}
        payload = {'prompt': f'{system_msg}\n\n{user_msg}','max_tokens':max_tokens,'temperature':temperature}
        r = requests.post(gb.rstrip('/'), headers=headers, json=payload, timeout=60); r.raise_for_status()
        data = r.json()
        for k in ['text','output','completion','result']:
            if k in data: return str(data[k]).strip()
        return json.dumps(data)[:4000]

    raise RuntimeError('No LLM credentials found. Set OPENAI_* or LLM_API_* in GitHub Secrets.')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--head', required=True)
    a = ap.parse_args()

    diff = git_diff(a.base, a.head)[:20000]
    if not diff.strip():
        with open('ai_review_report.md','w',encoding='utf-8') as f:
            f.write(f"# 🤖 AI Code Review Report\n\nNo code changes detected between {a.base} → {a.head}.\n")
        print("No code changes; wrote ai_review_report.md")
        return

    msg  = git_commit_message(a.head)
    docs = read_docs_snippets('docs')
    files = git_changed_files(a.base, a.head)
    files_block = "\n".join(f"- {p}" for p in files.splitlines() if p.strip())

    s,u  = build_prompt(msg, diff, docs, files_block)
    review = call_llm(s,u)

    with open('ai_review_report.md','w',encoding='utf-8') as f:
        f.write(f"# 🤖 AI Code Review Report\n\n**Base:** `{a.base}`  \n**Head:** `{a.head}`\n\n---\n\n{review}\n")
    print("Wrote ai_review_report.md")

if __name__=='__main__': main()
