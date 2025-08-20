\# AI-Enhanced CI/CD Code Reviewer + Git Insights



This repo demonstrates an AI-augmented code review pipeline (GitHub Actions + LLM) and a small \*\*Git Insights\*\* tool:



\- \*\*AI Review:\*\* On each PR/push, Actions runs `src/reviewer.py`, diffs changes, uses repo docs (RAG-lite), and generates a structured review (`ai\_review\_report.md`) + PR comment.

\- \*\*Git Insights:\*\* Standard-library Python tool that summarizes authors, files, and commit message themes.



