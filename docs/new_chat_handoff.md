# New Chat Handoff

Copy this into a new chat when continuing the project:

> We are building an AI Agent internship-oriented project named **Long-Memory Agent**. The course version targets long-term dialogue memory on the LoCoMo dataset. The core algorithms are recent window, summary memory, BM25-style retrieval, hybrid memory, and time-aware reranking. The planned metrics are Exact Match, token-level F1, retrieval hit, and error analysis. After the course deadline, we will extend it into an Agentic RAG system with `search_memory`, `write_memory`, `summarize_session` tools, workflow orchestration, JSONL run logs, FastAPI service, and GitHub project polish. The repository is `long_memory_agent_project`, and the public GitHub URL is https://github.com/lixiang-A/long-memory-agent-project.

## Current Repository

- Local path: `/Users/liz/Desktop/统计软件计算课程资料/long_memory_agent_project`
- GitHub: https://github.com/lixiang-A/long-memory-agent-project
- Main branch: `main`

## Current Code

- `src/long_memory_agent/loader.py`: dialogue sample loader
- `src/long_memory_agent/retrieval.py`: chunking, lexical retrieval, time-aware reranking
- `src/long_memory_agent/memory.py`: recent, summary, and hybrid memory builders
- `src/long_memory_agent/prompting.py`: QA prompt construction
- `src/long_memory_agent/metrics.py`: EM and token-F1 helpers
- `scripts/run_demo.py`: runnable demo

## Current Command

```bash
cd /Users/liz/Desktop/统计软件计算课程资料/long_memory_agent_project
PYTHONPATH=src python3 scripts/run_demo.py --strategy hybrid --question-id q1
```

## Course Requirements

The course requires:

1. data URL
2. dataset
3. four references
4. algorithms

We have documented these in:

- `docs/course_submission_plan.md`

## Internship-Oriented Plan

The agent internship roadmap is documented in:

- `docs/internship_agent_extension.md`

## Next Implementation Steps

1. Download or adapt LoCoMo `locomo10.json`.
2. Add a LoCoMo adapter.
3. Add a batch evaluation runner.
4. Run recent / summary / hybrid strategies.
5. Save experiment results to JSONL or CSV.
6. Produce the first result table.
7. Start drafting the method and experiment sections of the course paper.

