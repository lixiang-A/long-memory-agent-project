# Long Memory Agent Project

This is the course-version project scaffold for a long-context memory system built around three memory sources:

- recent window
- session summary
- retrieval memory

The goal is to support a paper-friendly experiment loop:

1. load multi-session dialogue data
2. build memory context with different strategies
3. assemble prompts for QA
4. compare baselines and the hybrid memory design

## Current scope

Implemented in this scaffold:

- generic dialogue loader
- lexical retriever
- simple time-aware reranking
- recent / summary / hybrid memory builders
- prompt assembly
- demo script
- exact match and token F1 helpers

Not implemented yet:

- remote LLM API call
- large-scale batch experiment runner
- dataset-specific adapters for every benchmark
- plotting

## Project layout

```text
long_memory_agent_project/
  data/demo/sample_dialogue.json
  scripts/run_demo.py
  src/long_memory_agent/
    loader.py
    memory.py
    metrics.py
    models.py
    prompting.py
    retrieval.py
```

## Quick start

From the workspace root:

```bash
PYTHONPATH=long_memory_agent_project/src python3 long_memory_agent_project/scripts/run_demo.py --strategy hybrid --question-id q1
```

Try other strategies:

```bash
PYTHONPATH=long_memory_agent_project/src python3 long_memory_agent_project/scripts/run_demo.py --strategy recent
PYTHONPATH=long_memory_agent_project/src python3 long_memory_agent_project/scripts/run_demo.py --strategy summary
```

## Next steps

- replace the demo file with a benchmark sample
- add a provider wrapper for model inference
- write experiment outputs to JSONL
- add ablations

## GitHub publishing

This folder is designed to work as an independent repository.

Suggested local workflow:

```bash
cd /Users/liz/Desktop/统计软件计算课程资料/long_memory_agent_project
git init -b main
git add .
git commit -m "Initial project scaffold"
```

Then create an empty GitHub repository and connect it:

```bash
git remote add origin git@github.com:<your-username>/long-memory-agent-project.git
git push -u origin main
```
