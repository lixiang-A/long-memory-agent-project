# AGENTS.md

## Project purpose

This repository contains a lightweight research-style project on long-context memory for Transformer-based dialogue systems and AI agents.

The course-version goal is:

- build a clean long-memory prototype
- compare memory strategies
- support a course paper with reproducible code

The follow-up goal is:

- polish the repository into a stronger GitHub portfolio project
- extend it toward agent memory and retrieval systems

## Recommended working rules

- Keep the scope focused on memory, retrieval, prompting, and evaluation.
- Prefer simple baselines before adding new methods.
- Do not introduce heavyweight infrastructure unless it directly helps the paper.
- Keep all experiments reproducible from command-line scripts.

## Repository layout

- `src/long_memory_agent/`: core package
- `scripts/`: runnable entry points
- `data/demo/`: small local examples

## Current command

```bash
PYTHONPATH=src python3 scripts/run_demo.py --strategy hybrid --question-id q1
```

## Near-term roadmap

1. Add a real benchmark adapter.
2. Add batch evaluation.
3. Add model-provider wrapper.
4. Save experiment outputs for the paper.
