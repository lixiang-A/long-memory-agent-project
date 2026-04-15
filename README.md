# Long Memory Agent Project

A lightweight research-style project on long-context memory for Transformer-based dialogue systems and AI agents.

## Overview

This project studies a practical question behind many LLM and agent systems:

**When the context window is limited, how can a model preserve long-term dialogue information more effectively?**

Instead of building a full agent stack, this repository focuses on one high-value module:

- recent context memory
- session summary memory
- retrieval memory
- time-aware reranking

That makes it suitable both as a course-paper prototype and as a portfolio project that can later grow into an agent memory system.

## Why This Project Matters

Long-running AI agents need more than strong single-turn generation. They also need to:

- preserve user goals across many turns
- recover important facts from old dialogue history
- balance global summaries with fine-grained evidence
- stay robust under limited context budgets

This repository is a small but focused implementation of that idea.

## Current Method

The current prototype compares three memory strategies:

1. `recent`
   Keeps only the latest dialogue turns.
2. `summary`
   Keeps recent turns plus session-level summaries.
3. `hybrid`
   Combines recent turns, session summaries, and retrieved history chunks with a simple time-aware reranking step.

## Implemented Components

- generic dialogue loader
- chunk-based lexical retrieval
- lightweight time-aware reranking
- memory bundle construction
- QA prompt assembly
- exact match and token F1 helpers
- runnable command-line demo

## Repository Layout

```text
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

## Quick Start

From the repository root:

```bash
PYTHONPATH=src python3 scripts/run_demo.py --strategy hybrid --question-id q1
```

Try the other memory settings:

```bash
PYTHONPATH=src python3 scripts/run_demo.py --strategy recent --question-id q2
PYTHONPATH=src python3 scripts/run_demo.py --strategy summary --question-id q3
```

## Example Output

The demo prints:

- the selected question
- the chosen memory strategy
- the assembled memory context
- the final QA prompt

This makes it easy to inspect how different memory strategies expose evidence to the model.

## Next Milestones

- adapt the loader to a real long-memory benchmark
- add batch evaluation across QA examples
- save experiment outputs to JSONL or CSV
- plug in a model provider for end-to-end answering
- add ablation experiments for summary, retrieval, and reranking

## Project Status

This repository is currently in the **course-project MVP** stage:

- the core structure is ready
- the demo pipeline runs
- the next step is to connect benchmark data and formal experiments

Later, this can be extended into a stronger agent-memory and RAG portfolio project.
