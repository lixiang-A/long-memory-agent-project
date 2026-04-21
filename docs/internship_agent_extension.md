# Internship-Oriented Agent Extension Plan

This document explains how the current long-memory course project can grow into an AI Agent internship portfolio project.

## 1. Positioning

The current project is not a complete agent platform yet.

It is currently a focused **agent memory module**:

- long dialogue state
- retrieval memory
- summary memory
- prompt context assembly
- basic evaluation

For a summer AI Agent internship, this is a good starting point, but we should extend it toward the current industry core:

- memory
- retrieval / RAG
- tool use
- workflow orchestration
- evaluation
- observability
- deployable API

We do not need to make every part deep at the beginning. But every part we include must be clear enough to explain in an interview.

## 2. Industry-Core Capability Map

Based on current agent tooling and documentation, a practical agent system usually includes the following modules.

### 2.1 Model Interface

What it does:

- calls an LLM
- handles messages and structured outputs
- manages token budget and model configuration

Interview explanation:

> The model interface isolates provider-specific APIs from the rest of the system. This makes it easier to swap models or compare different model settings.

### 2.2 Memory

What it does:

- stores short-term conversation state
- stores long-term user or task memory
- decides what to retrieve into the current context

Our current project already covers:

- recent memory
- summary memory
- retrieval memory
- time-aware retrieval

Interview explanation:

> I treat memory as a context-management problem. The agent cannot place everything into the context window, so the system must decide what to keep, compress, retrieve, and pass back to the model.

### 2.3 Retrieval / RAG

What it does:

- chunks documents or conversations
- indexes chunks
- retrieves relevant evidence
- grounds model generation

Our current project already covers:

- chunking
- BM25-style lexical retrieval
- top-k retrieval
- retrieved evidence assembly

Future upgrade:

- dense embedding retrieval
- hybrid retrieval
- reranking
- citation-style evidence output

Interview explanation:

> Retrieval is used as non-parametric memory. Instead of expecting the model to remember everything internally, we store historical evidence externally and retrieve it when needed.

### 2.4 Tool Use

What it does:

- lets the agent call functions
- connects the model to external operations
- supports search, calculators, file readers, APIs, databases, or custom tools

Future upgrade:

- add a simple `search_memory` tool
- add a `write_memory` tool
- add a `summarize_session` tool

Interview explanation:

> Tool use turns the model from a text generator into a controller. In this project, memory operations can be exposed as tools so the agent can decide when to search or update memory.

### 2.5 Workflow Orchestration

What it does:

- defines multi-step agent behavior
- controls transitions among retrieval, reasoning, tool calls, and final answer
- prevents the system from becoming a loose prompt chain

Future upgrade:

- build a simple workflow:
  1. classify question type
  2. retrieve memory
  3. decide whether evidence is enough
  4. answer with citations
  5. log result

Interview explanation:

> I would not rely on one long prompt for everything. I would split the agent into explicit steps so the system is easier to debug and evaluate.

### 2.6 Evaluation

What it does:

- measures whether the agent actually works
- separates retrieval errors from generation errors
- compares memory strategies

Our current project already covers:

- Exact Match
- token F1
- evidence hit intuition

Future upgrade:

- retrieval recall
- answer faithfulness
- latency and token cost
- per-question-type analysis

Interview explanation:

> For agent projects, evals are as important as prompts. I want to know whether failure comes from retrieval, memory compression, prompt design, or model reasoning.

### 2.7 Observability

What it does:

- logs what memory was retrieved
- logs prompts and responses
- tracks failure cases
- helps debugging

Future upgrade:

- save every run to JSONL
- save retrieved chunks
- save prompt length
- save model response
- save score and error type

Interview explanation:

> Without traces, agent behavior is hard to debug. I log the intermediate memory bundle and retrieval results so failures are inspectable.

### 2.8 Deployable API

What it does:

- exposes the project as a small backend service
- makes it usable by a front end or another agent system

Future upgrade:

- add FastAPI endpoint
- `POST /answer`
- `POST /memory/search`
- `POST /memory/update`

Interview explanation:

> The research prototype can be wrapped as an API so other systems can use it as a memory service.

## 3. Project Expansion Roadmap

### Stage 0: Current Course MVP

Goal:

- finish the course paper
- keep the project small and explainable

Must include:

- LoCoMo data adapter
- recent baseline
- summary baseline
- hybrid memory method
- basic metrics
- error analysis

### Stage 1: Internship-Ready Memory System

Goal:

- make the project interviewable

Add:

- batch evaluation script
- JSONL experiment logs
- retrieval hit rate
- README method diagram
- clearer examples

What you should be able to explain:

- why context windows fail
- why summary alone is not enough
- why retrieval alone is not enough
- how time-aware reranking works
- how to evaluate retrieval vs answer generation

### Stage 2: Agentic Memory Tools

Goal:

- connect memory to agent behavior

Add:

- `search_memory` tool
- `write_memory` tool
- `summarize_session` tool
- basic tool-call simulation

What you should be able to explain:

- why memory should be exposed as tools
- when the agent should retrieve
- when the agent should update memory
- how to avoid memory pollution

### Stage 3: Workflow Agent

Goal:

- show that this is not only a retrieval script

Add:

- question classifier
- retrieval decision node
- evidence check node
- answer generation node
- run logger

What you should be able to explain:

- why workflow beats one-shot prompting
- where state is stored
- where errors are logged
- how to debug a failed agent answer

### Stage 4: Demo API

Goal:

- make it look like a deployable engineering project

Add:

- FastAPI service
- `/answer` endpoint
- `/memory/search` endpoint
- simple request and response schema

What you should be able to explain:

- how the service receives a question
- how it retrieves memory
- how it constructs the prompt
- how it returns an answer with evidence

## 4. Interview Q&A Preparation

### Q1: What problem does this project solve?

It solves long-term dialogue memory under limited context windows. The system decides which historical information should be preserved, summarized, retrieved, and passed back to the model.

### Q2: Why not just use a long-context model?

Long-context models still face cost, latency, and distraction issues. Even if the full history fits, irrelevant or stale content can hurt answer quality. A memory system gives more explicit control.

### Q3: Why combine summary and retrieval?

Summary preserves global context but loses details. Retrieval preserves evidence but may miss the overall story. Combining them gives both global continuity and fine-grained grounding.

### Q4: What is your innovation?

For the course version, the innovation is a lightweight hybrid memory strategy with time-aware reranking. It is not a new foundation model, but it is a clear system-level improvement.

### Q5: How do you evaluate it?

I compare recent-only, summary-only, and hybrid memory strategies using long-dialogue QA. I track answer quality with EM/F1 and inspect retrieval evidence to identify whether failures come from retrieval, compression, or generation.

### Q6: How would you extend it into a real agent?

I would expose memory operations as tools, add a workflow layer for retrieve-read-answer-log, store traces for observability, and wrap the system in an API.

## 5. What We Must Avoid

For internship preparation, the project must not become only a paper implementation.

Avoid:

- only showing a prompt
- only showing a demo conversation
- no metrics
- no failure analysis
- no clear module boundaries
- no explanation of why each module exists

We want the project to show:

- engineering structure
- agent thinking
- evaluation awareness
- research motivation

## 6. Target Resume Bullet

Possible resume bullet after Stage 1:

> Built a long-context memory prototype for LLM agents, combining recent-window context, session summaries, BM25-style retrieval, and time-aware reranking for long-dialogue QA; implemented modular memory builders, prompt assembly, and evaluation utilities for comparing memory strategies.

Possible resume bullet after Stage 3:

> Extended the prototype into an agentic memory workflow with memory search/update tools, structured run logs, and retrieval/answer evaluation, enabling analysis of context-window failures, summary compression loss, and temporal retrieval errors.

## 7. Current Priority

For the next two weeks, the priority remains:

1. course submission completeness
2. real dataset adapter
3. baseline and hybrid memory evaluation
4. paper writing

After the course deadline, we upgrade the same project toward the internship-ready agent system.

