# Course Submission Plan

This document answers the four required items for the course assignment:

1. data URL
2. dataset
3. four references
4. algorithms

## 1. Data URL

Primary dataset repository:

- https://github.com/snap-research/locomo

Primary dataset file:

- https://github.com/snap-research/locomo/blob/main/data/locomo10.json

Raw JSON file:

- https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json

## 2. Dataset

The course project will use **LoCoMo**, a long-term conversational memory benchmark released with the ACL 2024 paper **Evaluating Very Long-Term Conversational Memory of LLM Agents**.

LoCoMo is suitable for this project because it is directly designed to evaluate memory in long-running dialogue agents.

Important dataset properties:

- The official repository releases `locomo10.json`.
- The dataset contains 10 long conversations.
- Each conversation spans multiple sessions.
- Each conversation is about 300 turns and 9K tokens on average.
- The conversations can span up to 35 sessions.
- Each sample includes long dialogue content, session summaries, observations, event summaries, QA annotations, and evidence fields when available.

The course version will focus on the QA task:

- input: long multi-session conversation
- question: memory-related query about the conversation
- output: answer grounded in dialogue history
- evidence: dialogue ids or turns containing supporting information

## 3. Four References

### Reference 1

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., and Polosukhin, I. (2017). **Attention Is All You Need**.

URL: https://arxiv.org/abs/1706.03762

Why it is used:

- This is the foundational Transformer paper.
- Our project is built around the limitation of Transformer-style context windows and the need for external memory.

### Reference 2

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Kuttler, H., Lewis, M., Yih, W., Rocktaschel, T., Riedel, S., and Kiela, D. (2020). **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**.

URL: https://arxiv.org/abs/2005.11401

Why it is used:

- This paper provides the retrieval-augmented generation foundation.
- Our retrieval memory module follows the same general idea: combine parametric model ability with external non-parametric memory.

### Reference 3

Maharana, A., Lee, D.-H., Tulyakov, S., Bansal, M., Barbieri, F., and Fang, Y. (2024). **Evaluating Very Long-Term Conversational Memory of LLM Agents**.

URL: https://arxiv.org/abs/2402.17753

Code and data:

- https://github.com/snap-research/locomo

Why it is used:

- This paper introduces LoCoMo, the primary dataset for this course project.
- It directly studies long-term conversational memory of LLM agents.

### Reference 4

Packer, C., Wooders, S., Lin, K., Fang, V., Patil, S. G., Stoica, I., and Gonzalez, J. E. (2023). **MemGPT: Towards LLMs as Operating Systems**.

URL: https://arxiv.org/abs/2310.08560

Why it is used:

- This paper frames long-context limitations as a memory-management problem.
- It supports our project motivation: an agent needs explicit memory tiers rather than relying only on the current context window.

## 4. Algorithms

The project will compare several memory strategies for long-dialogue QA.

### 4.1 Baseline 1: Recent Window

Only the latest turns are kept as context.

Purpose:

- Simulates the simplest short-term memory strategy.
- Shows what happens when early but important information is truncated.

Algorithm steps:

1. Load a long conversation.
2. Keep the latest `k` dialogue turns.
3. Build a QA prompt using only these recent turns.
4. Generate or inspect the answer.

### 4.2 Baseline 2: Summary Memory

Recent turns are combined with session-level summaries.

Purpose:

- Tests whether global summaries can preserve long-term information.
- Provides a stronger baseline than simple truncation.

Algorithm steps:

1. Load conversation and session summaries.
2. Keep the latest `k` turns.
3. Add the latest or all session summaries.
4. Build the QA prompt from summaries and recent turns.

### 4.3 Proposed Method: Hybrid Memory

Hybrid memory combines:

- recent window
- session summaries
- retrieval memory
- time-aware reranking

Purpose:

- Preserve local continuity through recent turns.
- Preserve global context through summaries.
- Recover fine-grained evidence through retrieval.
- Improve temporal questions through time-aware reranking.

Algorithm steps:

1. Load the conversation.
2. Split dialogue history into overlapping chunks.
3. Score chunks against the question using lexical retrieval.
4. Select top-k candidate chunks.
5. Apply time-aware reranking when the question contains temporal signals.
6. Build the memory context from summaries, retrieved chunks, and recent turns.
7. Build the final QA prompt.
8. Evaluate answer quality and retrieval evidence.

### 4.4 Retrieval Algorithm

The current implementation uses a lightweight BM25-style lexical retriever.

Inputs:

- query: current question
- documents: dialogue chunks

Output:

- top-k relevant dialogue chunks

Why this is suitable for the course version:

- It is simple and explainable.
- It does not require model training.
- It provides a strong enough baseline for a two-week course project.

### 4.5 Time-Aware Reranking

The reranker adjusts retrieval scores using temporal hints in the question.

Examples:

- early hints: `before`, `earlier`, `first`, `at the beginning`
- late hints: `after`, `later`, `recent`, `finally`, `last`

If the question asks about early events, earlier chunks receive a small bonus.

If the question asks about later events, later chunks receive a small bonus.

### 4.6 Evaluation Metrics

The initial evaluation will use:

- Exact Match
- token-level F1
- retrieval hit rate when evidence labels are available

The course paper will report:

- comparison across memory strategies
- examples where each strategy succeeds or fails
- error analysis for retrieval failure, summary loss, and temporal confusion

## Course Project Title

Recommended title:

**面向长期对话记忆的 Transformer 外部记忆增强方法研究**

English title:

**External Memory-Augmented Transformer for Long-Term Dialogue Memory**

