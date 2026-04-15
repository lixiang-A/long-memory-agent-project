from typing import List

from long_memory_agent.models import DialogueSample, DialogueTurn, MemoryBundle
from long_memory_agent.retrieval import LexicalRetriever, chunk_turns, rerank_with_time_awareness


def get_recent_turns(turns: List[DialogueTurn], window_size: int = 4) -> List[DialogueTurn]:
    return turns[-window_size:]


def build_recent_memory(sample: DialogueSample, window_size: int = 4) -> MemoryBundle:
    return MemoryBundle(
        strategy="recent",
        recent_turns=get_recent_turns(sample.turns, window_size=window_size),
        session_summaries=[],
        retrieved_chunks=[],
    )


def build_summary_memory(sample: DialogueSample, window_size: int = 4, max_summaries: int = 3) -> MemoryBundle:
    return MemoryBundle(
        strategy="summary",
        recent_turns=get_recent_turns(sample.turns, window_size=window_size),
        session_summaries=sample.session_summaries[-max_summaries:],
        retrieved_chunks=[],
    )


def build_hybrid_memory(
    sample: DialogueSample,
    question: str,
    window_size: int = 4,
    chunk_size: int = 3,
    overlap: int = 1,
    top_k: int = 3,
    max_summaries: int = 3,
) -> MemoryBundle:
    chunks = chunk_turns(sample.turns, chunk_size=chunk_size, overlap=overlap)
    retriever = LexicalRetriever(chunks)
    retrieved = retriever.retrieve(question, top_k=top_k)
    reranked = rerank_with_time_awareness(question, retrieved, max_turn_id=sample.turns[-1].turn_id)

    return MemoryBundle(
        strategy="hybrid",
        recent_turns=get_recent_turns(sample.turns, window_size=window_size),
        session_summaries=sample.session_summaries[-max_summaries:],
        retrieved_chunks=reranked,
    )
