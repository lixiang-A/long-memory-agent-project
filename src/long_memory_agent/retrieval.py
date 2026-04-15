import math
import re
from collections import Counter
from typing import Iterable, List

from long_memory_agent.models import DialogueChunk, DialogueTurn


TOKEN_RE = re.compile(r"[A-Za-z0-9_']+")
EARLY_HINTS = ("before", "earlier", "first", "initially", "at the beginning", "之前", "最先")
LATE_HINTS = ("after", "later", "recent", "latest", "finally", "last", "之后", "后来", "最后", "最近")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def chunk_turns(turns: Iterable[DialogueTurn], chunk_size: int = 3, overlap: int = 1) -> List[DialogueChunk]:
    turns = list(turns)
    chunks = []
    step = max(1, chunk_size - overlap)
    chunk_index = 1

    for start in range(0, len(turns), step):
        window = turns[start:start + chunk_size]
        if not window:
            continue
        chunk_text = "\n".join(f"{turn.speaker}: {turn.text}" for turn in window)
        chunks.append(
            DialogueChunk(
                chunk_id=f"chunk-{chunk_index}",
                start_turn_id=window[0].turn_id,
                end_turn_id=window[-1].turn_id,
                session_id=window[0].session_id,
                text=chunk_text,
            )
        )
        chunk_index += 1
        if start + chunk_size >= len(turns):
            break

    return chunks


class LexicalRetriever:
    def __init__(self, chunks: List[DialogueChunk]):
        self.chunks = chunks
        self.term_frequencies = []
        self.document_frequencies = Counter()
        self.avg_doc_len = 0.0

        total_len = 0
        for chunk in chunks:
            tokens = tokenize(chunk.text)
            total_len += len(tokens)
            tf = Counter(tokens)
            self.term_frequencies.append(tf)
            for term in tf.keys():
                self.document_frequencies[term] += 1

        if chunks:
            self.avg_doc_len = total_len / len(chunks)

    def score(self, query: str, index: int, k1: float = 1.5, b: float = 0.75) -> float:
        tokens = tokenize(query)
        tf = self.term_frequencies[index]
        doc_len = sum(tf.values()) or 1
        score = 0.0
        total_docs = len(self.chunks) or 1

        for token in tokens:
            if token not in tf:
                continue
            df = self.document_frequencies[token]
            idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)
            freq = tf[token]
            numerator = freq * (k1 + 1.0)
            denominator = freq + k1 * (1.0 - b + b * doc_len / (self.avg_doc_len or 1.0))
            score += idf * numerator / denominator

        return score

    def retrieve(self, query: str, top_k: int = 3) -> List[DialogueChunk]:
        scored = []
        for index, chunk in enumerate(self.chunks):
            candidate = DialogueChunk(
                chunk_id=chunk.chunk_id,
                start_turn_id=chunk.start_turn_id,
                end_turn_id=chunk.end_turn_id,
                session_id=chunk.session_id,
                text=chunk.text,
                base_score=self.score(query, index),
                reranked_score=self.score(query, index),
            )
            scored.append(candidate)

        scored.sort(key=lambda item: item.base_score, reverse=True)
        return scored[:top_k]


def rerank_with_time_awareness(query: str, chunks: List[DialogueChunk], max_turn_id: int, alpha: float = 0.15) -> List[DialogueChunk]:
    query_lower = query.lower()
    prefers_early = any(hint in query_lower for hint in EARLY_HINTS)
    prefers_late = any(hint in query_lower for hint in LATE_HINTS)

    reranked = []
    for chunk in chunks:
        midpoint = (chunk.start_turn_id + chunk.end_turn_id) / 2.0
        normalized_position = midpoint / max(max_turn_id, 1)

        if prefers_early:
            position_bonus = 1.0 - normalized_position
        elif prefers_late:
            position_bonus = normalized_position
        else:
            position_bonus = 0.5

        chunk.reranked_score = chunk.base_score + alpha * position_bonus
        reranked.append(chunk)

    reranked.sort(key=lambda item: item.reranked_score, reverse=True)
    return reranked
