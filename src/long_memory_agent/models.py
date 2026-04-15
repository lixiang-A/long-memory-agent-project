from dataclasses import dataclass, field
from typing import List


@dataclass
class DialogueTurn:
    turn_id: int
    session_id: int
    speaker: str
    text: str


@dataclass
class QuestionItem:
    question_id: str
    question: str
    answer: str
    support_turn_ids: List[int] = field(default_factory=list)


@dataclass
class DialogueChunk:
    chunk_id: str
    start_turn_id: int
    end_turn_id: int
    session_id: int
    text: str
    base_score: float = 0.0
    reranked_score: float = 0.0


@dataclass
class DialogueSample:
    sample_id: str
    title: str
    session_summaries: List[str]
    turns: List[DialogueTurn]
    questions: List[QuestionItem]


@dataclass
class MemoryBundle:
    strategy: str
    recent_turns: List[DialogueTurn]
    session_summaries: List[str]
    retrieved_chunks: List[DialogueChunk]

    def render(self) -> str:
        parts = [f"Strategy: {self.strategy}"]

        if self.session_summaries:
            summary_lines = [f"- {summary}" for summary in self.session_summaries]
            parts.append("Session summaries:\n" + "\n".join(summary_lines))

        if self.retrieved_chunks:
            chunk_lines = []
            for chunk in self.retrieved_chunks:
                header = (
                    f"- {chunk.chunk_id} "
                    f"(turns {chunk.start_turn_id}-{chunk.end_turn_id}, "
                    f"score={chunk.reranked_score:.3f})"
                )
                chunk_lines.append(header + "\n" + chunk.text)
            parts.append("Retrieved memory:\n" + "\n\n".join(chunk_lines))

        if self.recent_turns:
            turn_lines = [
                f"{turn.speaker}: {turn.text}"
                for turn in self.recent_turns
            ]
            parts.append("Recent dialogue window:\n" + "\n".join(turn_lines))

        return "\n\n".join(parts)
