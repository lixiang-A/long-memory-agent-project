import json
from pathlib import Path

from long_memory_agent.models import DialogueSample, DialogueTurn, QuestionItem


def _get_text(item):
    return item.get("text") or item.get("utterance") or item.get("content") or ""


def load_dialogue_sample(path: str) -> DialogueSample:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    turns = []
    for idx, item in enumerate(raw.get("turns", []), start=1):
        turns.append(
            DialogueTurn(
                turn_id=int(item.get("turn_id", idx)),
                session_id=int(item.get("session_id", 1)),
                speaker=str(item.get("speaker", "unknown")),
                text=_get_text(item).strip(),
            )
        )

    questions = []
    for idx, item in enumerate(raw.get("questions", []), start=1):
        questions.append(
            QuestionItem(
                question_id=str(item.get("question_id", f"q{idx}")),
                question=str(item.get("question", "")).strip(),
                answer=str(item.get("answer", "")).strip(),
                support_turn_ids=list(item.get("support_turn_ids", [])),
            )
        )

    return DialogueSample(
        sample_id=str(raw.get("sample_id", Path(path).stem)),
        title=str(raw.get("title", "")),
        session_summaries=list(raw.get("session_summaries", [])),
        turns=turns,
        questions=questions,
    )
