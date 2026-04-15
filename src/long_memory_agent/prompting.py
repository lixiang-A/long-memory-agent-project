from long_memory_agent.models import MemoryBundle


DEFAULT_SYSTEM = (
    "You are answering questions about a long multi-session dialogue. "
    "Use only the provided memory context. If the memory is insufficient, "
    "say that the evidence is insufficient."
)


def build_qa_prompt(question: str, memory: MemoryBundle) -> str:
    return (
        f"System instruction:\n{DEFAULT_SYSTEM}\n\n"
        f"Memory context:\n{memory.render()}\n\n"
        f"Question:\n{question}\n\n"
        "Answer with a concise grounded response."
    )
