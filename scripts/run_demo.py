import argparse
from pathlib import Path

from long_memory_agent.loader import load_dialogue_sample
from long_memory_agent.memory import build_hybrid_memory, build_recent_memory, build_summary_memory
from long_memory_agent.prompting import build_qa_prompt


def parse_args():
    parser = argparse.ArgumentParser(description="Run the long-memory project demo.")
    parser.add_argument(
        "--data",
        default="long_memory_agent_project/data/demo/sample_dialogue.json",
        help="Path to a dialogue sample JSON file.",
    )
    parser.add_argument(
        "--strategy",
        choices=("recent", "summary", "hybrid"),
        default="hybrid",
        help="Memory strategy to preview.",
    )
    parser.add_argument(
        "--question-id",
        default="q1",
        help="Question id to use from the sample file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sample = load_dialogue_sample(args.data)

    try:
        question = next(item for item in sample.questions if item.question_id == args.question_id)
    except StopIteration as exc:
        available = ", ".join(item.question_id for item in sample.questions)
        raise SystemExit(f"Unknown question id {args.question_id}. Available ids: {available}") from exc

    if args.strategy == "recent":
        memory = build_recent_memory(sample)
    elif args.strategy == "summary":
        memory = build_summary_memory(sample)
    else:
        memory = build_hybrid_memory(sample, question=question.question)

    prompt = build_qa_prompt(question.question, memory)

    print("=" * 80)
    print(f"Sample: {sample.sample_id} | Title: {sample.title}")
    print(f"Strategy: {args.strategy}")
    print(f"Question ({question.question_id}): {question.question}")
    print(f"Gold answer: {question.answer}")
    print("=" * 80)
    print(prompt)


if __name__ == "__main__":
    main()
