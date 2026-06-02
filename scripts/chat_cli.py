from __future__ import annotations

import argparse

from rag.pipelines.chat_pipeline import ChatPipeline


def print_top_files(top_files):
    if not top_files:
        print("[Top Files]: Không có file liên quan.\n")
        return

    print("[Top Files]:")
    for i, item in enumerate(top_files, 1):
        print(
            f"  {i}. {item['source_file']} | "
            f"best_score={item['best_score']:.4f} | "
            f"hits={item['hits']}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-turn RAG chat CLI.")
    parser.add_argument("--index-dir", default="indexes/default")
    args = parser.parse_args()

    pipeline = ChatPipeline(index_dir=args.index_dir)
    history = []

    print("=== MULTI-TURN RAG CHATBOT ===")
    print("Gõ 'exit' hoặc 'quit' để thoát.\n")

    while True:
        question = input("Bạn: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Thoát chương trình.")
            break

        if not question:
            print("Vui lòng nhập câu hỏi.")
            continue

        try:
            result = pipeline.chat(question=question, history=history)

            if result.get("show_rewritten_query"):
                print(f"\n[Rewritten Query]: {result['rewritten_query']}")
                print(f"[Used Rewrite]: {result['used_rewrite']}\n")

            print_top_files(result.get("top_files", []))

            if result.get("warning"):
                print(result["warning"])
                print()

            if result.get("mode"):
                print(f"[Mode]: {result['mode']}\n")

            print(f"Bot: {result['answer']}\n")

        except Exception as e:
            print(f"\n[ERROR] {e}\n")


if __name__ == "__main__":
    main()