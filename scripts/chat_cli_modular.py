from __future__ import annotations

import argparse
import json

from rag.pipelines.modular_chat_pipeline import ModularChatPipeline


def print_top_files(top_files):
    if not top_files:
        print("[Top Files]: Khong co file lien quan.\n")
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
    parser = argparse.ArgumentParser(description="Modular Multi-turn RAG chat CLI.")
    parser.add_argument("--index-dir", default="indexes/default")
    parser.add_argument(
        "--show-pipeline-state",
        action="store_true",
        help="In metadata debug cua modular pipeline sau moi luot chat.",
    )
    args = parser.parse_args()

    pipeline = ModularChatPipeline(index_dir=args.index_dir)
    history = []

    print("=== MODULAR MULTI-TURN RAG CHATBOT ===")
    print("Go 'exit' hoac 'quit' de thoat.\n")

    while True:
        question = input("Ban: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Thoat chuong trinh.")
            break

        if not question:
            print("Vui long nhap cau hoi.")
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

            if args.show_pipeline_state:
                print("[Pipeline State]:")
                print(
                    json.dumps(
                        result.get("pipeline_state", {}),
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                print()

            print(f"Bot: {result['answer']}\n")

        except Exception as exc:
            print(f"\n[ERROR] {exc}\n")


if __name__ == "__main__":
    main()
