from rag.modules.query_rewriting import NoRewrite
from rag.modules.reranking import NoReranker
from rag.modules.retrieval.schemas import RetrievalResult
from rag.pipelines.modular_chat_pipeline import ModularChatPipeline


class StubRetriever:
    def run(self, state):
        state["retrieval_results"] = [
            RetrievalResult(
                chunk_id="c1",
                text="Đây là nội dung văn bản pháp luật.",
                score=0.9,
                source="dense",
                metadata={
                    "chunk_id": "c1",
                    "source_file": "legal.pdf",
                    "raw_score": 0.1,
                },
                final_score=0.9,
            )
        ]
        state["retrieval"] = {
            "strategy": "stub",
            "output_count": 1,
        }
        return state


def fake_answer_fn(question, rewritten_query, docs, history):
    return {
        "answer": f"answer for {rewritten_query}",
        "mode": "grounded",
        "grounded": True,
        "warning": "",
    }


def test_modular_chat_pipeline_runs_with_injected_modules():
    pipeline = ModularChatPipeline(
        vectorstore=object(),
        history_selector=type("PassHistory", (), {"run": lambda self, state: {**state, "selected_history": state["history"]}})(),
        query_rewriter=NoRewrite(),
        retriever=StubRetriever(),
        reranker=NoReranker(),
        answer_fn=fake_answer_fn,
        top_k=1,
    )

    history = [{"role": "user", "content": "Câu trước"}]
    result = pipeline.chat(question="Câu hiện tại", history=history)

    assert result["rewritten_query"] == "Câu hiện tại"
    assert result["used_rewrite"] is False
    assert result["mode"] == "grounded"
    assert result["top_files"][0]["source_file"] == "legal.pdf"
    assert result["history"][-2]["content"] == "Câu hiện tại"
    assert result["history"][-1]["content"] == "answer for Câu hiện tại"
