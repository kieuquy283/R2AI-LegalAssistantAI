import json

import pytest

from rag.modules.retrieval.schemas import RetrievalResult
from rag.modules.reranking import (
    CrossEncoderReranker,
    NoReranker,
    RerankResult,
    Reranker,
)
from rag.modules.reranking.utils import normalize_scores


class FakeDocument:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


class FakeCrossEncoder:
    def __init__(self, scores=None, error=None):
        self.scores = list(scores or [])
        self.error = error
        self.calls = 0
        self.last_pairs = None

    def predict(self, pairs, batch_size=8, show_progress_bar=False):
        self.calls += 1
        self.last_pairs = list(pairs)
        if self.error is not None:
            raise self.error
        return self.scores[: len(pairs)]


def make_result(chunk_id, text, score, rank):
    return RetrievalResult(
        chunk_id=chunk_id,
        text=text,
        score=score,
        source="retrieval",
        metadata={"score": score, "chunk_id": chunk_id},
        retrieval_rank=rank,
    )


def test_no_reranker_preserves_order_and_scores():
    reranker = NoReranker()
    retrieval_results = [
        make_result("c1", "doc 1", 0.9, 1),
        make_result("c2", "doc 2", 0.7, 2),
    ]

    state = reranker.run({"retrieval_results": retrieval_results, "query": "RAG"})

    assert [item.chunk_id for item in state["reranked_results"]] == ["c1", "c2"]
    assert [item.final_score for item in state["reranked_results"]] == [0.9, 0.7]
    assert [item.rerank_rank for item in state["reranked_results"]] == [1, 2]
    assert state["reranking"]["rerank_applied"] is False
    assert state["reranking"]["strategy"] == "no_reranker"


def test_no_reranker_handles_documents_and_empty_results():
    reranker = NoReranker()
    docs = [
        FakeDocument("alpha", {"chunk_id": "d1", "score": 0.3}),
        FakeDocument("beta", {"chunk_id": "d2", "score": 0.2}),
    ]

    state = reranker.run({"documents": docs})
    empty_state = reranker.run({"documents": []})

    assert [item.chunk_id for item in state["reranked_results"]] == ["d1", "d2"]
    assert empty_state["reranked_results"] == []
    assert empty_state["reranking"]["output_count"] == 0


def test_no_reranker_does_not_instantiate_cross_encoder(monkeypatch):
    def fail_load_model(self):
        raise AssertionError("CrossEncoder should not be instantiated")

    monkeypatch.setattr(CrossEncoderReranker, "_load_model", fail_load_model)
    reranker = NoReranker()
    state = reranker.run({"retrieval_results": [make_result("c1", "doc", 0.5, 1)]})
    assert state["reranked_results"][0].source == "no_reranker"


def test_reranker_reads_state_priority_and_candidate_top_k():
    fake_model = FakeCrossEncoder(scores=[0.2, 3.0])
    reranker = Reranker(
        model="unused",
        model_name="fake-model",
        candidate_top_k=2,
        score_alpha=0.2,
    )
    reranker.model = fake_model

    docs = [
        make_result("c1", "doc 1", 0.9, 1),
        make_result("c2", "doc 2", 0.4, 2),
        make_result("c3", "doc 3", 0.1, 3),
    ]

    state = reranker.run(
        {
            "question": "wrong",
            "query": "fallback",
            "rewritten_query": "preferred query",
            "retrieved_docs": docs,
        }
    )

    assert fake_model.calls == 1
    assert fake_model.last_pairs == [("preferred query", "doc 1"), ("preferred query", "doc 2")]
    assert [item.chunk_id for item in state["reranked_results"]] == ["c2", "c1", "c3"]
    assert state["reranking"]["candidate_top_k"] == 2
    assert state["reranking"]["rerank_applied"] is True


def test_reranker_writes_metadata_and_handles_empty_results():
    reranker = Reranker(model=FakeCrossEncoder(scores=[]), model_name="fake-model")

    state = reranker.run({"rewritten_query": "query", "retrieval_results": []})

    assert state["reranked_results"] == []
    assert state["reranking"]["strategy"] == "reranker"
    assert state["reranking"]["input_count"] == 0
    assert state["reranking"]["output_count"] == 0


def test_reranker_empty_query_falls_back_to_no_reranker():
    docs = [make_result("c1", "doc 1", 0.6, 1)]
    reranker = Reranker(model=FakeCrossEncoder(scores=[1.0]), model_name="fake-model")

    state = reranker.run({"retrieval_results": docs, "query": ""})

    assert state["reranked_results"][0].source == "no_reranker"
    assert state["reranking"]["fallback_reason"] == "empty_query"
    assert state["reranking"]["rerank_applied"] is False


def test_reranker_prediction_error_falls_back():
    docs = [make_result("c1", "doc 1", 0.6, 1)]
    reranker = Reranker(
        model=FakeCrossEncoder(error=RuntimeError("boom")),
        model_name="fake-model",
    )

    state = reranker.run({"rewritten_query": "query", "retrieval_results": docs})

    assert state["reranked_results"][0].source == "no_reranker"
    assert state["reranking"]["fallback_reason"] == "prediction_error"


def test_score_normalization_modes():
    sigmoid_scores = normalize_scores([0.0, 2.0], strategy="sigmoid")
    minmax_scores = normalize_scores([5.0, 5.0], strategy="minmax")
    none_scores = normalize_scores([1.5, -0.5], strategy="none")

    assert all(0.0 <= score <= 1.0 for score in sigmoid_scores)
    assert minmax_scores == [0.5, 0.5]
    assert none_scores == [1.5, -0.5]


def test_rerank_result_to_dict_is_json_serializable():
    result = RerankResult(
        chunk_id="c1",
        text="doc",
        retrieval_score=0.4,
        rerank_score=0.7,
        final_score=0.64,
        metadata={"source": "doc"},
        retrieval_rank=2,
        rerank_rank=1,
        raw_rerank_score=2.3,
        normalized_rerank_score=0.7,
        rank_delta=1,
        reranker_name="fake-model",
    )

    payload = result.to_dict()
    json.dumps(payload)

    assert payload["reranker_name"] == "fake-model"
    assert payload["rank_delta"] == 1
