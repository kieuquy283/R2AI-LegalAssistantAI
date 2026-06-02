import json

from rag.modules.retrieval import DenseRetriever, FAISSRetriever, HybridRetriever, SparseRetriever
from rag.modules.retrieval.fusion import reciprocal_rank_fusion, weighted_fusion
from rag.modules.retrieval.schemas import RetrievalResult
from rag.modules.retrieval.utils import (
    deduplicate_results,
    filter_low_score_results,
    get_effective_score,
    is_active_result,
    tokenize_for_bm25,
)


class FakeDocument:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


class FakeVectorstore:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def similarity_search_with_score(self, query, k):
        self.calls.append({"query": query, "k": k})
        return list(self.responses.get(query, []))[:k]


class FailingSparseRetriever:
    def retrieve(self, query, top_k=None):
        raise RuntimeError("bm25 failed")


def make_doc(text, chunk_id, **metadata):
    base_metadata = {"chunk_id": chunk_id}
    base_metadata.update(metadata)
    return FakeDocument(text, base_metadata)


def make_result(chunk_id, text, score, source="dense", **kwargs):
    return RetrievalResult(
        chunk_id=chunk_id,
        text=text,
        score=score,
        source=source,
        metadata=kwargs.pop("metadata", {}),
        final_score=kwargs.pop("final_score", None),
        dense_score=kwargs.pop("dense_score", None),
        sparse_score=kwargs.pop("sparse_score", None),
        sources=kwargs.pop("sources", None),
        retriever_name=kwargs.pop("retriever_name", None),
        raw_score=kwargs.pop("raw_score", None),
        normalized_score=kwargs.pop("normalized_score", None),
    )


def test_faiss_retriever_reads_query_priority_and_candidate_k():
    vectorstore = FakeVectorstore(
        {
            "preferred": [
                (make_doc("doc a", "c1"), 0.2),
                (make_doc("doc b", "c2"), 0.4),
            ]
        }
    )
    retriever = FAISSRetriever(vectorstore=vectorstore, top_k=2, candidate_k=7)

    state = retriever.run({"question": "q1", "query": "q2", "rewritten_query": "preferred"})

    assert vectorstore.calls == [{"query": "preferred", "k": 7}]
    assert [item.chunk_id for item in state["retrieval_results"]] == ["c1", "c2"]
    assert state["retrieval"]["strategy"] == "faiss"
    assert state["retrieval"]["candidate_k"] == 7


def test_faiss_retriever_handles_empty_query():
    retriever = FAISSRetriever(vectorstore=FakeVectorstore({}), top_k=2)
    state = retriever.run({"query": ""})

    assert state["retrieval_results"] == []
    assert state["retrieval"]["output_count"] == 0


def test_faiss_retriever_converts_scores_filters_active_and_deduplicates():
    vectorstore = FakeVectorstore(
        {
            "query": [
                (make_doc("doc a", "c1", is_active=True), 0.1),
                (make_doc("doc duplicate", "c1", is_active=True), 0.3),
                (make_doc("doc inactive", "c3", is_active=False), 0.2),
                (make_doc("doc active no flag", "c4"), 0.4),
            ]
        }
    )
    retriever = FAISSRetriever(vectorstore=vectorstore, top_k=3, candidate_k=4, filter_active=True)

    state = retriever.run({"query": "query"})
    results = state["retrieval_results"]

    assert len(results) == 2
    assert results[0].chunk_id == "c1"
    assert results[0].raw_score == 0.1
    assert results[0].normalized_score is not None
    assert all(is_active_result(result) for result in results)
    assert state["retrieval"]["filtered_inactive_count"] == 1


def test_dense_retriever_backward_compatible_import():
    retriever = DenseRetriever(vectorstore=FakeVectorstore({}), top_k=1)
    assert isinstance(retriever, FAISSRetriever)


def test_hybrid_retriever_rrf_runs_dense_and_sparse_and_merges_sources():
    vectorstore = FakeVectorstore(
        {
            "doc": [
                (make_doc("doc a", "c1"), 0.1),
                (make_doc("doc b", "c2"), 0.3),
            ]
        }
    )
    documents = [
        make_doc("doc a", "c1"),
        make_doc("doc c", "c3"),
    ]
    retriever = HybridRetriever(
        vectorstore=vectorstore,
        documents=documents,
        top_k=3,
        candidate_k=2,
        fusion_type="rrf",
    )

    state = retriever.run({"rewritten_query": "doc"})
    results = state["retrieval_results"]

    assert state["retrieval"]["fusion_type"] == "rrf"
    assert state["retrieval"]["dense_count"] >= 1
    assert state["retrieval"]["sparse_count"] >= 1
    assert state["retrieval"]["per_query_counts"][0]["query"] == "doc"
    assert any("dense" in (result.sources or []) for result in results)
    assert any(result.chunk_id == "c1" and "sparse" in (result.sources or []) for result in results)
    assert state["retrieval"]["output_count"] == len(results)


def test_hybrid_retriever_weighted_uses_final_score_sorting():
    dense_results = [
        make_result("c1", "dense 1", 0.9, source="dense", dense_score=0.9, sources=["dense"]),
        make_result("c2", "dense 2", 0.4, source="dense", dense_score=0.4, sources=["dense"]),
    ]
    sparse_results = [
        make_result("c2", "dense 2", 1.0, source="sparse", sparse_score=1.0, sources=["sparse"]),
        make_result("c3", "sparse 3", 0.7, source="sparse", sparse_score=0.7, sources=["sparse"]),
    ]
    fused = weighted_fusion(dense_results, sparse_results, alpha=0.2)

    assert fused[0].chunk_id == "c2"
    assert fused[0].final_score is not None
    assert "dense" in fused[0].sources and "sparse" in fused[0].sources


def test_hybrid_retriever_supports_multi_query_and_sparse_fallback():
    vectorstore = FakeVectorstore(
        {
            "q1": [(make_doc("doc a", "c1"), 0.1)],
            "q2": [(make_doc("doc b", "c2"), 0.2)],
        }
    )
    dense_retriever = FAISSRetriever(vectorstore=vectorstore, top_k=2, candidate_k=2)
    retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=FailingSparseRetriever(),
        top_k=3,
        candidate_k=2,
        fusion_type="rrf",
    )

    state = retriever.run({"queries": ["q1", "q2"]})

    assert [item["query"] for item in state["retrieval"]["per_query_counts"]] == ["q1", "q2"]
    assert state["retrieval"]["sparse_fallback"] is True
    assert state["retrieval"]["sparse_error"] == "bm25 failed"
    assert len(state["retrieval_results"]) == 2


def test_hybrid_retriever_handles_empty_bm25_corpus():
    vectorstore = FakeVectorstore({"q": [(make_doc("doc a", "c1"), 0.1)]})
    retriever = HybridRetriever(vectorstore=vectorstore, documents=[], top_k=2, candidate_k=2)

    state = retriever.run({"query": "q"})

    assert state["retrieval"]["sparse_count"] == 0
    assert len(state["retrieval_results"]) == 1


def test_get_effective_score_prefers_final_score_and_dedup_uses_it():
    low = make_result("c1", "text", 0.9, final_score=0.2)
    high = make_result("c1", "text", 0.1, final_score=0.8)
    deduped = deduplicate_results([low, high])

    assert get_effective_score(high) == 0.8
    assert deduped[0].final_score == 0.8


def test_filter_low_score_results_uses_effective_score():
    kept = make_result("c1", "text", 0.1, final_score=0.9)
    dropped = make_result("c2", "text", 0.9, final_score=0.01)
    filtered = filter_low_score_results([kept, dropped], threshold=0.1)

    assert [item.chunk_id for item in filtered] == ["c1"]


def test_tokenize_for_bm25_returns_tokens_and_bigrams():
    tokens = tokenize_for_bm25("nhập kinh doanh chuyển khẩu")
    assert "nhập" in tokens
    assert "kinh_doanh" in tokens
    assert "chuyển_khẩu" in tokens


def test_is_active_result_defaults_true():
    active = make_result("c1", "text", 0.5, metadata={})
    inactive = make_result("c2", "text", 0.5, metadata={"is_active": False})
    assert is_active_result(active) is True
    assert is_active_result(inactive) is False


def test_rrf_and_retrieval_result_to_dict_json_safe():
    dense = [make_result("c1", "doc1", 0.8, source="dense", sources=["dense"])]
    sparse = [make_result("c1", "doc1", 0.7, source="sparse", sources=["sparse"])]
    fused = reciprocal_rank_fusion(dense, sparse, k=60)

    assert fused[0].final_score is not None
    json.dumps(fused[0].to_dict())
