from __future__ import annotations

import os
import time
from pathlib import Path
from threading import Lock
from typing import Any, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag.config.retrieval import DEFAULT_INDEX_DIR
from rag.evaluation.metrics import compute_retrieval_metrics
from rag.ingestion.indexing import index_exists
from rag.pipelines.factory import build_chat_pipeline
from rag.retrieval.query_rewriter import rewrite_query
from rag.retrieval.vectorstore import load_vectorstore
from rag.retrieval.retriever import extract_cids_from_docs, retrieve_documents
from rag.retrieval.ranking import filter_active_docs
from rag.utils.io import load_json, save_json


class ChatMessage(BaseModel):
    role: str
    content: str
    time: str | None = None
    metadata: dict[str, Any] | None = None


class ChatSession(BaseModel):
    id: str
    name: str
    createdAt: str
    messages: list[ChatMessage]


class ChatRequest(BaseModel):
    question: str
    history: list[dict] = []
    stream: bool = False


class ChatResponse(BaseModel):
    answer: str
    rewritten_query: str
    used_rewrite: bool
    show_rewritten_query: bool
    grounded: bool
    warning: str
    mode: str
    top_files: list[dict]
    history: list[dict]
    metadata: dict[str, Any] | None = None


class EvaluationStats(BaseModel):
    name: str
    top_k: int
    eval_path: str
    sample_count: int
    hit: float
    recall: float
    mrr: float


class EvaluationResponse(BaseModel):
    results: List[EvaluationStats]


app = FastAPI(title="Multi-turn RAG Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

default_index_dir = Path(DEFAULT_INDEX_DIR)
legal_index_dir = Path("indexes/legal")
faiss_index_dir = Path("faiss_index")
selected_index_dir = default_index_dir

if index_exists(legal_index_dir):
    selected_index_dir = legal_index_dir
elif not index_exists(default_index_dir) and index_exists(faiss_index_dir):
    selected_index_dir = faiss_index_dir
elif not index_exists(default_index_dir) and not index_exists(faiss_index_dir):
    selected_index_dir = default_index_dir

pipeline = build_chat_pipeline(index_dir=str(selected_index_dir))

ROOT_DIR = Path(__file__).resolve().parents[1]
SESSION_FILE_PATH = ROOT_DIR / "data" / "chat_sessions.json"
EVALUATION_CACHE_TTL_SECONDS = int(os.getenv("EVALUATION_CACHE_TTL_SECONDS", "900"))

_evaluation_cache_lock = Lock()
_evaluation_cache: dict[str, Any] = {
    "updated_at": 0.0,
    "key": None,
    "response": None,
}


def load_chat_sessions() -> list[Any]:
    sessions = load_json(SESSION_FILE_PATH, [])
    return sessions if isinstance(sessions, list) else []


def save_chat_sessions(sessions: list[Any]) -> None:
    save_json(SESSION_FILE_PATH, sessions)


@app.get("/sessions", response_model=list[ChatSession])
def get_sessions() -> list[ChatSession]:
    return load_chat_sessions()


@app.post("/sessions", response_model=list[ChatSession])
def save_sessions(sessions: list[ChatSession]) -> list[ChatSession]:
    save_chat_sessions([session.dict() for session in sessions])
    return sessions


def evaluate_single_turn(
    eval_path: str,
    vectorstore: Any,
    top_k: int = 10,
    max_samples: int | None = None,
) -> EvaluationStats:
    data = load_json(eval_path, [])
    if not isinstance(data, list) or not data:
        raise ValueError(f"Evaluation data is missing or invalid: {eval_path}")

    total = 0
    total_hit = 0.0
    total_recall = 0.0
    total_mrr = 0.0

    for sample in data:
        if not isinstance(sample, dict):
            continue

        question = str(sample.get("question", "")).strip()
        gt_cids = sample.get("ground_truth_cids", [])
        if not question or not gt_cids:
            continue

        docs = retrieve_documents(query=question, vectorstore=vectorstore, top_k=top_k)
        docs = filter_active_docs(docs, top_k=top_k)
        retrieved_cids = extract_cids_from_docs(docs)

        hit, recall, mrr = compute_retrieval_metrics(retrieved_cids, gt_cids)
        total += 1
        total_hit += float(hit)
        total_recall += float(recall)
        total_mrr += float(mrr)
        if max_samples is not None and total >= max_samples:
            break

    if total == 0:
        raise ValueError(f"No valid evaluation samples found in: {eval_path}")

    return EvaluationStats(
        name="Single-turn Retrieval",
        top_k=top_k,
        eval_path=eval_path,
        sample_count=total,
        hit=total_hit / total,
        recall=total_recall / total,
        mrr=total_mrr / total,
    )


def evaluate_multiturn(
    eval_path: str,
    vectorstore: Any,
    top_k: int = 10,
    use_rewrite: bool = True,
    max_samples: int | None = None,
) -> EvaluationStats:
    data = load_json(eval_path, [])
    if not isinstance(data, list) or not data:
        raise ValueError(f"Evaluation data is missing or invalid: {eval_path}")

    total = 0
    total_hit = 0.0
    total_recall = 0.0
    total_mrr = 0.0

    for sample in data:
        if not isinstance(sample, dict):
            continue

        question = str(sample.get("question", "")).strip()
        gt_cids = sample.get("ground_truth_cids", [])
        history = sample.get("history", [])
        if not question or not gt_cids:
            continue

        query = rewrite_query(question, history) if use_rewrite else question
        docs = retrieve_documents(query=query, vectorstore=vectorstore, top_k=top_k)
        docs = filter_active_docs(docs, top_k=top_k)
        retrieved_cids = extract_cids_from_docs(docs)

        hit, recall, mrr = compute_retrieval_metrics(retrieved_cids, gt_cids)
        total += 1
        total_hit += float(hit)
        total_recall += float(recall)
        total_mrr += float(mrr)
        if max_samples is not None and total >= max_samples:
            break

    if total == 0:
        raise ValueError(f"No valid evaluation samples found in: {eval_path}")

    name = "Multi-turn Rewrite" if use_rewrite else "Multi-turn No Rewrite"
    return EvaluationStats(
        name=name,
        top_k=top_k,
        eval_path=eval_path,
        sample_count=total,
        hit=total_hit / total,
        recall=total_recall / total,
        mrr=total_mrr / total,
    )


@app.post("/chat")
def chat(request: ChatRequest) -> Any:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if request.stream:
        import json
        if hasattr(pipeline, "chat_stream"):
            metadata, stream = pipeline.chat_stream(
                question=request.question,
                history=request.history,
            )
        else:
            result = pipeline.chat(question=request.question, history=request.history)
            def single_chunk_generator():
                yield f"data: {json.dumps({'event': 'metadata', 'data': result})}\n\n"
                yield f"data: {json.dumps({'event': 'token', 'data': result['answer']})}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(single_chunk_generator(), media_type="text/event-stream")

        def event_generator():
            yield f"data: {json.dumps({'event': 'metadata', 'data': metadata})}\n\n"
            for chunk in stream:
                token = getattr(chunk, "content", chunk)
                yield f"data: {json.dumps({'event': 'token', 'data': token})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")
    else:
        result = pipeline.chat(question=request.question, history=request.history)
        return result


@app.get("/evaluation", response_model=EvaluationResponse)
def get_evaluation(
    force_refresh: bool = Query(default=False),
    top_k: int = Query(default=10, ge=1, le=50),
    include_rewrite: bool = Query(default=True),
    max_samples: int | None = Query(default=None, ge=1),
) -> EvaluationResponse:
    index_dir = str(selected_index_dir)

    cache_key = (index_dir, top_k, include_rewrite, max_samples)
    now = time.time()

    with _evaluation_cache_lock:
        is_cache_valid = (
            not force_refresh
            and _evaluation_cache.get("response") is not None
            and _evaluation_cache.get("key") == cache_key
            and (now - float(_evaluation_cache.get("updated_at", 0.0))) <= EVALUATION_CACHE_TTL_SECONDS
        )
        if is_cache_valid:
            return _evaluation_cache["response"]

    try:
        vectorstore = load_vectorstore(index_dir=index_dir)
        single = evaluate_single_turn(
            eval_path="data/evaluation.json",
            vectorstore=vectorstore,
            top_k=top_k,
            max_samples=max_samples,
        )
        multiturn_no_rewrite = evaluate_multiturn(
            eval_path="data/multiturn_evaluation_filled.json",
            vectorstore=vectorstore,
            top_k=top_k,
            use_rewrite=False,
            max_samples=max_samples,
        )
        results = [single, multiturn_no_rewrite]
        if include_rewrite:
            multiturn_rewrite = evaluate_multiturn(
                eval_path="data/multiturn_evaluation_filled.json",
                vectorstore=vectorstore,
                top_k=top_k,
                use_rewrite=True,
                max_samples=max_samples,
            )
            results.append(multiturn_rewrite)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    response = EvaluationResponse(results=results)
    with _evaluation_cache_lock:
        _evaluation_cache["updated_at"] = now
        _evaluation_cache["key"] = cache_key
        _evaluation_cache["response"] = response
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.api:app", host="127.0.0.1", port=8000, reload=True)
