from __future__ import annotations

from typing import Any, Iterable, Sequence, Tuple


def hit_at_k(retrieved_cids: Sequence[Any], ground_truth_cids: Sequence[Any]) -> int:
    gt_set = set(ground_truth_cids)
    return int(any(cid in gt_set for cid in retrieved_cids))


def recall_at_k(retrieved_cids: Sequence[Any], ground_truth_cids: Sequence[Any]) -> float:
    gt_set = set(ground_truth_cids)
    if not gt_set:
        return 0.0
    return len(set(retrieved_cids) & gt_set) / len(gt_set)


def mrr(retrieved_cids: Sequence[Any], ground_truth_cids: Sequence[Any]) -> float:
    gt_set = set(ground_truth_cids)
    for rank, cid in enumerate(retrieved_cids, start=1):
        if cid in gt_set:
            return 1.0 / rank
    return 0.0


def compute_retrieval_metrics(
    retrieved_cids: Sequence[Any],
    ground_truth_cids: Sequence[Any],
) -> Tuple[int, float, float]:
    return (
        hit_at_k(retrieved_cids, ground_truth_cids),
        recall_at_k(retrieved_cids, ground_truth_cids),
        mrr(retrieved_cids, ground_truth_cids),
    )
