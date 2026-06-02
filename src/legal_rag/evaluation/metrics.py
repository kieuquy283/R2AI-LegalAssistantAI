from __future__ import annotations

from statistics import mean


def precision(pred: set[str], gold: set[str]) -> float:
    if not pred:
        return 0.0
    return len(pred & gold) / len(pred)


def recall(pred: set[str], gold: set[str]) -> float:
    if not gold:
        return 0.0
    return len(pred & gold) / len(gold)


def f2_score(p: float, r: float) -> float:
    if p == 0 and r == 0:
        return 0.0
    return (5 * p * r) / (4 * p + r)


def macro_average(values: list[float]) -> float:
    return mean(values) if values else 0.0
