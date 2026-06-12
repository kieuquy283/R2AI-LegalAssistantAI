from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List


# Legal code pattern
LEGAL_CODE_PATTERN = re.compile(r"\b\d+(?:/\d+)+/[A-Z0-9À-ỴĂÂĐÊÔƠƯ\-]+\b", re.IGNORECASE)
ARTICLE_PATTERN = re.compile(r"[ĐđDd][iíìị][eê][ều\s]+\d+", re.IGNORECASE)


def _extract_legal_refs(text: str) -> list[str]:
    """Extract legal references from text."""
    return [m.group(0) for m in LEGAL_CODE_PATTERN.finditer(text)]


def _extract_articles(text: str) -> list[str]:
    """Extract article references from text."""
    return [m.group(0) for m in ARTICLE_PATTERN.finditer(text)]


class ComprehensiveEvaluator:
    """Evaluate RAG outputs with comprehensive metrics."""

    def __init__(self, *, llm_client=None, auto_eval: bool = True) -> None:
        self.llm_client = llm_client
        self.auto_eval = auto_eval and bool(llm_client)

    # ==================== 1. CĂN CỨ CHÍNH XÁC PHÁP LUẬT ====================
    def evaluate_legal_accuracy(self, answer: str, contexts: list[dict]) -> dict[str, Any]:
        """Đánh giá tỷ lệ câu hỏi có ít nhất một điều luật được trích xuất đúng."""
        answer_refs = set(_extract_legal_refs(answer))
        answer_articles = set(_extract_articles(answer))
        
        context_refs = set()
        context_articles = set()
        for ctx in contexts:
            citation = str(ctx.get("citation") or ctx.get("doc_title", ""))
            article = str(ctx.get("article") or "")
            context_refs.update(_extract_legal_refs(citation))
            context_articles.update(_extract_articles(article))
        
        # Citation match
        matched_refs = answer_refs & context_refs
        matched_articles = answer_articles & context_articles
        
        has_citation = bool(matched_refs or matched_articles)
        
        return {
            "has_citation": has_citation,
            "answer_refs": list(answer_refs),
            "context_refs": list(context_refs),
            "matched_refs": list(matched_refs),
            "matched_articles": list(matched_articles),
            "score": 1.0 if has_citation else 0.0,
        }

    # ==================== 2. PRECISION / RECALL / F2 ====================
    def evaluate_precision_recall_f2(
        self, predicted_articles: list[str], gold_articles: list[str]
    ) -> dict[str, float]:
        """Compute precision, recall, and F2 score."""
        pred_set = set(str(a).strip() for a in predicted_articles)
        gold_set = set(str(a).strip() for a in gold_articles)
        
        if not pred_set:
            return {"precision": 0.0, "recall": 0.0, "f2": 0.0}
        
        tp = len(pred_set & gold_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # F2 = 5 * (P * R) / (4P + R)
        if precision + recall == 0:
            f2 = 0.0
        else:
            f2 = 5 * precision * recall / (4 * precision + recall)
        
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f2": round(f2, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    # ==================== 3. MRR (Mean Reciprocal Rank) ====================
    def evaluate_mrr(self, predicted_articles: list[str], gold_articles: list[str]) -> float:
        """Compute Mean Reciprocal Rank for article retrieval."""
        if not gold_articles:
            return 0.0
        
        gold_set = set(str(a).strip() for a in gold_articles)
        
        for rank, pred in enumerate(predicted_articles, start=1):
            if str(pred).strip() in gold_set:
                return 1.0 / rank
        
        return 0.0

    # ==================== 4. FAITHFULNESS ====================
    def evaluate_faithfulness(self, answer: str, contexts: list[dict]) -> dict[str, Any]:
        """Evaluate if answer is faithful to retrieved contexts."""
        # Extract claims from answer
        sentences = [s.strip() for s in re.split(r"[.!?\n]", answer) if len(s.strip()) > 10]
        
        context_text = "\n".join(str(ctx.get("content", "")) for ctx in contexts)
        
        faithful_count = 0
        total_count = len(sentences)
        
        for sentence in sentences:
            # Check if sentence keywords appear in context
            words = set(re.findall(r"\b\w{4,}\b", sentence.lower()))
            if not words:
                continue
            
            context_words = set(re.findall(r"\b\w{4,}\b", context_text.lower()))
            overlap = len(words & context_words)
            
            if overlap >= max(1, len(words) * 0.3):
                faithful_count += 1
        
        score = faithful_count / total_count if total_count > 0 else 0.0
        
        return {
            "faithfulness_score": round(score, 4),
            "faithful_sentences": faithful_count,
            "total_sentences": total_count,
        }

    # ==================== 5. COMPLETENESS ====================
    def evaluate_completeness(self, answer: str, question: str, contexts: list[dict]) -> dict[str, Any]:
        """Đánh giá tính đầy đủ & toàn diện."""
        # Check if answer covers multiple aspects of the question
        question_aspects = set(re.findall(r"\b\w{5,}\b", question.lower()))
        answer_aspects = set(re.findall(r"\b\w{5,}\b", answer.lower()))
        
        covered = len(question_aspects & answer_aspects)
        coverage = covered / len(question_aspects) if question_aspects else 0.0
        
        # Check if answer has multiple sections
        has_sections = bool(re.search(r"\n\d+[.\)]\s+", answer))
        
        # Check length adequacy
        length_score = min(1.0, len(answer) / 500)
        
        score = (coverage * 0.4 + (1.0 if has_sections else 0.3) * 0.3 + length_score * 0.3)
        
        return {
            "completeness_score": round(score, 4),
            "coverage": round(coverage, 4),
            "has_sections": has_sections,
            "length_score": round(length_score, 4),
        }

    # ==================== 6. PRACTICALITY ====================
    def evaluate_practicality(self, answer: str) -> dict[str, Any]:
        """Đánh giá tính thực tiễn – khả năng áp dụng."""
        # Check for practical indicators
        indicators = [
            "nên", "cần", "phải", "bước", "thực hiện", "áp dụng",
            "lưu ý", "lưu ý", "rủi ro", "khuyến nghị", "khuyến cáo"
        ]
        
        indicator_count = sum(1 for ind in indicators if ind in answer.lower())
        indicator_score = min(1.0, indicator_count / 3)
        
        # Check if answer has actionable advice
        has_actionable = bool(re.search(r"(nên|cần|phải|bước|thực hiện)\s+", answer.lower()))
        
        # Check for SME-friendly language
        sme_indicators = [
            "doanh nghiệp", "sme", "công ty", "tổ chức", "người dùng"
        ]
        sme_score = min(1.0, sum(1 for ind in sme_indicators if ind in answer.lower()) / 2)
        
        score = (indicator_score * 0.4 + (1.0 if has_actionable else 0.0) * 0.4 + sme_score * 0.2)
        
        return {
            "practicality_score": round(score, 4),
            "has_actionable_advice": has_actionable,
            "indicator_count": indicator_count,
        }

    # ==================== 7. CLARITY ====================
    def evaluate_clarity(self, answer: str) -> dict[str, Any]:
        """Đánh giá tính rõ ràng – dễ hiểu."""
        # Average sentence length
        sentences = [s.strip() for s in re.split(r"[.!?\n]", answer) if s.strip()]
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        
        # Score: shorter sentences = clearer
        length_score = 1.0 if avg_sentence_length <= 15 else max(0.0, 1.0 - (avg_sentence_length - 15) / 30)
        
        # Check for structured format
        has_structure = bool(re.search(r"\n\d+[.\)]\s+|\n[-*]\s+|\n\w+[:\s]+", answer))
        
        # Check for legal jargon density
        jargon_words = [
            "hereinafter", "whereas", "pursuant", "notwithstanding",
            "thereto", "hereby", "aforementioned"
        ]
        jargon_count = sum(1 for j in jargon_words if j in answer.lower())
        jargon_score = max(0.0, 1.0 - jargon_count * 0.2)
        
        score = (length_score * 0.4 + (1.0 if has_structure else 0.0) * 0.3 + jargon_score * 0.3)
        
        return {
            "clarity_score": round(score, 4),
            "avg_sentence_length": round(avg_sentence_length, 2),
            "has_structure": has_structure,
            "jargon_count": jargon_count,
        }

    # ==================== 8. LLM-AS-A-JUDGE (Optional) ====================
    def evaluate_with_llm(self, question: str, answer: str, contexts: list[dict]) -> dict[str, Any]:
        """Use LLM to evaluate answer quality (optional)."""
        if not self.auto_eval or not self.llm_client:
            return {"llm_evaluated": False}
        
        prompt = f"""Bạn là chuyên gia pháp lý. Đánh giá câu trả lời dựa trên thang điểm 1-10:

Câu hỏi: {question}

Câu trả lời: {answer}

Hãy đánh giá:
1. Căn cứ pháp luật chính xác (1-10)
2. Tính chính xác nội dung (1-10)
3. Tính đầy đủ & toàn diện (1-10)
4. Tính thực tiễn (1-10)
5. Tính rõ ràng dễ hiểu (1-10)

Trả lời dạng JSON với các trường: legal_accuracy, content_accuracy, completeness, practicality, clarity"""

        try:
            response = self.llm_client.generate(
                system_prompt="Bạn là chuyên gia pháp lý đánh giá chất lượng câu trả lời.",
                user_prompt=prompt,
                temperature=0.0,
            )
            if response:
                # Try to parse JSON from response
                json_match = re.search(r"\{.*?\}", response, re.DOTALL)
                if json_match:
                    scores = json.loads(json_match.group(0))
                    return {
                        "llm_evaluated": True,
                        **{k: float(v) / 10.0 for k, v in scores.items()},
                    }
        except Exception as exc:
            print(f"[LLM Eval] Error: {exc}")
        
        return {"llm_evaluated": False}

    # ==================== MAIN EVALUATION ====================
    def evaluate(self, question: str, answer: str, contexts: list[dict], gold_articles: list[str] | None = None) -> dict[str, Any]:
        """Run comprehensive evaluation."""
        t0 = time.perf_counter()
        
        # 1. Legal accuracy
        legal_acc = self.evaluate_legal_accuracy(answer, contexts)
        
        # 2. Faithfulness
        faith = self.evaluate_faithfulness(answer, contexts)
        
        # 3. Completeness
        complete = self.evaluate_completeness(answer, question, contexts)
        
        # 4. Practicality
        practical = self.evaluate_practicality(answer)
        
        # 5. Clarity
        clarity = self.evaluate_clarity(answer)
        
        # 6. Precision/Recall/F2 (if gold available)
        prf = self.evaluate_precision_recall_f2(
            _extract_articles(answer), gold_articles or []
        ) if gold_articles else {"precision": None, "recall": None, "f2": None}
        
        # 7. MRR (if gold available)
        mrr = self.evaluate_mrr(
            _extract_articles(answer), gold_articles or []
        ) if gold_articles else None
        
        # 8. LLM evaluation (optional)
        llm_scores = self.evaluate_with_llm(question, answer, contexts) if self.auto_eval else {"llm_evaluated": False}
        
        return {
            "legal_accuracy": legal_acc,
            "faithfulness": faith,
            "completeness": complete,
            "practicality": practical,
            "clarity": clarity,
            "precision_recall_f2": prf,
            "mrr": mrr,
            "llm_scores": llm_scores,
            "eval_time": round(time.perf_counter() - t0, 3),
        }


def evaluate_batch(
    results: list[dict],
    *,
    gold_data: list[dict] | None = None,
    llm_client=None,
    auto_eval: bool = True,
) -> dict[str, Any]:
    """Evaluate a batch of results."""
    evaluator = ComprehensiveEvaluator(llm_client=llm_client, auto_eval=auto_eval)
    
    metrics = {
        "legal_accuracy": [],
        "faithfulness": [],
        "completeness": [],
        "practicality": [],
        "clarity": [],
        "precision": [],
        "recall": [],
        "f2": [],
        "mrr": [],
    }
    
    for i, result in enumerate(results):
        question = result.get("question", "")
        answer = result.get("answer", "")
        contexts = result.get("final_contexts", []) or result.get("contexts", [])
        
        gold_articles = None
        if gold_data and i < len(gold_data):
            gold_articles = gold_data[i].get("expected_articles", []) or gold_data[i].get("relevant_articles", [])
        
        eval_result = evaluator.evaluate(question, answer, contexts, gold_articles)
        
        metrics["legal_accuracy"].append(eval_result["legal_accuracy"]["score"])
        metrics["faithfulness"].append(eval_result["faithfulness"]["faithfulness_score"])
        metrics["completeness"].append(eval_result["completeness"]["completeness_score"])
        metrics["practicality"].append(eval_result["practicality"]["practicality_score"])
        metrics["clarity"].append(eval_result["clarity"]["clarity_score"])
        
        prf = eval_result["precision_recall_f2"]
        if prf["precision"] is not None:
            metrics["precision"].append(prf["precision"])
            metrics["recall"].append(prf["recall"])
            metrics["f2"].append(prf["f2"])
        
        if eval_result["mrr"] is not None:
            metrics["mrr"].append(eval_result["mrr"])
    
    def _avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0
    
    return {
        "num_questions": len(results),
        "legal_accuracy": round(_avg(metrics["legal_accuracy"]), 4),
        "faithfulness": round(_avg(metrics["faithfulness"]), 4),
        "completeness": round(_avg(metrics["completeness"]), 4),
        "practicality": round(_avg(metrics["practicality"]), 4),
        "clarity": round(_avg(metrics["clarity"]), 4),
        "precision": round(_avg(metrics["precision"]), 4) if metrics["precision"] else None,
        "recall": round(_avg(metrics["recall"]), 4) if metrics["recall"] else None,
        "f2": round(_avg(metrics["f2"]), 4) if metrics["f2"] else None,
        "mrr": round(_avg(metrics["mrr"]), 4) if metrics["mrr"] else None,
    }
