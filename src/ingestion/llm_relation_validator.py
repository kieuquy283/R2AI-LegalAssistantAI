"""
llm_relation_validator.py — Triage-based LLM verification for legal graph edges.

Chỉ gọi LLM cho các trường hợp Regex có độ tin cậy THẤP hoặc cú pháp phức tạp.
Các trường hợp đơn giản (1 số hiệu, ngữ cảnh rõ ràng) → Auto-Pass.

Usage:
    python -m src.ingestion.llm_relation_validator --edges data/processed/legal_edges.jsonl --mode sample -n 200
    python -m src.ingestion.llm_relation_validator --mode all

Environment:
    OPENROUTER_API_KEY or OPENAI_API_KEY
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Constants ──
BATCH_SIZE = 10          # edges per LLM prompt
MAX_CONTEXT_CHARS = 800  # chars of source text around match
DEFAULT_MODEL = "google/gemma-4-31b-it:free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ── Triage heuristics ──
# Keywords that indicate complex/ambiguous context → flag for LLM
COMPLEX_CONTEXT_RE = re.compile(
    r"(?:ngoại\s+trừ|trừ\s+trường\s+hợp|được\s+áp\s+dụng\s+tương\s+tự\s+nhưng|"
    r"hoặc|tùy\s+thuộc\s+vào|tùy\s+theo|nếu|trường\s+hợp|"
    r"trừ\s+khi|ngoài\s+ra|đồng\s+thờì|cũng\s+nhu|"
    r"áp\s+dụng\s+cho|không\s+áp\s+dụng\s+cho|chỉ\s+áp\s+dụng)",
    re.IGNORECASE | re.UNICODE,
)

# Branching structure indicators
BRANCHING_RE = re.compile(
    r"(?:hoặc\s+(?:Điều|khoản|Nghị\s+định|Luật|Thông\s+tư)|"
    r"tùy\s+(?:thuộc\s+vào|theo)|"
    r"theo\s+quy\s+định\s+tại\s+(?:Điều\s+\d+\s+và\s+Điều|khoản\s+\d+\s+và\s+khoản))",
    re.IGNORECASE | re.UNICODE,
)

# Multiple doc numbers in one sentence
DOC_NUMBER_RE = re.compile(r"\b\d+/\d{4}/[A-ZĐ0-9\-]+\b")

# ── LLM Prompt ──
SYSTEM_PROMPT = (
    "Bạn là một chuyên gia Lập pháp và Kiểm định Dữ liệu Pháp luật Việt Nam. "
    "Nhiệm vụ của bạn là kiểm tra xem mối quan hệ dẫn chiếu do hệ thống tự động (Regex) trích xuất "
    "từ đoạn văn bản dưới đây có chính xác về mặt ngữ nghĩa pháp lý hay không."
)

USER_PROMPT_TEMPLATE = """# CONTEXT TEXT
"{context_text}"

# PROPOSED RELATIONSHIP TO VERIFY
- Văn bản gốc (Source Chunk ID): {source_chunk_id}
- Văn bản bị dẫn chiếu tới (Target ID dự kiến): {target_id}
- Loại quan hệ đề xuất: {proposed_relation_type} (Có thể là EXTERNAL_REF hoặc EXCLUDED_REF)

# VALIDATION RULES
1. TRUE (Chính xác): Nếu đoạn văn bản thực sự khẳng định rằng quy định, hành vi, hoặc mức phạt ở Văn bản gốc được dẫn chiếu, áp dụng, hoặc tuân theo điều khoản ở Văn bản đích.
2. FALSE (Sai):
   - Nếu Regex bắt nhầm số Điều của văn bản này ghép vào số hiệu của văn bản khác do câu có quá nhiều thực thể.
   - Nếu ngữ cảnh là phủ định hoàn toàn ("không liên quan", "không áp dụng") nhưng hệ thống lại đề xuất nhãn dẫn chiếu thông thường thay vì loại trừ.
3. EXCLUDED (Ngoại trừ): Nếu ngữ cảnh là ngoại lệ ("trừ trường hợp quy định tại Điều X...").

# OUTPUT FORMAT
Bạn BẮT BUỘC phải trả về định dạng JSON duy nhất dưới đây, không thêm bất kỳ lờì giải thích nào khác ngoài JSON:
{{
    "is_correct": true / false,
    "corrected_relation_type": "EXTERNAL_REF" / "EXCLUDED_REF" / "NONE",
    "confidence_score": 0.0 đến 1.0,
    "reason": "Giải thích ngắn gọn lý do bằng tiếng Việt"
}}
"""


class RegexConfidenceScorer:
    """Score regex-extracted edges for confidence. High confidence = Auto-Pass."""

    @staticmethod
    def score(edge: Dict[str, Any], context: str) -> Tuple[float, str]:
        """
        Returns (confidence_score, reason).
        confidence_score >= 0.95 → Auto-Pass (no LLM needed)
        confidence_score < 0.95 → Flagged for LLM review
        """
        score = 1.0
        reasons = []

        # 1. Count doc numbers in context
        doc_numbers = DOC_NUMBER_RE.findall(context)
        if len(doc_numbers) > 2:
            score -= 0.30
            reasons.append(f"multiple_doc_numbers({len(doc_numbers)})")
        elif len(doc_numbers) == 2:
            score -= 0.10
            reasons.append("two_doc_numbers")

        # 2. Check complex/negation context
        if COMPLEX_CONTEXT_RE.search(context):
            score -= 0.40
            reasons.append("complex_context")

        # 3. Check branching structure
        if BRANCHING_RE.search(context):
            score -= 0.35
            reasons.append("branching_structure")

        # 4. Check relation type
        relation = edge.get("relation_type", "")
        if relation == "EXCLUDED_REF":
            # EXCLUDED_REF from regex is usually correct, but worth double-checking
            score -= 0.05
            reasons.append("excluded_ref")

        # 5. Context length (too short = less confidence)
        if len(context) < 100:
            score -= 0.10
            reasons.append("short_context")

        score = max(0.0, score)
        reason_str = "; ".join(reasons) if reasons else "simple_explicit_ref"
        return score, reason_str


class TriageEngine:
    """Decide which edges need LLM and which can Auto-Pass."""

    def __init__(self, auto_pass_threshold: float = 0.95):
        self.auto_pass_threshold = auto_pass_threshold
        self.scorer = RegexConfidenceScorer()

    def triage(
        self, edges: List[Dict[str, Any]], contexts: Dict[str, str]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Returns (auto_pass_edges, flagged_edges).
        Each edge is enriched with regex_confidence and regex_confidence_reason.
        """
        auto_pass = []
        flagged = []

        for edge in edges:
            context = contexts.get(edge.get("source_id", ""), "")
            conf_score, conf_reason = self.scorer.score(edge, context)

            enriched = {
                **edge,
                "regex_confidence": conf_score,
                "regex_confidence_reason": conf_reason,
                "context_preview": context[:300],
            }

            if conf_score >= self.auto_pass_threshold:
                enriched["llm_verdict"] = "AUTO_PASS"
                enriched["llm_reason"] = f"Regex confidence {conf_score:.2f} >= threshold"
                auto_pass.append(enriched)
            else:
                flagged.append(enriched)

        return auto_pass, flagged


class LLMRelationValidator:
    """Validate legal graph edges using LLM only for flagged (low-confidence) cases."""

    def __init__(
        self,
        *,
        docs_path: str | Path = "data/processed/cleaned_documents_enriched.jsonl",
        chunks_path: str | Path = "data/processed/chunks.jsonl",
        edges_path: str | Path = "data/processed/legal_edges.jsonl",
        output_validated: str | Path = "data/processed/legal_edges_llm_validated.jsonl",
        output_rejected: str | Path = "data/processed/legal_edges_llm_rejected.jsonl",
        output_autopass: str | Path = "data/processed/legal_edges_autopass.jsonl",
        log_path: str | Path = "data/processed/llm_validator.log",
        model: str = DEFAULT_MODEL,
        batch_size: int = BATCH_SIZE,
        max_context: int = MAX_CONTEXT_CHARS,
        temperature: float = 0.0,
        auto_pass_threshold: float = 0.95,
    ) -> None:
        self.docs_path = Path(docs_path)
        self.chunks_path = Path(chunks_path)
        self.edges_path = Path(edges_path)
        self.output_validated = Path(output_validated)
        self.output_rejected = Path(output_rejected)
        self.output_autopass = Path(output_autopass)
        self.log_path = Path(log_path)

        self.batch_size = batch_size
        self.max_context = max_context
        self.model = model
        self.temperature = temperature
        self.api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY", "")

        self.triage = TriageEngine(auto_pass_threshold=auto_pass_threshold)
        self._client = None

        # Lazy-loaded source texts
        self.doc_texts: Dict[str, str] = {}
        self.chunk_texts: Dict[str, str] = {}

        # Stats
        self.stats = {
            "total_edges": 0,
            "auto_pass": 0,
            "flagged": 0,
            "llm_corrected": 0,
            "llm_confirmed": 0,
            "llm_rejected": 0,
            "llm_uncertain": 0,
            "api_calls": 0,
            "tokens_input_est": 0,
            "tokens_output_est": 0,
        }

    # ── OpenRouter client ──

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=self.api_key,
                default_headers={
                    "HTTP-Referer": "https://github.com/r2ai/rag-chatbot",
                    "X-Title": "R2AI Legal Relation Validator",
                },
            )
        return self._client

    def is_available(self) -> bool:
        return bool(self.api_key)

    # ── Text loading ──

    def _load_doc_texts(self) -> None:
        if self.doc_texts:
            return
        print("Loading doc texts...")
        with open(self.docs_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                doc_id = rec.get("doc_id") or rec.get("id")
                if doc_id:
                    self.doc_texts[str(doc_id)] = rec.get("cleaned_text", "")
        print(f"  Loaded {len(self.doc_texts)} doc texts")

    def _load_all_chunk_texts(self) -> None:
        """Preload all chunk texts into memory (one-time scan)."""
        if self.chunk_texts:
            return
        print(f"Preloading chunk texts from {self.chunks_path}...")
        count = 0
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                cid = rec.get("chunk_id") or rec.get("id")
                if cid:
                    self.chunk_texts[cid] = rec.get("content", "")
                    count += 1
                    if count % 500000 == 0:
                        print(f"  ...loaded {count} chunks")
        print(f"  Loaded {count} chunk texts")

    def _get_context(self, edge: Dict[str, Any]) -> str:
        source_id = edge.get("source_id", "")
        relation = edge.get("relation_type", "")

        if relation in ("GUIDED_BY", "REPLACES"):
            text = self.doc_texts.get(source_id, "")
            if relation == "GUIDED_BY":
                return text[: self.max_context]
            return text[-self.max_context :] if len(text) > self.max_context else text

        return self.chunk_texts.get(source_id, "")

    # ── LLM calling ──

    def _build_llm_prompt(self, edge: Dict[str, Any]) -> str:
        context = self._get_context(edge)
        if len(context) > self.max_context:
            context = context[: self.max_context] + "\n...[truncated]"

        return USER_PROMPT_TEMPLATE.format(
            context_text=context,
            source_chunk_id=edge.get("source_id", ""),
            target_id=edge.get("target_id", ""),
            proposed_relation_type=edge.get("relation_type", "UNKNOWN"),
        )

    def _call_llm_single(self, edge: Dict[str, Any]) -> Dict[str, Any]:
        """Call LLM for one edge. Returns parsed verdict dict."""
        user_prompt = self._build_llm_prompt(edge)
        self.stats["api_calls"] += 1
        self.stats["tokens_input_est"] += len(user_prompt) // 4

        try:
            client = self._get_client()
            t0 = time.perf_counter()
            response = client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            latency = time.perf_counter() - t0
            raw = response.choices[0].message.content if response.choices else None
            print(f"    [LLM] {latency:.1f}s | {edge.get('relation_type', '?')} | {edge.get('source_id', '')[:50]}...")
        except Exception as e:
            print(f"    [ERROR] OpenRouter: {e}")
            return {"is_correct": False, "corrected_relation_type": "NONE", "confidence_score": 0.0, "reason": f"API error: {e}"}

        if not raw:
            return {"is_correct": False, "corrected_relation_type": "NONE", "confidence_score": 0.0, "reason": "Empty response"}

        self.stats["tokens_output_est"] += len(raw) // 4
        return self._parse_llm_json(raw)

    def _parse_llm_json(self, raw: str) -> Dict[str, Any]:
        """Extract JSON from LLM response."""
        try:
            # Find JSON block
            match = re.search(r"\{[^{}]*\"is_correct\"[^{}]*\}", raw, re.DOTALL)
            if not match:
                # Try broader search
                match = re.search(r"\{.*\"is_correct\".*\}", raw, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
            else:
                data = json.loads(raw)

            return {
                "is_correct": bool(data.get("is_correct", False)),
                "corrected_relation_type": data.get("corrected_relation_type", "NONE"),
                "confidence_score": float(data.get("confidence_score", 0.0)),
                "reason": str(data.get("reason", "")),
            }
        except Exception as e:
            print(f"    [WARN] JSON parse failed: {e}")
            return {"is_correct": False, "corrected_relation_type": "NONE", "confidence_score": 0.0, "reason": f"Parse error: {e}"}

    def _call_llm_batch(self, edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Call LLM for a batch of edges ( batched into one prompt for efficiency)."""
        if not edges:
            return []

        # Build combined prompt
        blocks = []
        for idx, edge in enumerate(edges):
            context = self._get_context(edge)
            if len(context) > self.max_context:
                context = context[: self.max_context] + "\n...[truncated]"
            block = (
                f"--- Edge {idx} ---\n"
                f"Proposed: {edge.get('relation_type')} | {edge.get('source_id')} → {edge.get('target_id')}\n"
                f"Context: {context[:500]}\n"
            )
            blocks.append(block)

        combined_prompt = (
            SYSTEM_PROMPT + "\n\n"
            "Validate each edge below. Return JSON array:\n"
            '[{"edge_index": 0, "is_correct": true, "corrected_relation_type": "EXTERNAL_REF", "confidence_score": 0.95, "reason": "..."}, ...]\n\n'
            + "\n".join(blocks)
        )

        self.stats["api_calls"] += 1
        self.stats["tokens_input_est"] += len(combined_prompt) // 4

        try:
            client = self._get_client()
            t0 = time.perf_counter()
            response = client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[{"role": "user", "content": combined_prompt}],
            )
            latency = time.perf_counter() - t0
            raw = response.choices[0].message.content if response.choices else None
            print(f"  [LLM Batch] {len(edges)} edges in {latency:.1f}s")
        except Exception as e:
            print(f"  [ERROR] Batch API call failed: {e}")
            return [{"is_correct": False, "corrected_relation_type": "NONE", "confidence_score": 0.0, "reason": f"API error"} for _ in edges]

        if not raw:
            return [{"is_correct": False, "corrected_relation_type": "NONE", "confidence_score": 0.0, "reason": "Empty"} for _ in edges]

        self.stats["tokens_output_est"] += len(raw) // 4

        # Parse array response
        try:
            match = re.search(r"\[.*?\]", raw, re.DOTALL)
            if match:
                arr = json.loads(match.group(0))
            else:
                arr = json.loads(raw)

            results = []
            for i in range(len(edges)):
                item = next((x for x in arr if x.get("edge_index") == i), {})
                results.append({
                    "is_correct": bool(item.get("is_correct", False)),
                    "corrected_relation_type": item.get("corrected_relation_type", "NONE"),
                    "confidence_score": float(item.get("confidence_score", 0.0)),
                    "reason": str(item.get("reason", "")),
                })
            return results
        except Exception as e:
            print(f"  [WARN] Batch parse failed, falling back to single calls: {e}")
            # Fallback: call one by one
            return [self._call_llm_single(e) for e in edges]

    # ── Main validation ──

    def validate_edges(self, edges: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Run triage + LLM validation.
        Returns (validated, rejected, autopass).
        """
        # Preload chunk texts to avoid repeated file scans
        self._load_all_chunk_texts()

        # Build context map
        contexts = {}
        print("Building context map...")
        for edge in edges:
            sid = edge.get("source_id", "")
            if sid not in contexts:
                contexts[sid] = self._get_context(edge)

        # Triage
        print(f"\nTriage: scoring {len(edges)} edges...")
        auto_pass, flagged = self.triage.triage(edges, contexts)
        print(f"  Auto-Pass: {len(auto_pass)} | Flagged for LLM: {len(flagged)}")

        self.stats["auto_pass"] = len(auto_pass)
        self.stats["flagged"] = len(flagged)

        # Process flagged edges in batches
        validated = []
        rejected = []

        for i in range(0, len(flagged), self.batch_size):
            batch = flagged[i : i + self.batch_size]
            print(f"\n  LLM batch {i//self.batch_size + 1}/{(len(flagged)-1)//self.batch_size + 1} ({len(batch)} edges)")

            verdicts = self._call_llm_batch(batch)

            for edge, verdict in zip(batch, verdicts):
                enriched = {
                    **edge,
                    "llm_is_correct": verdict["is_correct"],
                    "llm_corrected_relation_type": verdict["corrected_relation_type"],
                    "llm_confidence_score": verdict["confidence_score"],
                    "llm_reason": verdict["reason"],
                }

                if verdict["is_correct"]:
                    # LLM confirmed or corrected relation type
                    if verdict["corrected_relation_type"] != edge.get("relation_type", ""):
                        enriched["relation_type"] = verdict["corrected_relation_type"]
                        self.stats["llm_corrected"] += 1
                        print(f"    [OK] CORRECTED -> {verdict['corrected_relation_type']}")
                    else:
                        self.stats["llm_confirmed"] += 1
                        print(f"    [OK] CONFIRMED")
                    validated.append(enriched)
                else:
                    self.stats["llm_rejected"] += 1
                    reason = verdict.get("reason", "")
                    # Safe print for Windows console
                    try:
                        print(f"    [REJECTED] {reason[:80]}")
                    except UnicodeEncodeError:
                        print(f"    [REJECTED] {reason[:80].encode('ascii', 'replace').decode()}")
                    rejected.append(enriched)

            time.sleep(0.3)  # Rate limit safety

        return validated, rejected, auto_pass

    # ── Run modes ──

    def run_sample(self, n: int = 200, seed: int = 42) -> Dict[str, Any]:
        self._load_doc_texts()

        print(f"Loading edges from {self.edges_path}...")
        all_edges = []
        with open(self.edges_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_edges.append(json.loads(line))
        print(f"  Total edges: {len(all_edges)}")
        self.stats["total_edges"] = len(all_edges)

        if n < len(all_edges):
            random.seed(seed)
            sample = random.sample(all_edges, n)
        else:
            sample = all_edges

        print(f"\n{'='*60}")
        print(f"TRIAGE-BASED LLM VALIDATION (sample={len(sample)})")
        print(f"Model: {self.model}")
        print(f"Auto-Pass threshold: {self.triage.auto_pass_threshold}")
        print(f"{'='*60}")

        t0 = time.time()
        validated, rejected, autopass = self.validate_edges(sample)
        elapsed = time.time() - t0

        # Write outputs
        self._write_jsonl(self.output_validated, validated + autopass)
        self._write_jsonl(self.output_rejected, rejected)
        self._write_jsonl(self.output_autopass, autopass)

        result = {
            "mode": "sample",
            "sample_size": len(sample),
            "elapsed_seconds": elapsed,
            **self.stats,
            "accuracy_estimate": (self.stats["auto_pass"] + self.stats["llm_confirmed"]) / len(sample) if sample else 0,
        }
        self._write_log(result)
        return result

    def run_all(self) -> Dict[str, Any]:
        self._load_doc_texts()

        print(f"Loading ALL edges from {self.edges_path}...")
        all_edges = []
        with open(self.edges_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_edges.append(json.loads(line))
        print(f"  Total edges: {len(all_edges)}")
        self.stats["total_edges"] = len(all_edges)

        print(f"\n{'='*60}")
        print(f"TRIAGE-BASED LLM VALIDATION (FULL)")
        print(f"Model: {self.model}")
        print(f"{'='*60}")

        t0 = time.time()
        validated, rejected, autopass = self.validate_edges(all_edges)
        elapsed = time.time() - t0

        self._write_jsonl(self.output_validated, validated + autopass)
        self._write_jsonl(self.output_rejected, rejected)
        self._write_jsonl(self.output_autopass, autopass)

        result = {
            "mode": "all",
            "total_edges": len(all_edges),
            "elapsed_seconds": elapsed,
            **self.stats,
        }
        self._write_log(result)
        return result

    # ── Helpers ──

    def _write_jsonl(self, path: Path, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  Written {len(rows)} rows → {path}")

    def _write_log(self, result: Dict[str, Any]) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


# ── CLI ──

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Triage-based LLM validation of legal graph edges")
    parser.add_argument("--edges", default="data/processed/legal_edges.jsonl")
    parser.add_argument("--docs", default="data/processed/cleaned_documents_enriched.jsonl")
    parser.add_argument("--chunks", default="data/processed/chunks.jsonl")
    parser.add_argument("--mode", choices=["sample", "all"], default="sample")
    parser.add_argument("-n", "--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--threshold", type=float, default=0.95, help="Auto-pass confidence threshold")
    parser.add_argument("--output-validated", default="data/processed/legal_edges_llm_validated.jsonl")
    parser.add_argument("--output-rejected", default="data/processed/legal_edges_llm_rejected.jsonl")
    parser.add_argument("--output-autopass", default="data/processed/legal_edges_autopass.jsonl")
    args = parser.parse_args()

    validator = LLMRelationValidator(
        docs_path=args.docs,
        chunks_path=args.chunks,
        edges_path=args.edges,
        output_validated=args.output_validated,
        output_rejected=args.output_rejected,
        output_autopass=args.output_autopass,
        model=args.model,
        batch_size=args.batch_size,
        auto_pass_threshold=args.threshold,
    )

    if not validator.is_available():
        print("ERROR: LLM not available. Set OPENROUTER_API_KEY or OPENAI_API_KEY.")
        sys.exit(1)

    if args.mode == "sample":
        result = validator.run_sample(n=args.sample_size, seed=args.seed)
    else:
        result = validator.run_all()

    print("\n" + "=" * 60)
    print("TRIAGE VALIDATION RESULT")
    print("=" * 60)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("=" * 60)


if __name__ == "__main__":
    main()
