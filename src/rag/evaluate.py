"""
RAG Evaluation Suite (Day 6 Deliverable)
Evaluates:
1. Retrieval Metrics: Hit Rate @ K and Mean Reciprocal Rank (MRR)
2. Generation Metrics: Hallucination Resistance & Safety Guardrails
"""

import json
import sys
from pathlib import Path

# Ensure root in sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag.rag_pipeline import RAGPipeline

INDEX_DIRECTORY = ROOT / "storage" / "rag"
QUESTIONS_FILE = ROOT / "data" / "test_questions.json"


def evaluate_rag():
    print("=" * 65)
    print("        RAG System Evaluation & Benchmarking (Day 6)")
    print("=" * 65)

    if not QUESTIONS_FILE.exists():
        print(f"Error: Evaluation file not found at {QUESTIONS_FILE}")
        print("Please create data/test_questions.json first.")
        return

    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    # Initialize RAG Pipeline
    pipeline = RAGPipeline(index_directory=INDEX_DIRECTORY)
    top_k = 3

    # Metrics trackers
    retrieval_hits = 0
    mrr_sum = 0.0
    answerable_count = 0
    hallucination_guarded = 0
    unanswerable_count = 0

    print(f"\nRunning benchmark on {len(test_cases)} test cases (Top-K = {top_k})...\n")

    for idx, item in enumerate(test_cases, start=1):
        query = item["question"]
        expected_doc = item.get("expected_source", "").lower()
        is_answerable = item.get("is_answerable", True)

        print(f"[{idx}/{len(test_cases)}] Query: {query}")

        # 1. Evaluate Retrieval
        retrieved_chunks = pipeline.retrieve(query, top_k=top_k)
        retrieved_names = [
            Path(c.get("source", "")).name.lower()
            for c in retrieved_chunks
        ]

        if is_answerable:
            answerable_count += 1
            rank = None
            for pos, name in enumerate(retrieved_names, start=1):
                if expected_doc in name:
                    rank = pos
                    break

            if rank is not None:
                retrieval_hits += 1
                mrr_sum += 1.0 / rank
                print(f"  ✓ Retrieval: HIT (Rank #{rank}) | Sources: {retrieved_names}")
            else:
                print(f"  ✗ Retrieval: MISS | Expected: {expected_doc} | Got: {retrieved_names}")

        # 2. Evaluate Generation & Hallucination Guardrail
        result = pipeline.answer_question(query, top_k=top_k)
        generated_answer = result["answer"].strip()
        print(f"  Model Response: \"{generated_answer[:90]}...\"")

        if not is_answerable:
            unanswerable_count += 1
            # Check if model correctly refused to answer
            guardrail_phrases = [
                "could not find",
                "cannot find",
                "not present",
                "not provided",
                "no relevant",
                "not mentioned",
            ]
            guarded = any(phrase in generated_answer.lower() for phrase in guardrail_phrases)
            if guarded:
                hallucination_guarded += 1
                print("  ✓ Hallucination Test: PASSED (Properly refused ungrounded query)")
            else:
                print("  ⚠ Hallucination Test: FAILED (Model attempted to hallucinate answer)")

        print("-" * 65)

    # Calculate final scores
    hit_rate = (retrieval_hits / answerable_count) if answerable_count > 0 else 0.0
    mrr = (mrr_sum / answerable_count) if answerable_count > 0 else 0.0
    guard_rate = (hallucination_guarded / unanswerable_count) if unanswerable_count > 0 else 1.0

    print("\n" + "=" * 65)
    print("                    BENCHMARK RESULTS")
    print("=" * 65)
    print(f"Total Test Cases Evaluated : {len(test_cases)}")
    print(f"In-Domain Questions        : {answerable_count}")
    print(f"Out-of-Domain Guard Tests  : {unanswerable_count}")
    print("-" * 65)
    print(f"🎯 Retrieval Hit Rate @ {top_k}  : {hit_rate:.1%}")
    print(f"📊 Mean Reciprocal Rank (MRR): {mrr:.4f}")
    print(f"🛡️ Hallucination Guard Rate : {guard_rate:.1%}")
    print("=" * 65)


if __name__ == "__main__":
    evaluate_rag()