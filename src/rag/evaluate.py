import json
from pathlib import Path

from src.rag.vector_store import VectorStore


ROOT = Path(__file__).resolve().parents[2]

INDEX_DIRECTORY = ROOT / "storage" / "rag"
QUESTIONS_FILE = ROOT / "data" / "test_questions.json"


def main():
    store = VectorStore()
    store.load(INDEX_DIRECTORY)

    with open(QUESTIONS_FILE, "r", encoding="utf-8") as file:
        questions = json.load(file)

    top_k = 4
    hits = 0
    reciprocal_rank_sum = 0

    print("\nRAG Retrieval Evaluation")
    print("=" * 60)

    for item in questions:
        query = item["question"]
        expected_source = item["expected_source"]

        results = store.search(query, top_k=top_k)

        retrieved_sources = [
            result["source"] for result in results
        ]

        rank = None

        for position, source in enumerate(retrieved_sources, start=1):
            if source == expected_source:
                rank = position
                break

        if rank is not None:
            hits += 1
            reciprocal_rank_sum += 1 / rank

        print(f"\nQuestion: {query}")
        print(f"Expected: {expected_source}")
        print(f"Retrieved: {retrieved_sources}")
        print(f"Correct: {'YES' if rank else 'NO'}")

    total = len(questions)

    if total == 0:
        print("No evaluation questions found.")
        return

    hit_rate = hits / total
    mrr = reciprocal_rank_sum / total

    print("\n" + "=" * 60)
    print(f"Total questions: {total}")
    print(f"Correct retrievals: {hits}")
    print(f"Hit Rate@{top_k}: {hit_rate:.2%}")
    print(f"MRR: {mrr:.4f}")


if __name__ == "__main__":
    main()