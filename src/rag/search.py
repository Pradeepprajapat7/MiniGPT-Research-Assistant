# semantic search 


from pathlib import Path

from src.rag.vector_store import VectorStore


ROOT = Path(__file__).resolve().parents[2]
INDEX_DIRECTORY = ROOT / "storage" / "rag"


def main():
    store = VectorStore()
    store.load(INDEX_DIRECTORY)

    print("RAG Semantic Search")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Ask a question: ").strip()

        if query.lower() in {"exit", "quit"}:
            break

        results = store.search(query, top_k=4)

        if not results:
            print("No matching passages found.\n")
            continue

        for number, result in enumerate(results, start=1):
            print(f"\nResult {number}")
            print(f"Source: {result['source']}")
            print(f"Page: {result['page']}")
            print(f"Similarity: {result['score']:.4f}")
            print(f"Text: {result['text']}")
            print("-" * 60)


if __name__ == "__main__":
    main()
