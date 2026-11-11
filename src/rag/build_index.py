
from pathlib import Path

from src.rag.pdf_loader import load_pdf
from src.rag.chunker import create_chunks
from src.rag.vector_store import VectorStore


ROOT = Path(__file__).resolve().parents[2]

PDF_DIRECTORY = ROOT / "data" / "documents"
INDEX_DIRECTORY = ROOT / "storage" / "rag"


def main():
    pdf_files = list(PDF_DIRECTORY.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDFs found in {PDF_DIRECTORY}"
        )

    all_documents = []

    for pdf_file in pdf_files:
        print(f"\nReading: {pdf_file.name}")
        all_documents.extend(load_pdf(pdf_file))

    if not all_documents:
        raise ValueError(
            "No valid PDF documents found. "
            "Please add non-empty PDFs with extractable text."
        )

    print("Extracted pages:", len(all_documents))

    chunks = create_chunks(
        all_documents,
        chunk_size=250,
        overlap=50
    )

    print("Created chunks:", len(chunks))

    store = VectorStore()

    total = store.build(chunks)
    store.save(INDEX_DIRECTORY)

    print("\nIndexing completed!")
    print("FAISS vectors:", total)
    print("Saved to:", INDEX_DIRECTORY)


if __name__ == "__main__":
    main()
