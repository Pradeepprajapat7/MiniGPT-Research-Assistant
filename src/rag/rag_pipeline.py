"""
RAG Pipeline: Connects FAISS Vector Store retrieval with Pretrained LLM generation.
Includes Token Budgeting to prevent CUDA out-of-bounds errors on 1024-token models (like GPT-2).
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# Ensure the project root is in sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pretrained_model import load_model
from src.rag.vector_store import VectorStore

INDEX_DIRECTORY = ROOT / "storage" / "rag"


class RAGPipeline:
    def __init__(
        self,
        index_directory: Path = INDEX_DIRECTORY,
        vector_store: Optional[VectorStore] = None,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        device: Optional[Any] = None,
    ):
        # 1. Load Vector Store
        if vector_store is not None:
            self.vector_store = vector_store
        else:
            print("Loading FAISS vector store...")
            self.vector_store = VectorStore()
            self.vector_store.load(index_directory)
            index = self.vector_store.index
            chunk_count = index.ntotal if index is not None else 0
            print(f"Loaded {chunk_count} chunks from {index_directory}.")

        # 2. Load Language Model
        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        else:
            print("Loading pretrained LLM and tokenizer...")
            self.tokenizer, self.model, self.device = load_model()
            print(f"Model loaded on device: {self.device}")

        # 3. Detect Model Max Sequence Length (Default 1024 for GPT-2)
        self.max_seq_len = getattr(
            self.model.config, "n_positions",
            getattr(self.model.config, "max_position_embeddings", 1024)
        )
        print(f"Model context limit: {self.max_seq_len} tokens")

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        similarity_threshold: float = 0.35,
    ) -> List[Dict[str, Any]]:
        """Retrieve chunks filtered by a minimum similarity threshold."""
        raw_results = self.vector_store.search(query, top_k=top_k)
        # Filter out low-confidence matches (out-of-domain queries)
        filtered_results = [
            chunk for chunk in raw_results
            if chunk.get("score", 0.0) >= similarity_threshold
        ]
        return filtered_results

    def pack_context_within_budget(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        max_context_tokens: int = 650,
    ) -> tuple[str, List[Dict[str, Any]]]:
        """
        Packs retrieved chunks into context, respecting the token budget.
        Drops or truncates chunks that would exceed the model's window.
        """
        included_chunks = []
        context_blocks = []
        current_tokens = 0

        for idx, chunk in enumerate(retrieved_chunks, start=1):
            source_file = Path(chunk.get("source", "Unknown")).name
            page_num = chunk.get("page", "N/A")
            text = chunk.get("text", "").strip()

            block = f"[Doc {idx} | {source_file} | Page {page_num}]: {text}"
            block_tokens = len(self.tokenizer.encode(block))

            if current_tokens + block_tokens <= max_context_tokens:
                context_blocks.append(block)
                included_chunks.append(chunk)
                current_tokens += block_tokens
            else:
                # If even 1 chunk is too big, take a partial slice of it
                remaining_tokens = max_context_tokens - current_tokens
                if remaining_tokens > 60:
                    truncated_text = text[: remaining_tokens * 3]
                    partial_block = f"[Doc {idx} | {source_file} | Page {page_num}]: {truncated_text}..."
                    context_blocks.append(partial_block)
                    included_chunks.append(chunk)
                break

        return "\n\n".join(context_blocks), included_chunks

    def build_qa_prompt(self, query: str, context: str) -> str:
        """Concise prompt format optimized for small context windows."""
        prompt = (
            "Answer the question accurately based ONLY on the context below. "
            "If the answer is not present, reply 'I could not find the answer in the documents.'\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )
        return prompt

    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 120,
    ) -> str:
        """
        Generate answer with safety truncation to ensure no CUDA out-of-bounds asserts.
        """
        # Leave headroom for generated tokens and special tokens
        max_input_tokens = max(32, self.max_seq_len - max_new_tokens - 10)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        ).to(self.device)

        input_token_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            output_tokens = getattr(self.model, "generate")(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Slice off prompt tokens to return ONLY the model's new answer
        generated_new_tokens = output_tokens[0][input_token_len:]
        answer = self.tokenizer.decode(
            generated_new_tokens,
            skip_special_tokens=True,
        ).strip()

        return answer

    def answer_question(
        self,
        query: str,
        top_k: int = 3,
        similarity_threshold: float = 0.35,
        max_new_tokens: int = 120,
    ) -> Dict[str, Any]:
        # Step 1: Retrieval with confidence threshold
        raw_chunks = self.retrieve(query, top_k=top_k, similarity_threshold=similarity_threshold)
        if not raw_chunks:
            return {
                "query": query,
                "answer": "I could not find the answer in the provided documents.",
                "sources": [],
                "context": "",
            }

        # Step 2: Pack the best chunks into the model's context window
        context, used_chunks = self.pack_context_within_budget(
            raw_chunks,
            max_context_tokens=650,
        )

        # Step 3: Build a QA prompt and generate an answer
        prompt = self.build_qa_prompt(query, context)
        answer = self.generate_response(prompt, max_new_tokens=max_new_tokens)

        sources = []
        for chunk in used_chunks:
            text = chunk.get("text", "").strip()
            preview = text[:180] + ("..." if len(text) > 180 else "")
            sources.append(
                {
                    "source": Path(chunk.get("source", "Unknown")).name,
                    "page": chunk.get("page", "N/A"),
                    "similarity_score": chunk.get("score", 0.0),
                    "preview": preview,
                }
            )

        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "context": context,
        }

    def summarize_document(
        self,
        document_name: Optional[str] = None,
        max_chunks: int = 3,
        max_new_tokens: int = 150,
    ) -> Dict[str, Any]:
        """Summarize excerpts while strictly respecting the 1024 token limit."""
        all_metadata = self.vector_store.metadata

        if document_name:
            selected_chunks = [
                m for m in all_metadata
                if document_name.lower() in Path(m.get("source", "")).name.lower()
            ]
            if not selected_chunks:
                return {
                    "document": document_name,
                    "summary": f"No chunks found for document matching '{document_name}'.",
                    "chunks_used": 0,
                }
        else:
            selected_chunks = all_metadata

        step = max(1, len(selected_chunks) // max_chunks)
        sampled_chunks = selected_chunks[::step][:max_chunks]

        # Pack within 600 tokens
        context, used = self.pack_context_within_budget(sampled_chunks, max_context_tokens=600)

        prompt = (
            "Summarize the key findings from these document excerpts in 2-3 concise sentences:\n\n"
            f"{context}\n\n"
            "Summary:"
        )

        summary = self.generate_response(prompt, max_new_tokens=max_new_tokens)

        return {
            "document": document_name or "All Indexed Documents",
            "summary": summary,
            "chunks_used": len(used),
            "sources": [
                {"source": Path(c.get("source", "")).name, "page": c.get("page", "N/A")}
                for c in used
            ],
        }


def main():
    print("=" * 65)
    print("      MiniGPT Research Assistant - Day 5 RAG Pipeline")
    print("=" * 65)

    pipeline = RAGPipeline()

    print("\nCommands:")
    print("  - Ask any question directly.")
    print("  - Type 'summarize' or 'summarize <pdf_name>' for document summary.")
    print("  - Type 'exit' or 'quit' to close.\n")

    while True:
        try:
            user_input = input("Research Assistant > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        # Summarization command
        if user_input.lower().startswith("summarize"):
            parts = user_input.split(maxsplit=1)
            doc_name = parts[1] if len(parts) > 1 else None
            print(f"\nGenerating summary for: {doc_name or 'primary documents'}...")
            result = pipeline.summarize_document(document_name=doc_name)
            print("\n" + "=" * 50)
            print(f"SUMMARY: {result['document']}")
            print("=" * 50)
            print(result["summary"])
            print(f"\n(Based on {result['chunks_used']} document chunks)")
            print("=" * 50 + "\n")
            continue

        # Regular Q&A
        print("\nRetrieving knowledge and generating answer...")
        response = pipeline.answer_question(user_input, top_k=2)

        print("\n" + "=" * 60)
        print("ANSWER:")
        print("=" * 60)
        print(response["answer"])

        print("\n" + "-" * 60)
        print("CITATIONS & SOURCES:")
        print("-" * 60)
        for i, src in enumerate(response["sources"], start=1):
            print(f"[{i}] {src['source']} (Page {src['page']}) | Similarity: {src['similarity_score']}")
            print(f"    Excerpt: {src['preview']}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()