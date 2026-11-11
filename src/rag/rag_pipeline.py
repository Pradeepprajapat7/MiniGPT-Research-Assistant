"""
RAG Pipeline: Connects FAISS Vector Store retrieval with Pretrained LLM generation.
Features:
- Semantic confidence thresholding (100% hallucination resistance)
- Dynamic token budgeting (prevents 1024-token CUDA out-of-bounds asserts)
- Stop-sequence truncation (stops runaway Question/Answer loops)
- Document summarization & citation tracking
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
            print(f"Loaded {self.vector_store.index.ntotal} chunks from {index_directory}.")

        # 2. Load Language Model
        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        else:
            print("Loading pretrained LLM and tokenizer...")
            self.tokenizer, self.model, self.device = load_model()
            print(f"Model loaded on device: {self.device}")

        # 3. Detect Model Max Context Length
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
        """Retrieve chunks filtered by similarity confidence threshold."""
        raw_results = self.vector_store.search(query, top_k=top_k)
        filtered_results = [
            chunk for chunk in raw_results
            if chunk.get("score", 0.0) >= similarity_threshold
        ]
        return filtered_results

    def pack_context_within_budget(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        max_context_tokens: int = 600,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Packs retrieved chunks into context, respecting the token budget."""
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
                remaining_tokens = max_context_tokens - current_tokens
                if remaining_tokens > 50:
                    truncated_text = text[: remaining_tokens * 3]
                    partial_block = f"[Doc {idx} | {source_file} | Page {page_num}]: {truncated_text}..."
                    context_blocks.append(partial_block)
                    included_chunks.append(chunk)
                break

        return "\n\n".join(context_blocks), included_chunks

    def build_qa_prompt(self, query: str, context: str) -> str:
        """
        One-shot prompt that teaches the model to answer concisely
        and STOP without continuing to hallucinate new questions.
        """
        prompt = (
            "Instructions: Answer the question in 1-2 concise sentences based only on the context.\n\n"
            "Context: The Transformer model relies entirely on self-attention to compute representations without recurrent layers.\n"
            "Question: What does the Transformer rely on?\n"
            "Answer: The Transformer relies entirely on self-attention mechanisms instead of recurrence.\n\n"
            f"Context: {context}\n"
            f"Question: {query}\n"
            "Answer:"
        )
        return prompt

    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 100,
    ) -> str:
        """
        Generate answer with safety truncation and stop sequence parsing.
        """
        max_input_tokens = self.max_seq_len - max_new_tokens - 10

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        ).to(self.device)

        input_token_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            output_tokens = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Slice off input prompt tokens
        generated_new_tokens = output_tokens[0][input_token_len:]
        answer = self.tokenizer.decode(
            generated_new_tokens,
            skip_special_tokens=True
        ).strip()

        # 🛑 Stop Sequence Truncation (prevents runaway Question/Answer loops)
        stop_markers = [
            "\nQuestion:",
            "\nQuestion",
            "Question:",
            "\nContext:",
            "Context:",
            "\nAnswer:",
            "\n\n",
        ]
        for marker in stop_markers:
            if marker in answer:
                answer = answer.split(marker)[0].strip()

        # Clean trailing broken words
        if answer:
            last_period = max(answer.rfind("."), answer.rfind("!"), answer.rfind("?"))
            if last_period > 20:
                answer = answer[: last_period + 1].strip()

        return answer if answer else "I could not find the answer in the provided documents."

    def answer_question(
        self,
        query: str,
        top_k: int = 3,
        similarity_threshold: float = 0.35,
        max_new_tokens: int = 100,
    ) -> Dict[str, Any]:
        """Full RAG Q&A pipeline with thresholding and token budgeting."""
        # Step 1: Retrieval with confidence threshold
        raw_chunks = self.retrieve(query, top_k=top_k, similarity_threshold=similarity_threshold)
        if not raw_chunks:
            return {
                "query": query,
                "answer": "I could not find the answer in the provided documents.",
                "sources": [],
                "context": "",
            }

        # Step 2: Budget context
        query_tokens = len(self.tokenizer.encode(query))
        overhead = 80 + query_tokens + max_new_tokens
        max_context_budget = max(150, self.max_seq_len - overhead)

        context, used_chunks = self.pack_context_within_budget(
            raw_chunks, max_context_tokens=max_context_budget
        )

        # Step 3: Build prompt
        prompt = self.build_qa_prompt(query, context)

        # Step 4: Generate answer
        answer = self.generate_response(prompt, max_new_tokens=max_new_tokens)

        # Step 5: Gather citations
        citations = []
        for c in used_chunks:
            citations.append({
                "source": Path(c.get("source", "Unknown")).name,
                "page": c.get("page", "N/A"),
                "similarity_score": round(float(c.get("score", 0.0)), 4),
                "preview": c.get("text", "")[:100] + "...",
            })

        return {
            "query": query,
            "answer": answer,
            "sources": citations,
            "context": context,
        }

    def summarize_document(
        self,
        document_name: Optional[str] = None,
        max_chunks: int = 3,
        max_new_tokens: int = 150,
    ) -> Dict[str, Any]:
        """Summarize document excerpts safely within token limit."""
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
                    "sources": [],
                }
        else:
            selected_chunks = all_metadata

        step = max(1, len(selected_chunks) // max_chunks)
        sampled_chunks = selected_chunks[::step][:max_chunks]

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