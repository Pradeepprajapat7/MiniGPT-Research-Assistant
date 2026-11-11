# Build the FAISS vector store


import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class VectorStore:

    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ):
        # CPU keeps GPU memory available for language models.
        self.model = SentenceTransformer(
            model_name,
            device="cpu"
        )
        self.index = None
        self.metadata = []

    def build(self, chunks):
        if not chunks:
            raise ValueError("No chunks provided.")

        texts = [chunk["text"] for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32
        )

        dimension = embeddings.shape[1]

        # Inner product on normalized vectors = cosine similarity.
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        self.metadata = chunks

        return self.index.ntotal

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        if self.index is None:
            raise ValueError("Build the index before saving.")

        faiss.write_index(
            self.index,
            str(directory / "faiss.index")
        )

        with open(
            directory / "metadata.json",
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                self.metadata,
                file,
                ensure_ascii=False,
                indent=2
            )

    def load(self, directory):
        directory = Path(directory)

        self.index = faiss.read_index(
            str(directory / "faiss.index")
        )

        with open(
            directory / "metadata.json",
            "r",
            encoding="utf-8"
        ) as file:
            self.metadata = json.load(file)

    def search(self, query, top_k=4):
        if self.index is None:
            raise ValueError("Load or build an index first.")

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                "FAISS index and metadata size do not match."
            )

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32
        )

        k = min(top_k, self.index.ntotal)

        if k == 0:
            return []

        scores, indices = self.index.search(query_embedding, k)

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            result = self.metadata[idx].copy()
            result["score"] = float(score)
            results.append(result)

        return results

        for score, index in zip(scores[0], indices[0]):
            if index == -1:
                continue

            result = self.metadata[index].copy()
            result["score"] = float(score)
            results.append(result)

        return results
