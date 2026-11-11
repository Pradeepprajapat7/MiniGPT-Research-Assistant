
def create_chunks(
    documents,
    chunk_size=250,
    overlap=50
):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size")

    chunks = []
    step = chunk_size - overlap

    for document in documents:
        words = document["text"].split()

        for start in range(0, len(words), step):
            chunk_words = words[start:start + chunk_size]

            if not chunk_words:
                continue

            chunks.append({
                "text": " ".join(chunk_words),
                "source": document["source"],
                "page": document["page"]
            })

            if start + chunk_size >= len(words):
                break

    return chunks
