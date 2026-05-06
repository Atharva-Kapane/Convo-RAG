import faiss
import numpy as np


def build_index(texts, embedder):
    embeddings = embedder.embed_texts(texts)
    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]

    # Normalize → better cosine similarity via L2
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(dim)  # cosine similarity
    index.add(embeddings)

    return index, embeddings