from fastembed import TextEmbedding

class Embedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        # This downloads a tiny (~90MB) ONNX version of the model 
        # instead of the massive PyTorch version.
        self.model = TextEmbedding(model_name)

    def embed_texts(self, texts):
        # fastembed.embed returns a generator, so we convert to a list
        # to maintain compatibility with FAISS logic.
        return list(self.model.embed(texts))

    def embed_single(self, text):
        # Same as above, but just for one string
        return list(self.model.embed([text]))[0]
