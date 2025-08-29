# embeddings.py
import os
import torch
from typing import List
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")

class E5Embeddings(Embeddings):
    """
    Wrapper per al model e5 multilingüe, amb els prefixos 'passage:' i 'query:' i normalització.
    """
    def __init__(self, model_name: str = EMBED_MODEL, normalize: bool = True, device: str | None = None):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        try:
            if self.device == "cuda":
                self.model = self.model.to(torch.float16)
                torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        self.normalize = normalize

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = [f"passage: {t}" for t in texts]
        vecs = self.model.encode(texts, normalize_embeddings=self.normalize, show_progress_bar=False)
        return [v.tolist() for v in vecs]

    def embed_query(self, text: str) -> List[float]:
        text = f"query: {text}"
        vec = self.model.encode([text], normalize_embeddings=self.normalize, show_progress_bar=False)[0]
        return vec.tolist()
