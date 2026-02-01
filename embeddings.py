# embeddings.py
import os
import logging
import torch
from typing import List
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")


class E5Embeddings(Embeddings):
    """
    Wrapper for E5 multilingual model with 'passage:' and 'query:' prefixes and normalization.
    """

    def __init__(self, model_name: str = EMBED_MODEL, normalize: bool = True, device: str | None = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.normalize = normalize

        logger.info(f"Loading embedding model: {model_name} on {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)

        # Optimize for GPU if available
        if self.device == "cuda":
            try:
                self.model = self.model.to(torch.float16)
                torch.set_float32_matmul_precision("high")
                logger.info("GPU optimization enabled (float16)")
            except Exception as e:
                logger.warning(f"GPU optimization failed: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents with 'passage:' prefix."""
        texts = [f"passage: {t}" for t in texts]
        vecs = self.model.encode(texts, normalize_embeddings=self.normalize, show_progress_bar=False)
        return [v.tolist() for v in vecs]

    def embed_query(self, text: str) -> List[float]:
        """Embed query with 'query:' prefix."""
        text = f"query: {text}"
        vec = self.model.encode([text], normalize_embeddings=self.normalize, show_progress_bar=False)[0]
        return vec.tolist()
