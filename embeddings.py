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



# ----------------- Embeddings e5 -----------------
# classe nova que hereta d'una interficie de LC anomenada Embeddings 
# """ LC espera un obj amb dues funcions claus: 
#     - embed_documents: entra list string surt list de vectors embeddeds
# """
# class E5Embeddings(Embeddings):
#     """
#     e5 multilingüe amb 'query:' / 'passage:' i normalització.
#     """
#     def __init__(self, model_name: str = EMBED_MODEL, normalize: bool = True, device: str | None = None):
#         # instanciem la classe (creem l'obj self) amb el model de embeding
#         # es important carregar el model al constructor aixi nomes es carrega una vegada 
#         # normalize: el model treu vectors i els normalitzem amb longitud 1 aixi la similitud cosinus es mes fiable
#         self.model = SentenceTransformer(model_name, device=device)
#         """
#         normalitzar evita que el “volum” del vector afecti el resultat. Ens quedem només amb la direcció, que és el que ens interessa per buscar documents semblants."""
#         self.normalize = normalize

#     # embeddings per documents (la meva base de coneixement)
#     # input: lista de textos(chunk) -> afegim el prefix passage davant de cada text. el model e5 va ser entrenat aixi
#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         texts = [f"passage: {t}" for t in texts]
#         # Model transforma cada text en un vector de nombres 
#         # Per al model multilingual-e5-base, el vector sol tenir 768 dimensions (768 nombres).
#         # normalize_embeddings=self.normalize fa la normalització que hem comentat.
#         vecs = self.model.encode(texts, normalize_embeddings=self.normalize, show_progress_bar=False)
#         # resultat es una llista de llista de vectors 
#         return [v.tolist() for v in vecs]

#     # embeddings per (consulta de l'usuari)
#     def embed_query(self, text: str) -> List[float]:
#         # afegim query coma prefix davant la pregunta ja que aixi ho requereix el model
#         text = f"query: {text}"
#         # tot i ser un sol text el model rep i retorna una llista per aixo fem [0] per treure el primer vector
#         vec = self.model.encode([text], normalize_embeddings=self.normalize, show_progress_bar=False)[0]
#         # retorna una llista amb un vector 
#         return vec.tolist()