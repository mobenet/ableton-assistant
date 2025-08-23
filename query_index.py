# query_index.py
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "ableton"
EMBED_MODEL = "intfloat/multilingual-e5-base"

class E5Embeddings(Embeddings):
    def __init__(self, model_name: str = EMBED_MODEL, normalize: bool = True, device: str | None = None):
        self.model = SentenceTransformer(model_name, device=device)
        self.normalize = normalize
    def embed_documents(self, texts):
        texts = [f"passage: {t}" for t in texts]
        vecs = self.model.encode(texts, normalize_embeddings=self.normalize, show_progress_bar=False)
        return [v.tolist() for v in vecs]
    def embed_query(self, text):
        vec = self.model.encode([f"query: {text}"], normalize_embeddings=self.normalize, show_progress_bar=False)[0]
        return vec.tolist()

def seconds_to_ts(s):
    # converteix segons a "XmYs"
    s = int(s)
    m, sec = divmod(s, 60)
    return f"{m}m{sec}s" if m else f"{sec}s"

if __name__ == "__main__":
    emb = E5Embeddings(model_name=EMBED_MODEL)
    vs = Chroma(collection_name=COLLECTION_NAME, persist_directory=CHROMA_DIR, embedding_function=emb)
    retriever = vs.as_retriever(search_kwargs={"k": 5})

    query = "How do I quantize audio in Ableton?"
    docs = retriever.invoke(query)

    print(f"\nTop {len(docs)} hits for: {query}\n")
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        src = meta.get("source", "")
        if meta.get("type") == "video":
            start = meta.get("start", 0)
            ts = seconds_to_ts(start)
            if "watch?v=" in src:
                src = f"{src}&t={int(start)}s"
            print(f"{i}. [VIDEO] {src}  [{ts}]")
        elif meta.get("type") == "manual":
            print(f"{i}. [MANUAL] {src}")
        print(d.page_content[:220].strip(), "...\n")
