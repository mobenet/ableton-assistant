# rag_chain.py
import os
from typing import List, Dict, Any

from langchain_chroma import Chroma
from langchain.docstore.document import Document

# LLMs
from langchain_openai import ChatOpenAI  # si tens OPENAI_API_KEY
from langchain_ollama import ChatOllama  # si tens Ollama local

# Embeddings (separat a embeddings.py)
from embeddings import E5Embeddings

from langsmith import traceable


CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "ableton")
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")

# LangSmith project (ja activat via .env)
_LS_PROJECT = os.getenv("LANGCHAIN_PROJECT", "ableton-assistant-dev")

SYSTEM_PROMPT = """You are Ableton Assistant. Answer using ONLY the provided context snippets.
If the answer is not in the context, say you don't have enough info and suggest searching the web.
Be concise and technical-but-friendly. Always include a 'Sources' list with URLs and timestamps if videos.
"""

# ----------------- Utils per mostrar fonts -----------------
def _ts(seconds: float | int) -> str:
    s = int(seconds)
    m, sec = divmod(s, 60)
    return f"{m}m{sec}s" if m else f"{sec}s"

def _source_from_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    typ = meta.get("type")
    if typ == "video":
        vid = meta.get("video_id")
        start = int(meta.get("start", 0))
        url = f"https://www.youtube.com/watch?v={vid}&t={start}s"
        return {
            "type": "video",
            "url": url,
            "video_id": vid,
            "start": meta.get("start"),
            "end": meta.get("end"),
            "timestamp": _ts(start),
        }
    # manual
    return {"type": "manual", "url": meta.get("source", "")}

def _dedupe_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for s in sources:
        key = (s.get("url"), s.get("timestamp"))
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out

@traceable(name="build_context_block")
def build_context_block(docs: List[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        src = _source_from_meta(meta)
        src_line = src.get("url", "")
        blocks.append(f"[{i}] {src_line}\n{d.page_content}")
    return "\n\n".join(blocks)

# ----------------- Carrega retriever + LLM -----------------
@traceable(name="load_retriever")
def load_retriever(k: int = 5):
    emb = E5Embeddings(model_name=EMBED_MODEL)
    vs = Chroma(collection_name=COLLECTION_NAME, 
                persist_directory=CHROMA_DIR, 
                embedding_function=emb)
    return vs.as_retriever(search_kwargs={"k": k})

@traceable(name="pick_llm")
def pick_llm():
    prefer_ollama = os.getenv("PREFER_OLLAMA", "true").lower() == "true"
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    try:
        return ChatOllama(model="llama3.1", temperature=0)
    except Exception:
        return None

# ----------------- Funció principal de resposta -----------------
@traceable(name="answer_query")
def answer_query(query: str, k: int = 5) -> Dict[str, Any]:
    """
    Flux:
        1) load_retriever -> recupera top-k
        2) build_context_block -> concatena context 
        3) pick_llm
        4) retorna answer + sources
    Tot traçat a LangSmith
    """

    run_config = {
        "tags": ["ableton", "rag", "no-rerank"],
        "metadata": {"k": k, "project": _LS_PROJECT},
        "run_name":"RAG-Answer"
    }

    retriever = load_retriever(k=k)

    # Recuperació vectorial simple
    docs: List[Document] = retriever.invoke(query, config=run_config)

    sources = _dedupe_sources([_source_from_meta(d.metadata or {}) for d in docs])
    context_text = build_context_block(docs)

    llm = pick_llm()
    if llm is None:
        # Fallback extractiu (sense model generatiu)
        snippet = "\n\n".join(d.page_content[:700] for d in docs)
        answer = (
            "I don't have a generative model configured. Here's a concise extract from the most relevant materials:\n\n"
            + snippet
            + "\n\nSources:\n"
            + "\n".join(
                f"• {s['url']}" if s.get("type") == "manual" else f"• {s['url']} [{s['timestamp']}]"
                for s in sources
            )
        )
        return {"answer": answer, "sources": sources}

    prompt = (
        SYSTEM_PROMPT
        + "\n\nUser question:\n"
        + query
        + "\n\nContext:\n"
        + context_text
        + "\n\nFinal answer (include 'Sources:' lines at the end):"
    )

    msg = llm.invoke(prompt, config=run_config)
    answer = msg.content.strip()

    # Assegura fonts al final si el model s’oblida
    if "Sources:" not in answer:
        answer += "\n\nSources:\n" + "\n".join(
            f"• {s['url']}" if s.get("type") == "manual" else f"• {s['url']} [{s['timestamp']}]"
            for s in sources
        )

    return {"answer": answer, "sources": sources}
