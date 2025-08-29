# rag.py
import os 
from typing import List, Dict, Any 
from functools import lru_cache

from langchain_chroma import Chroma
from langchain.docstore.document import Document 
from langchain_openai import ChatOpenAI

from embeddings import E5Embeddings 

from langsmith import traceable

from langchain.prompts import ChatPromptTemplate 
from langchain.schema.output_parser import StrOutputParser 
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda


CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "ableton")
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")

_LS_PROJECT = os.getenv("LANGCHAIN_PROJECT", "ableton-assistant-dev")

SYSTEM_PROMPT = """You are Ableton Assistant. Answer using ONLY the provided context snippets.
If the answer is not in the context, say you don't have enough info and suggest searching the web.
Be concise and technical-but-friendly. Always include a 'Sources' list with URLs and timestamps if videos.
"""

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
            "start": start,
            "end": meta.get("end"),
            "timestamp": _ts(start),
        }
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

def _format_docs(docs: List[Document]) -> str:
    lines = []
    for d in docs:
        meta = d.metadata or {}
        src = _source_from_meta(meta)
        url = src.get("url", "")
        excerpt = (d.page_content or "").strip().replace("\n", " ")
        excerpt = excerpt[:400] + ("..." if len(excerpt) > 400 else "")
        lines.append(f"- {url}\n  {excerpt}")
    return "\n".join(lines)

@lru_cache(maxsize=1)
def _vectorstore():
    emb = E5Embeddings(model_name=EMBED_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=emb,
    )

@traceable(name="load_retriever")
def load_retriever(k: int = 5):
    return _vectorstore().as_retriever(search_kwargs={"k": k})

@lru_cache(maxsize=1)
@traceable(name="pick_llm_openai")
def pick_llm():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set."
        )
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model_name, temperature=0)

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human",
     "User question:\n{question}\n\n"
     "Context:\n{context}\n\n"
     "Final answer (include 'Sources:' lines at the end):")
])

def build_chain(llm, retriever):
    return (
        {
            "question": RunnablePassthrough(),
            "context": retriever | RunnableLambda(_format_docs),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

@traceable(name="answer_query")
def answer_query(query: str, k: int = 5) -> Dict[str, Any]:
    run_config = {
        "tags": ["ableton", "rag", "lcel", "openai-only"],
        "metadata": {"k": k, "project": _LS_PROJECT},
        "run_name": "RAG-Answer"
    }

    retriever = load_retriever(k=k)
    llm = pick_llm()
    chain = build_chain(llm, retriever)

    answer = chain.invoke(query, config=run_config)

    docs: List[Document] = retriever.invoke(query, config=run_config)
    sources = _dedupe_sources([_source_from_meta(d.metadata or {}) for d in docs])

    if not docs:
        answer = (
            "I don't have enough info in the local index for that.\n"
            "Consider searching the web.\n\nSources:\n"
        )
        return {"answer": answer.strip(), "sources": []}
    # Garanteix 'Sources:' si l'LLM se n'oblida
    if "Sources:" not in answer:
        answer += "\n\nSources:\n" + "\n".join(
            f"• {s['url']}" if s.get("type") == "manual"
            else f"• {s['url']} [{s['timestamp']}]"
            for s in sources
        )

    return {"answer": answer.strip(), "sources": sources}


def stream_answer(query: str, k: int = 5):
    retriever = load_retriever(k=k)
    llm = pick_llm()
    chain = build_chain(llm, retriever)
    yield from chain.stream(query)
