import os
import json
import threading
from typing import Dict
from functools import lru_cache

# LangChain core
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.chains import LLMMathChain
#from langchain_community.tools import DuckDuckGoSearchResults
from duckduckgo_search import DDGS

from langchain_core.tools import tool
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

# RAG y configuración compartida
from rag import answer_query, CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL

# Vector store para chequeo de relevancia
from langchain_chroma import Chroma
from embeddings import E5Embeddings
from dotenv import load_dotenv
load_dotenv()

_AGENT_RUNNABLE = None
_AGENT_LOCK = threading.Lock()
# ------------------ LLM ---------------------------
@lru_cache(maxsize=1)
def _llm():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model, temperature=0)

# ------------------ Tools externas ------------------
# 1) DuckDuckGo: devuelve lista de resultados (title, link, snippet)
#ddg_tool = DuckDuckGoSearchResults(name="duckduckgo_search", max_results=5)

# 2) Calculator basada en LLM (interpreta expresiones “humanas”)
@lru_cache(maxsize=1)
def _math_chain():
    return LLMMathChain.from_llm(_llm(), verbose=False)

@tool("calculator", return_direct=False)
def calculator(expression: str) -> str:
    """LLM math calculator (e.g., '60000/120', '120*(2/3)')."""
    try:
        return _math_chain().run(expression)
    except Exception as e:
        return f"Calculation error: {e}"


# ------------------ RAG con chequeo de relevancia ------------------
@lru_cache(maxsize=1)
def _vs():
    emb = E5Embeddings(model_name=EMBED_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=emb
    )

from pydantic import BaseModel, Field

class RAGStrictInput(BaseModel):
    query: str = Field(..., description="Pregunta del usuario")
    k: int = Field(5, description="Top-k a recuperar")
    min_relevance: float = Field(0.2, description="Umbral mínimo de relevancia [0..1]")

@tool("ask_rag_strict", args_schema=RAGStrictInput, return_direct=False)
def ask_rag_strict(query: str, k: int = 5, min_relevance: float = 0.2) -> str:
    """
    Intenta responder con el índice (Chroma+E5).
    Si el mejor documento tiene relevancia < min_relevance => devuelve 'NO_CONTEXT'.
    En caso contrario, devuelve la respuesta final (con 'Sources:').
    """
    vs = _vs()
    try:
        docs_scores = vs.similarity_search_with_relevance_scores(query, k=k)
        # docs_scores: List[Tuple[Document, float]] con relevancia en [0..1]
    except Exception:
        # Si la versión de langchain_chroma no soporta relevancia, asumimos 0.5
        docs = vs.similarity_search(query, k=k)
        docs_scores = [(d, 0.5) for d in docs]

    if not docs_scores:
        return "NO_CONTEXT"

    top_score = docs_scores[0][1]
    if top_score is None or top_score < min_relevance:
        return "NO_CONTEXT"

    # Si hay contexto, delega en tu pipeline LCEL para la respuesta con citas
    try:
        result = answer_query(query, k=k)
        return result["answer"]
    except Exception as e:
        return f"No he podido consultar el índice RAG: {e}"

# ------------------ Tool de tempo específica (BPM ↔ ms por figura) ------------------
class TempoInput(BaseModel):
    bpm: float = Field(..., description="Tempo en beats por minuto (BPM)")
    note: str = Field("1/4", description="Figura: 1/1,1/2,1/4,1/8,1/16,1/32")
    feel: str = Field("straight", description="straight | dotted | triplet")

_NOTE_BASE = {
    "1/1": 4.0, "1/2": 2.0, "1/4": 1.0, "1/8": 0.5, "1/16": 0.25, "1/32": 0.125,
}
def _feel_mult(feel: str) -> float:
    f = feel.lower().strip()
    if f == "dotted":   return 1.5
    if f == "triplet":  return 2.0/3.0
    return 1.0

@tool("tempo_calculator", args_schema=TempoInput, return_direct=False)
def tempo_calculator(bpm: float, note: str = "1/4", feel: str = "straight") -> str:
    """
    Convierte BPM a milisegundos por figura (con puntillo o tresillo).
    Devuelve ms por negra, ms por la figura indicada y ms de un compás 4/4.
    """
    if bpm <= 0:
        return "El BPM debe ser > 0"
    if note not in _NOTE_BASE:
        return f"Figura desconocida: {note}. Usa una de: {', '.join(_NOTE_BASE)}"

    ms_black = 60000.0 / bpm
    ms_note = ms_black * _NOTE_BASE[note] * _feel_mult(feel)
    bar_4_4_ms = ms_black * 4.0
    return (
        f"BPM: {bpm}\nFigura: {note} ({feel})\n"
        f"Ms por negra: {ms_black:.2f} ms\n"
        f"Ms por {note} ({feel}): {ms_note:.2f} ms\n"
        f"1 compás 4/4: {bar_4_4_ms:.2f} ms\n"
    )

# ------------------ Tool audio: MIDI ↔ Hz (recomendada para Ableton) ------------------
class PitchInput(BaseModel):
    mode: str = Field(..., description="'midi_to_hz' o 'hz_to_midi'")
    value: str = Field(..., description="Nota MIDI (número 0-127 o nombre como A4/C#3) o frecuencia en Hz")

_A4 = 440.0
_A4_MIDI = 69

_NOTE_TO_SEMITONE = {
    "C":0,"C#":1,"Db":1,"D":2,"D#":3,"Eb":3,"E":4,"F":5,"F#":6,"Gb":6,"G":7,"G#":8,"Ab":8,"A":9,"A#":10,"Bb":10,"B":11
}
def _name_to_midi(name: str) -> int:
    # e.g. "C#3", "A4"
    name = name.strip()
    # separar letra(s) y octava
    i = 1
    if len(name) >= 2 and name[1] in ["#","b"]:
        i = 2
    pitch = name[:i]
    octv = int(name[i:])
    semitone = _NOTE_TO_SEMITONE[pitch]
    # MIDI: C4=60 → C0=12, por lo tanto:
    midi = 12 + semitone + (octv * 12)
    return midi

def _midi_to_hz(m: int) -> float:
    return _A4 * (2 ** ((m - _A4_MIDI)/12.0))

def _hz_to_midi(hz: float) -> float:
    from math import log2
    return _A4_MIDI + 12 * log2(hz/_A4)

@tool("pitch_converter", args_schema=PitchInput, return_direct=False)
def pitch_converter(mode: str, value: str) -> str:
    """
    Convierte entre MIDI y Hz.
    Ejemplos:
      - {'mode':'midi_to_hz','value':'69'} -> 440 Hz
      - {'mode':'midi_to_hz','value':'C#3'} -> 138.59 Hz
      - {'mode':'hz_to_midi','value':'55'}  -> 43 (≈ G1)
    """
    mode = mode.strip().lower()
    try:
        if mode == "midi_to_hz":
            if value.strip().isdigit():
                midi = int(value.strip())
            else:
                midi = _name_to_midi(value.strip())
            hz = _midi_to_hz(midi)
            return f"MIDI {midi} → {hz:.2f} Hz"
        elif mode == "hz_to_midi":
            hz = float(value.strip())
            midi = _hz_to_midi(hz)
            return f"{hz:.2f} Hz → MIDI {midi:.2f}"
        else:
            return "Modo no soportado. Usa 'midi_to_hz' o 'hz_to_midi'."
    except Exception as e:
        return f"Error en conversión: {e}"
@tool("web_search", return_direct=False)
def web_search(query: str, max_results: int = 5) -> str:
    """
    DuckDuckGo web search. Returns up to max_results results as JSON:
    [{title, url, snippet}]
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        simplified = [
            {
                "title": r.get("title"),
                "url": r.get("href") or r.get("link"),
                "snippet": r.get("body"),
            }
            for r in results[:max_results]
        ]
        return json.dumps(simplified, ensure_ascii=False)
    except Exception as e:
        return f"SEARCH_ERROR: {e}"

# --- en el prompt del agente, menciona web_search ---
AGENT_PROMPT = ChatPromptTemplate.from_messages([
    MessagesPlaceholder("agent_scratchpad"),
    ("system",
     "You are Ableton Assistant. Tool policy:\n"
     "1) Always call 'ask_rag_strict' first for Ableton/audio/manuals/indexed videos.\n"
     "2) If it returns 'NO_CONTEXT', use 'web_search'. The tool returns a JSON array of results "
     "with keys 'title', 'url', and 'snippet'. Extract relevant facts and CITE LINKS in 'Sources:'.\n"
     "3) Use 'tempo_calculator' or 'calculator' for timing math.\n"
     "4) Use 'pitch_converter' for tuning and note/frequency conversions.\n"
     "Always include a 'Sources:' section when you used RAG or the web. Add timestamps when available."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

# --- actualiza la lista de tools ---
TOOLS = [ask_rag_strict, web_search, calculator, tempo_calculator, pitch_converter]



def get_agent_runnable():
    """
    Retorna el runnable amb memòria, creant-lo només una vegada (singleton).
    Thread-safe amb un lock per evitar condicions de cursa en arrencada.
    """
    global _AGENT_RUNNABLE
    if _AGENT_RUNNABLE is None:
        with _AGENT_LOCK:
            if _AGENT_RUNNABLE is None:  # double-checked locking
                executor = build_agent()
                runnable = RunnableWithMessageHistory(
                    executor,
                    _get_session_history,
                    input_messages_key="input",
                    history_messages_key="chat_history",
                )
                _AGENT_RUNNABLE = runnable
    return _AGENT_RUNNABLE

def build_agent():
    llm = _llm()
    agent = create_openai_tools_agent(llm, TOOLS, AGENT_PROMPT)
    executor = AgentExecutor(agent=agent, tools=TOOLS, verbose=True, return_intermediate_steps=False)
    return executor

# Memoria por sesión (RAM; persiste mientras el proceso esté vivo)
_CHAT_STORE: Dict[str, InMemoryChatMessageHistory] = {}

def _get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _CHAT_STORE:
        _CHAT_STORE[session_id] = InMemoryChatMessageHistory()
    return _CHAT_STORE[session_id]

def build_agent_with_memory():
    executor = build_agent()
    runnable = RunnableWithMessageHistory(
        executor,
        _get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    return runnable

# ------------------ API sencilla ------------------
def agent_ask(user_input: str, session_id: str = "default") -> str:
    """
    Llama al agente con memoria. 'session_id' separa conversaciones paralelas.
    """
    runnable = get_agent_runnable()
    cfg = {"configurable": {"session_id": session_id}}
    out = runnable.invoke({"input": user_input}, config=cfg)
    return out["output"]

def reset_agent():
    global _AGENT_RUNNABLE
    with _AGENT_LOCK:
        _AGENT_RUNNABLE = None