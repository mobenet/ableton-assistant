import os
import json
import threading 
from typing import Dict
from functools import lru_cache 

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.chains import LLMMathChain
from ddgs import DDGS


from langchain_core.tools import tool
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

from rag import answer_query, CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL

from langchain_chroma import Chroma
from embeddings import E5Embeddings
from dotenv import load_dotenv
load_dotenv()

_AGENT_RUNNABLE = None 
_AGENT_LOCK = threading.Lock() 


@lru_cache(maxsize=1)
def _llm():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model, temperature=0)


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
    query: str = Field(..., description="User Question")
    k: int = Field(5, description="Top-k to recuperate")
    min_relevance: float = Field(0.2, description="Minimum relevance threshold [0..1]")

@tool("ask_rag_strict", args_schema=RAGStrictInput, return_direct=False)
def ask_rag_strict(query: str, k: int = 5, min_relevance: float = 0.2) -> str:
    """
    Intenta responder con el índice (Chroma+E5).
    Si el mejor documento tiene relevancia < min_relevance => devuelve 'NO_CONTEXT'.
    En caso contrario, devuelve la respuesta final (con 'Sources:').
    """
    vs = _vs()
    docs_scores = vs.similarity_search_with_relevance_scores(query, k=k) 
    
    if not docs_scores:
        return "NO_CONTEXT"

    top_score = docs_scores[0][1] or 0.0
    if top_score is None or top_score < min_relevance: 
        return "NO_CONTEXT"

    try:
        result = answer_query(query, k=k)
        return result["answer"]
    except Exception as e:
        return f"I could'nt consult the RAG index: {e}"

class TempoInput(BaseModel):
    bpm: float = Field(..., description="Tempo in beats per minute (BPM)")
    note: str = Field("1/4", description="Figure: 1/1,1/2,1/4,1/8,1/16,1/32")
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
    Convert BPMs to milliseconds per figure (with dotted or triplet).
    Returns ms per black, ms per indicated figure and ms of a compas 4/4
    """
    if bpm <= 0:
        return "BPM must be > 0"
    if note not in _NOTE_BASE:
        return f"Unknown figure: {note}. Use one of: {', '.join(_NOTE_BASE)}"

    ms_black = 60000.0 / bpm 
    ms_note = ms_black * _NOTE_BASE[note] * _feel_mult(feel)
    bar_4_4_ms = ms_black * 4.0
    return (
        f"BPM: {bpm}\nFigura: {note} ({feel})\n"
        f"Ms por negra: {ms_black:.2f} ms\n"
        f"Ms por {note} ({feel}): {ms_note:.2f} ms\n"
        f"1 compás 4/4: {bar_4_4_ms:.2f} ms\n"
    )

class PitchInput(BaseModel):
    mode: str = Field(..., description="'midi_to_hz' or 'hz_to_midi'")
    value: str = Field(..., description="MIDI note (number 0-127 or name as A4/C#3) or frequency in Hz")

_A4 = 440.0
_A4_MIDI = 69

_NOTE_TO_SEMITONE = {
    "C":0,"C#":1,"Db":1,"D":2,"D#":3,"Eb":3,"E":4,"F":5,"F#":6,"Gb":6,"G":7,"G#":8,"Ab":8,"A":9,"A#":10,"Bb":10,"B":11
}
def _name_to_midi(name: str) -> int:
    name = name.strip()
    i = 1
    if len(name) >= 2 and name[1] in ["#","b"]:
        i = 2
    pitch = name[:i]
    octv = int(name[i:])
    semitone = _NOTE_TO_SEMITONE[pitch]
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
    Converts between MIDI and Hz
    Examples:
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
            return "Non supported mode. Use 'midi_to_hz' or 'hz_to_midi'."
    except Exception as e:
        return f"Conversion error: {e}"
    
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


AGENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are Ableton Assistant.\n"
     "Tool policy:\n"
     "1) Always call 'ask_rag_strict' first for Ableton/audio/manuals/indexed videos..\n"
     "2) If it returns 'NO_CONTEXT', call 'web_search'. That tool returns a JSON string; "
     "   parse the JSON, extract relevant facts, and CITE LINKS under 'Sources:'\n"
     "3) For any BPM/tempo/note-length math, ALWAYS use 'tempo_calculator'.\n"
     "4) Use 'pitch_converter' for tuning and note/frequency conversions (MIDI↔Hz).\n"
     "Always include 'Sources:' when you used RAG or the web and add timestamps when available."),
    MessagesPlaceholder("chat_history"), 
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

TOOLS = [ask_rag_strict, tempo_calculator, pitch_converter, web_search]



def get_agent_runnable():
    """
    Returns the runnable with memory, creating it only once (singleton)
    Thread-safe ((si llegan varias peticiones a la vez)) with a lock to not get race conditions
    """
    global _AGENT_RUNNABLE
    if _AGENT_RUNNABLE is None:
        with _AGENT_LOCK:
            if _AGENT_RUNNABLE is None: 
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

_CHAT_STORE: Dict[str, InMemoryChatMessageHistory] = {} 
_CHAT_LOCK = threading.Lock()

MAX_TURNS = 20

def _get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    with _CHAT_LOCK:
        hist = _CHAT_STORE.get(session_id)
        if hist is None:
            hist = InMemoryChatMessageHistory()
            _CHAT_STORE[session_id] = hist
        if MAX_TURNS and len(hist.messages) > MAX_TURNS * 2:
            hist.messages = hist.messages[-MAX_TURNS*2:]
        return hist


def agent_ask(user_input: str, session_id: str = "default") -> str:
    runnable = get_agent_runnable()
    cfg = {"configurable": {"session_id": session_id}}
    out = runnable.invoke({"input": user_input}, config=cfg)
    return out["output"]

def reset_agent():
    reset_all_sessions()

def reset_session(session_id: str) -> None:
    with _CHAT_LOCK:
        _CHAT_STORE.pop(session_id, None)

def reset_all_sessions() -> None:
    global _AGENT_RUNNABLE
    with _AGENT_LOCK, _CHAT_LOCK:
        _CHAT_STORE.clear()
        _AGENT_RUNNABLE = None