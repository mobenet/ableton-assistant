import os # leer variables de entorno
import json # serializar i deserializar respuestas de duckduckgo
import threading # para el lock que garantiza crear el agente solo una vez (singleton thread-safe)
from typing import Dict
from functools import lru_cache # cachea funciones (LLM, vs) y evita recrearlas

# LangChain core
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
# define el prompt del agente (mensajes “system”, “human”, historial y “scratchpad”)
from langchain.agents import create_openai_tools_agent, AgentExecutor
# create_openai_tools_agent fabrica un agente que usa function calling de OpenAI con tus tools 
# agentexecutor ejecuta el agente y se encarga de la orquestacion 
from langchain_openai import ChatOpenAI
# cliente del modelo openai 
from langchain.chains import LLMMathChain
# mini cadena para calculos humanos 
#from langchain_community.tools import DuckDuckGoSearchResults -> problemas con langchain community wrapper por eso usamos 
from ddgs import DDGS
# el cliente directo a DDg 

from langchain_core.tools import tool # decorador que registra una función como herramienta invocable por el agente.
from langchain_core.runnables.history import RunnableWithMessageHistory # envuelve tu ejecutor con memoria por sesión.
from langchain_core.chat_history import InMemoryChatMessageHistory # almacén en RAM del historial (no persistente en disco).

# RAG y configuración compartida
from rag import answer_query, CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL

# Vector store para chequeo de relevancia
from langchain_chroma import Chroma
from embeddings import E5Embeddings
from dotenv import load_dotenv
load_dotenv()

#Variables globales para cachear un único “runnable” del agente en el proceso (mejor rendimiento) y para hacerlo thread-safe (si llegan varias peticiones a la vez).
_AGENT_RUNNABLE = None # Variables módulo-globales para guardar el runnable del agente una sola vez
_AGENT_LOCK = threading.Lock() # El lock evita condiciones de carrera si dos peticiones lo inicializan a la vez.
# ------------------ LLM ---------------------------
@lru_cache(maxsize=1) # Crea el cliente OpenAI una vez (gracias al @lru_cache).
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
# calculator es una tool que el agente puede llamar para aritmética (útil para BPM → ms, etc.).
# Usa LLMMathChain (interpreta expresiones tipo “60000/120”).
# Devuelve texto (el agente necesitará parsearlo si quiere reutilizarlo; normalmente solo lo inserta en su explicación).

# ------------------ RAG con chequeo de relevancia ------------------
@lru_cache(maxsize=1) # i just use this to check if there is relevance before calling  my rag, if there is none i call websearch 
def _vs():
    emb = E5Embeddings(model_name=EMBED_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=emb
    )

from pydantic import BaseModel, Field

class RAGStrictInput(BaseModel): # In this example, RAGStrictInput is a model with 3 fields:
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
    docs_scores = vs.similarity_search_with_relevance_scores(query, k=k) # Return docs and relevance scores in the range [0, 1].
    # docs_scores: List[Tuple[Document, float]] con relevancia en [0..1]
    
    if not docs_scores:
        return "NO_CONTEXT"

    top_score = docs_scores[0][1] or 0.0
    if top_score is None or top_score < min_relevance: # he ido cambiando min_relevance para ir probando soluciones 
        # -> si min_relevance <= 0.5 todos pasan, pero si min_relevance > 0.5 todos los docs fallan
        return "NO_CONTEXT"

    # Si hay contexto, delega en tu pipeline LCEL para la respuesta con citas
    try:
        result = answer_query(query, k=k)
        return result["answer"]
    except Exception as e:
        return f"I could'nt consult the RAG index: {e}"

# ------------------ Tool de tempo específica (BPM ↔ ms por figura) ------------------
class TempoInput(BaseModel):
    bpm: float = Field(..., description="Tempo in beats per minute (BPM)")
    note: str = Field("1/4", description="Figure: 1/1,1/2,1/4,1/8,1/16,1/32")
    feel: str = Field("straight", description="straight | dotted | triplet")
# diccionario que dice cuántas negras dura cada figura
_NOTE_BASE = {
    "1/1": 4.0, "1/2": 2.0, "1/4": 1.0, "1/8": 0.5, "1/16": 0.25, "1/32": 0.125,
}
# _feel_mult(feel) devuelve:
# 1.5 si es dotted (con puntillo, dura ×1.5),
# 2/3 ≈ 0.6667 si es triplet (tresillo),
# 1.0 si es straight (recto).
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

    # Calcula los milisegundos por negra (quarter note).
    # En casi todos los DAWs (Ableton incluido) el tempo BPM está definido sobre la negra.
    ms_black = 60000.0 / bpm # 120 bpm -> 60 000 ms (1minute) / 120 = 500 ms por negra 
    # Calcula los milisegundos de la figura solicitada aplicando, si procede, puntillo o tresillo (feel).
    ms_note = ms_black * _NOTE_BASE[note] * _feel_mult(feel)
    # por ejemplo a 100 BPM:
    # ms_black = 600 ms
    # Una corchea con puntillo (note="1/8", feel="dotted"):
    # base: 0.5 (media negra) → 600 * 0.5 = 300 ms.
    # puntillo ×1.5 → 300 * 1.5 = 450 ms.
    bar_4_4_ms = ms_black * 4.0
    # Duración de un compás 4/4 (4 negras): ms_por_negra × 4.
    # A 120 BPM: 500 * 4 = 2000 ms.
    return (
        f"BPM: {bpm}\nFigura: {note} ({feel})\n"
        f"Ms por negra: {ms_black:.2f} ms\n"
        f"Ms por {note} ({feel}): {ms_note:.2f} ms\n"
        f"1 compás 4/4: {bar_4_4_ms:.2f} ms\n"
    )
    # Mini-ejemplos rápidos:
    # 120 BPM, note="1/16", feel="straight"
    # ms_black = 60000/120 = 500 ms
    # 1/16 = 0.25 negras → 500 * 0.25 = 125 ms
    # Compás 4/4: 2000 ms

    # 90 BPM, note="1/8", feel="triplet"
    # ms_black = 60000/90 ≈ 666.67 ms
    # base 0.5 → 333.33 ms; tresillo ×2/3 → 222.22 ms
# ------------------ Tool audio: MIDI ↔ Hz (recomendada para Ableton) ------------------
class PitchInput(BaseModel):
    mode: str = Field(..., description="'midi_to_hz' or 'hz_to_midi'")
    value: str = Field(..., description="MIDI note (number 0-127 or name as A4/C#3) or frequency in Hz")

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
    return _A4 * (2 ** ((m - _A4_MIDI)/12.0)) # Fórmula estándar: f=440⋅2^(m−69)/12                

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
    
@tool("web_search", return_direct=False) # decorador registra la función como herramienta del agente
# return_direct=False significa que el agente sigue razonando tras la tool y confecciona un mensaje final para el usuario (no se devuelve tal cual lo que saca la tool).
def web_search(query: str, max_results: int = 5) -> str:
    """
    DuckDuckGo web search. Returns up to max_results results as JSON:
    [{title, url, snippet}]
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results)) # Usa DDGS().text(...) (paquete ddgs) para buscar en DuckDuckGo.
        simplified = [ # Convierte los resultados del motor a un formato ligero y estable: una lista de dicts con title, url, snippet.
            {
                "title": r.get("title"),
                "url": r.get("href") or r.get("link"),
                "snippet": r.get("body"),
            }
            for r in results[:max_results]
        ]
        return json.dumps(simplified, ensure_ascii=False) # Devuelve un str JSON, no un objeto Python. ¿Por qué? Porque a los agentes les resulta fácil “leer” JSON desde la tool y pegar partes; además te ahorras fugas de tipos si alguna tool cambiara de versión.
    except Exception as e:
        return f"SEARCH_ERROR: {e}" # Si algo falla, devuelve SEARCH_ERROR: ... para que el agente lo vea y pueda reaccionar.


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
    MessagesPlaceholder("chat_history"), # aquí LangChain insertará el historial de la conversación (memoria), para dar contexto.
    ("human", "{input}"), # la pregunta actual del usuario.
    MessagesPlaceholder("agent_scratchpad"), # aquí LangChain insertará el “scratchpad” del agente, es decir, sus pensamientos intermedios y llamadas a tools.
])

TOOLS = [ask_rag_strict, tempo_calculator, pitch_converter, web_search]


# ------------------ Agente con memoria por sesión (RAM) ------------------
# Singleton thread-safe: crea el runnable del agente solo una vez (la primera vez que se llama a get_agent_runnable).
def get_agent_runnable():
    """
    Returns the runnable with memory, creating it only once (singleton)
    Thread-safe ((si llegan varias peticiones a la vez)) with a lock to not get race conditions
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
    agent = create_openai_tools_agent(llm, TOOLS, AGENT_PROMPT) # genera un agente function-calling usando tu prompt y tools.
    executor = AgentExecutor(agent=agent, tools=TOOLS, verbose=True, return_intermediate_steps=False) # Crea elñ ejecutor del agente. 
    # Envuelve al agente; tú le pasarás {"input": "..."} y él se ocupará de decidir qué tool llamar, leer la respuesta, y producir la salida final.
    return executor

# Memoria por sesión (RAM; persiste mientras el proceso esté vivo)
# Guarda un historial por session_id en memoria RAM (vive mientras el proceso esté corriendo).
_CHAT_STORE: Dict[str, InMemoryChatMessageHistory] = {} # Diccionario en RAM que guarda un historial por session_id.
_CHAT_LOCK = threading.Lock()

# límit de torns (user+assistant = 2 missatges per torn)
MAX_TURNS = 20

def _get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    with _CHAT_LOCK:
        hist = _CHAT_STORE.get(session_id)
        if hist is None:
            hist = InMemoryChatMessageHistory()
            _CHAT_STORE[session_id] = hist
        # trim si supera el límit
        if MAX_TURNS and len(hist.messages) > MAX_TURNS * 2:
            hist.messages = hist.messages[-MAX_TURNS*2:]
        return hist




# ------------------ API sencilla de altonoivel  ------------------
"""Función “facade”: la llamas desde tu endpoint /ask.
Pasa el session_id en config, que RunnableWithMessageHistory usa para recuperar el historial correcto.
Devuelve solo el texto de respuesta (out["output"]), porque AgentExecutor está configurado para no incluir pasos intermedios."""
def agent_ask(user_input: str, session_id: str = "default") -> str:
    """
    Llama al agente con memoria. 'session_id' separa conversaciones paralelas.
    """
    runnable = get_agent_runnable()
    cfg = {"configurable": {"session_id": session_id}}
    out = runnable.invoke({"input": user_input}, config=cfg)
    return out["output"]

# Resetea el agente (por ejemplo, para tests o si quieres borrar todas las memorias)
"""Borra el singleton en memoria; la siguiente llamada a get_agent_runnable() lo reconstruye (útil si cambiaste env vars, tools, prompt, etc. sin reiniciar el proceso).
No borra los historiales de _CHAT_STORE; si quieres “borrón y cuenta nueva” por sesión, limpia también ese dict."""
def reset_agent():
    reset_all_sessions()

def reset_session(session_id: str) -> None:
    """Esborra només la memòria d'una sessió."""
    with _CHAT_LOCK:
        _CHAT_STORE.pop(session_id, None)

def reset_all_sessions() -> None:
    """Esborra tota la memòria i reinicia l'agent singleton."""
    global _AGENT_RUNNABLE
    with _AGENT_LOCK, _CHAT_LOCK:
        _CHAT_STORE.clear()
        _AGENT_RUNNABLE = None