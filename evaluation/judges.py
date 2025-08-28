# evaluation/judges.py
import json
from typing import List, Dict, Any
from langsmith import traceable
from langchain_openai import ChatOpenAI

# ---------- Juez de "Correctness" con referencia ----------
@traceable(name="judge.correctness")
def judge_correctness(
    llm: ChatOpenAI,
    question: str,
    answer: str,
    reference: str,
) -> Dict[str, Any]:
    """
    Return: {verdict: CORRECT|PARTIAL|INCORRECT, score:0..1, justification:str}
    """
    sys = (
        "You are a strict QA judge. Compare the assistant's answer to the gold reference.\n"
        "Evaluate factual correctness ONLY (ignore style). "
        "Return a JSON object with fields: verdict (CORRECT|PARTIAL|INCORRECT), score (0..1), justification."
    )
    usr = f"""
        Question:
        {question}

        Assistant answer:
        {answer}

        Gold reference:
        {reference}

        Respond ONLY valid JSON.
    """
    msg = llm.invoke([("system", sys), ("user", usr)])
    # Intenta parsear JSON; si no, construye fallo robusto
    try:
        data = json.loads(msg.content)
        # sanea valores
        v = str(data.get("verdict","")).upper()
        if v not in {"CORRECT","PARTIAL","INCORRECT"}:
            v = "PARTIAL"
        s = float(data.get("score", 0.0))
        s = min(max(s, 0.0), 1.0)
        j = str(data.get("justification",""))
        return {"verdict": v, "score": s, "justification": j}
    except Exception:
        return {"verdict":"PARTIAL","score":0.0,"justification":"Judge output not JSON."}

# ---------- Juez de "Groundedness" con contexto RAG ----------
@traceable(name="judge.groundedness")
def judge_groundedness(
    llm: ChatOpenAI,
    question: str,
    answer: str,
    contexts: List[str],   # cada item: "[- URL] snippet..."
) -> Dict[str, Any]:
    """
    Returns: {verdict: SUPPORTED|PARTIALLY_SUPPORTED|NOT_SUPPORTED, score:0..1, cites:[indices], justification:str}
    """
    sys = (
        "You are a grounding judge. Determine whether the assistant's answer is supported by the provided context snippets. "
        "Be conservative: unsupported claims => NOT_SUPPORTED. Partially supported => PARTIALLY_SUPPORTED. "
        "Return JSON with keys: verdict (SUPPORTED|PARTIALLY_SUPPORTED|NOT_SUPPORTED), score (0..1), cites (array of snippet indexes), justification."
    )
    ctx_str = "\n\n".join([f"[{i}] {c}" for i, c in enumerate(contexts)])
    usr = f"""
        Question:
        {question}

        Assistant answer:
        {answer}

        Context snippets:
        {ctx_str}

        Rules:
        - Only rely on the snippets provided above.
        - If unsure, choose NOT_SUPPORTED.
        - Respond ONLY valid JSON.
    """
    msg = llm.invoke([("system", sys), ("user", usr)])
    try:
        data = json.loads(msg.content)
        v = str(data.get("verdict","")).upper()
        if v not in {"SUPPORTED","PARTIALLY_SUPPORTED","NOT_SUPPORTED"}:
            v = "PARTIALLY_SUPPORTED"
        s = float(data.get("score", 0.0))
        s = min(max(s, 0.0), 1.0)
        cites = data.get("cites", [])
        if not isinstance(cites, list): cites = []
        j = str(data.get("justification",""))
        return {"verdict": v, "score": s, "cites": cites, "justification": j}
    except Exception:
        return {"verdict":"PARTIALLY_SUPPORTED","score":0.0,"cites":[],"justification":"Judge output not JSON."}
