# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from rag_chain import answer_query
from agent_tools import agent_ask  # el que hicimos

app = FastAPI()

class QAReq(BaseModel):
    question: str
    mode: str = "auto"          # "auto" | "rag" | "agent"
    session_id: Optional[str] = "default"
    k: int = 5
    min_relevance: float = 0.2   # para ask_rag_strict si usas "agent"/"auto"

@app.post("/qa")
def qa(req: QAReq):
    if req.mode == "rag":
        res = answer_query(req.question, k=req.k)
        return {"mode": "rag", **res}

    # agent (auto o agent)
    out = agent_ask(req.question, session_id=req.session_id)
    return {"mode": "agent", "answer": out}
