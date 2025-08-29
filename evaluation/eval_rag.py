# eval/eval_rag.py
import os, json
from pathlib import Path
from typing import List
from langchain_openai import ChatOpenAI
from langsmith import traceable
from dotenv import load_dotenv
load_dotenv()

from rag import load_retriever, pick_llm, build_chain  
from .judges import judge_correctness, judge_groundedness   

DATA = Path(__file__).with_name("eval_data.jsonl")

def _format_docs_for_judge(docs) -> List[str]:
    out = []
    for d in docs:
        url = d.metadata.get("source") or d.metadata.get("url") or ""
        start = d.metadata.get("start")
        if start is not None:
            url = f"{url}&t={int(start)}s" if "watch?v=" in url else url
        txt = (d.page_content or "").strip().replace("\n", " ")
        if len(txt) > 400: txt = txt[:400] + "..."
        out.append(f"{url}\n{txt}")
    return out

@traceable(name="eval_rag.run_sample")
def run_sample(sample: dict, llm_judge: ChatOpenAI):
    q = sample["input"]
    ref = sample.get("reference","")
    k = int(sample.get("k", 5))

    retriever = load_retriever(k=k)
    chain = build_chain(pick_llm(), retriever)
    answer = chain.invoke(q)

    docs = retriever.invoke(q)
    ctx = _format_docs_for_judge(docs)

    correctness = judge_correctness(llm_judge, q, answer, ref) if ref else None
    grounded   = judge_groundedness(llm_judge, q, answer, ctx)

    return {
        "question": q,
        "answer": answer,
        "reference": ref or None,
        "correctness": correctness,
        "groundedness": grounded,
        "k": k,
    }

def main():
    judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    rows = []
    with open(DATA, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            res = run_sample(sample, judge_llm)
            rows.append(res)

    total = len(rows)
    corr = [r["correctness"]["score"] for r in rows if r["correctness"]]
    grnd = [r["groundedness"]["score"] for r in rows]
    avg_corr = sum(corr)/len(corr) if corr else None
    avg_grnd = sum(grnd)/len(grnd) if grnd else 0.0

    print(f"[EVAL] samples={total}  avg_correctness={avg_corr}  avg_groundedness={avg_grnd}")
    out = Path(__file__).with_name("eval_rag_results.jsonl")
    with open(out, "w", encoding="utf-8") as w:
        for r in rows: w.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[EVAL] wrote {out}")

if __name__ == "__main__":
    main()


# ómo interpretas y usas los resultados

# Correctness promedio (0–1): si tienes referencia. Mete un umbral (ej. ≥ 0.8) para aprobar.

# Groundedness promedio (0–1): clave para evitar alucinaciones. Umbral alto (ej. ≥ 0.9).

# Gating: si el promedio o el % de “SUPPORTED” baja de tu umbral, falla la evaluación (CI).

# Inspeccionar casos malos: revisa en LangSmith los runs con scores bajos; verás la pregunta, tu respuesta, los contextos y el dictamen del juez con explicación.

# Buenas prácticas:

# Juez con temperatura 0.

# Modelo juez ≠ modelo generador (si puedes) para reducir sesgo.

# Usa muestras humanas para calibrar los umbrales (mira 20–50 casos).

# Guarda los prompts de juez en control de versiones.