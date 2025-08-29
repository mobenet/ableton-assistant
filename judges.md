

# What the Judges Evaluate

1. **Correctness**
   *Question:* *“Does the system’s answer match the gold/reference answer?”*
   *Signal:* Compare **system answer** vs **gold** (semantic match, not exact string).

2. **Groundedness**
   *Question:* *“Is the answer fully supported by the retrieved RAG context (no hallucinations)?”*
   *Signal:* Compare **system answer** vs **RAG context** (the top-k snippets).

**Scoring scale** (simple and interpretable): **{0, 0.5, 1}**

* **1** = fully correct / fully grounded
* **0.5** = partially correct / partially grounded
* **0** = incorrect / ungrounded

---

# Evaluation Dataset

A **JSONL** file, e.g. `evaluation/eval_data.jsonl`, one object per line:

```json
{"question": "What is global quantization in Ableton?", "gold": "Short, correct reference answer...", "k": 5}
```

---

# rag.py – Flow

1. **Call your RAG**

   ```python
   from rag import answer_query, load_retriever
   res = answer_query(sample["question"], k=sample.get("k", 5))
   answer_text = res["answer"]
   ```

2. **Build the judge context**

   * Retrieve top-k docs and concatenate a compact **context\_text** (trimmed to \~1–1.2k chars) from their `page_content`.

3. **Call the judges** (from `judges.py`)

   * `judge_correctness(question, answer_text, gold)`
   * `judge_groundedness(answer_text, context_text)`

4. **Parse** both judges’ JSON to get **score** (0/0.5/1) + **rationale** text.

5. **Aggregate** results (averages, histograms) and **log** to **LangSmith** with tags like `"eval"`, `"judge_correctness"`, `"judge_groundedness"`.

6. **Save** a CSV/JSONL with per-item scores + rationales and **print** summary metrics.

---

# judges.py – How Each Judge Is Implemented

**Design goals**

* Deterministic & consistent: `temperature=0`
* Output is **strict JSON** to parse easily
* Groundedness judge is forced to use **ONLY** the provided context



# Reading the Results

* If **Correctness** is high but **Groundedness** is low → the model is “right for the wrong reason” (hallucinating or relying on prior knowledge). Improve retrieval quality (`k`, chunking, relevance threshold) or force the agent to cite context.
* If **Groundedness** is high but **Correctness** is low → your **gold** might be too strict or the answer is incomplete; refine gold or prompt.

