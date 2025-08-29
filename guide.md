

# 1) Title — “Chatbleton: an Ableton Live RAG Assistant”
“Hi! I’m going to show **Chatbleton**, a RAG assistant focused on Ableton Live. It answers production questions using curated manuals and YouTube transcripts, can browse the web when it has no context, remembers the chat session, and supports **voice conversation**.”

---

# 2) Problem & Goal

“Learning Ableton is fragmented: official manuals, scattered tutorials, and slang. My goal: a **single assistant** that (1) returns grounded answers with sources and timestamps; (2) works in English/Spanish/Catalan; (3) supports text and voice; (4) is simple to deploy and demo.”

---

# 3) System Overview (Architecture)

* **Data/Indexing**: split manual and video transcripts into small time windows, embed with **multilingual-e5-base**, store in **Chroma**.
* **Backend (Flask)**: endpoints `/chat`, `/chat_stream`, `/stt`, `/tts`, `/voice(_json)`, plus health & session reset.
* **App (React)**: clean ChatGPT-style UI, markdown rendering, sources open in new tabs, and **Speak ON/OFF** toggle.
  The agent uses **LangChain** tools: a strict RAG tool, a web search fallback, and music-specific tools (tempo calculator, MIDI↔Hz). Session memory is handled in-process.”

---

# 4) Data → Chunks → Embeddings → Vector Store

* **Manual chunks** in `data/manual_chunks/manual_chunks.json`.
* **YouTube transcripts** in `data/transcripts/*.json`.
  I convert transcripts into **overlapping time windows** (`--win 40 --stride 30`), so search returns not only text but **exact timestamps** (e.g., `&t=70s`).
  Then I embed with **E5** and store in **Chroma**.
  Command:
  `python build_index.py --rebuild --win 40 --stride 30`
  This prints how many docs were indexed.”

---

# 5) Retrieval & Answering (RAG)

“In `rag.py` I build a **compact context** from top-k documents with URLs and brief excerpts. The system prompt forces: ‘**Answer only from context, else say you don’t have enough info**’ and **always show Sources**.
I also **dedupe** sources and, for videos, attach timestamps like `1m10s`.
We keep the answer text clean; the frontend renders markdown and links open in new tabs.”

---

# 6) Agent with Tools (Why not only RAG?)

“Some queries aren’t in my index. So I wrapped RAG inside an **agent** (`agent_tools.py`) with a **tool policy**:

1. Try `ask_rag_strict` first. If **relevance < threshold** or no docs, return `NO_CONTEXT`.
2. Then call **`web_search`** (DuckDuckGo via the `duckduckgo_search` library — the official LC tool was flaky).
3. Music helpers: **`tempo_calculator`** (BPM↔ms with straight/dotted/triplet) and **`pitch_converter`** (MIDI↔Hz).
   The agent also uses **session memory** with `RunnableWithMessageHistory`, so the conversation is coherent within a `session_id`.”

---


# 8) Frontend (React)

“The UI is **minimal** and **fast**:

* Left/right bubbles for assistant/user.
* **`react-markdown` + `remark-gfm`** to render bold/lists/tables.
* **Speak ON/OFF**: when ON, text answers are read out; when OFF, just text.
* Mic button: press-to-record → `/stt` → text shown → `/chat` → answer shown; if Speak ON, we call `/tts` in parallel to talk back. We also added a simple **stop** mechanism to interrupt playback when toggling Speak OFF.”
* if the user 

---

# 9) Voice Flow (Simple “parallel” UX)

**What I’ll say (45s):**
“To keep it snappy, after STT returns the transcript we immediately paint the user message. Then we call `/chat` for the answer. As soon as the text arrives, we show it; **if Speak ON**, we fire `/tts` **without awaiting it**, so audio starts while the text is already on screen. Toggling Speak OFF stops the audio element.”

---

# 10) Telemetry & Evaluation

**What I’ll say (1 min):**
“For telemetry I use **LangSmith** tags, logging: the question, whether `similarity_search_with_relevance_scores` was used, and which branch (RAG vs. web).
For evaluation we used **LLM-as-Judge** to score:

* **Correctness** (does the answer match a gold label?), and
* **Groundedness** (is the answer supported by retrieved context?).
  This gives a quick pass/fail signal without hand-labeling every run.”

---

# 11) Challenges & Fixes

**What I’ll say (1 min):**

* **Windows / pip / numpy build**: hitting compiler errors; pinned versions and used prebuilt wheels.
* **Chroma `.persist()`** mismatch: new `langchain_chroma` manages persistence once you pass `persist_directory`; just removing `vs.persist()` fixed it.
* **DuckDuckGo tool**: the LangChain community tool was unreliable; switched to **`duckduckgo_search` (DDGS)** directly.
* **Duplicate sources**: removed source-prompts duplication.
* **Spanish queries**: ensured **multilingual E5**, consistent “query:”/“passage:” prefixes, and windowed transcripts so Spanish/Catalan terms still hit.
* **Streaming UI** quirks: when tight on time we fell back to **ask-once**; stream stays in code for later.

---

# 12) Demo (2–3 minutes)

**What I’ll say & Do:**

1. **Text**: “What’s the difference between global quantization and clip quantization in Ableton?”

   * Show answer + sources/timestamps.
   * Toggle **Speak ON** → send a new text → it reads aloud while rendering markdown.
2. **Voice**: press mic, say: “Set a 120 BPM dotted eighth-note delay time.”

   * Show transcript, then answer; if Speak ON, it talks.
3. **No-context example**: Ask something not in index (e.g., “Who invented the Rubik’s Cube?”)

   * Agent should go RAG→NO\_CONTEXT→**web\_search** and return links.

---

# 13) What’s Next

**What I’ll say (30s):**
“Next steps:

* Real streaming end-to-end;
* Better re-ranking & hybrid (BM25 + dense);
* Fine-tuned voice with interruptible TTS streams;
* Hardening deployment behind Cloudflare with a stable origin and HTTPS;
* More robust eval sets and dashboards in LangSmith.”

---

# 14) How to Run (very brief)

**What I’ll say (30s):**
“Steps:

1. `python -m venv .venv && source .venv/bin/activate` (or Windows `.venv\Scripts\activate`)
2. `pip install -r requirements.txt`
3. Copy `.env` with `OPENAI_API_KEY` and model vars
4. `python build_index.py --rebuild`
5. `python app.py` and open the frontend (Vite dev or build + served by Flask)
   Optionally expose with `cloudflared tunnel --url http://localhost:8000`.”

---

## Live Demo Checklist (backup)

* `.env` present; OpenAI key set.
* `chroma_db/` exists after indexing.
* Flask up on `:8000`; frontend points to the same origin or CORS enabled.
* Cloudflared prints your public URL; paste it into phone browser.

---

## Q\&A Ammo (quick answers)

* **Why E5?** Multilingual, good retrieval quality, simple “query:/passage:” format, normalized embeddings → stable cosine.
* **Why agent over plain RAG?** Graceful fallback to web when index lacks coverage; plus dedicated music tools.
* **Memory?** In-process **RunnableWithMessageHistory** keyed by `session_id` so parallel chats don’t mix.
* **LLM-as-Judge—how reliable?** It’s a fast proxy. We combine correctness + groundedness and sample check failures.
* **Windows install pains?** We pinned/used wheels and avoided building numpy/torch from source.
* **Why no heavy vector DB?** **Chroma** local is perfect for this size and demo; can swap later.

---

