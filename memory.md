
---

## 1) What we mean by “memory”

* **Chat memory** = the running conversation history (user + assistant turns) that we pass back to the LLM so it can keep context (“as we discussed…”, pronouns, follow-ups).
* **Not** your knowledge base: the Chroma vector store (manual + YouTube windows) is **static** content for retrieval, not conversation memory.

---

## 2) Where memory lives (server-side, per session)

All memory is held **in RAM** on the Python backend, scoped by a **session\_id**.

* In `agent_tools.py` we create a global, thread-safe agent with **per-session chat history**:

```python
# Global store of histories (in memory)
_CHAT_STORE: Dict[str, InMemoryChatMessageHistory] = {}

def _get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _CHAT_STORE:
        _CHAT_STORE[session_id] = InMemoryChatMessageHistory()
    return _CHAT_STORE[session_id]
```

* We wrap the agent with **RunnableWithMessageHistory**, telling LangChain which field is the user input and which holds the running history:

```python
runnable = RunnableWithMessageHistory(
    executor,                      # your AgentExecutor
    _get_session_history,          # returns the per-session buffer
    input_messages_key="input",
    history_messages_key="chat_history",
)
```

* Every time the frontend calls the agent, we pass the **session\_id**:

```python
def agent_ask(user_input: str, session_id: str = "default") -> str:
    cfg = {"configurable": {"session_id": session_id}}
    out = get_agent_runnable().invoke({"input": user_input}, config=cfg)
    return out["output"]
```

LangChain automatically appends the user message and the agent’s reply to the `InMemoryChatMessageHistory` for that `session_id`.

---

## 3) How the frontend controls memory

* Your React app includes a `sessionId` (currently `"default"`), and sends it to the backend:

  * **Text chat:** `POST /chat` with `{"question": "...", "session_id": "default"}`
  * **Stream chat:** `GET /chat_stream?q=...&session_id=default`
  * **Voice flow:** endpoints also pass `session_id`.
* Result: you can have parallel conversations by using different `session_id` values (e.g., per browser tab or per user).

**Tip for demo:** generate a random `sessionId` per tab, and show that context doesn’t leak between tabs.

---

## 4) Resetting memory

You exposed two control endpoints in `app.py`:

* `POST /reset_session` → clears only that session’s history.
* `POST /reset_all` → clears **all** sessions.

Internally they call helpers in `agent_tools.py` that drop entries from `_CHAT_STORE`.

---

## 5) Lifecycle & limits

* Memory persists **only while the Python process is running**. Restarting the server clears everything (this is by design for the assignment).
* There’s **no hard cap** yet; if a session gets very long, prompts get bigger (slower, costlier). If needed, you can:

  * Keep only the **last N messages**, or
  * Summarize earlier turns and keep a compact summary in the history.

---

## 6) Why this design (vs other options)

* We used **LangChain’s `RunnableWithMessageHistory`** instead of older memory classes because it plugs directly into the new **tool-calling agent** API and is simple, explicit, and stable.
* We deliberately **separated chat memory from RAG retrieval**:

  * RAG uses Chroma to fetch relevant snippets.
  * The agent uses chat memory to hold the conversation state (what you just asked, clarifications, tool decisions).
  * This separation keeps your knowledge base clean and reproducible.

---

## 7) What memory actually changes at runtime

* Follow-up questions work: “and what about warping?” references the previous turn because the model receives the **full chat history** for that `session_id`.
* Tool choice improves: the agent sees it already tried RAG and might go to `web_search` next, or use `tempo_calculator` on a follow-up, because the prior tool steps are part of the history.
* Voice and text share memory when they share the same `session_id` (you can start on mic, continue in text, and vice versa).

---

## 8) How to demonstrate it

1. Open the app in **two tabs** with different `session_id` values (or change it in code temporarily).
2. In Tab A: ask a two-part question (second turn relies on the first). It remembers.
3. In Tab B: ask only the second part first. It **doesn’t** remember—different session, clean memory.
4. Show `POST /reset_session` and repeat: the assistant forgets.

---

## 9) Known limitations / future work

* **Persistence**: currently RAM only. To persist across restarts, write histories to a DB/Redis (and reload on boot).
* **Privacy**: don’t store PII indefinitely; provide UI “Clear chat” (you already have endpoints).
* **Trimming**: add a max token/messages policy to keep prompts small and fast.

---

### One-sentence summary you can say

> “Each browser session has its own in-memory chat history on the server, keyed by `session_id`. We wrap the agent with `RunnableWithMessageHistory`, so every turn is added automatically; this gives the model context for follow-ups without polluting the vector database, and we expose endpoints to reset a single session or all sessions.”
