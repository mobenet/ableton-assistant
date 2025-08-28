import React, { useEffect, useMemo, useRef, useState } from "react";
import "./app.css";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm"; 

const DEFAULT_SERVER = import.meta?.env?.VITE_API_BASE ?? window.location.origin;
console.log("[CHATBLETON] API base:", DEFAULT_SERVER);
export default function App() {
  const [serverUrl] = useState(DEFAULT_SERVER);
  const [sessionId] = useState("default");
  const voice = "alloy";      // voz de /tts
  const audioFmt = "mp3";     // mp3|wav|flac|pcm
  const language = "en";      // idioma sugerido a /stt

  const [messages, setMessages] = useState([
    { role: "assistant", text: "Hi! I’m Chatbleton, your Ableton Assistant :). Ask me anything about Live, clips, warping, quantization… or hold the mic to speak." }
  ]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [speakAll, setSpeakAll] = useState(false);

  // mic state
  const [recording, setRecording] = useState(false);
  const [recSupported, setRecSupported] = useState(true);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const pressStartedAtRef = useRef(0);

  const listRef = useRef(null);
  const audioRef = useRef(null);
  const assistantIdxRef = useRef(null);

  const api = useMemo(() => makeApi(serverUrl), [serverUrl]);

  useEffect(() => {
    const el = listRef.current; if (el) el.scrollTop = el.scrollHeight;
  }, [messages, sending]);

  const push = (m) => setMessages((x) => [...x, m]);

  // ---- TTS (servidor /tts) ----
  async function speak(text) {
    if (!text) return;
    try {
      const blob = await api.postJsonForBlob("/tts", { text, voice, format: audioFmt });
      const url = URL.createObjectURL(blob);
      const el = audioRef.current || new Audio();
      if (!audioRef.current) audioRef.current = el;
      el.src = url;
      await el.play().catch(()=>{});
    } catch (e) {
      push({ role: "assistant", text: `TTS failed: ${e.message}` });
    }
  }
  async function onSend(e) {
    e?.preventDefault?.();
    const q = input.trim();
    if (!q || sending) return;
    setSending(true);
    push({ role: "user", text: q });
    setInput("");
    try {
      await askOnce(q);
    } catch (err) {
      push({ role: "assistant", text: "Error: " + err.message });
    } finally {
      setSending(false);
    }
  }
  async function askOnce(q) {
    const res = await api.postJson("/chat", {
      question: q,
      session_id: sessionId,
    });
    const answer = res?.answer || "No answer.";
    push({ role: "assistant", text: answer });
    if (speakAll) await speak(answer);
  }
    
  // ---- Enviar por texto ----
  // async function onSend(e) {
  //   e?.preventDefault?.();
  //   const q = input.trim(); if (!q || sending) return;
  //   setSending(true);
  //   push({ role: "user", text: q });
  //   setInput("");
  //   try {
  //     await streamAnswer(q);
  //   } finally {
  //     setSending(false);
  //   }
  // }
  async function streamAnswer(q) {
    // 1) placeholder d’assistent i acumulador
    const aIdx = pushAndGetIndex({ role: "assistant", text: "" });
    let acc = "";

    try {
      // 2) stream des del backend amb memòria per sessió
      await api.getTextStream(`/chat_stream?q=${encodeURIComponent(q)}&session_id=${encodeURIComponent(sessionId)}`,
        (chunk) => {
          acc += chunk;
          setMessages(prev => {
            const next = [...prev];
            const cur = next[aIdx] || { role: "assistant", text: "" };
            next[aIdx] = { ...cur, text: (cur.text || "") + chunk };
            return next;
          });
        }
      );

      // 3) quan acaba el stream, opcional: parlar
      if (speakAll && acc.trim()) {
        await speak(acc);
      }
    } catch (err) {
      setMessages(prev => {
        const next = [...prev];
        next[aIdx] = { role: "assistant", text: `Stream error: ${err.message}` };
        return next;
      });
    }
  }

  

  // ---- Mic: preparación ----
  async function ensureRecorder() {
    if (mediaRecorderRef.current) return mediaRecorderRef.current;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const opts = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? { mimeType: "audio/webm;codecs=opus" } : undefined;
      const rec = new MediaRecorder(stream, opts);

      chunksRef.current = [];
      rec.ondataavailable = (ev) => { if (ev.data?.size) chunksRef.current.push(ev.data); };
      rec.onstart = () => setRecording(true);

      rec.onstop = async () => {
        setRecording(false);
        // ¿pulsación muy corta? => instrucción de “mantener”
        const dt = Date.now() - (pressStartedAtRef.current || 0);
        if (dt < 300) {
          push({ role: "assistant", text: "To talk, keep the mic button pressed." });
          chunksRef.current = [];
          return;
        }
        // build the audio blob we just recorded
        const blob = new Blob(chunksRef.current, { type: rec.mimeType || "audio/webm" });
        chunksRef.current = [];

        try {
          // 1) STT
          const form = new FormData(); 
          form.append("audio", blob, "q.webm");
          
          const stt = await fetch(
            `${serverUrl.replace(/\/$/, "")}/voice?session_id=${encodeURIComponent(sessionId)}&voice=${encodeURIComponent(voice)}&format=${encodeURIComponent(audioFmt)}`,
            { method: "POST", body: form }
          );
          // const q = (stt?.text || "").trim();
          // if (!q) { push({ role: "assistant", text: "I couldn't hear anything. Try again." }); return; }
          if (!stt.ok) {
            let e = null; try { e = await stt.json(); } catch {}
            throw new Error(e?.error || `${stt.status} ${stt.statusText}`);
          }
          // 2) Mostrar transcripción
          const audioBlob = await stt.blob();
          const url = URL.createObjectURL(audioBlob);
          const el = audioRef.current || new Audio();
          if (!audioRef.current) audioRef.current = el;
          el.src = url;
          await el.play().catch(() => {});

        } catch (e) {
          push({ role: "assistant", text: `Voice flow failed: ${e.message}` });
        }
      };

      mediaRecorderRef.current = rec; 
      return rec;
    } catch (e) {
      setRecSupported(false);
      push({ role: "assistant", text: "Mic permission denied or unsupported browser." });
      throw e;
    }
  }

  async function startRec() {
    try {
      const rec = await ensureRecorder();
      if (rec.state !== "inactive") return;
      pressStartedAtRef.current = Date.now();
      rec.start();
    } catch {}
  }
  function stopRec() {
    const r = mediaRecorderRef.current;
    if (r && r.state === "recording") r.stop();
  }

  function pushAndGetIndex(msg) {
  let idx;
  setMessages(prev => {
    const next = [...prev, msg];
    idx = next.length - 1;
    return next;
  });
  return idx;
}

  return (
    <div className="app">
      <header className="header">
        <div className="header-inner">
          <div className="brand">
            <div className="brand-icon" aria-hidden>
              <div></div><div></div><div></div><div></div>
            </div>
            <div>Chatbleton - Your Ableton Assistant</div>
          </div>
          <div className="header-actions">
            <button className={"btn " + (speakAll ? "btn-accent" : "")}
                    onClick={() => setSpeakAll(v=>!v)}>
              {speakAll ? "Speak: ON" : "Speak: OFF"}
            </button>
          </div>
        </div>
      </header>

      <main className="main">
        <div className="container">
          <div ref={listRef} className="timeline">
            {messages.map((m, i) => (
              <div className={"row " + (m.role === "user" ? "user" : "")} key={i}>
                <div className="bubble">
                  <div className="role">{m.role === "user" ? "You" : "Chatbleton"}</div>
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    linkTarget="_blank"
                    components={{
                        a: (props) => (
                          <a {...props} target="_blank" rel="noopener noreferrer" />
                        ),
                      }}
                    >
                    {m.text || ""}
                  </ReactMarkdown>
                </div>
              </div>
            ))}

            {sending && (
              <div className="row">
                <div className="bubble"><span className="loader"></span> thinking…</div>
              </div>
            )}
          </div>
        </div>
      </main>

      <footer className="footer">
        <div className="inputbar">
          <form onSubmit={onSend} className="input-wrap">
            <textarea
              className="textarea"
              placeholder="Message Ableton Assistant…"
              value={input}
              onChange={(e)=>setInput(e.target.value)}
              onKeyDown={(e)=>{ if (e.key==='Enter' && !e.shiftKey) { e.preventDefault(); onSend(); } }}
            />
            <div className="actions">
              {recSupported ? (
                <button type="button"
                        title="Hold to talk"
                        className={"iconbtn " + (recording ? "ping" : "")}
                        onMouseDown={startRec}
                        onMouseUp={stopRec}
                        onMouseLeave={stopRec}
                        onTouchStart={(e)=>{e.preventDefault(); startRec();}}
                        onTouchEnd={stopRec}>
                  🎙️
                </button>
              ) : (
                <button type="button" className="iconbtn" title="Mic unavailable" disabled>🚫</button>
              )}
              <button type="submit" title="Send" className="iconbtn">➤</button>
            </div>
          </form>
          <div className="small">Press Enter to send • Hold mic to speak</div>
        </div>
      </footer>

      <audio ref={audioRef} hidden />
    </div>
  );
}

function makeApi(baseUrl) {
  const url = (p) => baseUrl.replace(/\/$/, "") + p;
  return {
    async getTextStream(path, onChunk) {
      const res = await fetch(url(path), {
        method: "GET",
        headers: { "Accept": "text/plain", "Cache-Control": "no-cache" },
        cache: "no-store",
      });
      if (!res.ok) {
        let e=null; try{e=await res.json();}catch{}
        throw new Error(e?.error || `${res.status} ${res.statusText}`);
      }

      const END = "<<END_OF_MESSAGE>>";

      // Fallback si no hay ReadableStream
      if (!res.body || !res.body.getReader) {
        const full = await res.text();
        const cut = full.split(END)[0];
        if (cut) onChunk(cut);
        return;
      }

      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buffer = "";

      try {
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += dec.decode(value, { stream: true });

          // procesa en “trocitos” por si llega más de uno
          let idx;
          while ((idx = buffer.indexOf(END)) !== -1) {
            const part = buffer.slice(0, idx);
            if (part) onChunk(part);
            // descarta lo consumido + marcador y rompe (hemos terminado)
            buffer = buffer.slice(idx + END.length);
            await reader.cancel().catch(()=>{});
            return;
          }

          // si todavía no hay END, emite lo que tengas y vacía
          if (buffer.length > 0) {
            onChunk(buffer);
            buffer = "";
          }
        }
      } finally {
        try { await reader.cancel(); } catch {}
      }
    },
    async postJson(path, body) {
      const res = await fetch(url(path), { method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify(body||{}) });
      if (!res.ok) { let e=null; try{e=await res.json();}catch{} throw new Error(e?.error || `${res.status} ${res.statusText}`); }
      return res.json();
    },
    async postJsonForBlob(path, body) {
      const res = await fetch(url(path), { method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify(body||{}) });
      if (!res.ok) { let e=null; try{e=await res.json();}catch{} throw new Error(e?.error || `${res.status} ${res.statusText}`); }
      return res.blob();
    },
    async postFormJson(path, form) {
      const res = await fetch(url(path), { method: "POST", body: form });
      if (!res.ok) { let e=null; try{e=await res.json();}catch{} throw new Error(e?.error || `${res.status} ${res.statusText}`); }
      return res.json();
    },
  };
}
