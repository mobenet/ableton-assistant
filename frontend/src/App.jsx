import React, { useEffect, useMemo, useRef, useState } from "react";
import "./app.css";

const DEFAULT_SERVER = window.location.origin;

export default function App() {
  const [serverUrl] = useState(DEFAULT_SERVER);
  const [sessionId] = useState("default");
  const mode = "agent";
  const k = 5;
  const voice = "alloy";
  const audioFmt = "mp3";
  const language = "en";

  const [messages, setMessages] = useState([
    { role: "assistant", text: "Hi! I’m your Ableton Assistant. Ask me anything about Live, clips, warping, quantization… or hold the mic to speak." }
  ]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const listRef = useRef(null);

  const [recording, setRecording] = useState(false);
  const [recSupported, setRecSupported] = useState(true);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const audioRef = useRef(null);
  const [speakAll, setSpeakAll] = useState(false);

  const api = useMemo(() => makeApi(serverUrl), [serverUrl]);

  useEffect(() => {
    const el = listRef.current; if (el) el.scrollTop = el.scrollHeight;
  }, [messages, sending]);

  const push = (m) => setMessages((x) => [...x, m]);

  async function speak(text) {
    if (!text) return;
    try {
      const blob = await api.postJsonForBlob("/tts", { text, voice, format: audioFmt });
      const url = URL.createObjectURL(blob);
      const el = audioRef.current || new Audio();
      if (!audioRef.current) audioRef.current = el;
      el.src = url;
      await el.play().catch(()=>{});
    } catch (e) { console.warn("TTS failed:", e.message); }
  }

  async function onSend(e) {
    e?.preventDefault?.();
    const q = input.trim(); if (!q || sending) return;
    setSending(true);
    push({ role: "user", text: q });
    setInput("");
    try {
      const res = await api.postJson("/ask", { question: q, session_id: sessionId, mode, k });
      const answer = res?.answer || "No answer.";
      push({ role: "assistant", text: answer, sources: res?.sources });
      if (speakAll) speak(answer);
    } catch (err) {
      push({ role: "assistant", text: "Error: " + err.message });
    } finally { setSending(false); }
  }

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
        const blob = new Blob(chunksRef.current, { type: rec.mimeType || "audio/webm" });
        chunksRef.current = [];
        try {
          const form = new FormData(); form.append("audio", blob, "q.webm");
          const stt = await api.postFormJson(`/stt?language=${encodeURIComponent(language)}`, form);
          const q = (stt?.text || "").trim();
          if (!q) { push({ role: "assistant", text: "I couldn't hear anything. Try again." }); return; }
          push({ role: "user", text: q });
          const res = await api.postJson("/ask", { question: q, session_id: sessionId, mode, k });
          const answer = res?.answer || "No answer.";
          push({ role: "assistant", text: answer, sources: res?.sources });
          if (speakAll) speak(answer);
        } catch (e) {
          push({ role: "assistant", text: `Voice flow failed: ${e.message}` });
        }
      };
      mediaRecorderRef.current = rec; return rec;
    } catch (e) {
      setRecSupported(false);
      push({ role: "assistant", text: "Mic permission denied or unsupported browser." });
      throw e;
    }
  }
  async function startRec() { try { const r = await ensureRecorder(); if (r.state === "inactive") r.start(); } catch {} }
  function stopRec() { const r = mediaRecorderRef.current; if (r && r.state === "recording") r.stop(); }

  return (
    <div className="app">
      <header className="header">
        <div className="header-inner">
          <div className="brand">
            <div className="brand-icon" aria-hidden>
              <div></div><div></div><div></div><div></div>
            </div>
            <div>Ableton Assistant</div>
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
                  <div className="role">{m.role === "user" ? "You" : "Assistant"}</div>
                  <div style={{whiteSpace:"pre-wrap", lineHeight:1.5}}>{m.text}</div>
                  {Array.isArray(m.sources) && m.sources.length > 0 && (
                    <div className="sources">
                      {m.sources.map((s, j) => (
                        <a className="chip" key={j} href={s.url} target="_blank" rel="noreferrer">
                          ▲<span style={{maxWidth:220, overflow:"hidden", textOverflow:"ellipsis", display:"inline-block", verticalAlign:"bottom"}}>{s.url}</span>
                          {s.timestamp ? <span style={{opacity:.7}}>[{s.timestamp}]</span> : null}
                        </a>
                      ))}
                    </div>
                  )}
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
