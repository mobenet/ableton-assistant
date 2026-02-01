import React, { useCallback, useEffect, useMemo, useRef, useState, memo } from "react";
import "./app.css";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const DEFAULT_SERVER = import.meta?.env?.VITE_API_BASE ?? "";
console.log("[CHATBLETON] API base:", DEFAULT_SERVER || "(relative URLs)");

// Error Boundary Component
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="error-boundary">
          <h2>Something went wrong</h2>
          <p>{this.state.error?.message || "Unknown error"}</p>
          <button onClick={() => window.location.reload()}>Reload</button>
        </div>
      );
    }
    return this.props.children;
  }
}

// Memoized Message Component to prevent unnecessary re-renders
const Message = memo(function Message({ role, text }) {
  return (
    <div className={"row " + (role === "user" ? "user" : "")}>
      <div className="bubble">
        <div className="role">{role === "user" ? "You" : "Chatbleton"}</div>
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            a: (props) => (
              <a {...props} target="_blank" rel="noopener noreferrer" />
            ),
          }}
        >
          {text || ""}
        </ReactMarkdown>
      </div>
    </div>
  );
});

// Loading indicator component
const LoadingBubble = memo(function LoadingBubble() {
  return (
    <div className="row">
      <div className="bubble">
        <span className="loader"></span> thinking...
      </div>
    </div>
  );
});

export default function App() {
  const [serverUrl] = useState(DEFAULT_SERVER);
  const [sessionId] = useState("default");
  const voice = "alloy";
  const audioFmt = "mp3";
  const language = "en";

  const [messages, setMessages] = useState([
    {
      role: "assistant",
      text: "Hi! I'm Chatbleton, your Ableton Assistant. Ask me anything about Live, clips, warping, quantization... or hold the mic to speak.",
    },
  ]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [speakAll, setSpeakAll] = useState(false);

  // Mic state
  const [recording, setRecording] = useState(false);
  const [recSupported, setRecSupported] = useState(true);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const pressStartedAtRef = useRef(0);

  const listRef = useRef(null);
  const audioRef = useRef(null);
  const speakAllRef = useRef(false);

  const api = useMemo(() => makeApi(serverUrl), [serverUrl]);

  // Auto-scroll to bottom
  useEffect(() => {
    const el = listRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, sending]);

  // Keep speakAllRef in sync
  useEffect(() => {
    speakAllRef.current = speakAll;
  }, [speakAll]);

  const push = useCallback((m) => setMessages((x) => [...x, m]), []);

  const stripSources = useCallback((t) => {
    return (t || "").replace(/\n+?Sources:\s*[\s\S]*$/i, "").trim();
  }, []);

  // Track current audio URL for cleanup
  const currentAudioUrlRef = useRef(null);

  // TTS function with proper memory management
  const speak = useCallback(
    async (text) => {
      const clean = stripSources(text);
      if (!clean) return;

      try {
        if (!speakAllRef.current) return;

        const blob = await api.postJsonForBlob("/tts", {
          text: clean,
          voice,
          format: audioFmt,
        });

        // Check again after async operation
        if (!speakAllRef.current) return;

        const el = audioRef.current || new Audio();
        if (!audioRef.current) audioRef.current = el;

        // Cleanup previous audio URL to prevent memory leak
        if (currentAudioUrlRef.current) {
          URL.revokeObjectURL(currentAudioUrlRef.current);
        }

        const url = URL.createObjectURL(blob);
        currentAudioUrlRef.current = url;

        el.src = url;
        el.onended = () => {
          // Cleanup URL after playback ends
          if (currentAudioUrlRef.current === url) {
            URL.revokeObjectURL(url);
            currentAudioUrlRef.current = null;
          }
        };

        await el.play().catch((err) => console.warn("Audio play error:", err));
      } catch (e) {
        console.error("TTS error:", e);
        // Don't spam chat with TTS errors, just log
      }
    },
    [api, stripSources]
  );

  // Chat function with streaming support
  const askOnce = useCallback(
    async (q) => {
      let fullAnswer = "";
      let streamStarted = false;

      try {
        // Use streaming
        await api.postJsonStream(
          "/chat",
          { question: q, session_id: sessionId, stream: true },
          (token) => {
            // On first token, add message and switch to streaming mode
            if (!streamStarted) {
              streamStarted = true;
              setStreaming(true);
              setMessages((prev) => [...prev, { role: "assistant", text: "" }]);
            }
            fullAnswer += token;
            // Update last message with accumulated text
            setMessages((prev) => {
              const updated = [...prev];
              updated[updated.length - 1] = { role: "assistant", text: fullAnswer };
              return updated;
            });
          }
        );

        // TTS after complete
        if (speakAllRef.current && fullAnswer) {
          speak(fullAnswer);
        }
      } catch (err) {
        // Fallback to non-streaming if SSE fails
        console.warn("Streaming failed, falling back:", err);
        setStreaming(false);
        try {
          const res = await api.postJson("/chat", {
            question: q,
            session_id: sessionId,
          });
          const answer = res?.answer || "No answer.";
          // Add message if streaming never started
          if (!streamStarted) {
            setMessages((prev) => [...prev, { role: "assistant", text: answer }]);
          } else {
            setMessages((prev) => {
              const updated = [...prev];
              updated[updated.length - 1] = { role: "assistant", text: answer };
              return updated;
            });
          }
          if (speakAllRef.current) speak(answer);
        } catch (fallbackErr) {
          const errorText = "Error: " + fallbackErr.message;
          if (!streamStarted) {
            setMessages((prev) => [...prev, { role: "assistant", text: errorText }]);
          } else {
            setMessages((prev) => {
              const updated = [...prev];
              updated[updated.length - 1] = { role: "assistant", text: errorText };
              return updated;
            });
          }
        }
      } finally {
        setStreaming(false);
      }
    },
    [api, sessionId, speak]
  );

  // Send handler
  const onSend = useCallback(
    async (e) => {
      e?.preventDefault?.();
      const q = input.trim();
      if (!q || sending) return;
      setSending(true);
      setMessages((prev) => [...prev, { role: "user", text: q }]);
      setInput("");
      try {
        await askOnce(q);
      } catch (err) {
        setMessages((prev) => [...prev, { role: "assistant", text: "Error: " + err.message }]);
      } finally {
        setSending(false);
      }
    },
    [input, sending, askOnce]
  );

  // Toggle speak with proper cleanup
  const toggleSpeak = useCallback(() => {
    setSpeakAll((prev) => {
      if (prev) {
        // Turning OFF - stop and cleanup
        const el = audioRef.current;
        if (el) {
          try {
            el.pause();
            el.currentTime = 0;
            el.src = "";
          } catch (err) {
            console.error("Audio cleanup error:", err);
          }
        }
        // Cleanup URL
        if (currentAudioUrlRef.current) {
          URL.revokeObjectURL(currentAudioUrlRef.current);
          currentAudioUrlRef.current = null;
        }
      }
      return !prev;
    });
  }, []);

  // Mic: prepare recorder
  const ensureRecorder = useCallback(async () => {
    if (mediaRecorderRef.current) return mediaRecorderRef.current;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const opts = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? { mimeType: "audio/webm;codecs=opus" }
        : undefined;
      const rec = new MediaRecorder(stream, opts);

      chunksRef.current = [];
      rec.ondataavailable = (ev) => {
        if (ev.data?.size) chunksRef.current.push(ev.data);
      };
      rec.onstart = () => setRecording(true);

      rec.onstop = async () => {
        setRecording(false);

        // Too short click -> hint
        const dt = Date.now() - (pressStartedAtRef.current || 0);
        if (dt < 300) {
          push({
            role: "assistant",
            text: "To talk, keep the mic button pressed.",
          });
          chunksRef.current = [];
          return;
        }

        const blob = new Blob(chunksRef.current, {
          type: rec.mimeType || "audio/webm",
        });
        chunksRef.current = [];

        // Validate blob size
        if (blob.size < 1000) {
          push({
            role: "assistant",
            text: "Recording too short. Hold the button longer.",
          });
          return;
        }

        setSending(true);

        try {
          // STT - transcribe audio
          const form = new FormData();
          form.append("audio", blob, "recording.webm");
          const stt = await api.postFormJson(
            `/stt?language=${encodeURIComponent(language)}`,
            form
          );
          const transcript = (stt?.text || "").trim();

          if (!transcript) {
            push({
              role: "assistant",
              text: "I couldn't hear anything. Try again.",
            });
            return;
          }

          // Show user's transcribed message
          push({ role: "user", text: transcript });

          // Chat - get response (reuse askOnce logic)
          const res = await api.postJson("/chat", {
            question: transcript,
            session_id: sessionId,
          });
          const answer = (res?.answer || "").trim();

          push({ role: "assistant", text: answer || "(no answer)" });

          // TTS - speak response if enabled
          if (speakAllRef.current && answer) {
            speak(answer);
          }
        } catch (e) {
          const errorMsg = e.message || "Unknown error";
          console.error("Voice flow error:", e);

          // User-friendly error messages
          if (errorMsg.includes("transcribe") || errorMsg.includes("audio")) {
            push({ role: "assistant", text: "Couldn't understand the audio. Please try again." });
          } else if (errorMsg.includes("network") || errorMsg.includes("fetch")) {
            push({ role: "assistant", text: "Connection error. Check your internet." });
          } else {
            push({ role: "assistant", text: `Voice error: ${errorMsg}` });
          }
        } finally {
          setSending(false);
        }
      };

      mediaRecorderRef.current = rec;
      return rec;
    } catch (e) {
      setRecSupported(false);
      push({
        role: "assistant",
        text: "Mic permission denied or unsupported browser.",
      });
      throw e;
    }
  }, [api, sessionId, push, speak]);

  const startRec = useCallback(async () => {
    try {
      const rec = await ensureRecorder();
      if (rec.state !== "inactive") return;
      pressStartedAtRef.current = Date.now();
      rec.start();
    } catch (err) {
      console.error("startRec error:", err);
    }
  }, [ensureRecorder]);

  const stopRec = useCallback(() => {
    const r = mediaRecorderRef.current;
    if (r && r.state === "recording") r.stop();
  }, []);

  // Handle input change
  const handleInputChange = useCallback((e) => setInput(e.target.value), []);

  // Handle key down
  const handleKeyDown = useCallback(
    (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        onSend();
      }
    },
    [onSend]
  );

  // Handle touch start
  const handleTouchStart = useCallback(
    (e) => {
      e.preventDefault();
      startRec();
    },
    [startRec]
  );

  return (
    <ErrorBoundary>
      <div className="app">
        <header className="header">
          <div className="header-inner">
            <div className="brand">
              <div className="brand-icon" aria-hidden>
                <div></div>
                <div></div>
                <div></div>
                <div></div>
              </div>
              <div>Chatbleton - Your Ableton Assistant</div>
            </div>
            <div className="header-actions">
              <button
                className={"btn " + (speakAll ? "btn-accent" : "")}
                onClick={toggleSpeak}
              >
                {speakAll ? "Speak: ON" : "Speak: OFF"}
              </button>
            </div>
          </div>
        </header>

        <main className="main">
          <div className="container">
            <div ref={listRef} className="timeline">
              {messages.map((m, i) => (
                <Message key={i} role={m.role} text={m.text} />
              ))}
              {sending && !streaming && <LoadingBubble />}
            </div>
          </div>
        </main>

        <footer className="footer">
          <div className="inputbar">
            <form onSubmit={onSend} className="input-wrap">
              <textarea
                className="textarea"
                placeholder="Message Ableton Assistant..."
                value={input}
                onChange={handleInputChange}
                onKeyDown={handleKeyDown}
              />
              <div className="actions">
                {recSupported ? (
                  <button
                    type="button"
                    title="Hold to talk"
                    className={"iconbtn " + (recording ? "ping" : "")}
                    onMouseDown={startRec}
                    onMouseUp={stopRec}
                    onMouseLeave={stopRec}
                    onTouchStart={handleTouchStart}
                    onTouchEnd={stopRec}
                  >
                    <MicIcon />
                  </button>
                ) : (
                  <button
                    type="button"
                    className="iconbtn"
                    title="Mic unavailable"
                    disabled
                  >
                    <NoMicIcon />
                  </button>
                )}
                <button type="submit" title="Send" className="iconbtn">
                  <SendIcon />
                </button>
              </div>
            </form>
            <div className="small">Press Enter to send - Hold mic to speak</div>
          </div>
        </footer>

        <audio ref={audioRef} hidden />
      </div>
    </ErrorBoundary>
  );
}

// Icon components (avoiding emoji for better accessibility)
const MicIcon = memo(() => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
    <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm-1-9c0-.55.45-1 1-1s1 .45 1 1v6c0 .55-.45 1-1 1s-1-.45-1-1V5zm6 6c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z" />
  </svg>
));

const NoMicIcon = memo(() => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
    <path d="M19 11h-1.7c0 .74-.16 1.43-.43 2.05l1.23 1.23c.56-.98.9-2.09.9-3.28zm-4.02.17c0-.06.02-.11.02-.17V5c0-1.66-1.34-3-3-3S9 3.34 9 5v.18l5.98 5.99zM4.27 3L3 4.27l6.01 6.01V11c0 1.66 1.33 3 2.99 3 .22 0 .44-.03.65-.08l1.66 1.66c-.71.33-1.5.52-2.31.52-2.76 0-5.3-2.1-5.3-5.1H5c0 3.41 2.72 6.23 6 6.72V21h2v-3.28c.91-.13 1.77-.45 2.54-.9L19.73 21 21 19.73 4.27 3z" />
  </svg>
));

const SendIcon = memo(() => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
    <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
  </svg>
));

// API helper
function makeApi(baseUrl) {
  const url = (p) => baseUrl.replace(/\/$/, "") + p;

  return {
    async postJson(path, body) {
      const fullUrl = url(path);
      console.log("[API] POST", fullUrl);
      const res = await fetch(fullUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body || {}),
      });
      if (!res.ok) {
        let e = null;
        try {
          e = await res.json();
        } catch (parseErr) {
          console.warn("Could not parse error response:", parseErr);
        }
        throw new Error(e?.error || `${res.status} ${res.statusText}`);
      }
      return res.json();
    },

    // Streaming POST with Server-Sent Events
    async postJsonStream(path, body, onToken) {
      const fullUrl = url(path);
      console.log("[API] POST Stream", fullUrl);

      const res = await fetch(fullUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body || {}),
      });

      if (!res.ok) {
        let e = null;
        try {
          e = await res.json();
        } catch (parseErr) {
          console.warn("Could not parse error response:", parseErr);
        }
        throw new Error(e?.error || `${res.status} ${res.statusText}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // Parse SSE messages
        const lines = buffer.split("\n");
        buffer = lines.pop() || ""; // Keep incomplete line in buffer

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6));
              if (data.token) {
                onToken(data.token);
              }
              if (data.done) {
                return;
              }
              if (data.error) {
                throw new Error(data.error);
              }
            } catch (parseErr) {
              // Ignore parse errors for incomplete JSON
            }
          }
        }
      }
    },

    async postJsonForBlob(path, body) {
      const res = await fetch(url(path), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body || {}),
      });
      if (!res.ok) {
        let e = null;
        try {
          e = await res.json();
        } catch (parseErr) {
          console.warn("Could not parse error response:", parseErr);
        }
        throw new Error(e?.error || `${res.status} ${res.statusText}`);
      }
      return res.blob();
    },

    async postFormJson(path, form) {
      const res = await fetch(url(path), { method: "POST", body: form });
      if (!res.ok) {
        let e = null;
        try {
          e = await res.json();
        } catch (parseErr) {
          console.warn("Could not parse error response:", parseErr);
        }
        throw new Error(e?.error || `${res.status} ${res.statusText}`);
      }
      return res.json();
    },
  };
}
