# app.py
import os, io, tempfile, base64
from functools import partial
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory, send_file, Response, stream_with_context
from flask_cors import CORS
from openai import OpenAI

from agent_tools import agent_ask, reset_all_sessions
from agent_tools import reset_session as _reset_session


load_dotenv()
client = OpenAI()

app = Flask(__name__)
CORS(app)

# ---------- Utils ----------
def _read_audio_from_request() -> tuple[str, io.BufferedReader]:
    """
    Devuelve (cleanup_path, file_handle) listo para pasar a OpenAI STT.
    Cierra/borra el tmp en quien lo llame.
    """
    cleanup_path = None
    file_obj = None
    if "audio" in request.files:  # multipart/form-data
        up = request.files["audio"]
        # Guardar SIEMPRE a disco para evitar el error de FileStorage
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            up.save(tmp)
            cleanup_path = tmp.name
        file_obj = open(cleanup_path, "rb")
    else:
        payload = request.get_json(silent=True) or {}
        b64 = payload.get("audio_b64")
        if not b64:
            raise ValueError("Provide audio via multipart 'audio' or JSON 'audio_b64'.")
        raw = base64.b64decode(b64)
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            tmp.write(raw)
            cleanup_path = tmp.name
        file_obj = open(cleanup_path, "rb")
    return cleanup_path, file_obj

# ---------- Health ----------
@app.get("/health")
def health():
    return jsonify({"status": "ok", "agent": True, "voice": True})

# ---------- Chat con streaming (AGENTE) ----------
@app.get("/chat_stream")
def chat_stream():
    """
    Uso:
      GET /chat_stream?q=...&session_id=demo
    Respuesta: text/plain con chunks de texto (streaming sencillo).
    """
    q = (request.args.get("q") or "").strip()
    session_id = (request.args.get("session_id") or "default").strip()
    if not q:
        return jsonify({"error": "Missing 'q'"}), 400
    END = "\n<<END_OF_MESSAGE>>"
    def gen():
        # 1) Pedimos la respuesta completa al agente
        try:
            answer = agent_ask(q, session_id=session_id) or ""
        except Exception as e:
            yield f"[error] {e}{END}"
            return
        # 2) La “troceamos” para que el front la reciba en tiempo real
        CHUNK = 256
        for i in range(0, len(answer), CHUNK):
            yield answer[i:i+CHUNK] + "\n"
        yield END
    return Response(
        stream_with_context(gen()),
        mimetype="text/plain",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
# --- Chat sense streaming (AGENT) ---
@app.post("/chat")
def chat_once():
    """
    Body JSON: { "question": "...", "session_id": "default" }
    Resposta:  { "answer": "..." }
    """
    payload = request.get_json(force=True) or {}
    q = (payload.get("question") or "").strip()
    session_id = (payload.get("session_id") or "default").strip()
    if not q:
        return jsonify({"error": "Missing 'question'"}), 400
    try:
        answer = agent_ask(q, session_id=session_id) or ""
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
# ---------- Round-trip de VOZ ----------
@app.post("/voice")
def voice_roundtrip():
    """
    1) Recibe audio (multipart 'audio' o JSON 'audio_b64')
    2) STT (OpenAI) -> texto
    3) Agente -> respuesta texto
    4) TTS (OpenAI) -> audio (mp3 por defecto)
    """
    session_id = (request.args.get("session_id") or "default").strip()
    stt_model = os.getenv("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe")
    tts_model = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
    voice = (request.args.get("voice") or "alloy").strip()
    fmt = (request.args.get("format") or "mp3").strip().lower()

    mime_by_fmt = {"mp3": "audio/mpeg", "wav": "audio/wav", "flac": "audio/flac", "pcm": "audio/wave"}
    mimetype = mime_by_fmt.get(fmt, "audio/mpeg")
    ext = fmt if fmt in mime_by_fmt else "mp3"

    cleanup_path = None
    out_path = None
    try:
        # 1) Audio IN
        cleanup_path, file_obj = _read_audio_from_request()

        # 2) STT
        tr = client.audio.transcriptions.create(model=stt_model, file=file_obj)
        file_obj.close()
        question = getattr(tr, "text", None) or (tr.get("text") if isinstance(tr, dict) else None)
        if not question:
            return jsonify({"error": "STT returned no text"}), 422

        # 3) Agente
        answer = agent_ask(question, session_id=session_id)

        # 4) TTS
        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp_out:
            out_path = tmp_out.name

        with client.audio.speech.with_streaming_response.create(
            model=tts_model, voice=voice, input=answer, response_format=fmt
        ) as resp:
            resp.stream_to_file(out_path)

        buf = io.BytesIO()
        with open(out_path, "rb") as f:
            buf.write(f.read())
        buf.seek(0)

        return send_file(buf, mimetype=mimetype, as_attachment=False, download_name=f"answer.{ext}")

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"voice failed: {e}"}), 500
    finally:
        try:
            if cleanup_path and os.path.exists(cleanup_path):
                os.remove(cleanup_path)
        except Exception:
            pass
        try:
            if out_path and os.path.exists(out_path):
                os.remove(out_path)
        except Exception:
            pass

@app.post("/reset_session")
def reset_session_route():
    payload = request.get_json(silent=True) or {}
    session_id = (payload.get("session_id") or request.args.get("session_id") or "default").strip()
    _reset_session(session_id)
    return jsonify({"ok": True, "session_id": session_id})
@app.post("/reset_all")

def reset_all_route():
    reset_all_sessions()
    return jsonify({"ok": True})
# ---------- Servir frontend (opcional) ----------
FRONT_DIST = os.path.join(os.path.dirname(__file__), "frontend", "dist")

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    """
    Sirve el build de React (si existe). Deja este bloque al final.
    """
    file_path = os.path.join(FRONT_DIST, path)
    if path and os.path.exists(file_path):
        return send_from_directory(FRONT_DIST, path)
    index = os.path.join(FRONT_DIST, "index.html")
    if os.path.exists(index):
        return send_from_directory(FRONT_DIST, "index.html")
    return jsonify({"status": "ok", "frontend": False})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
