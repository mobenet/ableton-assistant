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

def _read_audio_from_request() -> tuple[str, io.BufferedReader]:
    cleanup_path = None
    file_obj = None
    if "audio" in request.files:
        up = request.files["audio"]
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

@app.get("/health")
def health():
    return jsonify({"status": "ok", "agent": True, "voice": True})

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
    
@app.post("/tts")
def tts_openai():
    payload = request.get_json(force=True) or {}
    text  = (payload.get("text") or "").strip()
    voice = (payload.get("voice") or "alloy").strip()
    fmt   = (payload.get("format") or "mp3").strip().lower()
    model = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")

    if not text:
        return jsonify({"error": "Missing 'text'"}), 400

    mime_by_fmt = {"mp3":"audio/mpeg","wav":"audio/wav","flac":"audio/flac","pcm":"audio/wave"}
    mimetype = mime_by_fmt.get(fmt, "audio/mpeg")
    ext = fmt if fmt in mime_by_fmt else "mp3"

    try:
        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
            tmp_path = tmp.name

        with client.audio.speech.with_streaming_response.create(
            model=model, voice=voice, input=text, response_format=fmt
        ) as resp:
            resp.stream_to_file(tmp_path)

        buf = io.BytesIO()
        with open(tmp_path, "rb") as f:
            buf.write(f.read())
        buf.seek(0)
        os.remove(tmp_path)

        return send_file(buf, mimetype=mimetype, as_attachment=False, download_name=f"tts.{ext}")
    except Exception as e:
        return jsonify({"error": f"TTS failed: {e}"}), 500


@app.post("/stt")
def stt_openai():
    """
    Speech-to-Text.
    Accepta:
      - multipart/form-data amb 'audio' (webm/wav/mp3…)
      - JSON amb {"audio_b64": "<base64>"}
    Paràmetres opcionals (query o JSON): language, prompt
    Retorna: {"text": "...", "model": "...", "language": "..."}
    """
    stt_model = os.getenv("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe")

    payload = request.get_json(silent=True) or {}
    language = (request.args.get("language") or payload.get("language") or "").strip()
    prompt   = (request.args.get("prompt")   or payload.get("prompt")   or "").strip()

    cleanup_path = None
    file_obj = None
    try:
        cleanup_path, file_obj = _read_audio_from_request()

        kwargs = {"model": stt_model, "file": file_obj}
        if language:
            kwargs["language"] = language
        if prompt:
            kwargs["prompt"] = prompt

        tr = client.audio.transcriptions.create(**kwargs)
        text = getattr(tr, "text", None)
        if text is None and isinstance(tr, dict):
            text = tr.get("text")
        if not text:
            return jsonify({"error": "STT returned no text"}), 422

        return jsonify({"text": text, "model": stt_model, "language": language or None})

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"STT failed: {e}"}), 500
    finally:
        try:
            if file_obj:
                file_obj.close()
        except Exception:
            pass
        try:
            if cleanup_path and os.path.exists(cleanup_path):
                os.remove(cleanup_path)
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
