# app.py
import io, os, sys, tempfile, base64
from openai import OpenAI
import io
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

from rag import answer_query  # fa el RAG
from agent_tools import agent_ask, reset_agent

load_dotenv()
client = OpenAI()

app = Flask(__name__)
CORS(app)


@app.get("/health")
def health():
    return jsonify({"status": "ok", "tts": bool(_tts)})

@app.post("/ask_text")
def ask_text():
    payload = request.get_json(force=True) or {}
    question = (payload.get("question" or "").strip())
    k = int(payload.get("k", 5))
    if not question:
        return jsonify({"error": "Missing 'question'"}), 400
    result = answer_query(question, k=k)
    return jsonify(result)


@app.post("/ask")
def ask():
    payload = request.get_json(force=True) or {}
    if payload.get("reset_agent"):
        reset_agent()
    question = (payload.get("question") or "").strip()
    mode = (payload.get("mode") or "agent").lower()  # "agent" | "rag"
    session_id = (payload.get("session_id") or "default").strip()
    
    k = int(payload.get("k", 5))  # usado sólo por /ask_text o dentro de RAG si lo adaptas

    if not question:
        return jsonify({"error": "Missing 'question'"}), 400

    try:
        if mode == "rag":
            res = answer_query(question, k=k)
            return jsonify({"mode": "rag", **res})
        # agent (por defecto)
        out = agent_ask(question, session_id=session_id)
        return jsonify({"mode": "agent", "answer": out})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    



@app.post("/ask_tts")
def ask_tts():
    """
    1) Respon la pregunta amb l'agent (o RAG si mode='rag')
    2) Converteix la resposta a veu amb OpenAI TTS
    3) Retorna àudio (mp3/wav/flac/pcm)
    """
    payload = request.get_json(force=True) or {}
    question   = (payload.get("question") or "").strip()
    session_id = (payload.get("session_id") or "default").strip()
    mode       = (payload.get("mode") or "agent").lower()   # "agent" | "rag"
    voice      = (payload.get("voice") or "alloy").strip()
    fmt        = (payload.get("format") or "mp3").strip().lower()
    tts_model  = (payload.get("tts_model") or os.getenv("OPENAI_TTS_MODEL","gpt-4o-mini-tts")).strip()

    if not question:
        return {"error": "Missing 'question'"}, 400

    # 1) Obtenir resposta
    if mode == "rag":
        answer = answer_query(question, k=int(payload.get("k", 5)))["answer"]
    else:
        answer = agent_ask(question, session_id=session_id)

    # 2) TTS amb OpenAI (streaming a fitxer temporal)
    mime_by_fmt = {"mp3":"audio/mpeg","wav":"audio/wav","flac":"audio/flac","pcm":"audio/wave"}
    mimetype = mime_by_fmt.get(fmt, "audio/mpeg")
    ext = fmt if fmt in mime_by_fmt else "mp3"

    try:
        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
            tmp_path = tmp.name

        with client.audio.speech.with_streaming_response.create(
            model=tts_model,          # ex: gpt-4o-mini-tts o tts-1
            voice=voice,              # ex: alloy
            input=answer,
            response_format=fmt
        ) as response:
            response.stream_to_file(tmp_path)

        buf = io.BytesIO()
        with open(tmp_path, "rb") as f:
            buf.write(f.read())
        buf.seek(0)
        os.remove(tmp_path)

        return send_file(buf, mimetype=mimetype, as_attachment=False, download_name=f"answer.{ext}")

    except Exception as e:
        return {"error": f"ask_tts failed: {e}"}, 500

@app.post("/tts")
def tts_openai():
    payload = request.get_json(force=True) or {}
    text   = (payload.get("text") or "").strip()
    voice  = (payload.get("voice") or "alloy").strip()       # veus integrades (ex: alloy, coral…)
    fmt    = (payload.get("format") or "mp3").strip().lower()  # mp3|wav|flac|pcm
    model  = (payload.get("model") or os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")).strip()

    if not text:
        return jsonify({"error": "Missing 'text'"}), 400

    # Tria el mime pel format
    mime_by_fmt = {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "flac": "audio/flac",
        "pcm": "audio/wave",  # PCM s'envia com WAV container habitualment
    }
    mimetype = mime_by_fmt.get(fmt, "audio/mpeg")
    ext = "mp3" if fmt not in mime_by_fmt else fmt

    # Fem servir la ruta “streaming_response” (recomanada als docs)
    # i escrivim a un fitxer temporal.
    try:
        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
            tmp_path = tmp.name

        with client.audio.speech.with_streaming_response.create(
            model=model,       # ex: gpt-4o-mini-tts o tts-1 / tts-1-hd
            voice=voice,       # ex: alloy
            input=text,
            response_format=fmt
        ) as response:
            response.stream_to_file(tmp_path)

        # Enviem el binari al client i esborrem el tmp
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
    model = (request.args.get("model") or os.getenv("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe")).strip()
    language = (request.args.get("language") or "").strip()  # opcional, ex: "es", "en"
    prompt = (request.args.get("prompt") or "").strip()      # opcional: “context” per millorar STT

    file_obj = None
    cleanup_path = None

    try:
        if "audio" in request.files:
            # via multipart/form-data
            file_obj = request.files["audio"]
        else:
            # via JSON base64
            payload = request.get_json(silent=True) or {}
            audio_b64 = payload.get("audio_b64")
            if not audio_b64:
                return jsonify({"error": "Provide audio via multipart 'audio' file or JSON 'audio_b64'"}), 400
            raw = base64.b64decode(audio_b64)
            # escriu a temporal
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(raw)
                cleanup_path = tmp.name
            file_obj = open(cleanup_path, "rb")

        # Crida a l’API de transcripcions
        # Models vàlids a dia d’avui: gpt-4o-transcribe, gpt-4o-mini-transcribe, whisper-1
        # https://platform.openai.com/docs/api-reference/audio
        kwargs = {"model": model, "file": file_obj}
        if language:
            kwargs["language"] = language
        if prompt:
            kwargs["prompt"] = prompt

        result = client.audio.transcriptions.create(**kwargs)
        # El SDK retorna un objecte amb .text (i camps addicionals segons model)
        text = getattr(result, "text", None) or result.get("text") if isinstance(result, dict) else None

        return jsonify({
            "model": model,
            "language": language or None,
            "text": text,
        })

    except Exception as e:
        return jsonify({"error": f"STT failed: {e}"}), 500

    finally:
        try:
            if cleanup_path:
                os.remove(cleanup_path)
        except Exception:
            pass

from agent_tools import agent_ask
from openai import OpenAI
import tempfile, io, os, base64

client = OpenAI()

@app.post("/ask_voice")
def ask_voice():
    """
    1) Rep audio (multipart 'audio' o JSON base64 'audio_b64')
    2) STT (OpenAI) -> text
    3) Agent -> answer text
    4) TTS (OpenAI) -> retorna àudio (mp3 per defecte)
    """
    # --- 1) Llegir audio de la request ---
    file_obj = None
    cleanup_path = None
    try:
        if "audio" in request.files:  # multipart/form-data
            file_obj = request.files["audio"]
        else:  # JSON base64
            payload = request.get_json(silent=True) or {}
            b64 = payload.get("audio_b64")
            if not b64:
                return {"error": "Provide audio via multipart 'audio' or JSON 'audio_b64'."}, 400
            raw = base64.b64decode(b64)
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
                tmp.write(raw)
                cleanup_path = tmp.name
            file_obj = open(cleanup_path, "rb")

        # Parametres opcionals
        session_id = (request.args.get("session_id") or "default").strip()
        stt_model = os.getenv("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe")
        tts_model = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
        voice = request.args.get("voice", "alloy")
        fmt = (request.args.get("format") or "mp3").lower()

        # --- 2) STT (OpenAI) ---
        tr = client.audio.transcriptions.create(
            model=stt_model,  # ex: gpt-4o-mini-transcribe o whisper-1
            file=file_obj
        )
        question = getattr(tr, "text", None) or (tr.get("text") if isinstance(tr, dict) else None)
        if not question:
            return {"error": "STT returned no text"}, 422

        # --- 3) Agent ---
        answer = agent_ask(question, session_id=session_id)

        # --- 4) TTS (OpenAI) ---
        mime_by_fmt = {"mp3":"audio/mpeg","wav":"audio/wav","flac":"audio/flac","pcm":"audio/wave"}
        mimetype = mime_by_fmt.get(fmt, "audio/mpeg")
        ext = fmt if fmt in mime_by_fmt else "mp3"

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
        os.remove(out_path)

        return send_file(buf, mimetype=mimetype, as_attachment=False, download_name=f"answer.{ext}")

    except Exception as e:
        return {"error": f"ask_voice failed: {e}"}, 500
    finally:
        try:
            if cleanup_path:
                os.remove(cleanup_path)
        except Exception:
            pass

FRONT_DIST = os.path.join(os.path.dirname(__file__), "frontend", "dist")

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    """
    Serveix el build de React. Mantén aquest bloc AL FINAL del fitxer
    perquè no eclipsi els endpoints de l'API.
    """
    file_path = os.path.join(FRONT_DIST, path)
    if path and os.path.exists(file_path):
        return send_from_directory(FRONT_DIST, path)
    return send_from_directory(FRONT_DIST, "index.html")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
