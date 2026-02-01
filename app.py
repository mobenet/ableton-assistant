# app.py
"""
Chatbleton - Ableton Live AI Assistant
Flask backend with RAG, voice capabilities, and music tools.
"""
import os
import io
import re
import sys
import json
import html
import logging
import tempfile
import base64
from functools import wraps

from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory, send_file, g

load_dotenv()

# =============================================================================
# Configuration
# =============================================================================

class Config:
    """Application configuration from environment variables."""

    # Required
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Optional with defaults
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_STT_MODEL = os.getenv("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe")
    OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
    PORT = int(os.getenv("PORT", 8000))
    DEBUG = os.getenv("DEBUG", "").lower() in ("1", "true")
    ENV = os.getenv("FLASK_ENV", "production")

    # Security
    MAX_QUESTION_LENGTH = int(os.getenv("MAX_QUESTION_LENGTH", 2000))
    MAX_AUDIO_SIZE_MB = int(os.getenv("MAX_AUDIO_SIZE_MB", 10))
    RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", 30))

    @classmethod
    def validate(cls):
        """Validate required configuration."""
        if not cls.OPENAI_API_KEY:
            logger.error("Missing required OPENAI_API_KEY")
            sys.exit(1)
        return True


# =============================================================================
# Logging Setup
# =============================================================================

logging.basicConfig(
    level=logging.DEBUG if Config.DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Validate configuration
Config.validate()
logger.info(f"Starting Chatbleton in {Config.ENV} mode")

# =============================================================================
# Flask App Setup
# =============================================================================

from flask_cors import CORS
from openai import OpenAI

from agent_tools import agent_ask, reset_all_sessions
from agent_tools import reset_session as _reset_session

client = OpenAI()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_AUDIO_SIZE_MB * 1024 * 1024

# CORS configuration
CORS(app, origins="*", supports_credentials=True)

# =============================================================================
# Rate Limiting (Simple in-memory implementation)
# =============================================================================

from collections import defaultdict
from time import time
import threading

class RateLimiter:
    """Simple in-memory rate limiter."""

    def __init__(self, requests_per_minute: int = 30):
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)
        self.lock = threading.Lock()

    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed for given key."""
        now = time()
        minute_ago = now - 60

        with self.lock:
            # Clean old requests
            self.requests[key] = [t for t in self.requests[key] if t > minute_ago]

            if len(self.requests[key]) >= self.requests_per_minute:
                return False

            self.requests[key].append(now)
            return True

    def get_client_key(self) -> str:
        """Get unique key for current client."""
        return request.headers.get('X-Forwarded-For', request.remote_addr) or 'unknown'


rate_limiter = RateLimiter(Config.RATE_LIMIT_PER_MINUTE)


def rate_limit(f):
    """Decorator to apply rate limiting."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        client_key = rate_limiter.get_client_key()
        if not rate_limiter.is_allowed(client_key):
            logger.warning(f"Rate limit exceeded for {client_key}")
            return jsonify({
                "error": "Rate limit exceeded. Please wait before making more requests."
            }), 429
        return f(*args, **kwargs)
    return decorated_function


# =============================================================================
# Security Helpers
# =============================================================================

def sanitize_input(text: str, max_length: int = None) -> str:
    """Sanitize user input to prevent injection attacks."""
    if not text:
        return ""

    # Remove null bytes and control characters (except newlines/tabs)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # Limit length
    if max_length:
        text = text[:max_length]

    return text.strip()


def sanitize_session_id(session_id: str) -> str:
    """Sanitize session ID to alphanumeric + limited special chars."""
    if not session_id:
        return "default"
    # Only allow alphanumeric, dash, underscore
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '', session_id)
    return sanitized[:50] or "default"


# =============================================================================
# Middleware
# =============================================================================

@app.before_request
def before_request():
    """Pre-request processing."""
    g.start_time = time()


@app.after_request
def after_request(response):
    """Add security headers and logging."""
    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

    # Don't expose server info in production
    if Config.ENV == "production":
        response.headers["Server"] = "Chatbleton"

    # Request logging
    if hasattr(g, 'start_time'):
        elapsed = (time() - g.start_time) * 1000
        logger.debug(f"{request.method} {request.path} - {response.status_code} ({elapsed:.1f}ms)")

    return response


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large errors."""
    return jsonify({"error": f"File too large. Maximum size is {Config.MAX_AUDIO_SIZE_MB}MB"}), 413


@app.errorhandler(429)
def rate_limit_exceeded(error):
    """Handle rate limit errors."""
    return jsonify({"error": "Rate limit exceeded"}), 429


@app.errorhandler(500)
def internal_error(error):
    """Handle internal errors without exposing details in production."""
    logger.exception("Internal server error")
    if Config.ENV == "production":
        return jsonify({"error": "An internal error occurred"}), 500
    return jsonify({"error": str(error)}), 500


# =============================================================================
# Audio Helpers
# =============================================================================

def read_audio_from_request() -> tuple[str, io.BufferedReader]:
    """
    Read audio from request (multipart or base64 JSON).

    Returns:
        Tuple of (cleanup_path, file_object)

    Raises:
        ValueError: If no audio provided or invalid format
    """
    cleanup_path = None
    file_obj = None

    if "audio" in request.files:
        up = request.files["audio"]
        if not up.filename:
            raise ValueError("Empty audio file")

        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            up.save(tmp)
            cleanup_path = tmp.name
        file_obj = open(cleanup_path, "rb")
    else:
        payload = request.get_json(silent=True) or {}
        b64 = payload.get("audio_b64")
        if not b64:
            raise ValueError("Provide audio via multipart 'audio' or JSON 'audio_b64'")

        try:
            raw = base64.b64decode(b64)
        except Exception:
            raise ValueError("Invalid base64 audio data")

        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            tmp.write(raw)
            cleanup_path = tmp.name
        file_obj = open(cleanup_path, "rb")

    return cleanup_path, file_obj


def cleanup_temp_file(path: str, file_obj: io.BufferedReader = None):
    """Safely cleanup temporary file and file object."""
    if file_obj:
        try:
            file_obj.close()
        except Exception:
            pass
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/health")
def health():
    """Health check endpoint for monitoring."""
    return jsonify({
        "status": "ok",
        "service": "chatbleton",
        "version": "1.0.0",
        "features": {
            "agent": True,
            "voice": True,
            "rag": True
        }
    })


@app.post("/chat")
@rate_limit
def chat():
    """
    Chat endpoint with session memory.

    Request JSON:
        - question (str): User's question (required)
        - session_id (str): Session identifier (optional, default: "default")
        - stream (bool): Enable streaming response (optional, default: false)

    Response JSON (non-streaming):
        - answer (str): Assistant's response

    Response (streaming):
        - Server-Sent Events with tokens
    """
    payload = request.get_json(force=True) or {}

    # Validate and sanitize input
    question = sanitize_input(
        payload.get("question", ""),
        max_length=Config.MAX_QUESTION_LENGTH
    )
    session_id = sanitize_session_id(payload.get("session_id", "default"))
    stream = payload.get("stream", False)

    if not question:
        return jsonify({"error": "Missing or empty 'question'"}), 400

    logger.info(f"Chat [{session_id}]: {question[:50]}... (stream={stream})")

    if stream:
        return stream_chat_response(question, session_id)

    try:
        answer = agent_ask(question, session_id=session_id) or ""
        return jsonify({"answer": answer})
    except Exception as e:
        logger.exception("Chat error")
        if Config.ENV == "production":
            return jsonify({"error": "Failed to process your question"}), 500
        return jsonify({"error": str(e)}), 500


def stream_chat_response(question: str, session_id: str):
    """Stream chat response using Server-Sent Events."""
    from flask import Response, stream_with_context

    def generate():
        try:
            # Get the full answer first (agent doesn't support streaming directly)
            answer = agent_ask(question, session_id=session_id) or ""

            # Stream it character by character with small chunks
            chunk_size = 4  # Characters per chunk
            for i in range(0, len(answer), chunk_size):
                chunk = answer[i:i + chunk_size]
                yield f"data: {json.dumps({'token': chunk})}\n\n"

            # Send done signal
            yield f"data: {json.dumps({'done': True})}\n\n"

        except Exception as e:
            logger.exception("Stream chat error")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )


@app.post("/tts")
@rate_limit
def tts():
    """
    Text-to-Speech endpoint.

    Request JSON:
        - text (str): Text to convert (required, max 4000 chars)
        - voice (str): Voice name (optional, default: "alloy")
        - format (str): Audio format (optional, default: "mp3")
        - speed (float): Speed 0.25-4.0 (optional, default: 1.0)

    Response: Audio file blob with appropriate MIME type
    """
    payload = request.get_json(force=True) or {}

    text = sanitize_input(payload.get("text", ""), max_length=4000)
    voice = sanitize_input(payload.get("voice", "alloy"), max_length=20)
    fmt = sanitize_input(payload.get("format", "mp3"), max_length=10).lower()

    # Validate speed (0.25 to 4.0)
    try:
        speed = float(payload.get("speed", 1.0))
        speed = max(0.25, min(4.0, speed))
    except (ValueError, TypeError):
        speed = 1.0

    if not text:
        return jsonify({"error": "Missing 'text'"}), 400

    if len(text) < 1:
        return jsonify({"error": "Text too short"}), 400

    # Validate voice
    valid_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    if voice not in valid_voices:
        voice = "alloy"

    # Validate format
    mime_map = {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "flac": "audio/flac",
        "opus": "audio/opus",
        "aac": "audio/aac"
    }
    if fmt not in mime_map:
        fmt = "mp3"
    mimetype = mime_map[fmt]

    logger.info(f"TTS: {len(text)} chars, voice={voice}, format={fmt}, speed={speed}")

    try:
        # Stream directly to memory buffer (no temp file needed)
        response = client.audio.speech.create(
            model=Config.OPENAI_TTS_MODEL,
            voice=voice,
            input=text,
            response_format=fmt,
            speed=speed
        )

        # Get audio content directly
        buf = io.BytesIO(response.content)
        buf.seek(0)

        # Add cache headers for identical requests
        resp = send_file(
            buf,
            mimetype=mimetype,
            as_attachment=False,
            download_name=f"speech.{fmt}"
        )
        resp.headers["Cache-Control"] = "private, max-age=3600"
        return resp

    except Exception as e:
        logger.exception("TTS error")
        error_msg = str(e)

        # Handle specific OpenAI errors
        if "rate_limit" in error_msg.lower():
            return jsonify({"error": "TTS rate limit reached. Please wait."}), 429
        if "invalid_api_key" in error_msg.lower():
            return jsonify({"error": "API configuration error"}), 500

        if Config.ENV == "production":
            return jsonify({"error": "Text-to-speech failed"}), 500
        return jsonify({"error": error_msg}), 500


@app.post("/stt")
@rate_limit
def stt():
    """
    Speech-to-Text endpoint.

    Request:
        - multipart/form-data with 'audio' file, OR
        - JSON with 'audio_b64' (base64 encoded audio)

    Supported formats: mp3, mp4, mpeg, mpga, m4a, wav, webm

    Optional params (query or JSON):
        - language: ISO 639-1 code (e.g., "en", "es", "ca")
        - prompt: Context prompt for better transcription (e.g., "Ableton, MIDI, sidechain")

    Response JSON:
        - text (str): Transcribed text
        - model (str): Model used
        - language (str): Detected/specified language
        - duration (float): Audio duration in seconds (if available)
    """
    payload = request.get_json(silent=True) or {}

    # Validate language code (ISO 639-1)
    language = sanitize_input(
        request.args.get("language") or payload.get("language", ""),
        max_length=5
    ).lower()

    # Validate language is a valid ISO code (basic check)
    valid_languages = ["en", "es", "ca", "de", "fr", "it", "pt", "nl", "pl", "ru", "zh", "ja", "ko"]
    if language and language not in valid_languages:
        language = ""  # Let Whisper auto-detect

    # Prompt helps with domain-specific terms
    prompt = sanitize_input(
        request.args.get("prompt") or payload.get("prompt", ""),
        max_length=500
    )

    # Default prompt for music production context
    if not prompt:
        prompt = "Ableton Live, MIDI, audio, DAW, synthesizer, compression, EQ"

    cleanup_path = None
    file_obj = None

    try:
        cleanup_path, file_obj = read_audio_from_request()

        # Validate file size (max 25MB for Whisper)
        file_obj.seek(0, 2)  # Seek to end
        file_size = file_obj.tell()
        file_obj.seek(0)  # Reset to beginning

        if file_size > 25 * 1024 * 1024:
            return jsonify({"error": "Audio file too large. Maximum 25MB."}), 413

        if file_size < 1000:  # Less than 1KB is probably empty/corrupt
            return jsonify({"error": "Audio file too small or empty."}), 400

        logger.info(f"STT: {file_size/1024:.1f}KB, language={language or 'auto'}")

        kwargs = {
            "model": Config.OPENAI_STT_MODEL,
            "file": file_obj,
            "prompt": prompt
        }
        if language:
            kwargs["language"] = language

        tr = client.audio.transcriptions.create(**kwargs)

        text = getattr(tr, "text", None)
        if text is None and isinstance(tr, dict):
            text = tr.get("text")

        if not text or not text.strip():
            return jsonify({
                "error": "No speech detected in audio",
                "hint": "Make sure to speak clearly and check microphone permissions"
            }), 422

        # Clean up transcription
        text = text.strip()

        logger.info(f"STT result: {len(text)} chars")

        return jsonify({
            "text": text,
            "model": Config.OPENAI_STT_MODEL,
            "language": language or "auto",
            "file_size_kb": round(file_size / 1024, 1)
        })

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.exception("STT error")
        error_msg = str(e)

        # Handle specific errors
        if "audio" in error_msg.lower() and "format" in error_msg.lower():
            return jsonify({"error": "Unsupported audio format. Use webm, mp3, or wav."}), 400
        if "rate_limit" in error_msg.lower():
            return jsonify({"error": "STT rate limit reached. Please wait."}), 429

        if Config.ENV == "production":
            return jsonify({"error": "Speech-to-text failed"}), 500
        return jsonify({"error": error_msg}), 500
    finally:
        cleanup_temp_file(cleanup_path, file_obj)


@app.post("/reset_session")
@rate_limit
def reset_session():
    """Reset chat history for a specific session."""
    payload = request.get_json(silent=True) or {}
    session_id = sanitize_session_id(
        payload.get("session_id") or request.args.get("session_id") or "default"
    )
    _reset_session(session_id)
    logger.info(f"Session reset: {session_id}")
    return jsonify({"ok": True, "session_id": session_id})


@app.post("/reset_all")
@rate_limit
def reset_all():
    """Reset all chat sessions (admin endpoint)."""
    reset_all_sessions()
    logger.info("All sessions reset")
    return jsonify({"ok": True})


# =============================================================================
# Frontend Serving
# =============================================================================

FRONT_DIST = os.path.join(os.path.dirname(__file__), "frontend", "dist")


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    """Serve React frontend build."""
    file_path = os.path.join(FRONT_DIST, path)
    if path and os.path.exists(file_path):
        return send_from_directory(FRONT_DIST, path)
    index = os.path.join(FRONT_DIST, "index.html")
    if os.path.exists(index):
        return send_from_directory(FRONT_DIST, "index.html")
    return jsonify({"status": "ok", "message": "API is running. Frontend not built."})


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    logger.info(f"Starting Chatbleton on port {Config.PORT}")
    app.run(
        host="0.0.0.0",
        port=Config.PORT,
        debug=Config.DEBUG
    )
