<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react&logoColor=black" alt="React">
  <img src="https://img.shields.io/badge/LangChain-1.2+-green?style=for-the-badge&logo=chainlink&logoColor=white" alt="LangChain">
  <img src="https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=for-the-badge&logo=openai&logoColor=white" alt="OpenAI">
  <img src="https://img.shields.io/badge/ChromaDB-Vector_Store-orange?style=for-the-badge" alt="ChromaDB">
</p>

<h1 align="center">Chatbleton</h1>

<p align="center">
  <strong>AI-Powered Ableton Live Assistant with Voice & RAG</strong>
</p>

<p align="center">
  An intelligent assistant that answers questions about Ableton Live using Retrieval-Augmented Generation (RAG) with voice input/output capabilities and specialized music production tools.
</p>

---

## Features

- **RAG-Powered Knowledge Base** - Searches through Ableton Live manuals and curated YouTube tutorials with timestamp references
- **Intelligent Agent** - Automatically falls back to web search (DuckDuckGo) when local knowledge is insufficient
- **Streaming Responses** - Real-time typing effect with Server-Sent Events (SSE)
- **Voice Interaction** - Speech-to-text input and text-to-speech output for hands-free operation
- **Music Production Tools** - Built-in tempo calculator (BPM → ms) and pitch converter (MIDI ↔ Hz)
- **Multilingual Support** - Works in English, Spanish and Catalan
- **Session Memory** - Maintains conversation context across interactions
- **Modern UI** - Dark theme with Ableton-inspired yellow accents

---

## Demo

```
You: How do I quantize audio clips in Ableton?

Chatbleton: To quantize audio clips in Ableton Live, you need to...
            [Detailed answer with timestamps]

            Sources:
            • https://youtube.com/watch?v=xxx&t=120s [2m0s]
            • Ableton Live Manual - Chapter 10
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Frontend (React 19)                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────────┐  │
│  │   Chat   │  │   Voice  │  │   TTS    │  │    Markdown    │  │
│  │   Input  │  │  Record  │  │  Output  │  │  + Sources     │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └───────┬────────┘  │
└───────┼─────────────┼─────────────┼────────────────┼───────────┘
        │             │             │                │
        ▼             ▼             ▼                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Backend (Flask + LangChain)                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    LangChain Agent                        │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐  │   │
│  │  │  RAG Tool   │  │ Web Search  │  │   Music Tools    │  │   │
│  │  │ (ChromaDB)  │  │ (DuckDuckGo)│  │ (Tempo/Pitch)    │  │   │
│  │  └──────┬──────┘  └──────┬──────┘  └────────┬─────────┘  │   │
│  └─────────┼────────────────┼──────────────────┼────────────┘   │
│            ▼                ▼                  ▼                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ E5 Embeddings│  │   OpenAI     │  │   OpenAI Whisper     │   │
│  │ (Multilingual)│ │   GPT-4o     │  │    STT / TTS         │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | React 19, Vite 7, react-markdown, remark-gfm |
| **Backend** | Python 3.10+, Flask 3.x, LangChain 1.2+ |
| **LLM** | OpenAI GPT-4o-mini |
| **Embeddings** | intfloat/multilingual-e5-base (HuggingFace) |
| **Vector Store** | ChromaDB (local, persistent) |
| **Voice** | OpenAI Whisper (STT), OpenAI TTS |
| **Web Search** | DuckDuckGo (ddgs) - fallback when RAG fails |

---

## Prerequisites

Before you begin, ensure you have:

- **Python 3.10+** installed
- **Node.js 18+** installed
- **OpenAI API key** ([Get one here](https://platform.openai.com/api-keys))
- **~4GB RAM** for embedding model

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/chatbleton.git
cd chatbleton
```

### 2. Set up environment variables

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your OpenAI API key
# Required: OPENAI_API_KEY=sk-your-key-here
```

### 3. Create Python virtual environment

```bash
# Create virtual environment
python -m venv .venv

# Activate it
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

> **Note:** First run will download the E5 embedding model (~400MB)

### 5. Build the vector index

```bash
python build_index.py --rebuild
```

This creates the ChromaDB vector store from:
- `data/manual_chunks/` - Ableton Live manual (pre-chunked)
- `data/transcripts/` - YouTube tutorial transcripts with timestamps

### 6. Build the frontend

```bash
cd frontend
npm install
npm run build
cd ..
```

### 7. Run the application

```bash
python app.py
```

Open **http://localhost:8000** in your browser.

---

## Project Structure

```
chatbleton/
├── app.py                 # Flask server, API endpoints, security
├── agent_tools.py         # LangChain agent with 4 tools
├── rag.py                 # RAG chain (retrieval + generation)
├── embeddings.py          # E5 multilingual embeddings wrapper
├── build_index.py         # Script to build ChromaDB index
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variables template
├── Dockerfile             # Production container
├── docker-compose.yml     # Container orchestration
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx        # Main React component
│   │   └── app.css        # Dark theme styles
│   ├── package.json       # Node dependencies
│   └── vite.config.js     # Vite configuration
│
├── data/
│   ├── manual_chunks/     # Ableton manual (JSON, pre-processed)
│   └── transcripts/       # YouTube transcripts (JSON with timestamps)
│
└── chroma_db/             # Vector database (generated by build_index.py)
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /health` | GET | Health check & feature flags |
| `POST /chat` | POST | Send question, receive answer |
| `POST /stt` | POST | Speech-to-text (audio → text) |
| `POST /tts` | POST | Text-to-speech (text → audio) |
| `POST /reset_session` | POST | Clear conversation history |

### Example: Chat Request

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I sidechain compress in Ableton?",
    "session_id": "user123"
  }'
```

**Response:**
```json
{
  "answer": "To set up sidechain compression in Ableton Live...\n\nSources:\n• https://youtube.com/watch?v=xxx&t=45s [0m45s]"
}
```

### Example: Streaming Chat Request

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is MIDI?",
    "session_id": "user123",
    "stream": true
  }'
```

**Response (Server-Sent Events):**
```
data: {"token": "MIDI"}
data: {"token": " sta"}
data: {"token": "nds "}
data: {"token": "for "}
...
data: {"done": true}
```

---

## Agent Tools

The LangChain agent has access to 4 specialized tools:

| Tool | Purpose | When Used |
|------|---------|-----------|
| `ask_rag_strict` | Query ChromaDB knowledge base | First attempt for any question |
| `web_search` | DuckDuckGo web search | When RAG returns NO_CONTEXT |
| `tempo_calculator` | BPM → milliseconds conversion | "What's 128 BPM in ms?" |
| `pitch_converter` | MIDI ↔ Hz conversion | "What frequency is C4?" |

---

## Security Features

| Feature | Description |
|---------|-------------|
| **Rate Limiting** | 30 requests/minute per IP (configurable) |
| **Input Sanitization** | Strips control characters, limits length |
| **File Size Limits** | Max 10MB for audio uploads |
| **Security Headers** | X-Frame-Options, X-Content-Type-Options, etc. |
| **Error Handling** | No stack traces exposed in production |

---

## Configuration

All settings are configured via environment variables. See `.env.example` for the full list.

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *required* | Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | Chat model |
| `PORT` | `8000` | Server port |
| `DEBUG` | `false` | Enable debug logging |
| `RATE_LIMIT_PER_MINUTE` | `30` | Max requests per IP |
| `MAX_QUESTION_LENGTH` | `2000` | Max characters per question |

---

## Docker Deployment (Optional)

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

---

## Development

```bash
# Run backend with debug mode
DEBUG=true python app.py

# Run frontend with hot reload (separate terminal)
cd frontend
npm run dev
```

Frontend dev server runs on `http://localhost:5173` with API proxy to backend.

---

## Data Sources

The knowledge base includes:

1. **Ableton Live Manual** - Official documentation, chunked for retrieval
2. **YouTube Tutorials** - Curated video transcripts with timestamps:
   - Workflow tips
   - Sound design techniques
   - Mixing & mastering guides

---

## Limitations

- Requires OpenAI API key (paid)
- Knowledge base is static (manual + pre-indexed videos)
- Voice features require microphone permissions
- No user authentication (single-user design)

---

## Future Improvements

- [ ] User authentication
- [ ] Custom knowledge base upload
- [ ] Fine-tuned embedding model
- [ ] Hybrid search (BM25 + dense vectors)

---

## License

MIT License - feel free to use for learning and personal projects.

---

## Acknowledgments

- [Ableton](https://www.ableton.com/) for creating an amazing DAW
- [LangChain](https://langchain.com/) for the agent framework
- [OpenAI](https://openai.com/) for GPT and Whisper APIs
- [ChromaDB](https://www.trychroma.com/) for the vector store

---

<p align="center">
  <strong>Built with passion for music production and AI</strong>
</p>
