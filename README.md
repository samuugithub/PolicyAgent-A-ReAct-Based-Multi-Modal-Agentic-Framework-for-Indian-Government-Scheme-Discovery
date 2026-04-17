# 🏛️ PolicyAgent — ReAct-Based Multi-Modal Agentic Framework for Indian Government Scheme Discovery

> An AI-powered web application that helps Indian citizens discover relevant government welfare schemes and assess their eligibility using natural language queries, multilingual support, and document analysis.

---

## 📌 Problem Statement

Government welfare schemes in India are distributed across thousands of PDF documents, making it extremely difficult for citizens to discover relevant schemes and assess their eligibility. **PolicyAgent** solves this by extracting, indexing, and intelligently querying these documents using a modular agentic AI pipeline.

---

## 🚀 Features

- 📄 **PDF Ingestion** — Upload government scheme PDFs (text-based or scanned via OCR)
- 🔍 **Hybrid Retrieval** — FAISS vector search + BM25 keyword search with cross-encoder reranking
- 🤖 **ReAct Agentic Pipeline** — `DocAgent → RetrievalAgent → ResponseAgent` for structured reasoning
- 🌐 **Multilingual Support** — Query and receive answers in Kannada, Hindi, or English
- 🎙️ **Voice Query** — Submit queries via speech and receive spoken audio responses (gTTS)
- 📊 **Auto Excel/HTML Export** — Extracts structured fields from PDFs and generates master reports
- ☁️ **Google Drive Sync** — Automatically syncs the master Excel report to/from Google Drive
- 🎓 **Document Recommender** — Upload personal documents (income cert, caste cert, marksheets) to get matched schemes
- 📈 **Evaluation Suite** — Full 5-dimensional evaluation matrix (retrieval, generation, agentic reasoning, OCR, latency)

---

## 🗂️ Project Structure

```
genai_project/
│
├── app.py              # Flask web server — routes only (thin wrapper)
├── agents.py           # ReAct agents: DocAgent → RetrievalAgent → ResponseAgent
├── llm.py              # Groq LLM client, token budget, prompts, field definitions
├── retrieval.py        # FAISS + BM25 hybrid retrieval, reranking, chunking
├── evaluation.py       # 5-dimension evaluation matrix (Cell-17)
├── logger.py           # Structured pipeline logging
├── start.py            # Startup launcher script
│
├── templates/
│   └── index.html      # Frontend UI
│
├── uploads/            # Uploaded PDFs and documents (auto-created)
├── outputs/            # Generated Excel and HTML reports (auto-created)
│
├── client_secrets.json # Google OAuth credentials (for Drive sync)
├── token.json          # Google OAuth token (auto-generated)
├── .env                # Environment variables (API keys)
├── .env.example        # Example environment config
└── requirements.txt    # Python dependencies
```

---

## ⚙️ Architecture

```
User Query
    │
    ▼
DocAgent          ← checks if document-level question
    │
    ▼
RetrievalAgent    ← FAISS + BM25 hybrid search + cross-encoder reranking
    │
    ▼
ResponseAgent     ← Groq LLM (llama / mixtral) generates final answer
    │
    ▼
Translation       ← deep-translator / Groq (Kannada ↔ Hindi ↔ English)
    │
    ▼
TTS (optional)    ← gTTS audio response
```

---

## 🛠️ Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/samuugithub/PolicyAgent-A-ReAct-Based-Multi-Modal-Agentic-Framework-for-Indian-Government-Scheme-Discovery.git
cd PolicyAgent-...
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** Requires Python 3.9+. For GPU acceleration, install the appropriate `torch` version for your CUDA setup.

### 3. Configure Environment Variables

Copy the example env file and fill in your Groq API key:

```bash
cp .env.example .env
```

Edit `.env`:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get your free Groq API key at [console.groq.com](https://console.groq.com).

### 4. (Optional) Set Up Google Drive Sync

To enable automatic sync of extracted scheme data to Google Drive:

1. Create a project in [Google Cloud Console](https://console.cloud.google.com)
2. Enable the **Google Drive API**
3. Download OAuth credentials as `client_secrets.json` and place it in the project root
4. On first run, a browser window will open for authentication — `token.json` is auto-generated

### 5. Run the Application

```bash
python start.py
# or
python app.py
```

Open your browser at: **http://localhost:5000**

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Main UI |
| `GET` | `/api/status` | App status (models, schemes loaded, Drive config) |
| `POST` | `/api/set-api-key` | Set Groq API key dynamically |
| `POST` | `/api/init-models` | Load embedding + reranker models |
| `POST` | `/api/upload-pdf` | Upload and index scheme PDFs |
| `POST` | `/api/extract` | Extract structured fields from indexed PDFs |
| `POST` | `/api/chat` | Natural language query (English) |
| `POST` | `/api/voice-query` | Multilingual voice query (Kannada/Hindi/English) |
| `POST` | `/api/tts` | Text-to-speech audio response |
| `POST` | `/api/recommend` | Upload personal documents → get matched schemes |
| `POST` | `/api/drive-sync` | Sync master Excel to/from Google Drive |
| `POST` | `/api/load-excel` | Load existing master Excel into memory |
| `GET` | `/api/schemes` | List all extracted schemes |
| `GET` | `/api/download-excel` | Download master Excel report |
| `GET` | `/api/download-html` | Download HTML report |
| `POST` | `/api/eval` | Run full 5-dimension evaluation suite |
| `GET` | `/api/logs` | View pipeline logs |

---

## 🧠 How It Works

### PDF Extraction Pipeline
1. PDF is uploaded and saved to `uploads/`
2. Text-based PDFs are parsed with `pdfplumber`; scanned PDFs use `pytesseract` OCR
3. Text is split into smart sections and chunked
4. FAISS vector index + BM25 index are built per PDF

### Field Extraction (Scheme Profiling)
For each PDF, the system extracts structured fields (scheme name, eligibility, benefits, deadlines, etc.) by asking targeted questions against the hybrid retrieval context using Groq LLM.

### Scheme Recommendation (Document Recommender)
1. User uploads personal documents (income cert, caste cert, marksheet, etc.)
2. Groq Vision OCR + Tesseract extract text from each document
3. LLM parses a student profile (income, category, percentage, course, etc.)
4. Each loaded scheme is scored (MATCH / PARTIAL / NO) against the profile
5. Top matching schemes are returned with explanations

### Multilingual Voice Flow
```
Voice Input (Kannada/Hindi)
    → Transliterate (if Romanized) → Translate to English
    → Smart Chat → English Answer
    → Translate back → Native Language Answer
    → gTTS Audio Response
```

---

## 📦 Key Dependencies

| Package | Purpose |
|---------|---------|
| `flask` | Web server |
| `groq` | LLM inference (llama / mixtral models) |
| `sentence-transformers` | Embedding model (multi-qa-MiniLM-L6) |
| `faiss-cpu` | Vector similarity search |
| `rank_bm25` | BM25 keyword retrieval |
| `pdfplumber` | Text PDF extraction |
| `pytesseract` | OCR for scanned PDFs |
| `deep-translator` | Google Translate wrapper |
| `gTTS` | Text-to-speech |
| `openpyxl` | Excel report generation |
| `google-api-python-client` | Google Drive integration |

---

## 🧪 Evaluation

Run the full evaluation suite via `/api/eval`. It measures:

1. **Retrieval Quality** — FAISS + BM25 Recall@3
2. **Generation Quality** — Groq faithfulness + relevance scoring
3. **Agentic Reasoning** — ReAct loop 6-check analysis
4. **Document Extraction** — OCR → structured field accuracy
5. **End-to-End Latency** — Query benchmark timing

---

## 👤 Author

Samruddhi Patil, Bhumika Lokare, Sushma V

---

## 📄 License

This project is open-source. Feel free to use, modify, and distribute with attribution.
