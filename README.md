# PolicyAgent-A-ReAct-Based-Multi-Modal-Agentic-Framework-for-Indian-Government-Scheme-Discovery
Government welfare schemes in India are dis- tributed across thousands of PDF documents, making it difficult for citizens to discover relevant schemes and assess their eligibility.
# PolicyAgent: ReAct-Based Multi-Modal Agentic Framework

## 📌 Project Overview

This project is an AI-powered system designed to help users discover and understand Indian government welfare schemes.

It uses a **ReAct-based multi-agent framework** where different agents collaborate to process user queries, retrieve relevant data, and generate intelligent responses.

---

## 🚀 Features

* Multi-agent architecture (Agent-based system)
* Retrieval-Augmented Generation (RAG)
* LLM-based intelligent responses
* Government scheme discovery system
* Web interface using HTML templates
* Modular and scalable design

---

## 🏗️ Architecture

The system follows a pipeline:

User Input → app.py → agents → retrieval → LLM → evaluation → response → UI

---

## 📂 Project Structure

* `app.py` → Main entry point of application
* `agents.py` → Handles agent workflow and reasoning
* `retrieval.py` → Fetches relevant scheme data
* `llm.py` → Generates responses using AI model
* `evaluation.py` → Evaluates response quality
* `logger.py` → Logging system
* `start.py` → Runs the application
* `templates/` → Frontend UI

---

## ⚙️ Installation

1. Clone the repository:
   git clone https://github.com/samuugithub/PolicyAgent-A-ReAct-Based-Multi-Modal-Agentic-Framework-for-Indian-Government-Scheme-Discovery.git

2. Navigate to folder:
   cd PolicyAgent

3. Install dependencies:
   pip install -r requirements.txt

4. Create `.env` file based on `.env.example`

---

## ▶️ Run the Project

```bash id="run123"
python app.py
```

---

## 🧠 How It Works

1. User enters query
2. Agents analyze the request
3. Retrieval module fetches relevant data
4. LLM generates response
5. Evaluation checks quality
6. Final output shown to user

---

## 🔒 Security

Sensitive files like `.env`, API keys, and tokens are excluded using `.gitignore`.

---

## 🎯 Future Scope

* Voice-based interaction
* Multi-language support
* Real-time government data integration

---

## 👩‍💻 Author

Samruddhi Patil
