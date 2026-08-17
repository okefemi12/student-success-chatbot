[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Backend-green)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-blue)](https://www.docker.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Machine%20Learning-orange)](https://xgboost.ai/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-blueviolet)](https://langchain.com/)
[![Pinecone](https://img.shields.io/badge/Pinecone-Vector%20DB-lightgrey)](https://www.pinecone.io/)


<img width="1918" height="969" alt="image" src="https://github.com/user-attachments/assets/890c4607-d289-4357-9938-19e73d6db2a3" />





# 🧠 Student Success Chatbot (ACE Bot)

A comprehensive **AI-powered academic assistant** designed to help students plan, track, and improve their academic performance.

The system combines a **Dockerized Flask backend** with a robust **XGBoost Machine Learning model** to predict student success probabilities. It features a fully serverless **LangChain RAG pipeline**, real-time sub-200ms TTS audio, automated study planning, document analysis (OCR), and gamification—all deployed securely on the cloud with 99.9% uptime.

---

## 🚀 Key Features

### 🤖 AI & Machine Learning
* **Performance Prediction:** Integrated XGBoost model predicts pass/fail probability based on live user data (study hours, attendance, sleep) achieving **94.4% accuracy** and an **AUC of 0.88**.
* **Pedagogical AI Tuning:** The LLM's conversational outputs are calibrated against a **Bloom’s Taxonomy** framework, generating a mathematically balanced ratio of foundational knowledge and complex problem-solving (e.g., 22.9% 'Creating', 18.8% 'Analyzing').
* **Adaptive Quizzes & Flashcards:** Automatically generates JSON-structured study materials from uploaded notes or chat topics using Google Gemini.
* **Content Recommendations:** Suggests targeted YouTube videos based on the user's specific course and identified weak subjects.

### 🧠 LangChain RAG Pipeline
* **Vector Database Integration:** Utilizes **Pinecone** Serverless to isolate contextual memory per chat session.
* **Smart Chunking & Embedding:** Extracts text from multi-modal documents, splits it via `RecursiveCharacterTextSplitter`, and embeds it using Google's `text-embedding-004`.
* **Contextual Retrieval:** Performs semantic similarity searches to inject precise, document-grounded context into the LLM without context-window bloat.

### 🛠️ Core Functionality
* **Real-Time Audio Service:** Features a sub-200ms streaming TTS service with custom exception-handling for automated API key rotation to bypass 429 strict rate limits.
* **Document Analysis:** Extracts text from PDFs, Word docs, PowerPoints, and Images using **PyMuPDF**, **Tesseract OCR**, and native Vision models.
* **Cloud Infrastructure:**
    * **Firestore:** Securely stores user profiles, chat history, vector namespaces, and study telemetry.
    * **Cloudinary:** Hosts uploaded documents and handles file format management.
* **Gamification:** Tracks streaks, awards badges, and levels up users based on study consistency.
* **Modular Architecture:** Clean, scalable code structure using Flask Blueprints.

---

## 🧩 Project Structure

The project is organized into a modular structure for maintainability and scalability:

```text
flask/ACE_bot/
├── app.py                   # Application entry point
├── config.py                # Environment configuration
├── extensions.py            # Database (Firestore) & Vector DB (Pinecone) init
├── key_manager.py           # API key rotation logic to bypass 429 rate limits
│
├── routes/                  # API Endpoints (Blueprints)
│   ├── auth.py              # User authentication, profile, & activity logging
│   ├── chat.py              # LangChain RAG, chat logic, & audio processing
│   ├── study_tools.py       # Summaries, Library, Reminders, & ML Predictions
│   └── quiz.py              # Quiz generation, flashcards, & scoring
│
├── services/                # Core Logic
│   ├── ai_engine.py         # LLM configuration (Gemini/Groq) & ML pipeline
│   ├── audio_service.py     # TTS & STT services with automatic key rotation
│   └── doc_processor.py     # OCR & file extraction (PDF, DOCX, IMG)
│
├── utils/                   # Helpers
│   └── helpers.py           # Token verification & data sanitization
│
├── model/                   # Machine Learning Artifacts
│   └── student_prediction_model.pkl  # Trained XGBoost model
│
├── Notebook/                # Data Science Work
│   └── prediction_model.ipynb        # Model training, ROC curves, & EDA
│
├── templates/               # HTML templates
├── requirements.txt         # Python dependencies
├── Dockerfile               # Container configuration
└── .dockerignore

## ⚙️ Installation (Local Development)

1. **Clone the repo**
   ```bash
   git clone https://github.com/okefemi12/student-success-chatbot.git
   cd student-success-chatbot/flask/ACE_bot
````

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate      # Mac/Linux
   venv\Scripts\activate         # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask app**

   ```bash
   python app.py
   ```

   The app will start at `http://127.0.0.1:5000`.

---

## 🐳 Running with Docker

1. **Build the image**

   ```bash
   docker build -t student-chatbot .
   ```

2. **Run the container**

   ```bash
   docker run -p 5000:5000 student-chatbot
   ```

3. Open in your browser:

   ```
   http://localhost:5000
   ```

---

## ☁️ Deployment (Render)

This app is pre-configured for Render:

* `Dockerfile` and `.dockerignore` are already set up at the repo root.
* Render automatically builds and exposes port `5000`.

If deploying manually:

1. Connect your GitHub repo to Render.
2. Choose **“Web Service” → “Docker”**.
3. Deploy.
4. Watch logs for:

   ```
   Detected open port 5000 — service is live!
   ```

---

## 🔒 Environment Variables

| Variable               | Description                                     |
| ---------------------- | ----------------------------------------------- |
| `FIREBASE_CREDENTIALS` | Path or JSON credentials for Firestore          |
| `PINECONE_API_KEY`     | API key for Vector Database storage             |
| `PINECONE_INDEX_NAME`  | Name of your Pinecone Index                     |
| `CLOUDINARY_URL`       | Cloudinary API connection URL                   |
| `GROQ_API_KEY`         | Whisper STT and Backup LLM key                  |
| `ACE_VOICE`            | Primary TTS Audio key                           |
| `JWT_SECRET`           | Secret key for JWT authentication               |
| `OPENAI_API_KEY`       | (Optional) For OpenAI or LangChain integrations |
| `PORT`                 | Port (Render sets this automatically)           |


(Note: The system utilizes an array of backup API keys to guarantee 99.9% uptime).
---

## 🧠 Model Info

The included model:

```
flask/ACE_bot/model/student_prediction_model.pkl
```

is an ML model used to predict student success probability or study recommendations.
Make sure this file is available when deploying (it’s included via `.dockerignore` exception).

---

## 🧪 API Endpoints (Example)

| Route                  | Method | Description             |
| ---------------------- | ------ | ----------------------- |
| `/register`            | POST   | Register a new user     |
| `/test-login`          | POST   | User login              |
| `/profile`             | GET    | Fetch profile info      |
| `/create_chat_session` | POST   | Start chat              |
| `/chat_summary_pdf`    | POST   | Upload and analyze PDF  |
| `/log_activity`        | POST   | Log user study activity |

---

## 🧰 Tech Stack

* **Backend:** Flask, Python
* **Infrastructure::** Docker, Gunicorn, Render
* **Storage:** Cloudinary
* **ML / AI:** Scikit-learn, XGBoost Classifier, LangChain, Gemini text-embedding-004, Groq
* **OCR & Audio:** PyMuPDF, Pytesseract, Whisper (STT)
* **Deployment:** Render + Docker , Cloud(soon)
