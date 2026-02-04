# 🧠 Student Success Chatbot (ACE Bot)

A comprehensive **AI-powered academic assistant** designed to help students plan, track, and improve their academic performance.

The system combines a **modular Flask backend** with a robust **XGBoost Machine Learning model** to predict student success probabilities (94.4% accuracy). It features real-time chat, automated study planning, document analysis (OCR), and gamification, all deployed securely on the cloud.

---

## 🚀 Key Features

### 🤖 AI & Machine Learning
* **Performance Prediction:** Integrated XGBoost model predicts pass/fail probability based on live user data (study habits, attendance, sleep).
* **Smart Tutoring:** Generates personalized study plans and flashcards using **Google Gemini** and **OpenAI**.
* **Adaptive Quizzes:** Automatically generates quizzes from uploaded notes (PDF/DOCX/Images) or chat topics.
* **Content Recommendations:** Suggests relevant YouTube videos based on the user's course and weak subjects.

### 🛠️ Core Functionality
* **Multi-Modal Chat:** Supports text and voice interactions (Speech-to-Text & Text-to-Speech).
* **Document Analysis:** Extracts text from PDFs, Word docs, PowerPoints, and Images using **PyMuPDF** & **Tesseract OCR**.
* **Cloud Storage:**
    * **Firestore:** Securely stores user profiles, chat history, and study logs.
    * **Cloudinary:** Hosts uploaded documents and handles file format management for the frontend.
* **Gamification:** Tracks streaks, awards badges, and levels up users based on study consistency.
* **Modular Architecture:** Clean, scalable "Modular Monolith" code structure using Flask Blueprints.

---

## 🧩 Project Structure

The project is organized into a modular structure for maintainability and scalability:

```text
flask/ACE_bot/
├── app.py                   # Application entry point
├── config.py                # Environment configuration
├── extensions.py            # Database & external service initialization
├── key_manager.py           # API key rotation logic for reliability
│
├── routes/                  # API Endpoints (Blueprints)
│   ├── auth.py              # User authentication, profile, & activity logging
│   ├── chat.py              # Chatbot logic, history, & audio processing
│   ├── study_tools.py       # Summaries, Library, Reminders, & Recommendations
│   └── quizzes.py           # Quiz generation, flashcards, & scoring
│
├── services/                # Core Logic
│   ├── ai_engine.py         # LLM integration (Gemini/Groq) & ML pipeline
│   ├── audio_service.py     # TTS & STT services
│   └── doc_processor.py     # OCR & file extraction (PDF, DOCX, IMG)
│
├── utils/                   # Helpers
│   └── helpers.py           # Token verification & data sanitization
│
├── model/                   # Machine Learning Artifacts
│   └── student_prediction_model.pkl  # XGBoost model
│
├── Notebook/                # Data Science Work
│   └── prediction-model.ipynb        # Model training & analysis notebook
│
├── templates/               # HTML templates
├── requirements.txt         # Python dependencies
├── Dockerfile               # Container configuration
└── .dockerignore
---

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
| `CLOUDINARY_URL`       | Cloudinary API connection URL                   |
| `JWT_SECRET`           | Secret key for JWT authentication               |
| `OPENAI_API_KEY`       | (Optional) For OpenAI or LangChain integrations |
| `PORT`                 | Port (Render sets this automatically)           |

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

* **Backend:** Flask (Python)
* **Database:** Firestore (Firebase)
* **Storage:** Cloudinary
* **ML / AI:** Scikit-learn, LangChain, OpenAI, Gemini, XGBOOST Classifier
* **OCR:** PyMuPDF, Pytesseract
* **Deployment:** Render + Docker , Cloud(soon)

