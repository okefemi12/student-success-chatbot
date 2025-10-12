
# 🧠 Student Success Chatbot (ACE Bot)

A smart **Flask-based chatbot** that helps students plan, track, and improve their academic performance.  
It supports **PDF and image uploads**, extracts study materials via OCR and NLP, and stores chat data securely in **Firestore**.  
Images and files are hosted on **Cloudinary**.

---

## 🚀 Features
- 📚 **AI-powered chat assistant** for academic help and study planning  
- 🧾 **PDF and image upload support** (text extraction via PyMuPDF & Tesseract OCR)  
- ☁️ **Firestore** for storing user chats, sessions, and metadata  
- 🖼️ **Cloudinary integration** for secure media hosting  
- 🧠 **ML model integration** (`student_prediction_model.pkl`) for performance prediction  
- 🔐 **JWT authentication** for user validation  
- 🧰 **LangChain + Gemini + Cohere + OpenAI** support for advanced reasoning  
- 🧑‍💻 **Deployed on Render** using Docker  

---

## 🧩 Project Structure
```

student-success-chatbot/
│
├── Dockerfile
├── .dockerignore
├── flask/
│   └── ACE_bot/
│       ├── app.py
│       ├── requirements.txt
│       ├── model/
│       │   └── student_prediction_model.pkl
│       ├── templates/
│       │   └── index.html
│       ├── static/               # (optional: JS/CSS files)
│       └── routes/               # (optional: organized endpoints)
└── README.md

````

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


