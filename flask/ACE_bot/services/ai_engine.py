import os
import requests
import joblib
import cohere
import google.generativeai as genai
from openai import OpenAI
from key_manager import KeyManager
from serpapi.google_search import GoogleSearch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "student_prediction_model.pkl")


# ML Model 
try:
    pipeline = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Warning: ML model not found at {MODEL_PATH}. Error: {e}")
    pipeline = None

groq_client = OpenAI(
                   api_key=os.getenv("GROQ_API_KEY"),
                   base_url="https://api.groq.com/openai/v1"
                )
#api keys
serper_api_key = os.getenv("SERPER_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

#LLM configuration 
genai.configure(api_key=os.getenv("GEMINI_API_KEY_1"))
model = genai.GenerativeModel(model_name="gemini-2.5-flash")
genai.configure(api_key=os.getenv("RECOMMEND_BACKUP1"))
rec_backup1 = genai.GenerativeModel(model_name="gemini-2.5-flash")
genai.configure(api_key=os.getenv("RECOMMEND_BACKUP2"))
rec_backup2 = genai.GenerativeModel(model_name="gemini-2.5-flash")
genai.configure(api_key=os.getenv("GEMINI_API_KEY_2"))
model_pdf = genai.GenerativeModel(model_name="gemini-2.5-flash")
genai.configure(api_key=os.getenv("GEMINI_API_KEY_3"))
model_cards= genai.GenerativeModel(model_name="gemini-2.5-flash")
genai.configure(api_key=os.getenv("GEMINI_API_KEY_4"))
reminder_model = genai.GenerativeModel("gemini-2.5-flash")

# -----------------------------------

FINAL_BACKUP = [
    os.environ.get("BACKUP_1"),             
    os.environ.get("BACKUP_2"),     
    os.environ.get("BACKUP_3"),    
    os.environ.get("BACKUP_4"),
    os.environ.get("BACKUP_5"),
    os.environ.get("BACKUP_6"),
]
backup_manager = KeyManager(FINAL_BACKUP, model_name="gemini-2.5-flash")

def rec_backup_response(prompt):
    try:
        return rec_backup1.generate_content(prompt)
    except Exception:
        try: 
            return rec_backup2.generate_content(prompt)
        except Exception as e: 
            return f"Error: {e}"

#Backup models 
def generate_backup_response(prompt):
    try:
        # Try Gemini
        return model.generate_content(prompt).text
    except Exception:
        try:
            # Try Cohere
            co = cohere.Client(os.getenv("COHERE_API_KEY"))
            response = co.chat(model="command-r-plus-08-2024", message=prompt)
            return response.text.strip()
        except Exception:
            try:
                client = OpenAI(
                   api_key=os.getenv("GROQ_API_KEY"),
                   base_url="https://api.groq.com/openai/v1"
                )

                groq_response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}]
                )

                return groq_response.choices[0].message.content
            except Exception as e:
                return f"All model attempts failed: {e}"

def search_web(query):
    headers = {
        "X-API-KEY": serper_api_key,  # replace this
        "Content-Type": "application/json"
    }

    data = {
        "q": query
    }

    response = requests.post("https://google.serper.dev/search", headers=headers, json=data)

    if response.status_code != 200:
        return [{"title": "Error", "link": f"Status code: {response.status_code}"}]

    results = response.json()

    links = []
    for result in results.get("organic", []):
        links.append({
            "title": result.get("title"),
            "link": result.get("link")
        })

    return links

def final_backup_response(prompt_or_content):
    """
    Tries to generate response using the backup key rotation.
    """
    try:
        # Calls the KeyManager to rotate through your 6 FINAL_BACKUP keys
        response = backup_manager.generate_content(prompt_or_content)
        return response.text
    except Exception as e:
        print(f"All 6 Backup Keys failed: {e}")
        return "Error: Service temporarily unavailable. Please try again later."

# --- NEW: EMBEDDING FUNCTION ---
def get_embedding(text):
    """Converts a string of text into a 768-dimensional vector using Gemini."""
    try:
        # We use text-embedding-004, Google's latest embedding model
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document",
        )
        return result['embedding']
    except Exception as e:
        print(f"Embedding error: {e}")
        return None