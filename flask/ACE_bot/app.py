import os
import random
from time import sleep
from flask import Flask, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
import cloudinary
import warnings
import extensions

# Import Blueprints
from routes.auth import auth_bp
from routes.chat import chat_bp
from routes.study_tools import study_bp
from routes.quiz import quiz_bp

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()

# Global config delay
sleep(random.uniform(1.5, 2.5)) 

app = Flask(__name__, template_folder="templates")
CORS(app)

# Cloudinary Config
cloudinary.config(
    cloud_name=os.getenv("CLOUD_NAME"),
    api_key=os.getenv("API_KEY"),
    api_secret=os.getenv("API_SECRET")
)

# Register Blueprints
app.register_blueprint(auth_bp)
app.register_blueprint(chat_bp)
app.register_blueprint(study_bp)
app.register_blueprint(quiz_bp)

@app.route("/")
def home():
    return jsonify({
        "status": "OK",
        "message": "Ace The Tutor backend API is live 🚀"
    })

@app.route("/test")
def index():
    firebase_config = {
        "apiKey": os.getenv("FIREBASE_API_KEY"),
        "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN"),
        "projectId": os.getenv("FIREBASE_PROJECT_ID"),
        "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET"),
        "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID"),
        "appId": os.getenv("FIREBASE_APP_ID"),
    }
    return render_template("index.html", firebase_config=firebase_config)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)