import os
import json
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv

load_dotenv()

#Initialise the firebase
if not firebase_admin._apps:
    firebase_json = os.getenv("FIREBASE_CREDENTIALS")
    if not firebase_json:
        raise ValueError("Missing FIREBASE_CREDENTIALS environment variable.")
    cred_dict = json.loads(firebase_json)
    cred = credentials.Certificate(cred_dict)
    firebase_admin.initialize_app(cred)

db = firestore.client()





