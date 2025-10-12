import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
from dotenv import load_dotenv
import os
import json
load_dotenv()

# ---------- Initialize Firebase ----------
if not firebase_admin._apps:
    firebase_json = os.getenv("FIREBASE_CREDENTIALS")
    if not firebase_json:
        raise ValueError("Missing FIREBASE_CREDENTIALS environment variable.")
    cred_dict = json.loads(firebase_json)
    cred = credentials.Certificate(cred_dict)
    firebase_admin.initialize_app(cred)
db = firestore.client()

data = []

# Fetch all users
users_ref = db.collection("users")
users = users_ref.stream()

for user in users:
    user_id = user.id
    chat_sessions_ref = users_ref.document(user_id).collection("chat_sessions")
    chat_sessions = chat_sessions_ref.stream()

    for session in chat_sessions:
        messages_ref = (
            chat_sessions_ref.document(session.id)
            .collection("messages")
            .order_by("timestamp")
        )
        messages = messages_ref.stream()

        for msg in messages:
            msg_data = msg.to_dict()
            # ✅ Extract AI responses (assistant role)
            if msg_data.get("role") == "assistant":
                data.append({
                    "question": msg_data.get("content", ""),
                    "bloom_level": ""  # you’ll manually label or auto-categorize later
                })

# Save to CSV
df = pd.DataFrame(data)
df.to_csv("ai_generated_questions.csv", index=False)

print(f"✅ Exported {len(data)} assistant responses to ai_generated_questions.csv")
