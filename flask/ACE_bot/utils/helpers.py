import re
import html
from datetime import datetime
import pytz
from flask import request
from firebase_admin import auth, firestore
from extensions import db

def verify_token_from_request():
    """Verify Firebase ID token from Authorization header (Bearer <token>) or JSON body idToken."""
    auth_header = request.headers.get("Authorization", "")
    token = None
    if auth_header.startswith("Bearer "):
        token = auth_header.split(" ", 1)[1]
    else:
        # fallback: token in JSON body (useful for some POSTs)
        try:
            token = (request.get_json() or {}).get("idToken")
        except Exception:
            token = None

    if not token:
        raise ValueError("Missing ID token")

    decoded = auth.verify_id_token(token)  # fixed
    return decoded


def update_streak(uid):
    """Update user's current and longest streak based on last active date."""
    user_ref = db.collection("users").document(uid)
    user_doc = user_ref.get()
    if not user_doc.exists:
        return

    data = user_doc.to_dict()
    today = datetime.now(pytz.UTC).date()

    last_active_str = data.get("last_active_date")
    current_streak = data.get("streak", 0)
    longest_streak = data.get("longest_streak", 0)

    if last_active_str:
        last_active_date = datetime.strptime(last_active_str, "%Y-%m-%d").date()
        days_diff = (today - last_active_date).days

        if days_diff == 0:
            # already updated today — no change
            return
        elif days_diff == 1:
            current_streak += 1  # continued streak
        else:
            current_streak = 1   # reset streak
    else:
        current_streak = 1  # first time

    longest_streak = max(longest_streak, current_streak)

    user_ref.update({
        "streak": current_streak,
        "longest_streak": longest_streak,
        "last_active_date": today.strftime("%Y-%m-%d")
    })

def clean_value(value):
    """Convert Firestore/special values into JSON-safe formats."""
    if isinstance(value, datetime):
        return value.isoformat()
    if value == firestore.SERVER_TIMESTAMP:
        return None  # placeholder until Firestore resolves
    return value

def safe_clean_dict(d):
    """Recursively clean dicts/lists for JSON serialization."""
    if isinstance(d, dict):
        return {k: safe_clean_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [safe_clean_dict(v) for v in d]
    else:
        return clean_value(d)
    

def clean_response(text):
    # Unescape HTML entities
    text = html.unescape(text)
    # Replace escaped newline and tab characters
    text = text.replace('\\n', '\n').replace('\\t', '\t')
    # Replace escaped quotes
    text = text.replace('\\"', '"').replace("\\'", "'")
    # Collapse multiple whitespaces
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  
    text = re.sub(r'^\* ', '• ', text, flags=re.MULTILINE) 

    # Convert markdown to HTML
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
    
    # Clean spacing
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\n\n+', '\n\n', text)
    return text.strip()