import pandas as pd
import google.generativeai as genai
import os
from dotenv import load_dotenv
import time

# ---------- Setup ----------
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY_3"))
model = genai.GenerativeModel("gemini-2.5-flash")

# ---------- Load exported data ----------
data = pd.read_csv("ai_generated_questions.csv")

# Ensure we have a 'question' column
if "question" not in data.columns:
    raise ValueError("CSV must have a 'question' column.")

bloom_levels = []

# ---------- Classify each response ----------
for i, q in enumerate(data["question"], 1):
    prompt = f"""
Classify the following response/question into one Bloom’s Taxonomy level:
Remembering, Understanding, Applying, Analyzing, Evaluating, or Creating.

Response: "{q}"

Return only one of these levels.
"""
    try:
        response = model.generate_content(prompt)
        level = response.text.strip().split()[0].capitalize()
        bloom_levels.append(level)
        print(f"{i}. {level} — {q[:60]}...")
        time.sleep(0.5)  # be polite to API
    except Exception as e:
        print(f"Error at row {i}: {e}")
        bloom_levels.append("Unknown")

# ---------- Save labeled data ----------
data["bloom_level"] = bloom_levels
data.to_csv("ai_generated_questions_labeled.csv", index=False)
print("\n✅ Saved labeled file: ai_generated_questions_labeled.csv")
