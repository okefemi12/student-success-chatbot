import json
import re
import os
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from difflib import SequenceMatcher
from sklearn.metrics import accuracy_score, f1_score
import google.generativeai as genai
from rouge_score import rouge_scorer

# ---------- Load Environment Variables ----------
load_dotenv()

# ---------- Configure Gemini ----------
genai.configure(api_key=os.getenv("GEMINI_API_KEY_2"))
model = genai.GenerativeModel(model_name="gemini-2.5-pro")

# ---------- Load Test Data ----------
with open("test_data.json", "r") as f:
    test_samples = json.load(f)

# ---------- Initialize Evaluators ----------
rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

y_true = []  # Ground truth (1 = correct)
y_pred = []  # Model prediction
rouge_scores = []

# ---------- Helper Function ----------
def keyword_overlap(ref, cand):
    ref_words = set(re.findall(r"\b\w+\b", ref.lower()))
    cand_words = set(re.findall(r"\b\w+\b", cand.lower()))
    if not ref_words:
        return 0
    overlap = len(ref_words & cand_words) / len(ref_words)
    return overlap

# ---------- Run Evaluation ----------
for i, sample in enumerate(test_samples, 1):
    question = sample["question"]
    correct_answer = sample["expected_answer"].strip()

    # Prompt for model
    prompt = (
        f"You are a concise academic assistant. "
        f"Answer the following question with a **direct, single-sentence and accurateacademic answer**, "
        f"similar to a short test response. "
        f"Do not explain or elaborate unless absolutely necessary. "
        f"If you don't know the answer, reply exactly with 'I don't know'.\n\n"
        f"Question: {question}"
    )

    response = model.generate_content(prompt)
    candidate = response.text.strip()

    print(f"\n--- Sample {i} ---")
    print("Question:", question)
    print("Correct Answer:", correct_answer)
    print("Model Response:", candidate)

    # ----- Compute Text Similarity Metrics -----
    if re.search(r"\bi don't know\b", candidate.lower()):
        pred_label = 0
        similarity = 0
    else:
        fuzzy_sim = SequenceMatcher(None, correct_answer.lower(), candidate.lower()).ratio()
        keyword_sim = keyword_overlap(correct_answer, candidate)
        rouge_result = rouge.score(correct_answer, candidate)
        rouge_l = rouge_result["rougeL"].fmeasure

        # Weighted average similarity (to make evaluation more forgiving)
        combined_score = (0.5 * fuzzy_sim) + (0.3 * keyword_sim) + (0.2 * rouge_l)
        pred_label = 1 if combined_score > 0.5 else 0  # 0.35 is a balanced threshold

    y_true.append(1)
    y_pred.append(pred_label)
    rouge_scores.append(rouge_result["rougeL"].fmeasure)

    print(f"Fuzzy: {fuzzy_sim:.2f} | Keyword: {keyword_sim:.2f} | ROUGE-L: {rouge_l:.2f}")
    print(f"Combined Score: {combined_score:.2f} | Predicted Label: {pred_label}")

# ---------- Compute Overall Metrics ----------
acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
avg_rouge = sum(rouge_scores) / len(rouge_scores)

print("\n=== Overall Evaluation ===")
print(f"Accuracy (ACC): {acc * 100:.2f}%")
print(f"F1-score: {f1 * 100:.2f}%")
print(f"Average ROUGE-L: {avg_rouge * 100:.2f}%")


