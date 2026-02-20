import json
import re
import mimetypes
import requests
from io import BytesIO
from datetime import datetime
from PIL import Image
from flask import Blueprint, request, jsonify
from firebase_admin import firestore
from extensions import db
from utils.helpers import verify_token_from_request, update_streak
from services.doc_processor import extract_text_from_pdf, extract_text_from_docx, extract_text_from_pptx, extract_text_from_excel, get_pdf_as_images
from services.ai_engine import model_cards, model_pdf, final_backup_response
import cloudinary.uploader

quiz_bp = Blueprint('quiz', __name__)

# --- HELPER FUNCTION FOR ROBUST JSON PARSING ---
def extract_json_array(raw_text):
    """Removes markdown and attempts to safely extract a JSON array."""
    try:
        # Remove markdown formatting if the AI added it
        cleaned = re.sub(r'```json|```', '', raw_text).strip()
        start = cleaned.find('[')
        end = cleaned.rfind(']')
        if start != -1 and end != -1:
            return json.loads(cleaned[start : end + 1])
        return json.loads(cleaned)
    except Exception as e:
        print(f"JSON Parsing Error: {e}\nRaw Text: {raw_text}")
        return []


# ==========================================
# QUIZ ROUTES
# ==========================================

@quiz_bp.route('/chat_quiz', methods=['POST'])
def create_chat_quiz():
    try:
        decoded = verify_token_from_request()
        uid = decoded["uid"]
        update_streak(uid)

        content = request.json.get("text", "")
        if not content:
            return jsonify({"error": "No input text provided"}), 400

        prompt = f"""
        You are ACE, a helpful AI tutor. Based on the content below, generate up to 10 multiple-choice quiz flashcards.
        
        Strictly follow this JSON schema:
        [
          {{
            "question": "string",
            "options": {{ "A": "...", "B": "...", "C": "...", "D": "..." }},
            "answer": "A"
          }}
        ]
        Do not add any Markdown formatting like ```json. Just return the raw JSON array.
        
        Content:
        {content}
        """

        try:
            response = model_cards.generate_content(prompt)
            quiz_data = extract_json_array(response.text)

        except Exception as e:
            print(f"Primary Gemini failed: {e}")
            try:
                backup_raw = final_backup_response(prompt)
                quiz_data = extract_json_array(backup_raw)
            except Exception as backup_error:
                print(f"Backup also failed: {backup_error}")
                quiz_data = []

        if not isinstance(quiz_data, list) or len(quiz_data) == 0:
            return jsonify({"error": "Could not generate valid quiz data from AI. Please try again."}), 500

        doc_ref = db.collection("users").document(uid).collection("quiz").document()
        doc_ref.set({
            "quiz": quiz_data,
            "created_at": datetime.utcnow().isoformat()
        })

        return jsonify({"quiz": quiz_data}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@quiz_bp.route('/media_quiz', methods=['POST'])
def media_quiz():
    try:
        decoded = verify_token_from_request()
        uid = decoded["uid"]
        update_streak(uid)

        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        filename = file.filename.lower()

        upload_result = cloudinary.uploader.upload(
            file,
            upload_preset="unsigned_quiz_uploads",
            folder=f"file_quizzes/{uid}",
            resource_type="auto"
        )

        file_url = upload_result["secure_url"]
        file_name = upload_result.get("original_filename", "document")

        extracted_text = ""
        image_parts = [] 

        try:
            if filename.endswith(".pdf"):
                extracted_text = extract_text_from_pdf(file_url)
                if not extracted_text.strip():
                    print("No text found. Falling back to Visual PDF reading via Gemini...")
                    image_parts = get_pdf_as_images(file_url, max_pages=8)
                    if image_parts:
                        extracted_text = "IMAGE_MODE"
            elif filename.endswith(".docx"):
                extracted_text = extract_text_from_docx(file_url)
            elif filename.endswith(".pptx"):
                extracted_text = extract_text_from_pptx(file_url)
            elif filename.endswith((".xls", ".xlsx")):
                extracted_text = extract_text_from_excel(file_url)
            elif filename.endswith((".png", ".jpg", ".jpeg", ".webp")):
                response = requests.get(file_url)
                image_parts = [Image.open(BytesIO(response.content))]
                extracted_text = "IMAGE_MODE" 
            else:
                return jsonify({"error": "Unsupported file type"}), 400
        except Exception as e:
            print(f"Extraction error: {e}")
            return jsonify({"error": "Failed to read document content"}), 400

        if not extracted_text.strip():
            return jsonify({"error": "No readable text found. If this is a scanned PDF or image, please upload a version with selectable text."}), 400

        base_prompt = (
            "You are FELIX, a helpful AI tutor. Based on the study material provided, "
            "generate a comprehensive multiple-choice quiz in JSON format.\n"
            "Rules for Quantity:\n"
            "- If the content is SHORT, generate up to 10 questions.\n"
            "- If the content is EXTENSIVE, generate up to 25 questions.\n"
            "- CRITICAL: Ensure you cover the BEGINNING, MIDDLE, and END of the content.\n\n"
            "Strictly follow this schema:\n"
            "[{\"question\": \"...\", \"options\": {\"A\": \"...\", \"B\": \"...\", \"C\": \"...\", \"D\": \"...\"}, \"answer\": \"A\"}]\n"
            "Return ONLY raw JSON. No Markdown."
        )

        try:
            if image_parts:
                payload = [base_prompt] + image_parts
                response = model_cards.generate_content(payload)
            else:
                full_prompt = f"{base_prompt}\n\nText:\n{extracted_text[:50000]}"
                response = model_cards.generate_content(full_prompt)

            quiz_data = extract_json_array(response.text)

        except Exception as e:
            print(f"Gemini quiz generation failed: {e}")
            try:
                if image_parts:
                    payload = [base_prompt] + image_parts
                    backup_raw = final_backup_response(payload)
                else:
                    backup_input = f"{base_prompt}\n\nText:\n{extracted_text[:50000]}"
                    backup_raw = final_backup_response(backup_input)
                
                quiz_data = extract_json_array(backup_raw)
            except Exception as backup_error:
                print(f"Backup also failed: {backup_error}")
                quiz_data = []

        if not isinstance(quiz_data, list) or len(quiz_data) == 0:
             return jsonify({"error": "Could not generate valid quiz. The document might be too complex."}), 500

        db.collection("users").document(uid).collection("quiz").add({
            "quiz": quiz_data,
            "created_at": datetime.utcnow().isoformat(),
            "source_file": file_name,
            "file_url": file_url
        })

        return jsonify({"quiz": quiz_data, "file_url": file_url}), 200

    except Exception as e:
        print("Error in /media_quiz:", str(e))
        return jsonify({"error": str(e)}), 500


@quiz_bp.route('/get_chat_quiz', methods=['GET'])
def get_chat_quiz():
    try:
        decoded = verify_token_from_request()
        uid = decoded["uid"]

        quiz_docs = db.collection("users").document(uid).collection("quiz").order_by("created_at").get()
        quiz_list = [{"id": doc.id, "quiz": doc.to_dict().get("quiz"), "created_at": doc.to_dict().get("created_at")} for doc in quiz_docs]

        return jsonify({"quiz": quiz_list}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@quiz_bp.route('/get_chat_quiz/<quiz_id>', methods=['GET'])
def get_single_quiz(quiz_id):
    try:
        decoded = verify_token_from_request()
        uid = decoded["uid"]
        update_streak(uid)

        doc = db.collection("users").document(uid).collection("quiz").document(quiz_id).get()
        if not doc.exists:
            return jsonify({"error": "Quiz not found"}), 404

        quiz_data = doc.to_dict()
        return jsonify({"id": doc.id, "quiz": quiz_data.get("quiz"), "created_at": quiz_data.get("created_at")}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@quiz_bp.route('/delete_quiz/<quiz_id>', methods=['DELETE'])
def delete_quiz(quiz_id):
    try:
        decoded = verify_token_from_request()
        uid = decoded["uid"]
        db.collection("users").document(uid).collection("quiz").document(quiz_id).delete()
        return jsonify({"ok": True, "message": "Quiz successfully deleted"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@quiz_bp.route('/save_quiz_score', methods=['POST'])
def save_quiz_score():
    try:
        decoded = verify_token_from_request()
        uid = decoded["uid"]
        data = request.json
        
        quiz_id = data.get("quiz_id") 
        score = data.get("score")
        total = data.get("total")

        db.collection("users").document(uid).collection("quiz_scores").document(quiz_id).set({
            "score": score,
            "total": total,
            "percentage": (score / total) * 100 if total > 0 else 0,
            "updated_at": datetime.utcnow()
        }, merge=True)

        return jsonify({"message": "Score saved successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==========================================
# FLASHCARD ROUTES
# ==========================================

@quiz_bp.route('/media_flashcards', methods=['POST'])
def media_flashcards():
    try:
        decoded = verify_token_from_request()
        uid = decoded["uid"]
        update_streak(uid)

        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        filename = file.filename.lower()

        upload_result = cloudinary.uploader.upload(
            file,
            upload_preset="unsigned_flashcards",
            folder=f"flashcards/{uid}",
            resource_type="auto"
        )

        file_url = upload_result["secure_url"]
        file_name = upload_result.get("original_filename", "document")

        extracted_text = ""
        image_parts = []

        try:
            if filename.endswith(".pdf"):
                extracted_text = extract_text_from_pdf(file_url)
                if not extracted_text.strip():
                    print("No text found. Falling back to Visual PDF reading via Gemini...")
                    image_parts = get_pdf_as_images(file_url, max_pages=8)
                    if image_parts:
                        extracted_text = "IMAGE_MODE"
            elif filename.endswith(".docx"):
                extracted_text = extract_text_from_docx(file_url)
            elif filename.endswith(".pptx"):
                extracted_text = extract_text_from_pptx(file_url)
            elif filename.endswith((".xls", ".xlsx")):
                extracted_text = extract_text_from_excel(file_url)
            elif filename.endswith((".png", ".jpg", ".jpeg", ".webp")):
                response = requests.get(file_url)
                image_parts = [Image.open(BytesIO(response.content))]
                extracted_text = "IMAGE_MODE" 
            else:
                return jsonify({"error": "Unsupported file type"}), 400
                
        except Exception as e:
            print(f"Extraction error: {e}")
            return jsonify({"error": "Failed to read document text"}), 400

        if not extracted_text.strip():
            return jsonify({"error": "No readable text found. If this is a scanned PDF or image, please upload a version with selectable text."}), 400

        base_prompt = (
            "You are ACE. Generate flashcards based on the study material in JSON format.\n"
            "Rules for Quantity:\n"
            "- If the content is SHORT, generate exactly 10-15 cards.\n"
            "- If the content is EXTENSIVE, generate up to 25-50 cards.\n"
            "- Ensure you cover the BEGINNING, MIDDLE, and END of the content.\n"
            "- Do not repeat facts.\n\n"
            "Strictly follow this schema:\n"
            "[{\"question\": \"...\", \"answer\": \"...\"}]\n"
            "Return ONLY raw JSON. No Markdown."
        )

        try:
            if image_parts:
                payload = [base_prompt] + image_parts
                response = model_pdf.generate_content(payload)
            else:
                full_prompt = f"{base_prompt}\n\nText:\n{extracted_text[:50000]}"
                response = model_pdf.generate_content(full_prompt)

            flashcards = extract_json_array(response.text)

        except Exception as e:
            print(f"Primary generation failed: {e}")
            try:
                if image_parts:
                    payload = [base_prompt] + image_parts
                    backup_raw = final_backup_response(payload)
                else:
                    backup_input = f"{base_prompt}\n\nText:\n{extracted_text[:50000]}"
                    backup_raw = final_backup_response(backup_input)
                
                flashcards = extract_json_array(backup_raw)
            except Exception as backup_error:
                print(f"Backup also failed: {backup_error}")
                flashcards = []

        if not isinstance(flashcards, list) or len(flashcards) == 0:
            return jsonify({"error": "AI could not generate valid flashcards from this file"}), 500
        
        if len(flashcards) > 50: 
            flashcards = flashcards[:50]

        db.collection("users").document(uid).collection("flashcards").add({
            "flashcards": flashcards,
            "created_at": datetime.utcnow().isoformat(),
            "source_file": file_name,
            "file_url": file_url
        })

        return jsonify({"flashcards": flashcards, "file_url": file_url}), 200

    except Exception as e:
        print(f"Media Route Error: {e}")
        return jsonify({"error": str(e)}), 500


@quiz_bp.route('/chat_flashcards', methods=['POST'])
def chat_flashcards():
    try:
        decoded = verify_token_from_request()
        uid = decoded['uid']
        update_streak(uid)
        
        data = request.get_json().get("text", "").strip()
        if not data:
            return jsonify({"error": "no text provided"}), 400

        flashcard_prompt = (
            "You are ACE, a helpful academic AI assistant. "
            "Based on the content below, generate flashcards in JSON format.\n"
            "Rules for Quantity:\n"
            "- If the content is SHORT (under 500 words), generate exactly 10 high-quality cards.\n"
            "- If the content is LONG, generate up to 25 cards.\n"
            "- Do not repeat facts just to fill space. Quality > Quantity.\n\n"
            "Strictly follow this JSON schema:\n"
            "[{\"question\": \"What is X?\", \"answer\": \"X is ...\"}, ...]\n"
            "Return ONLY raw JSON. No Markdown or explanation text.\n\n"
            f"Content:\n{data}"
        )

        try:
            response = model_cards.generate_content(flashcard_prompt)
            flashcards = extract_json_array(response.text)
        except Exception as e:
            print(f"Primary Flashcard generation failed: {e}")
            try:
                backup_raw = final_backup_response(flashcard_prompt)
                flashcards = extract_json_array(backup_raw)
            except:
                flashcards = []

        if not isinstance(flashcards, list) or len(flashcards) == 0:
             return jsonify({"error": "Could not generate valid flashcards."}), 500

        if len(flashcards) > 25:
            flashcards = flashcards[:25]

        db.collection("users").document(uid).collection("flashcards").add({
            "flashcards": flashcards,
            "created_at": datetime.utcnow().isoformat()
        })

        return jsonify({"flashcards": flashcards}), 200

    except Exception as e:
        print(f"Critical Route Error: {e}")
        return jsonify({"error": str(e)}), 500


@quiz_bp.route('/get_flashcards', methods=['GET'])
def get_flashcards():
    try:
        decoded = verify_token_from_request()
        uid = decoded["uid"]
        update_streak(uid)
        docs = db.collection("users").document(uid).collection("flashcards").order_by("created_at").get()
        return jsonify({"flashcards": [{"id": d.id, "flashcards": d.to_dict().get("flashcards", []), "created_at": d.to_dict().get("created_at")} for d in docs]}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@quiz_bp.route('/update_flashcards/<flashcard_id>', methods=['PUT'])
def update_flashcards(flashcard_id):
    try:
        decoded = verify_token_from_request()
        uid = decoded["uid"]
        cards = request.get_json().get("flashcards")
        if not cards or not isinstance(cards, list): return jsonify({"error": "Invalid data"}), 400
        db.collection("users").document(uid).collection("flashcards").document(flashcard_id).update({"flashcards": cards, "updated_at": firestore.SERVER_TIMESTAMP})
        return jsonify({"ok": True}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@quiz_bp.route('/flashcards/<flashcard_id>', methods=['DELETE'])
def delete_flashcards(flashcard_id):
    try:
        decoded = verify_token_from_request()
        uid = decoded['uid']
        db.collection("users").document(uid).collection("flashcards").document(flashcard_id).delete()
        return jsonify({ 'ok': True}), 200
    except Exception as e :
        return jsonify ({'error': str(e)}),400