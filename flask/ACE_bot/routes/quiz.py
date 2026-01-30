import json
import mimetypes
import requests
from io import BytesIO
from datetime import datetime
from PIL import Image
from flask import Blueprint, request, jsonify
from firebase_admin import firestore
from extensions import db
from utils.helpers import verify_token_from_request, update_streak
from services.doc_processor import extract_text_from_pdf, extract_text_from_docx, extract_text_from_pptx, extract_text_from_excel
from services.ai_engine import model_cards, model_pdf, final_backup_response
import cloudinary.uploader

quiz_bp = Blueprint('quiz', __name__)


@quiz_bp.route('/chat_quiz', methods=['POST'])
def create_chat_quiz():
    try:
        decoded = verify_token_from_request()
        uid = decoded["uid"]
        update_streak(uid)

        content = request.json.get("text", "")
        if not content:
            return jsonify({"error": "No input text provided"}), 400

        # --- 1. Stronger Prompt ---
        prompt = f"""
        You are ACE, a helpful AI tutor. Based on the content below, generate 10 quiz flashcards.
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

        quiz_data = []

        try:
            # --- 2. Primary Generation ---
            response = model_cards.generate_content(prompt)
            quiz_raw = response.text.strip()
            
            # Robust Parsing
            start_index = quiz_raw.find('[')
            end_index = quiz_raw.rfind(']')
            
            if start_index != -1 and end_index != -1:
                json_str = quiz_raw[start_index : end_index + 1]
                quiz_data = json.loads(json_str)
            else:
                raise ValueError("No JSON list found in response")

        except Exception as e:
            print(f"Primary Gemini failed: {e}")
            
            # --- 3. Backup Strategy (Added) ---
            try:
                # Call your global backup function
                backup_raw = final_backup_response(prompt)

                # Parse Backup Response
                start_index = backup_raw.find('[')
                end_index = backup_raw.rfind(']')
                
                if start_index != -1 and end_index != -1:
                    json_str = backup_raw[start_index : end_index + 1]
                    quiz_data = json.loads(json_str)
                else:
                    print("Backup returned invalid JSON")
            except Exception as backup_error:
                print(f"Backup also failed: {backup_error}")
                pass

        # --- 4. Validation & Saving ---
        # Ensure we have data (from either Primary or Backup)
        if not isinstance(quiz_data, list):
            quiz_data = []

        if not quiz_data:
            return jsonify({"error": "Failed to generate valid quiz data from AI."}), 500

        # 5. Save structured quiz to Firestore
        doc_ref = db.collection("users").document(uid).collection("quiz").document()
        doc_ref.set({
            "quiz": quiz_data,
            "created_at": datetime.utcnow().isoformat()
        })

        # 6. Return result
        return jsonify({"quiz": quiz_data}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@quiz_bp.route('/media_quiz', methods=['POST'])
def media_quiz():
    try:
        # --- 1. Auth & Setup ---
        decoded = verify_token_from_request()
        uid = decoded["uid"]
        update_streak(uid)

        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        filename = file.filename.lower()

        # --- 2. Upload to Cloudinary ---
        upload_result = cloudinary.uploader.upload(
            file,
            upload_preset="unsigned_quiz_uploads",
            folder=f"file_quizzes/{uid}",
            resource_type="auto"
        )

        file_url = upload_result["secure_url"]
        file_name = upload_result["original_filename"]

        # --- 3. Smart Extraction (Text or Image) ---
        extracted_text = ""
        image_part = None  # Stores the image object if we are in Image Mode

        try:
            if filename.endswith(".pdf"):
                extracted_text = extract_text_from_pdf(file_url)
            elif filename.endswith(".docx"):
                extracted_text = extract_text_from_docx(file_url)
            elif filename.endswith(".pptx"):
                extracted_text = extract_text_from_pptx(file_url)
            elif filename.endswith((".xls", ".xlsx")):
                extracted_text = extract_text_from_excel(file_url)
            
            # Image Support (Native Vision)
            elif filename.endswith((".png", ".jpg", ".jpeg", ".webp")):
                response = requests.get(file_url)
                image_part = Image.open(BytesIO(response.content))
                extracted_text = "IMAGE_MODE" 
            else:
                return jsonify({"error": "Unsupported file type"}), 400
        except Exception as e:
            print(f"Extraction error: {e}")
            return jsonify({"error": "Failed to read document content"}), 400

        if not extracted_text:
            return jsonify({"error": "No readable content found"}), 400

        # --- 4. AI Generation ---
        base_prompt = (
            "You are FELIX, a helpful AI tutor. Based on the study material provided,"
            "generate a comprehensive multiple-choice quiz in JSON format.\n"
            "Rules for Quantity:\n"
            "- If the content is SHORT (under 3 pages), generate 10-15 questions.\n"
            "- If the content is EXTENSIVE (long docs, whole chapters), generate 20-30 questions.\n"
            "- CRITICAL: Ensure you cover the BEGINNING, MIDDLE, and END of the content"
            "- Do not repeat questions.\n\n"
            "Strictly follow this schema:\n"
            "[{\"question\": \"...\", \"options\": {\"A\": \"...\", \"B\": \"...\", \"C\": \"...\", \"D\": \"...\"}, \"answer\": \"A\"}]\n"
            "Return ONLY raw JSON. No Markdown."
        )

        quiz_data = []

        try:
            # --- Primary Attempt ---
            if image_part:
                # Image Mode (Send Prompt + Image)
                response = model_cards.generate_content([base_prompt, image_part])
            else:
                # Text Mode 
                full_prompt = f"{base_prompt}\n\nText:\n{extracted_text[:50000]}"
                response = model_cards.generate_content(full_prompt)

            raw_response = response.text.strip()
            
            # Robust Parsing
            start = raw_response.find('[')
            end = raw_response.rfind(']')
            if start != -1 and end != -1:
                quiz_data = json.loads(raw_response[start : end + 1])
            else:
                raise ValueError("No JSON list found")

        except Exception as e:
            print(f"Gemini quiz generation failed: {e}")
            
            # --- Backup Strategy (Using KeyManager) ---
            try:
                # 1. Determine Input for Backup
                if image_part:
                    backup_input = [base_prompt, image_part]
                else:
                    backup_input = f"{base_prompt}\n\nText:\n{extracted_text[:50000]}"
                
                # 2. Call the NEW global backup function
                backup_raw = final_backup_response(backup_input) # <--- RENAMED CALL
                
                # 3. Parse Backup Response
                start = backup_raw.find('[')
                end = backup_raw.rfind(']')
                if start != -1 and end != -1:
                    quiz_data = json.loads(backup_raw[start : end + 1])
                else:
                    print("Backup returned invalid JSON")
            except Exception as backup_error:
                print(f"Backup also failed: {backup_error}")
                pass

        # --- 6. Validation ---
        if not isinstance(quiz_data, list) or len(quiz_data) == 0:
             return jsonify({"error": "Could not generate valid quiz."}), 500

        # --- 7. Save & Return ---
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

# ---------- Get All Quizzes ----------
@quiz_bp.route('/get_chat_quiz', methods=['GET'])
def get_chat_quiz():
    try:
        decoded = verify_token_from_request()
        uid = decoded["uid"]

        quiz_docs = db.collection("users").document(uid).collection("quiz").order_by("created_at").get()

        quiz_list = []
        for doc in quiz_docs:
            data = doc.to_dict()
            quiz_list.append({
                "id": doc.id,
                "quiz": data.get("quiz"),
                "created_at": data.get("created_at")
            })

        return jsonify({"quiz": quiz_list}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------- Get One Quiz by ID ----------
@quiz_bp.route('/get_chat_quiz/<quiz_id>', methods=['GET'])
def get_single_quiz(quiz_id):
    try:
        decoded = verify_token_from_request()
        uid = decoded["uid"]
        update_streak(uid)

        doc_ref = db.collection("users").document(uid).collection("quiz").document(quiz_id)
        doc = doc_ref.get()

        if not doc.exists:
            return jsonify({"error": "Quiz not found"}), 404

        quiz_data = doc.to_dict()
        return jsonify({
            "id": doc.id,
            "quiz": quiz_data.get("quiz"),
            "created_at": quiz_data.get("created_at")
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------- Delete Quiz ----------
@quiz_bp.route('/delete_quiz/<quiz_id>', methods=['DELETE'])
def delete_quiz(quiz_id):
    try:
        decoded = verify_token_from_request()
        uid = decoded["uid"]

        doc_ref = db.collection("users").document(uid).collection("quiz").document(quiz_id)
        doc = doc_ref.get()

        if not doc.exists:
            return jsonify({"error": "Quiz set not found"}), 404

        doc_ref.delete()
        return jsonify({"ok": True, "message": "Quiz successfully deleted"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@quiz_bp.route('/save_quiz_score', methods=['POST'])
def save_quiz_score():
    try:
        decoded = verify_token_from_request()
        uid = decoded["uid"]

        data = request.json
        quiz_id = data.get("quiz_id")  # Could be PDF ID or timestamp
        score = data.get("score")
        total = data.get("total")

        # Reference quiz scores collection
        score_ref = db.collection("users").document(uid).collection("quiz_scores").document(quiz_id)

        # Update if exists, else create new
        score_ref.set({
            "score": score,
            "total": total,
            "percentage": (score / total) * 100 if total > 0 else 0,
            "updated_at": datetime.utcnow()
        }, merge=True)

        return jsonify({"message": "Score saved successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500





@quiz_bp.route('/media_flashcards', methods=['POST'])
def media_flashcards():
    try:
        # --- 1. Auth & Setup ---
        decoded = verify_token_from_request()
        uid = decoded["uid"]
        update_streak(uid)

        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        filename = file.filename.lower()

        # --- 2. Extract text FIRST (before uploading) ---
        extracted_text = ""
        image_part = None

        try:
            if filename.endswith(".pdf"):
                # Read directly from upload
                file.seek(0)
                reader = PdfReader(file)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        extracted_text += page_text + "\n"
                
            elif filename.endswith(".docx"):
                file.seek(0)
                doc = Document(file)
                extracted_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
                
            elif filename.endswith(".pptx"):
                file.seek(0)
                prs = Presentation(file)
                for slide in prs.slides:
                    for shape in slide.shapes:
                        if hasattr(shape, "text"):
                            extracted_text += shape.text + "\n"
                            
            elif filename.endswith((".xls", ".xlsx")):
                file.seek(0)
                wb = load_workbook(file)
                for sheet in wb.worksheets:
                    for row in sheet.iter_rows(values_only=True):
                        row_text = " ".join(str(cell) for cell in row if cell)
                        if row_text.strip():
                            extracted_text += row_text + "\n"
            
            elif filename.endswith((".png", ".jpg", ".jpeg", ".webp")):
                file.seek(0)
                image_part = Image.open(file)
                extracted_text = "IMAGE_MODE"
            else:
                return jsonify({"error": "Unsupported file type"}), 400
                
        except Exception as e:
            print(f"Extraction error: {e}")
            return jsonify({"error": "Failed to read document text"}), 400

        if not extracted_text:
            return jsonify({"error": "No readable content found"}), 400

        # --- 3. NOW upload to Cloudinary for storage ---
        file.seek(0)  # Reset file pointer
        upload_result = cloudinary.uploader.upload(
            file,
            upload_preset="unsigned_flashcards",
            folder=f"flashcards/{uid}",
            resource_type="auto"
        )

        file_url = upload_result["secure_url"]
        file_name = upload_result["original_filename"]

        # --- 4. AI Generation ---
        base_prompt = (
            "You are ACE. Generate flashcards based on the study material in JSON format.\n"
            "Rules for Quantity:\n"
            "- If the content is SHORT, generate exactly 10-15 cards.\n"
            "- If the content is EXTENSIVE, generate up to 25-50 cards.\n"
            "-Ensure you cover the BEGINNING, MIDDLE, and END of the content"
            "- Do not repeat facts.\n\n"
            "Strictly follow this schema:\n"
            "[{\"question\": \"...\", \"answer\": \"...\"}]\n"
            "Return ONLY raw JSON. No Markdown."
        )

        flashcards = []

        try:
            # --- Primary Attempt ---
            if image_part:
                # Image Mode
                response = model_pdf.generate_content([base_prompt, image_part])
            else:
                # Text Mode (Limited to 7500 chars)
                final_prompt = f"{base_prompt}\n\nText:\n{extracted_text[:50000]}"
                response = model_pdf.generate_content(final_prompt)

            raw_response = response.text.strip()
            
            # Parsing
            start = raw_response.find('[')
            end = raw_response.rfind(']')
            if start != -1 and end != -1:
                flashcards = json.loads(raw_response[start : end + 1])
            else:
                raise ValueError("No JSON list found")

        except Exception as e:
            print(f"Primary generation failed: {e}")
            
            # --- Backup Strategy ---
            try:
                # 1. Determine Input for Backup
                if image_part:
                    # Multimodal Backup
                    backup_input = [base_prompt, image_part]
                else:
                    # Text Backup
                    backup_input = f"{base_prompt}\n\nText:\n{extracted_text[:50000]}"
                
                # 2. Call the global backup function
                backup_raw = final_backup_response(backup_input)

                # 3. Parse Backup Response
                start = backup_raw.find('[')
                end = backup_raw.rfind(']')
                if start != -1 and end != -1:
                    flashcards = json.loads(backup_raw[start : end + 1])
                else:
                    print("Backup returned invalid JSON")
            except Exception as backup_error:
                print(f"Backup also failed: {backup_error}")
                pass 

        # --- 5. Validation ---
        if not isinstance(flashcards, list):
            flashcards = []
        
        if len(flashcards) > 50: 
            flashcards = flashcards[:50]

        if not flashcards:
            return jsonify({"error": "AI could not generate flashcards from this file"}), 500

        # --- 6. Save & Return ---
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
            "You are ACE, a helpful academic AI assistant."
            "Based on the content below, generate flashcards in JSON format.\n"
            "Rules for Quantity:\n"
            "- If the content is SHORT (under 500 words), generate exactly 10 high-quality cards."
            "- If the content is LONG, generate up to 50 cards."
            "Strictly follow this JSON schema:\n"
            "[{\"question\": \"What is X?\", \"answer\": \"X is ...\"}, ...]\n"
            "Return ONLY raw JSON. No Markdown or explanation text.\n\n"
            f"Content:\n{data}"
        )

        flashcards = [] # Initialize variable

        # --- 1. Generation & Parsing ---
        try:
            response = model_cards.generate_content(flashcard_prompt)
            flashcard_raw = response.text.strip()
            
            # Robust Extraction: Find the list brackets directly
            start = flashcard_raw.find('[')
            end = flashcard_raw.rfind(']')
            
            if start != -1 and end != -1:
                json_str = flashcard_raw[start : end + 1]
                flashcards = json.loads(json_str)
            else:
                raise ValueError("No JSON list found in response")

        except Exception as e:
            print(f"Primary Flashcard generation failed: {e}")
            
            # --- 2. Backup Handling ---
            # Assuming generate_backup_response returns a string
            backup_raw = final_backup_response(flashcard_prompt)
            
            # Try to parse backup, or fail gracefully
            try:
                start = backup_raw.find('[')
                end = backup_raw.rfind(']')
                if start != -1 and end != -1:
                    flashcards = json.loads(backup_raw[start : end + 1])
                else:
                    # Fallback if backup is not JSON: wrap raw text in a single card
                    flashcards = [{"question": "Summary", "answer": backup_raw}]
            except:
                flashcards = [{"question": "Error", "answer": "Could not generate flashcards."}]

        # --- 3. Save & Return (NOW UN-INDENTED) ---
        # This code now runs regardless of whether the Try or Except block executed.
        
        # Validation: Ensure we actually have a list before saving
        if not isinstance(flashcards, list):
            flashcards = []

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

        flashcards = (
            db.collection("users")
              .document(uid)
              .collection("flashcards")
              .order_by("created_at")
              .get()
        )

        flashcard_sets = []
        for doc in flashcards:
            data = doc.to_dict()
            flashcard_sets.append({
                "id": doc.id,
                "flashcards": data.get("flashcards", []),
                "created_at": data.get("created_at")
            })

        return jsonify({"flashcards": flashcard_sets}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@quiz_bp.route('/update_flashcards/<flashcard_id>', methods=['PUT'])
def update_flashcards(flashcard_id):
    try:
        decoded = verify_token_from_request()
        uid = decoded["uid"]

        data = request.get_json() or {}
        new_flashcards = data.get("flashcards")

        if not new_flashcards or not isinstance(new_flashcards, list):
            return jsonify({"error": "Flashcards must be provided as a list"}), 400

        doc_ref = db.collection("users").document(uid).collection("flashcards").document(flashcard_id)
        doc = doc_ref.get()

        if not doc.exists:
            return jsonify({"error": "Flashcard set not found"}), 404

        doc_ref.update({
            "flashcards": new_flashcards,
            "updated_at": firestore.SERVER_TIMESTAMP
        })

        return jsonify({"ok": True, "message": "Flashcards updated"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@quiz_bp.route('/flashcards/<flashcard_id>', methods=['DELETE'])
def delete_flashcards(flashcard_id):
    try:
        decoded = verify_token_from_request()
        uid = decoded['uid']
        flashcard_ref = db.collection("users").document(uid).collection("flashcards").document(flashcard_id)
        flashcard_ref.delete()
        return jsonify({ 'ok': True, "message": "flashcard deleted sucessfully"})
    except Exception as e :
        return jsonify ({'error': str(e)}),400