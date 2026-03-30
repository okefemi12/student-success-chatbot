import mimetypes
import json
import re
import pandas as pd
import numpy as np
import cloudinary.uploader
from datetime import datetime
from flask import Blueprint, request, jsonify
from firebase_admin import firestore
from extensions import db
from io import BytesIO
from PIL import Image
from utils.helpers import verify_token_from_request, clean_response, update_streak, safe_clean_dict
from services.doc_processor import extract_text_from_pdf, extract_text_from_docx, extract_text_from_pptx, extract_text_from_excel,extract_text_from_txt
from services.ai_engine import model, model_pdf, reminder_model, generate_backup_response, final_backup_response, rec_backup_response, pipeline, YOUTUBE_API_KEY
import requests
from youtube_search import YoutubeSearch

study_bp = Blueprint('study', __name__)

def parse_study_plan(raw_text):
    study_plan = []
    for line in raw_text.splitlines():
        match = re.match(r".*Day\s*(\d+)[\:\-]?\s*(.+)", line, re.IGNORECASE)
        if match:
            topics = [t.strip() for t in re.split(r",|;", match.group(2)) if t.strip()]
            study_plan.append({"day": int(match.group(1)), "topics": topics})
    return study_plan

@study_bp.route('/media_summary', methods=['POST'])
def media_summary():
    try:
        # --- 1. Auth & Setup ---
        decoded = verify_token_from_request()
        uid = decoded["uid"]
        update_streak(uid)

        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        filename = file.filename.lower()
        
        # --- FIX 4: Accept Title from frontend ---
        title = request.form.get("title", "").strip()

        # Determine Mime Type (Crucial for passing images to Gemini)
        mime_type, _ = mimetypes.guess_type(filename)

        # --- 2. Upload to Cloudinary ---
        upload_result = cloudinary.uploader.upload(
            file,
            upload_preset="unsigned_summary_file",
            folder=f"file_summaries/{uid}",
            resource_type="auto"
        )
        file_url = upload_result["secure_url"]
        file_name = upload_result.get("original_filename", "document")
        
        # Fallback to filename if no title is provided
        if not title:
            title = file_name 

        # --- 3. Rewind File (Crucial!) ---
        file.seek(0)

        # --- 4. Extract Text OR Flag for Visual Mode ---
        extracted_text = ""
        is_visual_file = False 
        
        # --- FIX 2: Check for exceptions during extraction to catch errors immediately ---
        try:
            if filename.endswith(".pdf"):
                extracted_text += extract_text_from_pdf(file_url)
                if not extracted_text.strip():
                    is_visual_file = True
            
            elif filename.endswith(".docx"):
                extracted_text += extract_text_from_docx(file_url)
            elif filename.endswith(".pptx"):
                extracted_text += extract_text_from_pptx(file_url)
            elif filename.endswith((".xls", ".xlsx")):
                extracted_text += extract_text_from_excel(file_url)
            elif filename.endswith(".txt"):
                extracted_text += extract_text_from_txt(file_url)
            
            # Handle Standard Images
            elif filename.endswith((".jpg", ".jpeg", ".png", ".webp", ".heic")):
                is_visual_file = True
            else:
                return jsonify({"error": "Unsupported file type"}), 400
        except Exception as e:
            print(f"File extraction error: {e}")
            return jsonify({"error": "Failed to read the file content. The file might be corrupted."}), 400

        # Check if text extraction failed (and it's not an image)
        if not extracted_text.strip() and not is_visual_file:
            return jsonify({"error": "No readable text found in the document."}), 400

        # --- 5. Generate Summary (With 3-Layer Backup) ---
        final_answer = ""

        # === METHOD A: TEXT-BASED SUMMARY ===
        if extracted_text.strip() and not is_visual_file:
            words = extracted_text.split()
            
            # --- FIX: Skip chunking for normal-sized documents ---
            if len(words) < 2500:
                full_summary_prompt = f"Create a detailed, highly structured academic summary from the following notes:\n\n{extracted_text.strip()}"
            
            # --- Chunking for massive documents ---
            else:
                def chunk_text(text, max_words=2000):
                    words_list = text.split()
                    return [' '.join(words_list[i:i + max_words]) for i in range(0, len(words_list), max_words)]

                chunks = chunk_text(extracted_text)
                summaries = []
                
                # Summarize Chunks (With Backup Protection)
                for i, chunk in enumerate(chunks):
                    chunk_prompt = f"Summarize part {i+1} of this document clearly:\n\n{chunk.strip()}"
                    try:
                        response = model_pdf.generate_content(chunk_prompt)
                        summaries.append(response.text.strip())
                    except Exception as e:
                        print(f"Chunk {i+1} primary model failed: {e}")
                        try:
                            # Rescue the chunk using the backup model!
                            backup_resp = generate_backup_response(chunk_prompt)
                            summaries.append(backup_resp)
                        except Exception as e2:
                            print(f"Chunk {i+1} backup failed: {e2}")

                # Clean out any failed empty strings
                summaries = [s for s in summaries if s.strip()]
                
                # Prevent AI from receiving blank prompts
                if not summaries:
                    return jsonify({"error": "AI models failed to process the document text. It may be too complex or flagged by safety filters."}), 500

                # Final Compilation
                full_summary_prompt = (
                    "Create a detailed, highly structured academic summary by combining these section summaries:\n\n" +
                    "\n\n".join(summaries)
                )

            # --- Generate Final Answer ---
            try:
                # LAYER 1: Primary Model
                final_response = model.generate_content(full_summary_prompt)
                raw_answer = final_response.text.strip()
            except Exception as e:
                print(f"Primary Text Summary Failed: {e}")
                try:
                    # LAYER 2: First Backup
                    raw_answer = generate_backup_response(full_summary_prompt)
                except Exception as e2:
                    print(f"First Backup Failed: {e2}")
                    # LAYER 3: Final 6-Key Backup (Last Resort)
                    raw_answer = final_backup_response(full_summary_prompt)

            final_answer = clean_response(raw_answer)

        # === METHOD B: VISUAL SUMMARY (Images/Scanned PDFs) ===
        elif is_visual_file:
            try:
                # Prepare visual data
                file_data = file.read()
                visual_prompt = "You are Felix. Analyze this image/document. Provide a detailed academic summary."
                
                # Input for Gemini
                gemini_input = [visual_prompt, {"mime_type": mime_type, "data": file_data}]

                # LAYER 1: Primary Model
                response = model_pdf.generate_content(gemini_input)
                raw_answer = response.text.strip()

            except Exception as e:
                print(f"Primary Visual Summary Failed: {e}")
                file.seek(0)
                file_data = file.read()
                gemini_input = [visual_prompt, {"mime_type": mime_type, "data": file_data}]

                # --- FIX 1: Prevent sending bytes to Groq/Cohere text-only backup models ---
                # Since generate_backup_response uses text-only models, we skip straight to final_backup_response 
                # because your KeyManager rotates through Gemini keys which CAN handle visual data.
                try:
                    print("Skipping text-only backup, using Gemini visual backup keys...")
                    raw_answer = final_backup_response(gemini_input).strip()
                except Exception as e3:
                    print(f"Visual Backup Failed: {e3}")
                    raw_answer = ""
            
            final_answer = raw_answer

        else:
            return jsonify({"error": "Could not extract text from this file."}), 400

        # --- FIX 5: Validate AI output before saving to Database ---
        # Prevent saving "All model attempts failed" to the database
        if not final_answer or "All model attempts failed" in final_answer or "Error:" in final_answer:
            return jsonify({"error": "AI could not generate a valid summary from this file."}), 500

        # --- 6. Save to Firestore ---
        db.collection("users").document(uid).collection("summaries").add({
            "title": title, # FIX 4: Save the title
            "summary": final_answer,
            "created_at": datetime.utcnow().isoformat(),
            "source_file": file_name,
            "file_url": file_url
        })

        # --- FIX 3: Return standard structure (summary instead of response) ---
        return jsonify({
            "summary": final_answer, 
            "title": title,
            "file_url": file_url
        }), 200

    except Exception as e:
        print("Error in /media_summary:", str(e))
        return jsonify({"error": str(e)}), 500

@study_bp.route('/get_summaries', methods=['GET'])
def get_summaries():
    try:
        decoded = verify_token_from_request()
        uid = decoded["uid"]

        db = firestore.client()
        summaries_ref = db.collection("users").document(uid).collection("summaries").order_by("created_at", direction=firestore.Query.DESCENDING)
        summaries_docs = summaries_ref.stream()

        summaries = []
        for doc in summaries_docs:
            data = doc.to_dict()
            summaries.append({
                "id": doc.id,
                "summary": data.get("summary", ""),
                "created_at": data.get("created_at", "")
            })

        return jsonify({"summaries": summaries})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@study_bp.route('/delete_summaries/<summary_id>', methods=['DELETE'])
def delete_summary(summary_id):
    try:
        decoded = verify_token_from_request()
        uid = decoded['uid']
        doc_ref = db.collection("users").document(uid).collection("summaries").document(summary_id)
        doc = doc_ref.get()
        if not doc.exists:
            return jsonify({"error": "Quiz set not found"}), 404
        doc_ref.delete()
        return jsonify({"ok": True, "message": "summary successfully deleted"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@study_bp.route("/reminders/study-plan", methods=["POST"])
def create_study_plan():
    try:
        decoded = verify_token_from_request()
        uid = decoded["uid"]
        update_streak(uid)

        # Accept form-data
        title = request.form.get("title")
        start_date = request.form.get("start_date")
        due_date = request.form.get("due_date")
        description = request.form.get("description", "")
        uploaded_files = request.files.getlist("files")  # list of files

        file_links = []
        extracted_text = ""

        # ---------- Upload files to Cloudinary ----------
        for file in uploaded_files:
            upload_result = cloudinary.uploader.upload(
                file,
                upload_preset="unsigned_study_plans",
                folder=f"study_plans/{uid}",
                resource_type="auto"
            )
            file_url = upload_result["secure_url"]
            file_name = upload_result["original_filename"]

            file_links.append({"name": file_name, "url": file_url})

            # ---------- Extract text (PDF/Docs only) ----------
            # Note: We removed extract_text_from_image to avoid Tesseract crashes.
            # If you need image support here later, we can add Single-Image Native Vision.
            try:
                if file_name.lower().endswith(".pdf"):
                    extracted_text += extract_text_from_pdf(file_url)
                elif file_name.lower().endswith((".docx", ".doc")):
                    extracted_text += extract_text_from_docx(file_url)
            except Exception as e:
                print(f"Failed to extract text from {file_name}: {e}")

        # ---------- Generate study plan ----------
        prompt = f"""
        You are a helpful AI tutor.
        I have an upcoming test: {title}.
        Start date: {start_date}.
        Due date: {due_date}.
        Description: {description}.
        Notes/Materials: {extracted_text[:7500]} 

        Please create a day-by-day study plan starting from the Start date in this exact format:
        Day 1: Topic 1, Topic 2
        Day 2: Topic 3, Topic 4
        Day 3: Topic 5
        ...
        Continue until the Due date.
        """

        try:
            # Primary Generation
            response = reminder_model.generate_content(prompt)
            text_output = response.text
        except Exception as e:
            print(f"Primary Study Plan Gen failed: {e}")
            # Final Backup Logic
            text_output = final_backup_response(prompt)

        # Parse the result (whether from Primary or Backup)
        study_plan = parse_study_plan(text_output)

        # ---------- Save to Firestore ----------
        reminder_ref = db.collection("users").document(uid).collection("reminders").document()
        reminder_ref.set({
            "title": title,
            "due_date": due_date,
            "description": description,
            "files": file_links,
            "study_plan": study_plan,
            "completed": False,
            "created_at": firestore.SERVER_TIMESTAMP,
        })

        return jsonify({"ok": True, "study_plan": study_plan, "files": file_links})

    except Exception as e:
        print(f"Study Plan Error: {e}")
        return jsonify({"error": str(e)}), 400



# ---------- Get Reminders ----------
@study_bp.route("/get_reminders", methods=["GET"])
def get_reminders():
    try:
        decoded = verify_token_from_request()
        uid = decoded["uid"]

        reminders_ref = db.collection("users").document(uid).collection("reminders")
        docs = reminders_ref.order_by("created_at", direction=firestore.Query.DESCENDING).stream()

        reminders = []
        for doc in docs:
            data = doc.to_dict()
            data["id"] = doc.id
            reminders.append(data)

        return jsonify({"ok": True, "reminders": reminders})

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400



# ---------- Complete Reminder ----------
@study_bp.route("/reminders/<reminder_id>/complete", methods=["POST"])
def complete_reminder(reminder_id):
    try:
        decoded = verify_token_from_request()
        uid = decoded["uid"]

        reminder_ref = db.collection("users").document(uid).collection("reminders").document(reminder_id)
        reminder_ref.update({"completed": True})

        return jsonify({"ok": True, "message": "Reminder marked as completed"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ---------- Delete Reminder ----------
@study_bp.route("/reminders/<reminder_id>", methods=["DELETE"])
def delete_reminder(reminder_id):
    try:
        decoded = verify_token_from_request()
        uid = decoded["uid"]

        reminder_ref = db.collection("users").document(uid).collection("reminders").document(reminder_id)
        reminder_doc = reminder_ref.get()

        if reminder_doc.exists:
            reminder_data = reminder_doc.to_dict()

            # Optional: delete files from Cloudinary
            if "files" in reminder_data and isinstance(reminder_data["files"], list):
                for f in reminder_data["files"]:
                    try:
                        file_url = f.get("url")
                        if file_url:
                            # Extract public_id safely (assuming format: https://res.cloudinary.com/<cloud_name>/.../study_plans/<uid>/<public_id>.<ext>)
                            public_id = file_url.split("/")[-1].split(".")[0]
                            cloudinary.uploader.destroy(f"study_plans/{uid}/{public_id}")
                    except Exception as inner_e:
                        print(f"Cloudinary delete error: {inner_e}")

            # Delete Firestore doc
            reminder_ref.delete()

            return jsonify({"ok": True, "message": "Reminder and files deleted successfully"})
        else:
            return jsonify({"error": "Reminder not found"}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
@study_bp.route("/gamification/<uid>", methods=["GET", "POST"])
def gamification(uid):
    try:
        decoded = verify_token_from_request()
        if decoded["uid"] != uid:
            return jsonify({"error": "Unauthorized"}), 403

        user_ref = db.collection("users").document(uid)
        gamification_ref = user_ref.collection("meta").document("gamification")

        # Frontend sends badge, level, OR streaks updates
        if request.method == "POST":
            data = request.get_json() or {}
            add_badge = data.get("badge")
            new_level = data.get("level")
            
            # --- FIX 6: Handle Streak Updates ---
            streak_updates = {}
            if "streak" in data:
                streak_updates["streak"] = data["streak"]
            if "longest_streak" in data:
                streak_updates["longest_streak"] = data["longest_streak"]
            if streak_updates:
                # Update the main user profile where streaks are kept
                user_ref.update(streak_updates)
            # ------------------------------------

            # Fetch current gamification data
            doc = gamification_ref.get()
            gamification_data = doc.to_dict() if doc.exists else {
                "level": 1,
                "badges": []
            }

            # Frontend decides when to give badges
            if add_badge and add_badge not in gamification_data["badges"]:
                gamification_data["badges"].append(add_badge)

            # Optional manual level update
            if new_level and isinstance(new_level, int):
                gamification_data["level"] = new_level

            gamification_data["last_activity"] = firestore.SERVER_TIMESTAMP
            gamification_ref.set(gamification_data)

            return jsonify({"ok": True, "gamification": gamification_data}), 200

        # GET → Retrieve gamification info + streak from user profile
        elif request.method == "GET":
            doc = gamification_ref.get()
            gamification_data = doc.to_dict() if doc.exists else {
                "level": 1,
                "badges": []
            }

            # Add streak info from user profile
            user_doc = user_ref.get()
            streak_info = {"streak": 0, "longest_streak": 0}
            if user_doc.exists:
                user_data = user_doc.to_dict()
                streak_info["streak"] = user_data.get("streak", 0)
                streak_info["longest_streak"] = user_data.get("longest_streak", 0)

            gamification_data.update(streak_info)

            return jsonify({"gamification": gamification_data}), 200

    except Exception as e:
        print("Gamification error:", e)
        return jsonify({"error": str(e)}), 500


@study_bp.route("/library/<uid>", methods=["GET", "POST", "DELETE"])
def library(uid):
    try:
        decoded = verify_token_from_request()
        uid = decoded["uid"]
        update_streak(uid)
        user_library_ref = db.collection("users").document(uid).collection("library")

        # -----------------------
        # POST → Upload documents
        # -----------------------
        if request.method == "POST":
            title = request.form.get("title")
            uploaded_files = request.files.getlist("files")

            if not uploaded_files:
                return jsonify({"error": "No files uploaded"}), 400

            file_links = []
            for file in uploaded_files:
                upload_result = cloudinary.uploader.upload(
                    file,
                    upload_preset="unsigned_library",
                    folder=f"library/{uid}",
                    resource_type="auto",  # handles pdf, image, docx, etc.
                )
                file_links.append({
                    "name": upload_result.get("original_filename"),
                    "url": upload_result.get("secure_url"),
                    "type": upload_result.get("resource_type"),
                })

            doc_ref = user_library_ref.document()
            doc_ref.set({
                "title": title,
                "files": file_links,
                "created_at": firestore.SERVER_TIMESTAMP,
            })

            return jsonify({"ok": True, "message": "Files uploaded successfully", "file_links": file_links}), 200

        # -----------------------
        # GET → Retrieve all documents
        # -----------------------
        elif request.method == "GET":
            docs = user_library_ref.order_by("created_at", direction=firestore.Query.DESCENDING).stream()
            library_items = [{"id": doc.id, **doc.to_dict()} for doc in docs]
            return jsonify({"ok": True, "library": library_items}), 200

        # -----------------------
        # DELETE → Delete all library entries
        # -----------------------
        elif request.method == "DELETE":
            docs = user_library_ref.stream()
            for doc in docs:
                doc.reference.delete()
            return jsonify({"ok": True, "message": "All library documents deleted"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# -----------------------
# PUT → Update specific library document (title or files)
# -----------------------
@study_bp.route("/library_update/<uid>/<doc_id>", methods=["PUT"])
def update_library(uid, doc_id):
    try:
        decoded = verify_token_from_request()
        uid = decoded["uid"]
        title = request.form.get("title")
        uploaded_files = request.files.getlist("files")

        doc_ref = db.collection("users").document(uid).collection("library").document(doc_id)

        update_data = {}
        if title:
            update_data["title"] = title
        if uploaded_files:
            file_links = []
            for file in uploaded_files:
                upload_result = cloudinary.uploader.upload(
                    file,
                    upload_preset="unsigned_library",
                    folder=f"library/{uid}",
                    resource_type="auto",
                )
                file_links.append({
                    "name": upload_result.get("original_filename"),
                    "url": upload_result.get("secure_url"),
                    "type": upload_result.get("resource_type"),
                })
            update_data["files"] = firestore.ArrayUnion(file_links)

        if update_data:
            doc_ref.update(update_data)
            return jsonify({"ok": True, "message": "Library entry updated successfully"}), 200
        else:
            return jsonify({"error": "No updates provided"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# -----------------------
# DELETE → Delete specific library document
# -----------------------
@study_bp.route("/library_delete/<uid>/<doc_id>", methods=["DELETE"])
def delete_library_item(uid, doc_id):
    try:
        decoded = verify_token_from_request()
        uid = decoded["uid"]

        doc_ref = db.collection("users").document(uid).collection("library").document(doc_id)
        doc_ref.delete()

        return jsonify({"ok": True, "message": "Library document deleted successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@study_bp.route("/recommendations", methods=["GET"])
def recommend_content():
    try:
        decoded = verify_token_from_request()
        uid = decoded["uid"]

        # 1. Get user profile
        user_doc = db.collection("users").document(uid).get()
        if not user_doc.exists:
            return jsonify({"error": "User profile not found"}), 404
        user_data = user_doc.to_dict()

        course = user_data.get("course_of_study", "")
        subjects = user_data.get("subject", [])
        if isinstance(subjects, str):
            subjects = [s.strip() for s in subjects.split(",")]

        # Determine queries. Fallback to course or general tips if subjects are empty
        queries = subjects if subjects else ([course] if course else ["Study motivation and productivity tips"])

        # To prevent long loading times, limit to maximum 3 subjects at a time
        queries = queries[:3] 

        all_fetched_videos = []

        # 2. Fetch YouTube videos for EACH subject separately
        for query in queries:
            search_term = f"{course} {query}".strip() if course else query
            try:
                # Fetch top 3 videos per subject
                yt_results = YoutubeSearch(search_term, max_results=3).to_dict()
                for item in yt_results:
                    all_fetched_videos.append({
                        "subject": query, 
                        "title": item.get("title", "YouTube Video"),
                        "url": f"https://www.youtube.com{item.get('url_suffix')}"
                    })
            except Exception as e:
                print(f"YouTube Search Failed for {query}: {e}")
                continue

        # FALLBACK: If all YouTube searches failed
        if not all_fetched_videos:
            all_fetched_videos = [
                {"subject": "General Study", "title": "How to Study Effectively", "url": "https://www.youtube.com/watch?v=p60rN9JEapg"},
                {"subject": "General Study", "title": "Pomodoro Technique", "url": "https://www.youtube.com/watch?v=mNBmG24djoY"}
            ]

        # --- DYNAMIC DISTRIBUTION LOGIC ---
        if len(queries) > 1:
            instruction = "Please return a STRICT JSON list containing EXACTLY ONE best video recommendation for EACH subject."
        else:
            instruction = "Please return a STRICT JSON list containing the TOP 3 best video recommendations for this subject."

        # 3. Ask Gemini for JSON output
        prompt = f"""
        The student needs help with these subjects: {', '.join(queries)}.
        Here is a list of YouTube videos categorized by subject: 
        {json.dumps(all_fetched_videos, indent=2)}

        {instruction}
        Do not add Markdown (like ```json). Just the raw array.
        Each entry MUST have:
        - "title"
        - "url"
        - "reason" (Explain exactly why this video helps with that specific subject)
        """

        try:
            response = model.generate_content(prompt)
            raw_text = response.text.strip()
            cleaned = re.sub(r'```json|```', '', raw_text).strip()
            recommendations = json.loads(cleaned)
        except Exception as e:
            print(f"Primary Gemini failed: {e}")
            try:
                response = rec_backup_response(prompt)
                raw_text = response.text.strip()
                cleaned = re.sub(r'```json|```', '', raw_text).strip()
                recommendations = json.loads(cleaned)
            except Exception:
                # FINAL FALLBACK: Manually format based on the rules
                recommendations = []
                if len(queries) > 1:
                    # Pick 1 from each subject
                    added_subjects = set()
                    for v in all_fetched_videos:
                        if v.get('subject') not in added_subjects:
                            recommendations.append({
                                "title": v["title"], 
                                "url": v["url"], 
                                "reason": f"Highly recommended resource for {v.get('subject', 'your studies')}."
                            })
                            added_subjects.add(v.get('subject'))
                else:
                    # Pick top 3 for the single subject
                    recommendations = [
                        {"title": v["title"], "url": v["url"], "reason": f"Highly recommended resource for {v.get('subject', 'your studies')}."} 
                        for v in all_fetched_videos[:3]
                    ]

        # 4. Save recommendations to Firestore
        recs_ref = db.collection("users").document(uid).collection("recommendations")
        rec_doc = {
            "created_at": datetime.utcnow().isoformat(),
            "course": course,
            "subjects": queries,
            "recommendations": recommendations
        }
        recs_ref.add(rec_doc)

        return jsonify({
            "ok": True,
            "recommendations": recommendations,
            "raw_videos": all_fetched_videos
        }), 200

    except Exception as e:
        print(f"Recommendations Route Error: {e}")
        return jsonify({"error": str(e)}), 500

@study_bp.route("/predict_performance", methods=["GET"])
def predict_performance():
    try:
        from services.ai_engine import pipeline
        if pipeline is None:
            return jsonify({"error": "Prediction model is currently unavailable on the server."}), 503

        # Verify user authentication
        decoded = verify_token_from_request()
        uid = decoded["uid"]

        # 1. Fetch user data
        user_ref = db.collection("users").document(uid)
        user_doc = user_ref.get()
        if not user_doc.exists:
            return jsonify({"error": "User not found"}), 404

        user_data = user_doc.to_dict()

        # Compute age from date_of_birth
        default_age = 17
        dob_str = user_data.get("date_of_birth") 
        age = default_age
        if dob_str:
            try:
                dob = datetime.fromisoformat(dob_str).date()
            except Exception:
                try:
                    dob = datetime.strptime(dob_str, "%Y-%m-%d").date()
                except Exception:
                    dob = None
            if dob:
                today = datetime.utcnow().date()
                age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
                if age < 0: age = 0

        # --- THE DATA ADAPTER HACK ---
        # 1. Extract Real App Telemetry (calculated by /log_activity and gamification)
        raw_study_hours = user_data.get("study_hours_per_week", 0)
        raw_attendance = user_data.get("attendance_percentage", 0)
        raw_assignments = user_data.get("assignment_completed", 0) 
        streak = user_data.get("streak", 0)
        
        # 2. Map & Boost
        # Boost "attendance" slightly if they have a good login streak
        boosted_attendance = min(100, raw_attendance + (streak * 2))

        # Hardcode sleep to 8 hours so the ML model doesn't penalize them
        fake_sleep_hours = 8.0 

        gender_val = user_data.get("gender", "")
        gender_val = gender_val.lower() if isinstance(gender_val, str) else ""

        # 3. Format exactly as the .pkl file expects
        ml_input = {
            "study_hours_per_week": round(raw_study_hours / 40, 3),
            "sleep_hours_per_day": round(fake_sleep_hours / 10, 3), # Locked at 0.8
            "attendance_percentage": round(boosted_attendance / 100, 3),
            "assignments_completed": round(raw_assignments, 3),
            "participation_level": int(user_data.get("participation_level", 0)),
            "Age": int(age),
            "Gender": 1 if gender_val == "male" else 0,
            "StudyTimeWeekly": round(raw_study_hours, 2),
            "Absences": int(user_data.get("Absences", 0)),
            "Tutoring": int(user_data.get("Tutoring", 0))
        }

        df = pd.DataFrame([ml_input])

        # Run prediction
        prediction = pipeline.predict(df)
        proba = pipeline.predict_proba(df)[0]
        confidence = float(proba[1] * 100)
        fail_confidence = float((1 - proba[1]) * 100)

        # Generate App-Specific Feedback
        if int(prediction[0]) == 1:
            message = f"Our model predicts you'll PASS your next exam 🎯 (Confidence: {confidence:.2f}%). Keep up your current streaks and study volume!"
        else:
            message = f"The model predicts you might struggle on your next exam 📊 (Confidence: {fail_confidence:.2f}%). Let's improve your learning habits."

            reasons = []
            if ml_input["study_hours_per_week"] < 0.4: # Less than 16 hours a week
                reasons.append("log more focused study hours using the session tracker")
            if ml_input["attendance_percentage"] < 0.7:
                reasons.append("maintain your daily login streak and study sessions")
            if ml_input["assignments_completed"] < 0.5:
                reasons.append("take more AI quizzes to prove your knowledge retention")

            if reasons:
                message += " To improve, try to " + ", ".join(reasons) + "."

        def to_native(value):
            if isinstance(value, (np.floating, np.float32, np.float64)): return float(value)
            elif isinstance(value, (np.integer, np.int32, np.int64)): return int(value)
            elif isinstance(value, dict): return {k: to_native(v) for k, v in value.items()}
            elif isinstance(value, list): return [to_native(v) for v in value]
            else: return value

        prediction_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "prediction": int(prediction[0]),
            "confidence": float(confidence),
            "message": message,
            "input_features": to_native(ml_input)
        }

        user_ref.collection("predictions").add(prediction_data)

        return jsonify({
            "ok": True,
            "prediction": int(prediction[0]),
            "confidence": confidence,
            "message": message,
            "ml_input": to_native(ml_input)
        }), 200

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500