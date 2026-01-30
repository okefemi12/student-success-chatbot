from flask import jsonify, request,Blueprint
from datetime import datetime, timedelta
import pytz
from firebase_admin import auth, firestore
from extensions import db
from utils.helpers import verify_token_from_request, safe_clean_dict

auth_bp = Blueprint('auth', __name__)

@auth_bp.route("/register", methods=["POST"])
def register_user():
    try:
        decoded = verify_token_from_request()
        uid = decoded["uid"]
        email = decoded.get("email")

        data = request.get_json() or {}
        name = data.get("name") or decoded.get("name") or ""
        date_of_birth = data.get("date_of_birth")
        gender = data.get("gender")
        phone_number = data.get("phone_number") or data.get("phone")

  
        subject = data.get("subject") or []
        if isinstance(subject, str):
            subject = [s.strip() for s in subject.split(",") if s.strip()]

        course_of_study = data.get("course_of_study")
        country = data.get("country")
        school_type = data.get("school_type")
        school_name = data.get("school_name")
        degree = data.get("degree")

        if not name or not date_of_birth or not gender or not phone_number:
            return jsonify({"error": "Required fields: name, date_of_birth, gender, phone_number"}), 400

        user_doc_ref = db.collection("users").document(uid)
        user_doc = user_doc_ref.get()

        if not user_doc.exists:  
            # First time registration → create with defaults
            user_data = {
                "user_id": uid,
                "name": name,
                "email": email,
                "date_of_birth": date_of_birth,
                "gender": gender,
                "phone_number": phone_number,
                "subject": subject,
                "school_type": school_type,
                "school_name": school_name,
                "country": country,
                "degree": degree,
                "course_of_study": course_of_study,
                "created_at": firestore.SERVER_TIMESTAMP,
                "provider": decoded.get("firebase", {}).get("sign_in_provider"),

                # Initialize log activity fields
                "AttendanceDays": [],
                "Absences": 0,
                "streak": 0,
                "attendance_percentage": 0,
                "study_hours_per_week": 0,
                "last_weekly_reset": datetime.now(pytz.UTC).strftime("%Y-%m-%d"),
                "assignment_completed": 0,
                "Tutoring": 0,
                "sleep_hours_per_day": 8,
                "participation_level": 0
            }
            user_doc_ref.set(user_data)
        else:
            # If user already exists, only update profile info
            updates = {
                "name": name,
                "date_of_birth": date_of_birth,
                "gender": gender,
                "phone_number": phone_number,
                "subject": subject,
                "course_of_study": course_of_study,
                "country": country,
                "school_type": school_type,
                "school_name": school_name,
                "degree": degree,
            }
            updates = {k: v for k, v in updates.items() if v is not None}
            user_doc_ref.update(updates)
            user_data = {**user_doc.to_dict(), **updates}

        returned = user_data.copy()
        returned.pop("provider", None)

        return jsonify({"ok": True, "uid": uid, "profile": safe_clean_dict(returned)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@auth_bp.route("/profile", methods=["GET"])
def get_profile():
    try:
        decoded = verify_token_from_request()
        uid = decoded["uid"]

        doc = db.collection("users").document(uid).get()
        if not doc.exists:
            return jsonify({"error": "User profile not found"}), 404

        profile = doc.to_dict()
        profile.pop("provider", None)

        return jsonify({"ok": True, "profile": safe_clean_dict(profile)})
    except Exception as e:
        return jsonify({"error": str(e)}), 401
    
@auth_bp.route("/update_profile", methods=["POST"])
def update_profile():
    try:
        decoded = verify_token_from_request()
        uid = decoded["uid"]
        data = request.get_json() or {}

        
        allowed = [
            "name", "date_of_birth", "gender", "phone_number", "subject",
            "course_of_study", "country", "school_type", "school_name", "degree"
        ]
        updates = {}

        for k in allowed:
            if k in data:
                if k == "subject":
                    # Always store subject as list
                    subject = data["subject"]
                    if isinstance(subject, str):
                        subject = [s.strip() for s in subject.split(",") if s.strip()]
                    updates["subject"] = subject
                else:
                    updates[k] = data[k]

        if not updates:
            return jsonify({"error": "No updatable fields provided"}), 400

        db.collection("users").document(uid).update(updates)

        doc = db.collection("users").document(uid).get()
        profile = doc.to_dict() or {}
        profile.pop("provider", None)

        return jsonify({"ok": True, "profile": safe_clean_dict(profile)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@auth_bp.route("/delete-account", methods=["DELETE"])
def delete_account():
    try:
        decoded = verify_token_from_request()
        uid = decoded["uid"]

        db.collection("users").document(uid).delete()
        auth.delete_user(uid)

        return jsonify({"ok": True, "message": "Account deleted successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    

@auth_bp.route("/log_activity", methods=["POST"])
def log_activity():
    try:
        decoded = verify_token_from_request()
        uid = decoded["uid"]

        data = request.get_json(silent=True) or {}
        session_hours = data.get("session_hours")
        if session_hours is None:
            return jsonify({"error": "Missing session_hours"}), 400

        try:
            session_hours = float(session_hours)
        except Exception:
            return jsonify({"error": "session_hours must be a number"}), 400

        if session_hours <= 0:
            return jsonify({"error": "session_hours must be > 0"}), 400

        today_dt = datetime.now(pytz.UTC)
        today_date = today_dt.date()
        today_str = today_date.strftime("%Y-%m-%d")

        user_doc_ref = db.collection("users").document(uid)
        user_doc = user_doc_ref.get()
        if not user_doc.exists:
            return jsonify({"error": "User not found"}), 404

        user_data = user_doc.to_dict()
        attendance_days = user_data.get("AttendanceDays", [])

        # --- Update today's record ---
        today_record = next((d for d in attendance_days if d.get("date") == today_str), None)
        if today_record:
            today_record["hours_used"] = today_record.get("hours_used", 0) + session_hours
            # Mark attended if studied 30+ minutes
            today_record["attended"] = today_record["hours_used"] >= 0.5
        else:
            today_record = {
                "date": today_str,
                "hours_used": session_hours,
                "attended": session_hours >= 0.5 
            }
            attendance_days.append(today_record)

        # --- Attendance metrics ---
        parsed_dates = [datetime.strptime(d["date"], "%Y-%m-%d").date() for d in attendance_days]
        start_date = min(parsed_dates) if attendance_days else today_date
        total_days = (today_date - start_date).days + 1
        attended_days_count = sum(1 for d in attendance_days if d.get("attended"))
        absences = max(0, total_days - attended_days_count)
        attendance_percentage = round((attended_days_count / total_days) * 100, 2) if total_days > 0 else 0

        # --- Weekly reset ---
        last_reset_str = user_data.get("last_weekly_reset")
        last_reset_date = datetime.strptime(last_reset_str, "%Y-%m-%d").date() if last_reset_str else today_date
        if (today_date - last_reset_date).days >= 7:
            study_hours_per_week = session_hours
            last_reset_date = today_date
        else:
            one_week_ago = today_date - timedelta(days=7)
            study_hours_per_week = sum(
                d.get("hours_used", 0)
                for d in attendance_days
                if datetime.strptime(d["date"], "%Y-%m-%d").date() >= one_week_ago
            )

        # --- Fetch quiz performance for assignment completion ---
        quiz_scores_ref = db.collection("users").document(uid).collection("quiz_scores")
        quiz_docs = quiz_scores_ref.stream()
        percentages = [doc.to_dict().get("percentage", 0) for doc in quiz_docs]
        assignments_completed = round(sum(percentages) / (len(percentages) * 100), 3) if percentages else 0.0

        # --- Tutoring recommendation ---
        Tutoring = 1 if study_hours_per_week < 5 or attendance_percentage < 70 else 0

        # --- Participation Level (binary for model) ---
        participation_level = 1 if attendance_percentage >= 50 else 0

        # --- Sleep (default fallback) ---
        sleep_hours_per_day = user_data.get("sleep_hours_per_day", 8)

        # --- Update Firestore ---
        user_doc_ref.update({
            "AttendanceDays": attendance_days,
            "Absences": absences,
            "attendance_percentage": attendance_percentage,
            "study_hours_per_week": study_hours_per_week,
            "last_weekly_reset": last_reset_date.strftime("%Y-%m-%d"),
            "assignment_completed": assignments_completed,
            "participation_level": participation_level,
            "Tutoring": Tutoring
        })

        return jsonify({
            "ok": True,
            "updated_metrics": safe_clean_dict({
                "attendance_percentage": attendance_percentage,
                "study_hours_per_week": study_hours_per_week,
                "assignment_completed": assignments_completed,
                "participation_level": participation_level,
                "Tutoring": Tutoring
            })
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400
