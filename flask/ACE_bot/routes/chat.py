import uuid
import cloudinary.uploader
from flask import Blueprint, request, jsonify
from firebase_admin import firestore
from extensions import db
from utils.helpers import verify_token_from_request, clean_response, update_streak
from services.ai_engine import model, generate_backup_response, groq_client, search_web
from services.audio_service import generate_audio_with_retry
from services.doc_processor import extract_text_from_pdf, extract_text_from_docx, extract_text_from_pptx, extract_text_from_excel,extract_text_from_txt
from langchain.memory.buffer import ConversationBufferMemory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain.schema import HumanMessage, AIMessage

chat_bp = Blueprint('chat', __name__)

conversation_memories = {}

def get_memory(session_id):
    if session_id not in conversation_memories:
        history = InMemoryChatMessageHistory()
        conversation_memories[session_id] = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history",
            chat_memory=history
        )
    return conversation_memories[session_id]


@chat_bp.route("/chat/<session_id>/upload", methods=["POST"])
def upload_chat_document(session_id):
    try:
        decoded = verify_token_from_request()
        uid = decoded["uid"]
        
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        filename = file.filename.lower()

        # 1. Upload to Cloudinary for UI viewing
        upload_result = cloudinary.uploader.upload(
            file,
            upload_preset="chat-uploads", 
            folder=f"chat_docs/{uid}/{session_id}",
            resource_type="auto"
        )
        file_url = upload_result["secure_url"]
        file_name = upload_result.get("original_filename", "document")

        # 2. Extract Text
        extracted_text = ""
        try:
            if filename.endswith(".pdf"): extracted_text = extract_text_from_pdf(file_url)
            elif filename.endswith(".docx"): extracted_text = extract_text_from_docx(file_url)
            elif filename.endswith(".pptx"): extracted_text = extract_text_from_pptx(file_url)
            elif filename.endswith((".xls", ".xlsx")): extracted_text = extract_text_from_excel(file_url)
            elif filename.endswith((".txt")): extracted_text = extract_text_from_txt(file_url)
            else: return jsonify({"error": "Unsupported file type for chat extraction."}), 400
        except Exception as e:
            return jsonify({"error": "Failed to read document."}), 400

        if not extracted_text.strip():
            return jsonify({"error": "No readable text found in the document."}), 400

        # Limit text to 200,000 characters to fit safely inside a Firestore document
        extracted_text = extracted_text[:200000]

        # 3. Save FULL TEXT to Firestore (No Pinecone needed)
        session_ref = db.collection("users").document(uid).collection("chat_sessions").document(session_id)
        
        session_ref.collection("context").add({
            "filename": file_name,
            "file_url": file_url,
            "text": extracted_text,
            "timestamp": firestore.SERVER_TIMESTAMP
        })

        # 4. Save a system message to Firestore so it shows up in the chat UI
        session_ref.collection("messages").add({
            "role": "system", 
            "content": f"Uploaded document: {file_name}. You can now ask questions about it!",
            "file_url": file_url,
            "timestamp": firestore.SERVER_TIMESTAMP
        })

        return jsonify({"ok": True, "message": "Document text saved to chat memory!", "file_url": file_url}), 200

    except Exception as e:
        print(f"Chat upload error: {e}")
        return jsonify({"error": str(e)}), 500


# ==========================================
# CORE CHAT ROUTE
# ==========================================
@chat_bp.route("/create_chat_session", methods=["POST"])
def create_chat_session():
    try:
        decoded = verify_token_from_request()  
        uid = decoded["uid"]

        session_id = str(uuid.uuid4())
        session_ref = db.collection("users").document(uid).collection("chat_sessions").document(session_id)
        session_ref.set({
            "created_at": firestore.SERVER_TIMESTAMP,
            "title": None 
        })
        return jsonify({"ok": True, "session_id": session_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@chat_bp.route("/chat/<session_id>", methods=["POST"])
def chat(session_id):
    try:
        decoded = verify_token_from_request()
        uid = decoded['uid']
        update_streak(uid)

        data = request.get_json() or {}
        user_message = data.get("message", "").strip()
        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        # Check if session exists + auto-set title
        session_ref = db.collection("users").document(uid).collection("chat_sessions").document(session_id)
        session_doc = session_ref.get()
        if session_doc.exists:
            session_data = session_doc.to_dict()
            if not session_data.get("title"):
                title = user_message[:50] + ("..." if len(user_message) > 50 else "")
                session_ref.update({"title": title})
        else:
            return jsonify({"error": "Invalid session_id"}), 404

        # Load LangChain memory
        memory = get_memory(session_id)
        if not memory.chat_memory.messages:
            docs = session_ref.collection("messages").order_by("timestamp", direction=firestore.Query.ASCENDING).limit(20).stream()
            for d in docs:
                m = d.to_dict()
                if m["role"] == "user":
                    memory.chat_memory.add_user_message(m["content"])
                elif m["role"] == "assistant":
                    memory.chat_memory.add_ai_message(m["content"])

        context = ""
        if memory.chat_memory.messages:
            context = "\n\nConversation History:\n"
            for msg in memory.chat_memory.messages[-10:]:
                if isinstance(msg, HumanMessage):
                    context += f"User: {msg.content}\n"
                elif isinstance(msg, AIMessage):
                    context += f"Assistant: {msg.content}\n"

        profile_context = ""
        if not memory.chat_memory.messages: 
            profile_doc = db.collection("users").document(uid).get()
            profile_data = profile_doc.to_dict() if profile_doc.exists else {}
            if profile_data:
                profile_context = (
                    f"User profile:\n"
                    f"- Name: {profile_data.get('name', 'N/A')}\n"
                    f"- Course of Study: {profile_data.get('course_of_study', 'N/A')}\n"
                    f"- Subject: {profile_data.get('subject', 'N/A')}\n"
                    f"- Degree: {profile_data.get('degree', 'N/A')}\n"
                )

        # ---> FETCH FULL DOCUMENT CONTEXT FROM FIRESTORE <---
        docs_context = ""
        context_refs = session_ref.collection("context").stream()
        for doc in context_refs:
            doc_data = doc.to_dict()
            docs_context += f"\n--- Uploaded Document: {doc_data.get('filename')} ---\n"
            docs_context += f"{doc_data.get('text')}\n"

        if docs_context:
            docs_context = "\nREFERENCE MATERIALS (The user has uploaded these files. You CAN read them. The full text is provided here):\n" + docs_context

        # Gemini prompt with strict rules for the tutoring system 
        gemini_prompt = (
            "You are FELIX, a helpful AI study tutor.\n"
            "Your goal is to help students deeply understand and apply academic concepts, not just recall them.\n\n"
            "CRITICAL INSTRUCTION: You DO have the ability to read and analyze documents. "
            "If the user uploaded a document, its entire text is provided below in the REFERENCE MATERIALS section. "
            "NEVER say 'I don't have the ability to view files' or 'I am a text-based AI'. "
            "If the user asks 'what is this document about', read the Reference Materials and provide a detailed summary.\n\n"
            "Follow these steps carefully for each question:\n"
            "1. Start by explaining the key concept in simple, clear language.\n"
            "2. If the question involves a calculation, proof, or problem, show the full worked solution step by step.\n"
            "3. After solving, explain what the result means and give a quick real-life or practical connection.\n"
            "4. If the user only asks for understanding (not solving), focus on clear explanations and short examples.\n"
            "5. Keep your tone encouraging, like a supportive tutor.\n\n"
            "Guidelines:\n"
            "• Focus strictly on academic and study-related questions — if off-topic, politely steer the conversation back to studying.\n"
            "• If a question is unclear, ask for clarification.\n"
            "• Do not greet the user if there is already Conversation History provided below.\n"
            "• Use rich Markdown formatting (bolding, italics, headers, lists, code blocks) to make your response structured and easy to read.\n"
            "• Use plain ASCII math notation (e.g., x^2, sqrt(y)).\n\n"
            f"{profile_context}\n"
            f"{docs_context}\n"  
            f"{context}\n\n"
            f"Current question: {user_message}\n\n"
            "Now think carefully and respond as FELIX. Finish with a clear final answer and ONE short follow-up question to test their understanding. "
            "If a topic is highly complex, recommend that the user reviews additional resources."
        )

        try:
            gemini_response = model.generate_content(gemini_prompt)
            raw_answer = gemini_response.text.strip()
            answer = raw_answer
        except Exception as e:
            print(f"Gemini failed: {e}")
            raw_answer = generate_backup_response(gemini_prompt)
            answer = raw_answer

        # Optional: add web search links
        if any(keyword in user_message.lower() for keyword in ["how to", "explain", "definition", "what is", "research", "video", "reference"]):
            web_links = search_web(user_message)
            if web_links:
                limited_links = web_links[:3]
                links_text = "\n\nUseful Links:\n"
                for link in limited_links:
                    title = link.get("title", "View Resource")
                    url = link.get("link", "")
                    links_text += f"• {title}: {url}\n"
                answer += links_text
                answer = clean_response(answer)

        memory.chat_memory.add_user_message(user_message)
        memory.chat_memory.add_ai_message(answer)
        
        chat_ref = session_ref.collection("messages")
        chat_ref.add({"role": "user", "content": user_message, "timestamp": firestore.SERVER_TIMESTAMP})
        chat_ref.add({"role": "assistant", "content": answer, "timestamp": firestore.SERVER_TIMESTAMP})

        return jsonify({"response": answer, "session_id": session_id})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@chat_bp.route("/chat/<session_id>/audio", methods=["POST"])
def generate_audio(session_id):
    """
    Generate audio for a specific message in the chat session.
    """
    try:
        # Verify user authentication
        decoded = verify_token_from_request()
        uid = decoded['uid']
        
        # Verify session belongs to user
        session_ref = (
            db.collection("users")
            .document(uid)
            .collection("chat_sessions")
            .document(session_id)
        )
        session_doc = session_ref.get()
        if not session_doc.exists:
            return jsonify({"error": "Invalid session_id"}), 404
        
        # Get text to convert to audio
        data = request.get_json() or {}
        text = data.get("text", "").strip()
        message_id = data.get("message_id")  # Optional: to update specific message
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Generate audio with automatic retry/fallback
        try:
            audio_base64, mime_type = generate_audio_with_retry(text)
            
        except Exception as e:
            print(f"TTS generation failed after all retries: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({"error": f"TTS generation failed: {str(e)}"}), 500
        
        # Optional: Update Firestore message with audio data if message_id provided
        if message_id and audio_base64:
            try:
                message_ref = session_ref.collection("messages").document(message_id)
                message_ref.update({
                    "audio": audio_base64,
                    "mime_type": mime_type,
                    "timestamp": firestore.SERVER_TIMESTAMP
                })
            except Exception as e:
                print(f" Failed to update message with audio: {e}")
        
        # Return audio data
        return jsonify({
            "audio": audio_base64,
            "mime_type": mime_type,
            "message_id": message_id
        })

    except Exception as e:
        print(f" Error in audio generation route: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
# ==========================================
# OTHER ROUTES (Audio, Clear, History)
# ==========================================
@chat_bp.route("/chat_audio/<session_id>", methods=["POST"])
def chat_audio(session_id):
    try:
        decoded = verify_token_from_request()
        uid = decoded["uid"]
        update_streak(uid)

        # Check audio
        if "audio" not in request.files:
            return jsonify({"error": "No audio file uploaded"}), 400

        audio_file = request.files["audio"]

        # Save temporarily to memory or file system
        audio_bytes = audio_file.read()

        # Step 1: Transcribe with Groq Whisper
        transcription = groq_client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=("audio.wav", audio_bytes)
        )
        user_message = transcription.text.strip()

        # Step 2: Save user message to Firestore
        message_id = str(uuid.uuid4())
        message_doc = {
            "role": "user",
            "content": user_message,
            "timestamp": firestore.SERVER_TIMESTAMP,
        }
        db.collection("users").document(uid).collection("chat_sessions") \
            .document(session_id).collection("messages") \
            .document(message_id).set(message_doc)

        # Step 3: Generate chatbot response
        gemini_prompt = f"User said (from audio): {user_message}"
        try:
            gemini_response = model.generate_content(gemini_prompt)
            raw_answer = gemini_response.text.strip()
            answer = clean_response(raw_answer)
        except Exception as e:
            print(f"Gemini failed: {e}")
            raw_answer = generate_backup_response(gemini_prompt)
            answer = clean_response(raw_answer)

        # Step 4: Save bot response to Firestore
        bot_message_id = str(uuid.uuid4())
        bot_message_doc = {
            "role": "assistant",
            "content": answer,
            "created_at": firestore.SERVER_TIMESTAMP,
        }
        db.collection("users").document(uid).collection("chat_sessions") \
            .document(session_id).collection("messages") \
            .document(bot_message_id).set(bot_message_doc)

        return jsonify({
            "ok": True,
            "transcription": user_message,
            "answer": answer
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@chat_bp.route('/chat/clear-memory', methods=['POST'])
def clear_memory():
    try:
        decoded = verify_token_from_request()
        uid = decoded["uid"]

        data = request.get_json() or {}
        session_id = data.get("session_id")
        if not session_id:
            return jsonify({"error": "Missing session_id"}), 400

        if session_id in conversation_memories:
            conversation_memories[session_id].clear()

        # Delete messages
        chat_ref = db.collection("users").document(uid).collection("chat_sessions").document(session_id).collection("messages")
        docs = chat_ref.stream()
        for d in docs:
            d.reference.delete()
            
        # Delete context (the uploaded files)
        context_ref = db.collection("users").document(uid).collection("chat_sessions").document(session_id).collection("context")
        c_docs = context_ref.stream()
        for c in c_docs:
            c.reference.delete()

        return jsonify({"ok": True, "message": f"Memory cleared for session {session_id}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@chat_bp.route('/chat/get-history', methods=['POST'])
def get_history():
    try:
        decoded = verify_token_from_request()
        uid = decoded["uid"]

        data = request.get_json() or {}
        session_id = data.get("session_id")
        if not session_id:
            return jsonify({"error": "Missing session_id"}), 400

        session_ref = db.collection("users").document(uid).collection("chat_sessions").document(session_id)
        session_doc = session_ref.get()
        if not session_doc.exists:
            return jsonify({"error": "Invalid session_id"}), 404
        session_data = session_doc.to_dict()

        chat_ref = session_ref.collection("messages")
        docs = chat_ref.order_by("timestamp", direction=firestore.Query.ASCENDING).stream()

        messages = []
        for d in docs:
            m = d.to_dict()
            messages.append({
                "role": m.get("role"),
                "content": m.get("content"),
                "file_url": m.get("file_url"), 
                "timestamp": m.get("timestamp").isoformat() if m.get("timestamp") else None
            })

        return jsonify({
            "session_id": session_id,
            "title": session_data.get("title"),
            "messages": messages,
            "total_messages": len(messages)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500