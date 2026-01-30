import uuid
from flask import Blueprint, request, jsonify
from firebase_admin import firestore
from extensions import db
from utils.helpers import verify_token_from_request, clean_response, update_streak
from services.ai_engine import model, generate_backup_response, groq_client, search_web
from services.audio_service import generate_audio_with_retry
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

@chat_bp.route("/create_chat_session", methods=["POST"])
def create_chat_session():
    try:
        decoded = verify_token_from_request()  
        uid = decoded["uid"]

        # Generate unique session ID
        session_id = str(uuid.uuid4())

        # Save session metadata in Firestore
        session_ref = (
            db.collection("users")
            .document(uid)
            .collection("chat_sessions")
            .document(session_id)
        )
        session_ref.set({
            "created_at": firestore.SERVER_TIMESTAMP,
            "title": None  # will be auto-set on first user message
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

        # 🔹 Check if session exists + auto-set title if missing
        session_ref = (
            db.collection("users")
            .document(uid)
            .collection("chat_sessions")
            .document(session_id)
        )
        session_doc = session_ref.get()
        if session_doc.exists:
            session_data = session_doc.to_dict()
            if not session_data.get("title"):
                title = user_message[:50] + ("..." if len(user_message) > 50 else "")
                session_ref.update({"title": title})
        else:
            return jsonify({"error": "Invalid session_id"}), 404

        # 🔹 Load LangChain memory
        memory = get_memory(session_id)
        if not memory.chat_memory.messages:
            docs = (
                session_ref.collection("messages")
                .order_by("timestamp", direction=firestore.Query.ASCENDING)
                .limit(20)
                .stream()
            )
            for d in docs:
                m = d.to_dict()
                if m["role"] == "user":
                    memory.chat_memory.add_user_message(m["content"])
                elif m["role"] == "assistant":
                    memory.chat_memory.add_ai_message(m["content"])

        # 🔹 Build conversation context
        context = ""
        if memory.chat_memory.messages:
            context = "\n\nConversation History:\n"
            for msg in memory.chat_memory.messages[-10:]:
                if isinstance(msg, HumanMessage):
                    context += f"User: {msg.content}\n"
                elif isinstance(msg, AIMessage):
                    context += f"Assistant: {msg.content}\n"

        # 🔹 Fetch user profile
        profile_doc = db.collection("users").document(uid).get()
        profile_data = profile_doc.to_dict() if profile_doc.exists else {}

        profile_context = ""
        if not memory.chat_memory.messages:  # Only for first user input
            profile_doc = db.collection("users").document(uid).get()
            profile_data = profile_doc.to_dict() if profile_doc.exists else {}
            if profile_data:
                profile_context = (
                    f"User profile:\n"
                    f"- Name: {profile_data.get('name', 'N/A')}\n"
                    f"- Course of Study: {profile_data.get('course_of_study', 'N/A')}\n"
                    f"- Subject: {profile_data.get('subject', 'N/A')}\n"
                    f"- Degree: {profile_data.get('degree', 'N/A')}\n"
                    f"- school_type: {profile_data.get('school_type', 'N/A')}\n"
                    f"- school_name: {profile_data.get('school_name', 'N/A')}\n"
                    f"- country: {profile_data.get('country', 'N/A')}\n"
                )
        

        # 🔹 Build Gemini prompt
        gemini_prompt = (
            "You are FELIX, a helpful intelligent AI study tutor and assistant. Your task is to assist with studying. "
            "Your goal is to help students deeply understand and apply academic concepts, not just recall them.\n\n"
    
            "Follow these steps carefully for each question:\n"
            "• Start by explaining the key concept in simple, clear language.\n"
            "• If the question involves a calculation, proof, or problem, show the full worked solution step by step.\n"
            "• After solving, explain what the result means and give a quick real-life or practical connection.\n"
            "• If the user only asks for understanding (not solving), focus on clear explanations and short examples.\n"
            "• Keep your tone encouraging, like a supportive tutor.\n\n"
    
    
            "Guidelines:\n"
            "• Focus strictly on academic and study-related questions — avoid casual conversation.\n"
            "• If a question is unclear or off-topic, politely ask for clarification.\n"
            "• When the user provides complex questions, break them into simpler parts before answering.\n"
            "• Only greet the user with their name in the very first message of a new session.\n"
            "• Don't mention their name or greet again after the your first response.\n"
            "• Format your response with clear paragraphs and bullet points using the • symbol.\n"
            "• Avoid markdown or special characters.\n"
            "• Use previous context to maintain continuity.\n\n"
            "• Use plain ASCII math notation (sqrt, ^, /, lim_{x->0}, etc.).\n"
            
    
            f"{profile_context}\n"
            f"{context}\n\n"
            f"Current question: {user_message}\n\n"
    
            "Now think carefully and respond as FELIX — explain the concept briefly, then solve step by step if needed, "
            "and finish with a clear final answer and one short follow-up question."
            "If a topic is complex, recommend that the user reviews additional resources.\n"
       )


        try:
            gemini_response = model.generate_content(gemini_prompt)
            raw_answer = gemini_response.text.strip()
            answer = clean_response(raw_answer)
        except Exception as e:
            print(f"Gemini failed: {e}")
            raw_answer = generate_backup_response(gemini_prompt)
            answer = clean_response(raw_answer)

        # 🔹 Optional: add web search links
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
        chat_ref.add({
            "role": "user",
            "content": user_message,
            "timestamp": firestore.SERVER_TIMESTAMP
        })
        chat_ref.add({
            "role": "assistant",
            "content": answer,
            "timestamp": firestore.SERVER_TIMESTAMP
        })

        return jsonify({
            "response": answer,
            "session_id": session_id
        })

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
                    "audio_generated_at": firestore.SERVER_TIMESTAMP
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

        chat_ref = (
            db.collection("users")
            .document(uid)
            .collection("chat_sessions")
            .document(session_id)
            .collection("messages")
        )
        docs = chat_ref.stream()
        for d in docs:
            d.reference.delete()

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

        # 🔹 Fetch session metadata
        session_ref = (
            db.collection("users")
            .document(uid)
            .collection("chat_sessions")
            .document(session_id)
        )
        session_doc = session_ref.get()
        if not session_doc.exists:
            return jsonify({"error": "Invalid session_id"}), 404
        session_data = session_doc.to_dict()

        # 🔹 Fetch messages
        chat_ref = session_ref.collection("messages")
        docs = chat_ref.order_by("timestamp", direction=firestore.Query.ASCENDING).stream()

        messages = []
        for d in docs:
            m = d.to_dict()
            messages.append({
                "role": m.get("role"),
                "content": m.get("content"),
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
            "created_at": firestore.SERVER_TIMESTAMP,
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