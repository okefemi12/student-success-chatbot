import os
import io
import time
import base64
import wave

TTS_API_KEYS = [
    os.environ.get("ACE_VOICE"),              # Primary
    os.environ.get("ACE_VOICE_BACKUP_1"),     # Backup 1
    os.environ.get("ACE_VOICE_BACKUP_2"),     # Backup 2
    os.environ.get("ACE_VOICE_BACKUP_3"),     # Backup 3
]
TTS_API_KEYS = [key for key in TTS_API_KEYS if key]

current_tts_key_index = 0

def convert_to_wav(pcm_data, mime_type):
    """
    Convert raw PCM audio data to WAV format.
    
    Args:
        pcm_data: Raw PCM audio bytes
        mime_type: MIME type string (e.g., "audio/L16;rate=24000")
    
    Returns:
        bytes: WAV file data
    """
    try:
        # Parse sample rate from mime_type (e.g., "audio/L16;rate=24000")
        sample_rate = 24000  # default
        if "rate=" in mime_type:
            rate_str = mime_type.split("rate=")[1].split(";")[0]
            sample_rate = int(rate_str)
        
        # PCM parameters
        num_channels = 1  # mono
        sample_width = 2  # 16-bit = 2 bytes
        
        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(num_channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_data)
        
        # Get WAV data
        wav_buffer.seek(0)
        return wav_buffer.read()
        
    except Exception as e:
        print(f"WAV conversion error: {e}")
        # Return original data if conversion fails
        return pcm_data
    

def generate_audio_with_retry(text, max_retries=None):
    """
    Generate audio with automatic API key rotation on quota OR network errors.
    """
    global current_tts_key_index
    
    if max_retries is None:
        max_retries = len(TTS_API_KEYS)
    
    last_error = None
    
    for attempt in range(max_retries):
        try:
            from google import genai
            from google.genai import types
            
            # Get current TTS API key
            current_key = TTS_API_KEYS[current_tts_key_index]
            print(f"Attempting with TTS API key #{current_tts_key_index + 1}")
            
            tts_client = genai.Client(api_key=current_key)
            
            tts_contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=text)]
                )
            ]
            
            tts_config = types.GenerateContentConfig(
                response_modalities=["audio"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name="Charon"
                        )
                    )
                )
            )
            
            audio_chunks = []
            audio_mime_type = None
            
            # Generate stream
            for chunk in tts_client.models.generate_content_stream(
                model="gemini-2.5-flash-preview-tts", 
                contents=tts_contents,
                config=tts_config
            ):
                if (chunk.candidates and len(chunk.candidates) > 0 and 
                    chunk.candidates[0].content and chunk.candidates[0].content.parts):
                    
                    part = chunk.candidates[0].content.parts[0]
                    
                    if hasattr(part, 'inline_data') and part.inline_data:
                        if hasattr(part.inline_data, 'data') and part.inline_data.data:
                            audio_chunks.append(part.inline_data.data)
                            
                            if audio_mime_type is None and hasattr(part.inline_data, 'mime_type'):
                                audio_mime_type = part.inline_data.mime_type
            
            if not audio_chunks:
                raise Exception("No audio chunks received (Stream Empty)")
            
            combined_audio = b"".join(audio_chunks)
            
            # Convert to WAV if needed (ensure convert_to_wav is defined in your code)
            if audio_mime_type and "audio/L" in audio_mime_type:
                wav_data = convert_to_wav(combined_audio, audio_mime_type)
                audio_base64 = base64.b64encode(wav_data).decode("utf-8")
                mime_type = "audio/wav"
            else:
                audio_base64 = base64.b64encode(combined_audio).decode("utf-8")
                mime_type = audio_mime_type or "audio/wav"
            
            print(f"TTS success with API key #{current_tts_key_index + 1}")
            return audio_base64, mime_type
                
        except Exception as e:
            error_msg = str(e).lower()
            last_error = e
            
            # Check for Quota limits OR Network Connection errors
            is_quota_error = any(k in error_msg for k in ['quota', 'rate limit', 'resource exhausted', '429'])
            is_network_error = any(k in error_msg for k in ['getaddrinfo', 'connecterror', 'socket', 'errno 11002', '11001'])
            
            if is_quota_error:
                print(f"API key #{current_tts_key_index + 1} exhausted. Switching...")
                current_tts_key_index = (current_tts_key_index + 1) % len(TTS_API_KEYS)
                
            elif is_network_error:
                print(f"Network Error (DNS/Connection) with key #{current_tts_key_index + 1}: {e}")
                print("Waiting 3 seconds for internet to stabilize...")
                time.sleep(3) # Wait before retrying
                # We do NOT switch keys for network errors, we just retry the loop (or you can switch if you prefer)
                # If you want to switch keys anyway:
                # current_tts_key_index = (current_tts_key_index + 1) % len(TTS_API_KEYS)
                
            else:
                # Actual code error (not network/quota), so we crash
                print(f"❌ Unrecoverable error with TTS API key #{current_tts_key_index + 1}: {e}")
                raise e
    
    raise Exception(f"TTS generation failed after {max_retries} attempts. Last error: {last_error}")