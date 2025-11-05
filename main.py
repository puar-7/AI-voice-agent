import os
import io
import asyncio
from datetime import datetime
from typing import List, Dict
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from vocode.streaming.models.agent import ChatGPTAgentConfig
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.telephony import TwilioConfig
from vocode.streaming.telephony.server.base import (
    TelephonyServer,
    TwilioInboundCallConfig,
)
from vocode.streaming.models.synthesizer import (
    ElevenLabsSynthesizerConfig,
    AudioEncoding,
)

from gtts import gTTS
from groq import Groq

# --- 1) FastAPI app ---
app = FastAPI()

# Add CORS middleware for web demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2) Agent prompt ---
AGENT_PROMPT = """
You are a friendly, casual AI assistant named 'Arya'.
- Start conversations casually in Hindi or English based on user's language
- Keep replies short (1-2 sentences) and natural
- Maintain context of the conversation
- Be warm and helpful
- If asked about construction project, say: "The foundation work is complete, and we're on schedule to start framing next week."
- For Hindi, respond naturally in Hinglish or Hindi
- Remember what user said earlier in the conversation
"""

# --- 3) Environment variables ---
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
YOUR_TWILIO_PHONE_NUMBER = os.environ.get("YOUR_TWILIO_PHONE_NUMBER", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
ELEVEN_LABS_API_KEY = os.environ.get("ELEVEN_LABS_API_KEY", "")
RENDER_EXTERNAL_URL = os.environ.get("RENDER_EXTERNAL_URL", "http://localhost:8000")

# Initialize Groq client - CRITICAL: Do this BEFORE Twilio setup
groq_client = None
try:
    if GROQ_API_KEY:
        groq_client = Groq(api_key=GROQ_API_KEY)
        print("✅ Groq client initialized successfully")
    else:
        print("⚠️ GROQ_API_KEY not found!")
except Exception as e:
    print(f"❌ Failed to initialize Groq: {e}")

# --- 4) Session storage for conversation history ---
conversation_sessions: Dict[str, List[Dict]] = {}

# --- 5) Pydantic models ---
class ChatMessage(BaseModel):
    message: str
    language: str = "en"
    session_id: str = "default"

class ChatResponse(BaseModel):
    response: str
    timestamp: str

# --- 6) Configure Twilio if credentials exist ---
startup_error = None
try:
    if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and YOUR_TWILIO_PHONE_NUMBER:
        agent_config = ChatGPTAgentConfig(
            initial_message=BaseMessage(text=" "),
            prompt_preamble=AGENT_PROMPT,
            model_name="llama-3.3-70b-versatile",  # Updated model
            allow_agent_to_be_interrupted=True,
            openai_api_key=GROQ_API_KEY,
            openai_api_base="https://api.groq.com/openai/v1",
        )

        ELEVEN_LABS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Rachel
        synthesizer_config = ElevenLabsSynthesizerConfig(
            api_key=ELEVEN_LABS_API_KEY,
            voice_id=ELEVEN_LABS_VOICE_ID,
            sampling_rate=8000,
            audio_encoding=AudioEncoding.MULAW,
        )

        twilio_config = TwilioConfig(
            account_sid=TWILIO_ACCOUNT_SID,
            auth_token=TWILIO_AUTH_TOKEN,
        )

        telephony_server = TelephonyServer(
            base_url=RENDER_EXTERNAL_URL,
            config_manager=None,
            inbound_call_configs=[
                TwilioInboundCallConfig(
                    url="/inbound_call",
                    agent_config=agent_config,
                    synthesizer_config=synthesizer_config,
                    twilio_config=twilio_config,
                )
            ],
        )
        app.include_router(telephony_server.get_router())
        print("✅ Twilio integration enabled")
    else:
        print("⚠️ Twilio credentials not found, phone calls disabled")

except Exception as e:
    startup_error = e
    print(f"Startup Error: {startup_error}")

# --- 7) Health check ---
@app.get("/")
def root():
    if startup_error:
        return {"error": f"Startup failed: {startup_error}"}
    return {
        "message": "🎙️ Arya AI Voice Agent is running!",
        "endpoints": {
            "demo": "/demo",
            "chat": "/chat (POST)",
            "tts": "/tts?text=hello&lang=en",
            "test": "/test"
        },
        "features": {
            "groq_llm": bool(GROQ_API_KEY),
            "twilio_calls": bool(TWILIO_ACCOUNT_SID),
            "tts": True
        }
    }

# --- 8) Chat endpoint with LLM ---
@app.post("/chat", response_model=ChatResponse)
async def chat(data: ChatMessage):
    """
    Chat endpoint that uses Groq LLM with conversation memory.
    """
    try:
        if not groq_client:
            print("❌ ERROR: Groq client is None!")
            print(f"   GROQ_API_KEY exists: {bool(GROQ_API_KEY)}")
            print(f"   GROQ_API_KEY length: {len(GROQ_API_KEY) if GROQ_API_KEY else 0}")
            raise HTTPException(status_code=500, detail="Groq API not configured - check logs")
        
        # Get or create session history
        if data.session_id not in conversation_sessions:
            conversation_sessions[data.session_id] = []
        
        session_history = conversation_sessions[data.session_id]
        
        # Add user message to history
        session_history.append({
            "role": "user",
            "content": data.message
        })
        
        # Keep only last 10 messages for context
        if len(session_history) > 10:
            session_history = session_history[-10:]
            conversation_sessions[data.session_id] = session_history
        
        # Prepare messages for Groq
        messages = [
            {"role": "system", "content": AGENT_PROMPT}
        ] + session_history
        
        print(f"🔄 Calling Groq API with {len(messages)} messages...")
        
        # Call Groq API with updated model
        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",  # Updated to latest model
            temperature=0.7,
            max_tokens=150,
            top_p=0.9,
        )
        
        response_text = chat_completion.choices[0].message.content.strip()
        
        print(f"✅ Got response: {response_text[:50]}...")
        
        # Add assistant response to history
        session_history.append({
            "role": "assistant",
            "content": response_text
        })
        
        return ChatResponse(
            response=response_text,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ CHAT ERROR DETAILS:")
        print(f"   Error Type: {type(e).__name__}")
        print(f"   Error Message: {str(e)}")
        print(f"   Groq Client Status: {groq_client is not None}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        
        # Fallback response
        fallback_responses = {
            "en": "I'm having trouble processing that. Could you rephrase?",
            "hi": "Maaf kijiye, main samajh nahi paayi. Kripya dubara kahein?"
        }
        return ChatResponse(
            response=fallback_responses.get(data.language, fallback_responses["en"]),
            timestamp=datetime.now().isoformat()
        )

@app.delete("/chat/session/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation history for a session."""
    if session_id in conversation_sessions:
        del conversation_sessions[session_id]
    return {"message": "Session cleared"}

# --- 9) Enhanced demo page with React component ---
@app.get("/demo", response_class=HTMLResponse)
def demo_page():
    """Serve the full interactive demo."""
    if startup_error:
        return HTMLResponse(f"<pre>Startup failed: {startup_error}</pre>", status_code=500)
    
    # Try to load demo.html if it exists
    try:
        import os
        demo_path = os.path.join(os.path.dirname(__file__), "demo.html")
        if os.path.exists(demo_path):
            with open(demo_path, "r", encoding="utf-8") as f:
                return HTMLResponse(f.read())
    except Exception as e:
        print(f"Could not load demo.html: {e}")
    
    # Fallback: Return simple demo page with instructions
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
    <title>Arya Voice Demo</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: system-ui; max-width: 800px; margin: 40px auto; padding: 20px; }
        h1 { color: #667eea; }
        .info { background: #f0f0f0; padding: 20px; border-radius: 8px; margin: 20px 0; }
        pre { background: #2d2d2d; color: #f8f8f8; padding: 15px; border-radius: 5px; overflow-x: auto; }
        a { color: #667eea; text-decoration: none; font-weight: bold; }
        a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <h1>🎙️ Arya Voice Agent Demo</h1>
    
    <div class="info">
        <h2>⚠️ Demo UI File Missing</h2>
        <p>The <code>demo.html</code> file is not deployed. You have two options:</p>
        <ol>
            <li><strong>Use the simple test page:</strong> <a href="/test">Go to /test</a></li>
            <li><strong>Deploy demo.html:</strong> Add it to your git repo and push</li>
        </ol>
    </div>
    
    <h2>API Testing</h2>
    <p>You can test the chat API directly using curl or Postman:</p>
    <pre>curl -X POST https://ai-voice-agent-2dri.onrender.com/chat \\
  -H "Content-Type: application/json" \\
  -d '{
    "message": "Hello Arya!",
    "language": "en",
    "session_id": "test123"
  }'</pre>
    
    <h2>Quick Links</h2>
    <ul>
        <li><a href="/test">Simple TTS Test Page</a></li>
        <li><a href="/docs">API Documentation</a></li>
        <li><a href="/info">System Info</a></li>
    </ul>
</body>
</html>
    """)

# --- 10) TTS using gTTS ---
async def generate_gtts(text: str, lang: str = 'en') -> bytes:
    """Generate speech using Google Text-to-Speech."""
    def _generate():
        fp = io.BytesIO()
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp.read()
    
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _generate)

@app.get("/tts")
async def tts(
    text: str = Query("Hello! This is Arya speaking."),
    lang: str = Query("en", description="Language code: 'en' or 'hi'")
):
    """Text-to-speech endpoint using Google TTS."""
    try:
        if not text or len(text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Text required")
        
        if len(text) > 5000:
            raise HTTPException(status_code=400, detail="Text too long")
        
        # Map language codes
        lang_map = {
            'en': 'en',
            'hi': 'hi',
            'en-in': 'en',
            'en-us': 'en',
            'en-gb': 'en'
        }
        lang = lang_map.get(lang, 'en')
        
        print(f"🔊 Generating TTS: '{text[:50]}...' [{lang}]")
        audio_bytes = await generate_gtts(text.strip(), lang)
        
        if not audio_bytes:
            raise HTTPException(status_code=500, detail="TTS failed")
        
        print(f"✅ TTS: {len(audio_bytes)} bytes")
        
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "inline",
                "Cache-Control": "no-cache"
            }
        )
        
    except Exception as e:
        print(f"❌ TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 11) Original test page ---
TEST_HTML = """<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Arya Test</title>
    <style>
      body { font-family: system-ui; padding: 20px; max-width: 600px; margin: 0 auto; }
      input { width: 100%; padding: 10px; margin: 10px 0; }
      button { padding: 10px 20px; background: #6366f1; color: white; border: none; border-radius: 5px; cursor: pointer; }
      button:hover { background: #4f46e5; }
    </style>
  </head>
  <body>
    <h2>🎙️ Arya TTS Test</h2>
    <input id="text" type="text" placeholder="Type something..." value="Hello! I am Arya." />
    <select id="lang">
      <option value="en">English</option>
      <option value="hi">Hindi</option>
    </select>
    <button onclick="speak()">▶ Speak</button>
    <audio id="player" controls style="width: 100%; margin-top: 20px;"></audio>
    <script>
      async function speak() {
        const text = document.getElementById('text').value;
        const lang = document.getElementById('lang').value;
        const url = `/tts?text=${encodeURIComponent(text)}&lang=${lang}`;
        const res = await fetch(url);
        const blob = await res.blob();
        document.getElementById('player').src = URL.createObjectURL(blob);
        document.getElementById('player').play();
      }
    </script>
  </body>
</html>"""

@app.get("/test", response_class=HTMLResponse)
def test_page():
    return HTMLResponse(TEST_HTML)

# --- 12) Info endpoints ---
@app.get("/info")
async def info():
    """System information."""
    return {
        "agent": "Arya",
        "version": "2.0",
        "features": {
            "llm": "Groq (llama-3.1-70b)" if groq_client else "Disabled",
            "tts": "Google TTS (gTTS)",
            "languages": ["English", "Hindi"],
            "phone_calls": bool(TWILIO_ACCOUNT_SID),
            "conversation_memory": True
        },
        "endpoints": {
            "chat": "/chat",
            "tts": "/tts",
            "demo": "/demo",
            "test": "/test",
            "test_groq": "/test-groq"
        }
    }

@app.get("/test-groq")
async def test_groq():
    """Simple test to verify Groq API works"""
    try:
        if not groq_client:
            return {
                "success": False,
                "error": "Groq client not initialized",
                "key_set": bool(GROQ_API_KEY),
                "key_length": len(GROQ_API_KEY) if GROQ_API_KEY else 0
            }
        
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": "Say hello in one word"}],
            model="llama-3.3-70b-versatile",  # Updated model
            max_tokens=10
        )
        
        return {
            "success": True,
            "response": response.choices[0].message.content,
            "model": "llama-3.1-70b-versatile",
            "message": "✅ Groq API is working!"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
