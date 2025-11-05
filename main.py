import os
import io
import asyncio
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse

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

# --- 1) FastAPI app ---
app = FastAPI()

# --- 2) Agent prompt ---
AGENT_PROMPT = """
You are a friendly, casual AI assistant. Your name is 'Arya'.
- Start conversation casually in Hindi or English.
- Keep replies short and natural.
- Maintain context of the conversation.
- If asked about construction project, say: "The foundation work is complete, and we're on schedule to start framing next week."
"""

# --- 3) Environment variables ---
TWILIO_ACCOUNT_SID = os.environ["TWILIO_ACCOUNT_SID"]
TWILIO_AUTH_TOKEN = os.environ["TWILIO_AUTH_TOKEN"]
YOUR_TWILIO_PHONE_NUMBER = os.environ["YOUR_TWILIO_PHONE_NUMBER"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
ELEVEN_LABS_API_KEY = os.environ.get("ELEVEN_LABS_API_KEY", "")
RENDER_EXTERNAL_URL = os.environ["RENDER_EXTERNAL_URL"]

# --- 4) Configure LLM + TTS for Twilio PSTN ---
startup_error = None
try:
    agent_config = ChatGPTAgentConfig(
        initial_message=BaseMessage(text=" "),
        prompt_preamble=AGENT_PROMPT,
        model_name="llama3-70b-8192",
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

except Exception as e:
    startup_error = e
    print(f"Startup Error: {startup_error}")

# --- 5) Health check ---
@app.get("/")
def root():
    if startup_error:
        return {"error": f"Startup failed: {startup_error}"}
    return {"message": "AI Voice Agent is running! 🎙️"}

# --- 6) HTML test page ---
TEST_HTML = """<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Arya Voice Test</title>
    <style>
      body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
        padding: 24px;
        max-width: 780px;
        margin: 0 auto;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
      }
      .card {
        background: white;
        border-radius: 16px;
        padding: 32px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
      }
      h2 {
        margin-top: 0;
        color: #667eea;
      }
      input {
        width: 100%;
        padding: 12px;
        font-size: 16px;
        border: 2px solid #e0e0e0;
        border-radius: 8px;
        box-sizing: border-box;
        margin: 16px 0;
      }
      input:focus {
        outline: none;
        border-color: #667eea;
      }
      button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 32px;
        font-size: 16px;
        border-radius: 8px;
        cursor: pointer;
        font-weight: 600;
        transition: transform 0.2s;
      }
      button:hover {
        transform: translateY(-2px);
      }
      button:active {
        transform: translateY(0);
      }
      button:disabled {
        opacity: 0.6;
        cursor: not-allowed;
      }
      audio {
        width: 100%;
        margin-top: 16px;
      }
      .lang-selector {
        display: flex;
        gap: 8px;
        margin: 16px 0;
      }
      .lang-btn {
        padding: 8px 16px;
        background: white;
        border: 2px solid #667eea;
        color: #667eea;
        border-radius: 6px;
        cursor: pointer;
        font-size: 14px;
        transition: all 0.2s;
      }
      .lang-btn.active {
        background: #667eea;
        color: white;
      }
      .error {
        color: #dc3545;
        margin-top: 16px;
        padding: 12px;
        background: #ffe6e6;
        border-radius: 8px;
        white-space: pre-wrap;
      }
      .success {
        color: #28a745;
        margin-top: 8px;
      }
    </style>
  </head>
  <body>
    <div class="card">
      <h2>🎙️ Arya Voice Test</h2>
      <p>Test the AI voice assistant in your browser.</p>
      
      <div class="lang-selector">
        <button class="lang-btn active" onclick="setLang('en')">English</button>
        <button class="lang-btn" onclick="setLang('hi')">हिंदी</button>
      </div>
      
      <input id="text" type="text" placeholder="Type something for Arya to say..." 
             value="Hey! Main Arya bol rahi hoon. How can I help?" />
      
      <button id="speakBtn" onclick="speak()">▶ Speak</button>
      
      <audio id="player" controls style="display:none;"></audio>
      
      <div id="status"></div>
      <pre id="error" class="error" style="display:none;"></pre>
    </div>
    
    <script>
      let currentLang = 'en';
      
      function setLang(lang) {
        currentLang = lang;
        document.querySelectorAll('.lang-btn').forEach(btn => {
          btn.classList.remove('active');
        });
        event.target.classList.add('active');
        
        if (lang === 'hi') {
          document.getElementById('text').value = 'नमस्ते! मैं आर्या हूँ। मैं आपकी कैसे मदद कर सकती हूँ?';
        } else {
          document.getElementById('text').value = 'Hey! I am Arya. How can I help you?';
        }
      }
      
      async function speak() {
        const errorEl = document.getElementById('error');
        const statusEl = document.getElementById('status');
        const btn = document.getElementById('speakBtn');
        const player = document.getElementById('player');
        
        errorEl.style.display = 'none';
        statusEl.innerHTML = '';
        
        try {
          btn.disabled = true;
          btn.textContent = '⏳ Generating...';
          
          const text = document.getElementById('text').value || 'Hello from Arya!';
          const url = `/tts?text=${encodeURIComponent(text)}&lang=${currentLang}`;
          
          const res = await fetch(url);
          
          if (!res.ok) {
            const msg = await res.text();
            throw new Error(msg);
          }
          
          const blob = await res.blob();
          const audioUrl = URL.createObjectURL(blob);
          
          player.src = audioUrl;
          player.style.display = 'block';
          await player.play();
          
          statusEl.innerHTML = '<span class="success">✓ Audio generated successfully!</span>';
          
        } catch (e) {
          errorEl.textContent = 'Error: ' + (e.message || e);
          errorEl.style.display = 'block';
        } finally {
          btn.disabled = false;
          btn.textContent = '▶ Speak';
        }
      }
      
      // Allow Enter key to trigger speak
      document.getElementById('text').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') speak();
      });
    </script>
  </body>
</html>"""

@app.get("/test", response_class=HTMLResponse)
def test_page():
    if startup_error:
        return HTMLResponse(f"<pre>Startup failed: {startup_error}</pre>", status_code=500)
    return HTMLResponse(TEST_HTML)

# --- 7) TTS using gTTS (Google Text-to-Speech) ---
async def generate_gtts(text: str, lang: str = 'en') -> bytes:
    """
    Generate speech using Google Text-to-Speech.
    This runs in a thread pool to avoid blocking.
    """
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
    lang: str = Query("en", description="Language code: 'en' for English, 'hi' for Hindi")
):
    """
    Text-to-speech endpoint using Google TTS.
    
    Supported languages:
    - en: English
    - hi: Hindi
    - en-in: English (Indian accent)
    """
    try:
        # Validate input
        if not text or len(text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Text parameter is required")
        
        if len(text) > 5000:
            raise HTTPException(status_code=400, detail="Text too long (max 5000 characters)")
        
        # Validate language
        supported_langs = ['en', 'hi', 'en-in', 'en-us', 'en-gb']
        if lang not in supported_langs:
            lang = 'en'  # Default to English
        
        # Generate audio
        print(f"Generating TTS for: '{text[:50]}...' in language: {lang}")
        audio_bytes = await generate_gtts(text.strip(), lang)
        
        if not audio_bytes or len(audio_bytes) == 0:
            raise HTTPException(status_code=500, detail="Generated audio is empty")
        
        print(f"✓ TTS generated successfully: {len(audio_bytes)} bytes")
        
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "inline",
                "Cache-Control": "no-cache",
                "Content-Length": str(len(audio_bytes))
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"TTS error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"TTS generation failed: {str(e)}"
        )

@app.get("/tts-info")
async def tts_info():
    """
    Information about TTS capabilities.
    """
    return {
        "provider": "Google Text-to-Speech (gTTS)",
        "supported_languages": {
            "en": "English (US)",
            "en-in": "English (Indian)",
            "hi": "Hindi",
            "en-gb": "English (British)",
            "en-us": "English (American)"
        },
        "features": [
            "No API key required",
            "Reliable on cloud platforms",
            "Natural sounding voice",
            "Free to use"
        ],
        "usage": "/tts?text=Hello&lang=en"
    }
