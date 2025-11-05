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

import edge_tts  # no API key needed
# from groq import Groq  # optional; not used directly

# --- 1) FastAPI app ---
app = FastAPI()

# --- 2) Agent prompt (as in your original) ---
AGENT_PROMPT = """
You are a friendly, casual AI assistant. Your name is 'Arya'.
- Start conversation casually in Hindi or English.
- Keep replies short and natural.
- Maintain context of the conversation.
- If asked about construction project, say: "The foundation work is complete, and we're on schedule to start framing next week."
"""

# --- 3) Environment variables (same names you already use) ---
TWILIO_ACCOUNT_SID = os.environ["TWILIO_ACCOUNT_SID"]
TWILIO_AUTH_TOKEN = os.environ["TWILIO_AUTH_TOKEN"]
YOUR_TWILIO_PHONE_NUMBER = os.environ["YOUR_TWILIO_PHONE_NUMBER"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
ELEVEN_LABS_API_KEY = os.environ.get("ELEVEN_LABS_API_KEY", "")  # optional for PSTN path
RENDER_EXTERNAL_URL = os.environ["RENDER_EXTERNAL_URL"]  # e.g., https://<service>.onrender.com

# --- 4) Configure LLM + TTS (for Twilio PSTN) and mount Telephony routes ---
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

    # Voice used for Twilio calls (8k MULAW is correct for PSTN)
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
    return {"message": "AI Voice Agent is running!"}

# --- 6) Minimal HTML page to hear the voice in Chrome (no phone needed) ---
TEST_HTML = """<!doctype html>
<html>
  <head><meta charset="utf-8"><title>Voice Test</title></head>
  <body style="font-family: system-ui; padding: 24px; max-width: 780px;">
    <h2>AI Voice Test (Browser TTS)</h2>
    <p>Type text and click <b>Speak</b> to hear the voice.</p>
    <input id="t" size="70" value="Hey! Main Arya bol rahi hoon. How can I help?" />
    <button onclick="speak()">Speak</button>
    <p style="margin-top:16px"><audio id="player" controls></audio></p>
    <pre id="err" style="color:#b00; white-space:pre-wrap;"></pre>
    <script>
      async function speak() {
        document.getElementById('err').textContent = '';
        try {
          const text = encodeURIComponent(document.getElementById('t').value || 'Hello from Arya!');
          const res = await fetch('/tts?text=' + text);
          if (!res.ok) {
            const msg = await res.text();
            document.getElementById('err').textContent = 'Error: ' + msg;
            return;
          }
          const blob = await res.blob();
          const url = URL.createObjectURL(blob);
          const a = document.getElementById('player');
          a.src = url;
          await a.play();
        } catch (e) {
          document.getElementById('err').textContent = 'Error: ' + (e && e.message ? e.message : e);
        }
      }
    </script>
  </body>
</html>"""

@app.get("/test", response_class=HTMLResponse)
def test_page():
    if startup_error:
        return HTMLResponse(f"<pre>Startup failed: {startup_error}</pre>", status_code=500)
    return HTMLResponse(TEST_HTML)

# --- 7) Browser TTS using edge-tts (no API key) ---
# Choose a voice. Good options:
#   "en-US-AriaNeural" (English, natural)
#   "hi-IN-SwaraNeural" (Hindi)
EDGE_TTS_VOICE = os.environ.get("EDGE_TTS_VOICE", "en-US-AriaNeural")

async def synth_edge_tts(text: str) -> bytes:
    communicate = edge_tts.Communicate(text, voice=EDGE_TTS_VOICE)
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    buf.seek(0)
    return buf.read()

@app.get("/tts")
async def tts(text: str = Query("Hello! This is Arya speaking.")):
    try:
        audio_bytes = await synth_edge_tts(text)
        return StreamingResponse(iter([audio_bytes]), media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS error: {e}")
