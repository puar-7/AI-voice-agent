import os
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from vocode.streaming.models.agent import ChatGPTAgentConfig
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.telephony import TwilioConfig
from vocode.streaming.telephony.server.base import TelephonyServer, TwilioInboundCallConfig
from vocode.streaming.models.synthesizer import ElevenLabsSynthesizerConfig, AudioEncoding
import requests

# --- 1. FastAPI app ---
app = FastAPI()

# --- 2. Agent prompt (unchanged) ---
AGENT_PROMPT = """
You are a friendly, casual AI assistant. Your name is 'Arya'.
- Start conversation casually in Hindi or English.
- Keep replies short and natural.
- Maintain context of the conversation.
- If asked about construction project, say: "The foundation work is complete, and we're on schedule to start framing next week."
"""

# --- 3. Env vars (same as before) ---
TWILIO_ACCOUNT_SID = os.environ["TWILIO_ACCOUNT_SID"]
TWILIO_AUTH_TOKEN = os.environ["TWILIO_AUTH_TOKEN"]
YOUR_TWILIO_PHONE_NUMBER = os.environ["YOUR_TWILIO_PHONE_NUMBER"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
ELEVEN_LABS_API_KEY = os.environ["ELEVEN_LABS_API_KEY"]
RENDER_EXTERNAL_URL = os.environ["RENDER_EXTERNAL_URL"]

# --- 4. Configure LLM + TTS + Telephony (same as before) ---
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

    # For Twilio calls (8k mulaw)
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

# --- 5. Health check ---
@app.get("/")
def root():
    if startup_error:
        return {"error": f"Startup failed: {startup_error}"}
    return {"message": "AI Voice Agent is running!"}

# --- 6. Simple browser test page to HEAR the voice now ---
TEST_HTML = """<!doctype html>
<html>
  <head><meta charset="utf-8"><title>Voice Test</title></head>
  <body style="font-family: system-ui; padding: 24px">
    <h2>AI Voice Test (ElevenLabs)</h2>
    <p>Type something and click Speak:</p>
    <input id="t" size="70" value="Hey! Main Arya bol rahi hoon. How can I help?"/>
    <button onclick="speak()">Speak</button>
    <p><audio id="player" controls></audio></p>
    <script>
      async function speak() {
        const text = encodeURIComponent(document.getElementById('t').value);
        const res = await fetch('/tts?text=' + text);
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.getElementById('player');
        a.src = url;
        a.play();
      }
    </script>
  </body>
</html>"""

@app.get("/test", response_class=HTMLResponse)
def test_page():
    if startup_error:
        return HTMLResponse(f"<pre>Startup failed: {startup_error}</pre>", status_code=500)
    return HTMLResponse(TEST_HTML)

# --- 7. TTS endpoint: returns MP3 synthesized by ElevenLabs ---
@app.get("/tts")
def tts(text: str = Query("Hello! This is Arya speaking.")):
    try:
        voice_id = "21m00Tcm4TlvDq8ikWAM"
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {
            "xi-api-key": ELEVEN_LABS_API_KEY,
            "accept": "audio/mpeg",
            "content-type": "application/json",
        }
        payload = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.7},
            # browser-friendly MP3; doesn’t affect Twilio path
            "output_format": "mp3_22050_32"
        }
        r = requests.post(url, headers=headers, json=payload, stream=True, timeout=60)
        r.raise_for_status()
        return StreamingResponse(r.iter_content(chunk_size=4096), media_type="audio/mpeg")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
