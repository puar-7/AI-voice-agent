import os
import io
import asyncio
import subprocess
import tempfile
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

import edge_tts

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

# --- 4) Configure LLM + TTS for Twilio ---
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

    ELEVEN_LABS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"
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

# --- 6) HTML test page ---
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

# --- 7) TTS with fallback strategy ---
EDGE_TTS_VOICE = os.environ.get("EDGE_TTS_VOICE", "en-US-JennyNeural")
USE_CLI_METHOD = os.environ.get("USE_CLI_METHOD", "true").lower() == "true"

async def synth_edge_tts_cli(text: str) -> bytes:
    """
    Use edge-tts CLI tool via subprocess.
    This sometimes works better on restricted networks.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        output_file = tmp.name
    
    try:
        # Run edge-tts as a subprocess
        process = await asyncio.create_subprocess_exec(
            "edge-tts",
            "--text", text,
            "--voice", EDGE_TTS_VOICE,
            "--write-media", output_file,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30.0)
        
        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            raise RuntimeError(f"edge-tts CLI failed: {error_msg}")
        
        # Read the generated file
        with open(output_file, "rb") as f:
            audio_data = f.read()
        
        if len(audio_data) == 0:
            raise RuntimeError("Generated audio file is empty")
        
        return audio_data
        
    finally:
        # Clean up temp file
        try:
            if os.path.exists(output_file):
                os.unlink(output_file)
        except:
            pass

async def synth_edge_tts_library(text: str) -> bytes:
    """
    Use edge-tts Python library (original method).
    """
    communicate = edge_tts.Communicate(
        text=text.strip(),
        voice=EDGE_TTS_VOICE,
        rate="+0%",
        pitch="+0Hz"
    )
    
    buf = io.BytesIO()
    audio_received = False
    
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
            audio_received = True
    
    if not audio_received:
        raise RuntimeError(f"No audio received from edge-tts")
    
    buf.seek(0)
    return buf.read()

@app.get("/tts")
async def tts(text: str = Query("Hello! This is Arya speaking.")):
    """
    Text-to-speech with multiple fallback methods.
    """
    try:
        if not text or len(text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Text parameter is required")
        
        if len(text) > 5000:
            raise HTTPException(status_code=400, detail="Text too long (max 5000 characters)")
        
        audio_bytes = None
        errors = []
        
        # Try CLI method first if enabled
        if USE_CLI_METHOD:
            try:
                print("Trying edge-tts CLI method...")
                audio_bytes = await synth_edge_tts_cli(text)
                print("CLI method succeeded!")
            except Exception as e:
                errors.append(f"CLI method failed: {e}")
                print(errors[-1])
        
        # Fallback to library method
        if audio_bytes is None:
            try:
                print("Trying edge-tts library method...")
                audio_bytes = await synth_edge_tts_library(text)
                print("Library method succeeded!")
            except Exception as e:
                errors.append(f"Library method failed: {e}")
                print(errors[-1])
        
        if audio_bytes is None or len(audio_bytes) == 0:
            raise HTTPException(
                status_code=500,
                detail=f"All TTS methods failed. Errors: {'; '.join(errors)}"
            )
        
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "inline",
                "Cache-Control": "no-cache"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"TTS endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"TTS error: {str(e)}"
        )

@app.get("/tts-test")
async def tts_test():
    """
    Diagnostics endpoint to test TTS configuration.
    """
    results = {
        "voice": EDGE_TTS_VOICE,
        "use_cli_method": USE_CLI_METHOD,
        "tests": {}
    }
    
    test_text = "Hello, this is a test."
    
    # Test CLI method
    if USE_CLI_METHOD:
        try:
            await synth_edge_tts_cli(test_text)
            results["tests"]["cli_method"] = "✓ Success"
        except Exception as e:
            results["tests"]["cli_method"] = f"✗ Failed: {str(e)}"
    
    # Test library method
    try:
        await synth_edge_tts_library(test_text)
        results["tests"]["library_method"] = "✓ Success"
    except Exception as e:
        results["tests"]["library_method"] = f"✗ Failed: {str(e)}"
    
    return results

@app.get("/voices")
async def list_voices():
    """List available voices."""
    try:
        voices = await edge_tts.list_voices()
        return {
            "current_voice": EDGE_TTS_VOICE,
            "total_voices": len(voices),
            "recommended_voices": {
                "english_female": ["en-US-JennyNeural", "en-US-AriaNeural"],
                "english_male": ["en-US-GuyNeural", "en-US-ChristopherNeural"],
                "hindi_female": ["hi-IN-SwaraNeural"],
                "hindi_male": ["hi-IN-MadhurNeural"]
            }
        }
    except Exception as e:
        return {
            "error": f"Failed to list voices: {e}",
            "current_voice": EDGE_TTS_VOICE
        }
