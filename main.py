import os
from fastapi import FastAPI
from vocode.streaming.models.agent import AgentConfig
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.telephony import TwilioConfig
from vocode.streaming.telephony.server.base import TelephonyServer
from vocode.streaming.models.synthesizer import ElevenLabsSynthesizerConfig
from vocode.streaming.models.audio import AudioEncoding
from groq import Groq

# --- 1. NEW IMPORT ---
# We need this new config object for the TelephonyServer
from vocode.streaming.telephony.config import TelephonyServerConfig

# --- 2. Initialize the FastAPI Server ---
app = FastAPI()

# --- 3. Define Your Agent's Prompt (The "Brain") ---
AGENT_PROMPT = """
You are a friendly, casual AI assistant. Your name is 'Arya'.
- **Rule 1:** Start the conversation with a casual greeting in Hindi or English. Examples: "Hey, what's up?" or "Namaste, kaise ho?". Pick one and only one.
- **Rule 2:** Keep your replies short and conversational. Use a warm, human-like, locally relatable tone.
- **Rule 3:** You MUST remember the conversation.
- **Rule 4 (Optional):** If the user asks about 'the project' or 'the construction', give them this update: "The foundation work is complete, and we're on schedule to start framing next week."
"""

# --- 4. Load All Your API Keys (Secrets) ---
TWILIO_ACCOUNT_SID = os.environ["TWILIO_ACCOUNT_SID"]
TWILIO_AUTH_TOKEN = os.environ["TWILIO_AUTH_TOKEN"]
YOUR_TWILIO_PHONE_NUMBER = os.environ["YOUR_TWILIO_PHONE_NUMBER"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
ELEVEN_LABS_API_KEY = os.environ["ELEVEN_LABS_API_KEY"]
RENDER_EXTERNAL_URL = os.environ["RENDER_EXTERNAL_URL"]
PORT = int(os.environ.get("PORT", 8000))

# --- 5. Configure Your Voice Agent ---
startup_error = None

try:
    # Configure LLM (High-Performance with Groq)
    agent_config = AgentConfig(
        initial_message=BaseMessage(text=" "),
        prompt_preamble=AGENT_PROMPT,
        model_name="llama3-70b-8192",
        allow_agent_to_be_interrupted=True,
        openai_api_key=GROQ_API_KEY,
        openai_base_url="https://api.groq.com/openai/v1/",
    )

    # Configure Voice (High-Realism)
    ELEVEN_LABS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM" # 'Rachel' voice
    
    synthesizer_config = ElevenLabsSynthesizerConfig(
        api_key=ELEVEN_LABS_API_KEY,
        voice_id=ELEVEN_LABS_VOICE_ID,
        sampling_rate=8000,
        audio_encoding=AudioEncoding.MULAW
    )

    # Configure Phone Number (Twilio)
    twilio_config = TwilioConfig(
        account_sid=TWILIO_ACCOUNT_SID,
        auth_token=TWILIO_AUTH_TOKEN,
        phone_number=YOUR_TWILIO_PHONE_NUMBER
    )

    # --- 6. THIS BLOCK IS CHANGED (THE FIX) ---
    # Bundle all settings into the TelephonyServerConfig
    server_config = TelephonyServerConfig(
        base_url=RENDER_EXTERNAL_URL,
        agent_config=agent_config,
        synthesizer_config=synthesizer_config,
        twilio_config=twilio_config,
        port=PORT
    )

    # Create the server by passing the single config object
    telephony_server = TelephonyServer(config=server_config)
    # --- END OF CHANGED BLOCK ---

    # --- 7. Add the Server's Routes to FastAPI ---
    app.include_router(telephony_server.get_router())

except Exception as e:
    startup_error = e
    print(f"Error setting up configuration: {startup_error}")

# --- 8. Health check endpoint ---
@app.get("/")
def root():
    if startup_error:
        return {"error": f"Failed to initialize server: {startup_error}"}
    return {"message": "AI Voice Agent is running!"}
