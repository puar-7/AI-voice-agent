import os
from fastapi import FastAPI
from vocode.streaming.models.agent import ChatGPTAgentConfig
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.telephony import TwilioConfig
from vocode.streaming.telephony.server.base import TelephonyServer, TwilioInboundCallConfig
from vocode.streaming.models.synthesizer import ElevenLabsSynthesizerConfig, AudioEncoding
from groq import Groq

# --- 1. Initialize the FastAPI Server ---
app = FastAPI()

# --- 2. Define Your Agent's Prompt (The "Brain") ---
AGENT_PROMPT = """
You are a friendly, casual AI assistant. Your name is 'Arya'.
- **Rule 1:** Start the conversation with a casual greeting in Hindi or English. Examples: "Hey, what's up?" or "Namaste, kaise ho?". Pick one and only one.
- **Rule 2:** Keep your replies short and conversational. Use a warm, human-like, locally relatable tone.
- **Rule 3:** You MUST remember the conversation.
- **Rule 4 (Optional):** If the user asks about 'the project' or 'the construction', give them this update: "The foundation work is complete, and we're on schedule to start framing next week."
"""

# --- 3. Load All Your API Keys (Secrets) ---
TWILIO_ACCOUNT_SID = os.environ["TWILIO_ACCOUNT_SID"]
TWILIO_AUTH_TOKEN = os.environ["TWILIO_AUTH_TOKEN"]
YOUR_TWILIO_PHONE_NUMBER = os.environ["YOUR_TWILIO_PHONE_NUMBER"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
ELEVEN_LABS_API_KEY = os.environ["ELEVEN_LABS_API_KEY"]
RENDER_EXTERNAL_URL = os.environ["RENDER_EXTERNAL_URL"]

# --- 4. Configure Your Voice Agent ---
startup_error = None

try:
    # Configure LLM (High-Performance with Groq)
    agent_config = ChatGPTAgentConfig(
        initial_message=BaseMessage(text=" "),
        prompt_preamble=AGENT_PROMPT,
        model_name="llama3-70b-8192",
        allow_agent_to_be_interrupted=True,
        openai_api_key=GROQ_API_KEY,
        openai_api_base="https://api.groq.com/openai/v1",
    )

    # Configure Voice (High-Realism)
    ELEVEN_LABS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # 'Rachel' voice
    
    synthesizer_config = ElevenLabsSynthesizerConfig(
        api_key=ELEVEN_LABS_API_KEY,
        voice_id=ELEVEN_LABS_VOICE_ID,
        sampling_rate=8000,
        audio_encoding=AudioEncoding.MULAW
    )

    # Configure Twilio
    twilio_config = TwilioConfig(
        account_sid=TWILIO_ACCOUNT_SID,
        auth_token=TWILIO_AUTH_TOKEN,
    )

    # --- 5. Create the Telephony Server with proper configuration ---
    telephony_server = TelephonyServer(
        base_url=RENDER_EXTERNAL_URL,
        config_manager=None,  # Optional: for dynamic configuration
        inbound_call_configs=[
            TwilioInboundCallConfig(
                url="/inbound_call",
                agent_config=agent_config,
                synthesizer_config=synthesizer_config,
                twilio_config=twilio_config,
            )
        ],
        logger=None,
    )

    # --- 6. Add the Server's Routes to FastAPI ---
    app.include_router(telephony_server.get_router())

except Exception as e:
    startup_error = e
    print(f"Error setting up configuration: {startup_error}")

# --- 7. Health check endpoint ---
@app.get("/")
def root():
    if startup_error:
        return {"error": f"Failed to initialize server: {startup_error}"}
    return {"message": "AI Voice Agent is running!"}
