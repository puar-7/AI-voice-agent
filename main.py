import os
from fastapi import FastAPI
from vocode.streaming.models.agent import AgentConfig
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.telephony import TwilioConfig
from vocode.streaming.telephony.server import TelephonyServer
from vocode.streaming.models.synthesizer import ElevenLabsSynthesizerConfig

# --- 1. Initialize the FastAPI Server ---
# This is the high-performance web framework
app = FastAPI()

# --- 2. Define Your Agent's Prompt (The "Brain") ---
# This is where you score on Context (20%) and Creativity (20%)
AGENT_PROMPT = """
You are a friendly, casual AI assistant. Your name is 'Arya'.
- **Rule 1:** Start the conversation with a casual greeting in Hindi or English. Examples: "Hey, what's up?" or "Namaste, kaise ho?". Pick one and only one.
- **Rule 2:** Keep your replies short and conversational. Use a warm, human-like, locally relatable tone.
- **Rule 3:** You MUST remember the conversation.
- **Rule 4 (Optional):** If the user asks about 'the project' or 'the construction', give them this update: "The foundation work is complete, and we're on schedule to start framing next week."
"""

# --- 3. Load All Your API Keys (Secrets) ---
# Render will provide these from the 'Environment' tab
TWILIO_ACCOUNT_SID = os.environ["TWILIO_ACCOUNT_SID"]
TWILIO_AUTH_TOKEN = os.environ["TWILIO_AUTH_TOKEN"]
YOUR_TWILIO_PHONE_NUMBER = os.environ["YOUR_TWILIO_PHONE_NUMBER"] # Your Twilio number
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
ELEVEN_LABS_API_KEY = os.environ["ELEVEN_LABS_API_KEY"]
RENDER_EXTERNAL_URL = os.environ["RENDER_EXTERNAL_URL"] # Render sets this automatically!
PORT = int(os.environ.get("PORT", 8000)) # Render sets this automatically

# --- 4. Configure Your Voice Agent ---
try:
    # Configure LLM (High-Performance)
    agent_config = AgentConfig(
        initial_message=BaseMessage(text=" "), # The prompt will trigger the *real* first message
        prompt_preamble=AGENT_PROMPT,
        model_name="gpt-4o-mini",
        allow_agent_to_be_interrupted=True, # Critical for realism!
        openai_api_key=OPENAI_API_KEY
    )

    # Configure Voice (High-Realism)
    # NOTE: You MUST change this Voice ID to one from your ElevenLabs account.
    # This ID (21m00Tcm4TlvDq8ikWAM) is for the 'Rachel' voice.
    ELEVEN_LABS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM" 
    
    synthesizer_config = ElevenLabsSynthesizerConfig.from_telephone_input_device(
        api_key=ELEVEN_LABS_API_KEY,
        voice_id=ELEVEN_LABS_VOICE_ID,
    )

    # Configure Phone Number (Twilio)
    twilio_config = TwilioConfig(
        account_sid=TWILIO_ACCOUNT_SID,
        auth_token=TWILIO_AUTH_TOKEN,
        phone_number=YOUR_TWILIO_PHONE_NUMBER
    )

    # --- 5. Create the Server ---
    telephony_server = TelephonyServer(
        base_url=RENDER_EXTERNAL_URL,
        agent_config=agent_config,
        synthesizer_config=synthesizer_config,
        twilio_config=twilio_config,
        port=PORT
    )

    # --- 6. Add the Server's Routes to FastAPI ---
    # This connects Vocode to your FastAPI app
    app.include_router(telephony_server.get_router())

    # Health check endpoint
    @app.get("/")
    def root():
        return {"message": "AI Voice Agent is running!"}

except Exception as e:
    print(f"Error setting up configuration: {e}")
    # Add a fallback route to report error on startup
    @app.get("/")
    def error():
        return {"error": f"Failed to initialize server: {e}"}
