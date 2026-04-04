"""
Voice-to-text service using OpenAI Whisper API.
Receives voice messages from Telegram, transcribes them.
"""
import io
import logging
from openai import AsyncOpenAI
from config import config

logger = logging.getLogger(__name__)

client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)


async def transcribe_voice(voice_bytes: bytes, filename: str = "voice.ogg") -> str:
    """
    Transcribe voice message bytes to text using Whisper.
    """
    try:
        audio_file = io.BytesIO(voice_bytes)
        audio_file.name = filename

        response = await client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="ru",  # primary language — Russian
            response_format="text",
        )
        text = response.strip()
        logger.info(f"Transcribed voice: {text[:80]}...")
        return text
    except Exception as e:
        logger.error(f"Whisper transcription failed: {e}")
        raise
