"""
Configuration — all secrets via environment variables.
"""
import os
from dataclasses import dataclass


@dataclass
class Config:
    # Telegram
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    ALLOWED_USER_IDS: list[int] = None  # restrict to your Telegram ID(s)

    # OpenAI (Whisper + GPT for task parsing)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")

    # Notion
    NOTION_TOKEN: str = os.getenv("NOTION_TOKEN", "")
    NOTION_DATABASE_ID: str = os.getenv("NOTION_DATABASE_ID", "")  # tasks DB
    NOTION_METRICS_DB_ID: str = os.getenv("NOTION_METRICS_DB_ID", "")  # metrics DB

    # Reminders
    REMINDER_CHECK_INTERVAL_MINUTES: int = int(os.getenv("REMINDER_CHECK_INTERVAL", "30"))
    DAILY_DIGEST_HOUR: int = int(os.getenv("DAILY_DIGEST_HOUR", "9"))  # 9 AM
    TIMEZONE: str = os.getenv("TIMEZONE", "Europe/Moscow")

    def __post_init__(self):
        raw_ids = os.getenv("ALLOWED_USER_IDS", "")
        if raw_ids:
            self.ALLOWED_USER_IDS = [int(x.strip()) for x in raw_ids.split(",")]
        else:
            self.ALLOWED_USER_IDS = []


config = Config()
