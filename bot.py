"""
Main entry point — starts the Telegram bot with all handlers and scheduler.
"""
import asyncio
import logging
import sys

from aiogram import Bot, Dispatcher
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties

from config import config
from handlers.messages import router as messages_router
from services.scheduler import ReminderScheduler

# ── Logging ───────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ── Bot setup ─────────────────────────────────────────────────────

async def main():
    # Validate config
    if not config.TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN is not set!")
        sys.exit(1)
    if not config.OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY is not set!")
        sys.exit(1)
    if not config.NOTION_TOKEN:
        logger.error("NOTION_TOKEN is not set!")
        sys.exit(1)

    # Create bot and dispatcher
    bot = Bot(
        token=config.TELEGRAM_BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher()

    # Register handlers
    dp.include_router(messages_router)

    # Start scheduler
    scheduler = ReminderScheduler(bot)
    scheduler.start()

    logger.info("Bot is starting...")
    logger.info(f"Allowed users: {config.ALLOWED_USER_IDS or 'ALL'}")
    logger.info(f"Timezone: {config.TIMEZONE}")
    logger.info(f"Daily digest at: {config.DAILY_DIGEST_HOUR}:00")

    # Notify admin on startup
    for uid in config.ALLOWED_USER_IDS:
        try:
            await bot.send_message(
                uid,
                "🟢 <b>Бот запущен!</b>\n\n"
                "Отправь /start чтобы увидеть команды.\n"
                "Или просто отправь голосовое с задачей 🎙",
            )
        except Exception:
            pass

    try:
        await dp.start_polling(bot)
    finally:
        scheduler.stop()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
