"""
Scheduler — deadline reminders, daily digests, task digests every 3 hours.
Uses APScheduler with asyncio.
"""
import logging
from datetime import datetime
import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from config import config
from services.notion_client import notion
from services.ai_parser import analyze_progress

logger = logging.getLogger(__name__)


class ReminderScheduler:
    """Manages all scheduled jobs: reminders, digests, check-ins."""

    def __init__(self, bot):
        self.bot = bot
        self.tz = pytz.timezone(config.TIMEZONE)
        self.scheduler = AsyncIOScheduler(timezone=self.tz)
        self.user_ids = config.ALLOWED_USER_IDS

    def start(self):
        """Register all jobs and start the scheduler."""

        # 1. Morning digest at 4:30 (Uruguay time)
        self.scheduler.add_job(
            self._daily_digest,
            CronTrigger(hour=4, minute=30),
            id="daily_digest",
            replace_existing=True,
        )

        # 2. Task digest every 3 hours from 7:30 to 19:30 (Uruguay)
        # Hours: 7:30, 10:30, 13:30, 16:30, 19:30
        self.scheduler.add_job(
            self._task_digest,
            CronTrigger(hour="7,10,13,16,19", minute=30),
            id="task_digest_3h",
            replace_existing=True,
        )

        # 3. Evening summary at 21:00
        self.scheduler.add_job(
            self._evening_checkin,
            CronTrigger(hour=21, minute=0),
            id="evening_checkin",
            replace_existing=True,
        )

        self.scheduler.start()
        logger.info(
            f"Scheduler started (TZ: {config.TIMEZONE}). "
            f"Jobs: morning 4:30, digest 7:30/10:30/13:30/16:30/19:30, evening 21:00"
        )

    async def _send_to_all(self, text: str):
        """Send message to all allowed users."""
        for user_id in self.user_ids:
            try:
                await self.bot.send_message(
                    user_id, text, parse_mode="HTML", disable_web_page_preview=True
                )
            except Exception as e:
                logger.error(f"Failed to send to {user_id}: {e}")

    async def _task_digest(self):
        """Send task digest every 3 hours with active tasks grouped by project."""
        try:
            summary = await notion.get_active_tasks_summary()
            now = datetime.now(self.tz).strftime("%H:%M")
            msg = f"<b>Задачи — {now}</b>\n\n{summary}"
            await self._send_to_all(msg)
        except Exception as e:
            logger.error(f"Task digest failed: {e}")

    async def _daily_digest(self):
        """Send morning daily digest."""
        try:
            all_tasks = await notion.get_all_tasks()
            metrics = await notion.get_recent_metrics()

            total = len(all_tasks)
            done = len([t for t in all_tasks if t["status"] == "Готово"])
            in_progress = len([t for t in all_tasks if t["status"] == "В работе"])
            blocked = len([t for t in all_tasks if t["status"] == "Заблокирована"])
            new_count = len([t for t in all_tasks if t["status"] == "Новая"])

            overdue = await notion.get_overdue_tasks()
            upcoming = await notion.get_upcoming_deadlines(days=3)

            msg = (
                f"<b>Доброе утро — {datetime.now(self.tz).strftime('%d.%m.%Y')}</b>\n\n"
                f"<b>Задачи:</b> {total} всего\n"
                f"  Новых: {new_count}\n"
                f"  В работе: {in_progress}\n"
                f"  Готово: {done}\n"
                f"  Заблокировано: {blocked}\n"
            )

            if overdue:
                msg += f"\n<b>Просрочено: {len(overdue)}</b>\n"
                for t in overdue[:5]:
                    msg += f"  {t['title']}  ({t['project']})\n"

            if upcoming:
                msg += f"\n<b>Ближайшие дедлайны ({len(upcoming)}):</b>\n"
                for t in upcoming[:5]:
                    msg += f"  {t['title']} — {t['deadline']}\n"

            msg += "\nПродуктивного дня!"

            await self._send_to_all(msg)
        except Exception as e:
            logger.error(f"Daily digest failed: {e}")

    async def _evening_checkin(self):
        """Evening check-in: summary + ask for updates."""
        try:
            summary = await notion.get_active_tasks_summary()
            msg = (
                "<b>Вечерний итог</b>\n\n"
                f"{summary}\n\n"
                "Закрыл задачу? — /done\n"
                "Есть обновления? — скажи голосовым."
            )
            await self._send_to_all(msg)
        except Exception as e:
            logger.error(f"Evening checkin failed: {e}")

    def stop(self):
        """Shutdown scheduler gracefully."""
        self.scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped")
