"""
Scheduler — deadline reminders, daily digests, progress check-ins.
Uses APScheduler with asyncio.
"""
import logging
from datetime import datetime
import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
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

        # 1. Check overdue tasks every N minutes
        self.scheduler.add_job(
            self._check_overdue,
            IntervalTrigger(minutes=config.REMINDER_CHECK_INTERVAL_MINUTES),
            id="check_overdue",
            replace_existing=True,
        )

        # 2. Daily morning digest
        self.scheduler.add_job(
            self._daily_digest,
            CronTrigger(hour=config.DAILY_DIGEST_HOUR, minute=0),
            id="daily_digest",
            replace_existing=True,
        )

        # 3. Evening progress check-in (ask for metrics update)
        self.scheduler.add_job(
            self._evening_checkin,
            CronTrigger(hour=19, minute=0),
            id="evening_checkin",
            replace_existing=True,
        )

        # 4. Upcoming deadlines alert (twice a day)
        self.scheduler.add_job(
            self._deadline_alert,
            CronTrigger(hour="9,15", minute=30),
            id="deadline_alert",
            replace_existing=True,
        )

        self.scheduler.start()
        logger.info("Scheduler started with all jobs")

    async def _send_to_all(self, text: str):
        """Send message to all allowed users."""
        for user_id in self.user_ids:
            try:
                await self.bot.send_message(
                    user_id, text, parse_mode="HTML", disable_web_page_preview=True
                )
            except Exception as e:
                logger.error(f"Failed to send to {user_id}: {e}")

    async def _check_overdue(self):
        """Check for overdue tasks and notify."""
        try:
            overdue = await notion.get_overdue_tasks()
            if not overdue:
                return

            lines = ["🔴 <b>Просроченные задачи:</b>\n"]
            for t in overdue:
                priority_emoji = {"high": "🔥", "medium": "⚡", "low": "📌"}.get(
                    t["priority"], "📌"
                )
                deadline = t["deadline"] or "без срока"
                lines.append(
                    f"{priority_emoji} <b>{t['title']}</b>\n"
                    f"   📁 {t['project']} | ⏰ {deadline}\n"
                    f"   <a href='{t['url']}'>Открыть в Notion</a>"
                )

            await self._send_to_all("\n".join(lines))
        except Exception as e:
            logger.error(f"Overdue check failed: {e}")

    async def _daily_digest(self):
        """Send morning daily digest."""
        try:
            all_tasks = await notion.get_all_tasks()
            metrics = await notion.get_recent_metrics()

            # Stats
            total = len(all_tasks)
            done = len([t for t in all_tasks if t["status"] == "Готово"])
            in_progress = len([t for t in all_tasks if t["status"] == "В работе"])
            blocked = len([t for t in all_tasks if t["status"] == "Заблокирована"])
            new = len([t for t in all_tasks if t["status"] == "Новая"])

            # AI analysis
            analysis = await analyze_progress(all_tasks, metrics)

            msg = (
                f"☀️ <b>Доброе утро! Дайджест на {datetime.now(self.tz).strftime('%d.%m.%Y')}</b>\n\n"
                f"📊 <b>Задачи:</b> {total} всего\n"
                f"   ✅ Готово: {done}\n"
                f"   🔄 В работе: {in_progress}\n"
                f"   🆕 Новых: {new}\n"
                f"   🚫 Заблокировано: {blocked}\n\n"
                f"🤖 <b>Анализ AI:</b>\n{analysis}"
            )

            await self._send_to_all(msg)
        except Exception as e:
            logger.error(f"Daily digest failed: {e}")

    async def _evening_checkin(self):
        """Evening check-in: ask user for progress update."""
        msg = (
            "🌙 <b>Вечерний чек-ин</b>\n\n"
            "Как прошёл день? Запиши голосовым:\n"
            "• Какие задачи продвинулись?\n"
            "• Есть обновления по цифрам/метрикам?\n"
            "• Что заблокировано?\n\n"
            "Я разберу и обновлю доску 📋"
        )
        await self._send_to_all(msg)

    async def _deadline_alert(self):
        """Alert about upcoming deadlines (next 3 days)."""
        try:
            upcoming = await notion.get_upcoming_deadlines(days=3)
            if not upcoming:
                return

            lines = ["⏳ <b>Дедлайны в ближайшие 3 дня:</b>\n"]
            for t in upcoming:
                priority_emoji = {"high": "🔥", "medium": "⚡", "low": "📌"}.get(
                    t["priority"], "📌"
                )
                lines.append(
                    f"{priority_emoji} <b>{t['title']}</b> — {t['deadline']}\n"
                    f"   📁 {t['project']} | {t['status']}\n"
                    f"   <a href='{t['url']}'>Открыть</a>"
                )

            await self._send_to_all("\n".join(lines))
        except Exception as e:
            logger.error(f"Deadline alert failed: {e}")

    def stop(self):
        """Shutdown scheduler gracefully."""
        self.scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped")
