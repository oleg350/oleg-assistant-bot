"""
Telegram message handlers — text, voice, commands.
"""
import logging
from aiogram import Router, F, Bot
from aiogram.types import Message, CallbackQuery
from aiogram.filters import Command
from aiogram.utils.keyboard import InlineKeyboardBuilder

from config import config
from services.whisper import transcribe_voice
from services.ai_parser import extract_tasks, extract_metrics, classify_intent, analyze_progress
from services.notion_client import notion

logger = logging.getLogger(__name__)
router = Router()


# ── Access guard ──────────────────────────────────────────────────

def is_allowed(user_id: int) -> bool:
    if not config.ALLOWED_USER_IDS:
        return True  # no restriction if list is empty
    return user_id in config.ALLOWED_USER_IDS


# ── /start ────────────────────────────────────────────────────────

@router.message(Command("start"))
async def cmd_start(message: Message):
    if not is_allowed(message.from_user.id):
        return await message.answer("⛔ Нет доступа.")

    await message.answer(
        "👋 <b>Привет! Я твой AI-ассистент.</b>\n\n"
        "Что я умею:\n"
        "🎙 <b>Голосовые</b> — скажи задачу, я разберу и добавлю на доску\n"
        "📋 /tasks — список всех задач\n"
        "✅ /done — отметить задачу выполненной\n"
        "📊 /progress — анализ прогресса по проектам\n"
        "📈 /metrics — последние метрики\n"
        "🔥 /overdue — просроченные задачи\n"
        "⏳ /upcoming — ближайшие дедлайны\n\n"
        "Просто отправь голосовое или текст — я пойму что нужно 🧠",
        parse_mode="HTML",
    )


# ── /tasks ────────────────────────────────────────────────────────

@router.message(Command("tasks"))
async def cmd_tasks(message: Message):
    if not is_allowed(message.from_user.id):
        return

    await message.answer("📋 Загружаю задачи...")
    tasks = await notion.get_all_tasks()

    if not tasks:
        return await message.answer("Задач пока нет. Отправь голосовое — добавлю!")

    # Group by project
    projects = {}
    for t in tasks:
        proj = t["project"] or "Общее"
        projects.setdefault(proj, []).append(t)

    lines = ["📋 <b>Все задачи:</b>\n"]
    status_emoji = {
        "Новая": "🆕",
        "В работе": "🔄",
        "Готово": "✅",
        "Заблокирована": "🚫",
    }

    for proj, proj_tasks in projects.items():
        lines.append(f"\n📁 <b>{proj}</b>")
        for t in proj_tasks:
            emoji = status_emoji.get(t["status"], "📌")
            priority = {"high": "🔥", "medium": "⚡", "low": "📌"}.get(t["priority"], "")
            deadline = f" | ⏰ {t['deadline']}" if t["deadline"] else ""
            lines.append(f"  {emoji}{priority} {t['title']}{deadline}")

    await message.answer("\n".join(lines), parse_mode="HTML")


# ── /done ─────────────────────────────────────────────────────────

@router.message(Command("done"))
async def cmd_done(message: Message):
    if not is_allowed(message.from_user.id):
        return

    # Get non-done tasks
    tasks = await notion.get_all_tasks()
    active = [t for t in tasks if t["status"] != "Готово"]

    if not active:
        return await message.answer("Все задачи уже выполнены! 🎉")

    builder = InlineKeyboardBuilder()
    for t in active[:20]:  # Max 20 buttons
        label = f"{'🔥' if t['priority'] == 'high' else '📌'} {t['title'][:40]}"
        builder.button(text=label, callback_data=f"complete:{t['id']}")
    builder.adjust(1)

    await message.answer(
        "✅ Какую задачу отметить выполненной?",
        reply_markup=builder.as_markup(),
    )


@router.callback_query(F.data.startswith("complete:"))
async def cb_complete_task(callback: CallbackQuery):
    page_id = callback.data.split(":", 1)[1]
    try:
        await notion.update_task_status(page_id, "Готово")
        await callback.answer("✅ Задача выполнена!")
        await callback.message.edit_text("✅ Задача отмечена как выполненная!")
    except Exception as e:
        logger.error(f"Failed to complete task: {e}")
        await callback.answer("Ошибка при обновлении", show_alert=True)


# ── /progress ─────────────────────────────────────────────────────

@router.message(Command("progress"))
async def cmd_progress(message: Message):
    if not is_allowed(message.from_user.id):
        return

    await message.answer("🤖 Анализирую прогресс...")
    tasks = await notion.get_all_tasks()
    metrics = await notion.get_recent_metrics()
    analysis = await analyze_progress(tasks, metrics)
    await message.answer(f"📊 <b>Анализ прогресса:</b>\n\n{analysis}", parse_mode="HTML")


# ── /metrics ──────────────────────────────────────────────────────

@router.message(Command("metrics"))
async def cmd_metrics(message: Message):
    if not is_allowed(message.from_user.id):
        return

    metrics = await notion.get_recent_metrics()
    if not metrics:
        return await message.answer("Метрик пока нет. Скажи голосовым цифры — я запишу!")

    lines = ["📈 <b>Последние метрики:</b>\n"]
    for m in metrics[:20]:
        lines.append(
            f"  📁 {m['project']} | <b>{m['metric_name']}</b>: "
            f"{m['value']} {m['unit']} ({m['date']})"
        )

    await message.answer("\n".join(lines), parse_mode="HTML")


# ── /overdue ──────────────────────────────────────────────────────

@router.message(Command("overdue"))
async def cmd_overdue(message: Message):
    if not is_allowed(message.from_user.id):
        return

    overdue = await notion.get_overdue_tasks()
    if not overdue:
        return await message.answer("Просроченных задач нет! 🎉")

    lines = ["🔴 <b>Просроченные задачи:</b>\n"]
    for t in overdue:
        lines.append(
            f"  🔥 <b>{t['title']}</b> — дедлайн был {t['deadline']}\n"
            f"     📁 {t['project']} | <a href='{t['url']}'>Открыть</a>"
        )

    await message.answer("\n".join(lines), parse_mode="HTML", disable_web_page_preview=True)


# ── /upcoming ─────────────────────────────────────────────────────

@router.message(Command("upcoming"))
async def cmd_upcoming(message: Message):
    if not is_allowed(message.from_user.id):
        return

    upcoming = await notion.get_upcoming_deadlines(days=7)
    if not upcoming:
        return await message.answer("На ближайшую неделю дедлайнов нет 🏖")

    lines = ["⏳ <b>Ближайшие дедлайны (7 дней):</b>\n"]
    for t in upcoming:
        lines.append(
            f"  ⚡ <b>{t['title']}</b> — {t['deadline']}\n"
            f"     📁 {t['project']} | {t['status']}"
        )

    await message.answer("\n".join(lines), parse_mode="HTML")


# ── Voice messages ────────────────────────────────────────────────

@router.message(F.voice)
async def handle_voice(message: Message, bot: Bot):
    if not is_allowed(message.from_user.id):
        return

    await message.answer("🎙 Слушаю...")

    # Download voice file
    file = await bot.get_file(message.voice.file_id)
    voice_bytes = await bot.download_file(file.file_path)
    voice_data = voice_bytes.read()

    # Transcribe
    text = await transcribe_voice(voice_data)
    await message.answer(f"📝 <i>Расшифровка:</i>\n{text}", parse_mode="HTML")

    # Process the text
    await _process_text(message, text)


# ── Text messages ─────────────────────────────────────────────────

@router.message(F.text)
async def handle_text(message: Message):
    if not is_allowed(message.from_user.id):
        return

    # Skip commands (already handled above)
    if message.text.startswith("/"):
        return

    await _process_text(message, message.text)


# ── Core processing logic ────────────────────────────────────────

async def _process_text(message: Message, text: str):
    """Central text processing — classify intent and act."""

    intent = await classify_intent(text)
    logger.info(f"Intent: {intent} for text: {text[:60]}")

    if intent == "new_tasks":
        await _handle_new_tasks(message, text)
    elif intent == "update_metrics":
        await _handle_metrics_update(message, text)
    elif intent == "check_progress":
        await cmd_progress(message)
    elif intent == "complete_task":
        await cmd_done(message)
    elif intent == "list_tasks":
        await cmd_tasks(message)
    elif intent == "help":
        await cmd_start(message)
    else:
        # Fallback: try to extract tasks anyway, might be mixed content
        tasks = await extract_tasks(text)
        metrics = await extract_metrics(text)

        if tasks:
            await _handle_new_tasks(message, text, pre_extracted=tasks)
        elif metrics:
            await _handle_metrics_update(message, text, pre_extracted=metrics)
        else:
            await message.answer(
                "🤔 Не нашёл задач или метрик в сообщении.\n"
                "Попробуй сказать конкретнее или используй /help"
            )


async def _handle_new_tasks(message: Message, text: str, pre_extracted=None):
    """Extract and create tasks."""
    tasks = pre_extracted or await extract_tasks(text)

    if not tasks:
        return await message.answer("Не удалось разобрать задачи. Попробуй переформулировать.")

    created = []
    for task in tasks:
        try:
            result = await notion.create_task(task)
            created.append(task)
        except Exception as e:
            logger.error(f"Failed to create task '{task['title']}': {e}")

    if created:
        lines = [f"✅ <b>Создано {len(created)} задач(и):</b>\n"]
        for t in created:
            priority_emoji = {"high": "🔥", "medium": "⚡", "low": "📌"}.get(t["priority"], "📌")
            deadline = f" | ⏰ до {t['deadline']}" if t.get("deadline") else ""
            lines.append(
                f"  {priority_emoji} <b>{t['title']}</b>\n"
                f"     📁 {t.get('project', 'Общее')}{deadline}"
            )
        lines.append("\n📋 Задачи добавлены на доску в Notion!")
        await message.answer("\n".join(lines), parse_mode="HTML")
    else:
        await message.answer("⚠️ Не удалось создать задачи. Проверь подключение к Notion.")


async def _handle_metrics_update(message: Message, text: str, pre_extracted=None):
    """Extract and save metrics."""
    metrics = pre_extracted or await extract_metrics(text)

    if not metrics:
        return await message.answer("Не нашёл метрик в сообщении.")

    saved = []
    for m in metrics:
        try:
            await notion.add_metric(m)
            saved.append(m)
        except Exception as e:
            logger.error(f"Failed to save metric: {e}")

    if saved:
        lines = [f"📈 <b>Записано {len(saved)} метрик(и):</b>\n"]
        for m in saved:
            lines.append(
                f"  📁 {m.get('project', 'Общее')} | "
                f"<b>{m['metric_name']}</b>: {m['value']} {m.get('unit', '')}"
            )
        await message.answer("\n".join(lines), parse_mode="HTML")
    else:
        await message.answer("⚠️ Не удалось сохранить метрики.")
