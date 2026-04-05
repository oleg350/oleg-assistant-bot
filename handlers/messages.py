"""
Telegram message handlers — text, voice, commands.
"""
import logging
from collections import deque
from datetime import datetime, date, timedelta

from aiogram import Router, F, Bot
from aiogram.types import Message, CallbackQuery
from aiogram.filters import Command
from aiogram.utils.keyboard import InlineKeyboardBuilder

from config import config
from services.whisper import transcribe_voice
from services.ai_parser import (
    extract_tasks, extract_metrics, classify_intent, analyze_progress,
    extract_rename, extract_project_name, match_task_from_text, KNOWN_PROJECTS,
)
from services.notion_client import notion

logger = logging.getLogger(__name__)
router = Router()

# ── Conversation history (per-user, in-memory) ──────────────────
MAX_HISTORY = 10
_user_history: dict[int, deque] = {}


def _add_to_history(user_id: int, role: str, text: str):
    if user_id not in _user_history:
        _user_history[user_id] = deque(maxlen=MAX_HISTORY)
    _user_history[user_id].append({"role": role, "text": text[:500]})


def _get_history(user_id: int) -> list[dict]:
    return list(_user_history.get(user_id, []))


# ── Pending state for follow-up flows ───────────────────────────
# Stores tasks waiting for deadline/project from user
_pending_tasks: dict[int, list[dict]] = {}
# Stores rename flow state
_rename_state: dict[int, str] = {}
# Stores subtask flow state: {user_id: task_id}
_subtask_state: dict[int, str] = {}


# ── Access guard ──────────────────────────────────────────────────

def is_allowed(user_id: int) -> bool:
    if not config.ALLOWED_USER_IDS:
        return True
    return user_id in config.ALLOWED_USER_IDS


# ── Helpers ──────────────────────────────────────────────────────

PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2, "": 3}
PRIORITY_LABEL = {"high": "!!!", "medium": "!!", "low": "!"}
STATUS_LABEL = {"Новая": "new", "В работе": "wip", "Готово": "done", "Заблокирована": "block"}


def _fmt_deadline_short(deadline_str: str | None) -> str:
    """Return a compact deadline string for table display."""
    if not deadline_str:
        return "нет"
    try:
        dl = datetime.fromisoformat(deadline_str).date()
        today = date.today()
        delta = (dl - today).days
        if delta < 0:
            return f"-{abs(delta)}д!"
        elif delta == 0:
            return "сегодня!"
        elif delta == 1:
            return "завтра"
        elif delta <= 7:
            return f"{delta}д"
        else:
            return dl.strftime("%d.%m")
    except (ValueError, TypeError):
        return str(deadline_str)


def _fmt_task_table(tasks: list[dict], show_project: bool = False) -> str:
    """Format tasks as a clean monospace table."""
    if not tasks:
        return "<i>Нет задач</i>"
    lines = []
    for i, t in enumerate(tasks, 1):
        prio = PRIORITY_LABEL.get(t.get("priority", ""), "")
        dl = _fmt_deadline_short(t.get("deadline"))
        title = t["title"][:38]
        if show_project:
            proj = (t.get("project") or "")[:10]
            lines.append(f"  {i}. {title}  [{proj}]  {dl}  {prio}")
        else:
            lines.append(f"  {i}. {title}  {dl}  {prio}")
    return "\n".join(lines)


def _is_overdue(t: dict) -> bool:
    dl = t.get("deadline")
    if not dl:
        return False
    try:
        return datetime.fromisoformat(dl).date() < date.today()
    except (ValueError, TypeError):
        return False


def _sort_tasks(tasks: list[dict]) -> list[dict]:
    """Sort tasks: closest deadline first, then by priority. No deadline = last."""
    return sorted(tasks, key=lambda t: (
        t.get("deadline") or "9999-99-99",
        PRIORITY_ORDER.get(t.get("priority", ""), 3),
    ))


def _group_by_project(tasks: list[dict]) -> dict[str, list[dict]]:
    projects = {}
    for t in tasks:
        proj = t.get("project") or "Общее"
        projects.setdefault(proj, []).append(t)
    return projects


# ── /start ────────────────────────────────────────────────────────

@router.message(Command("start"))
async def cmd_start(message: Message):
    if not is_allowed(message.from_user.id):
        return await message.answer("⛔ Нет доступа.")

    await message.answer(
        "<b>Привет! Я твой AI-ассистент.</b>\n\n"
        "Голосовым или текстом:\n"
        "  — добавить задачу / подзадачу\n"
        "  — отметить выполнение\n"
        "  — обновить метрики\n"
        "  — добавить / переименовать проект\n\n"
        "Команды:\n"
        "  /tasks — список задач\n"
        "  /done — отметить задачу\n"
        "  /projects — проекты\n"
        "  /progress — анализ прогресса\n\n"
        "Дедлайн и проект спрошу, если не указаны.",
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

    await _render_tasks(message, tasks, show_done=False)

    builder = InlineKeyboardBuilder()
    done_count = len([t for t in tasks if t["status"] == "Готово"])
    active_count = len(tasks) - done_count
    builder.button(text=f"Активные ({active_count})", callback_data="tasks_filter:active")
    builder.button(text=f"Выполненные ({done_count})", callback_data="tasks_filter:done")
    builder.button(text="Все задачи", callback_data="tasks_filter:all")
    builder.adjust(2, 1)

    await message.answer("Фильтры:", reply_markup=builder.as_markup())


@router.callback_query(F.data.startswith("tasks_filter:"))
async def cb_tasks_filter(callback: CallbackQuery):
    filter_type = callback.data.split(":", 1)[1]
    await callback.answer()

    tasks = await notion.get_all_tasks()
    if not tasks:
        return await callback.message.answer("Задач пока нет.")

    if filter_type == "active":
        await _render_tasks(callback.message, tasks, show_done=False)
    elif filter_type == "done":
        done_tasks = [t for t in tasks if t["status"] == "Готово"]
        if not done_tasks:
            return await callback.message.answer("Выполненных задач нет.")
        lines = [f"<b>Выполненные задачи — {len(done_tasks)}</b>\n"]
        for proj, proj_tasks in sorted(_group_by_project(done_tasks).items()):
            lines.append(f"<b>{proj}</b>")
            for i, t in enumerate(proj_tasks, 1):
                lines.append(f"  {i}. <s>{t['title']}</s>")
            lines.append("")
        await callback.message.answer("\n".join(lines), parse_mode="HTML")
    else:
        await _render_tasks(callback.message, tasks, show_done=True)


async def _render_tasks(message: Message, tasks: list[dict], show_done: bool):
    """Render tasks grouped by project. Within each project: active sorted by deadline, then done."""
    all_projects = _group_by_project(tasks)

    active_total = len([t for t in tasks if t["status"] != "Готово"])
    done_total = len([t for t in tasks if t["status"] == "Готово"])
    overdue_total = len([t for t in tasks if t["status"] != "Готово" and _is_overdue(t)])

    if not active_total and not show_done:
        return await message.answer("Все задачи выполнены!")

    lines = [f"<b>Задачи — {len(tasks)}</b>"]
    if overdue_total:
        lines.append(f"Просрочено: {overdue_total}")
    lines.append("")

    for proj in sorted(all_projects.keys()):
        proj_tasks = all_projects[proj]
        active = [t for t in proj_tasks if t["status"] != "Готово"]
        done = [t for t in proj_tasks if t["status"] == "Готово"]

        if not active and not show_done:
            continue

        lines.append(f"<b>{proj}</b>")

        if active:
            sorted_active = _sort_tasks(active)
            lines.append(_fmt_task_table(sorted_active))

        if done and show_done:
            for t in done:
                lines.append(f"  <s>{t['title']}</s>")
        elif done and not show_done:
            lines.append(f"  <i>+ {len(done)} выполн.</i>")

        lines.append("")

    await message.answer("\n".join(lines), parse_mode="HTML")


# ── /done ─────────────────────────────────────────────────────────

@router.message(Command("done"))
async def cmd_done(message: Message):
    if not is_allowed(message.from_user.id):
        return

    tasks = await notion.get_all_tasks()
    active = _sort_tasks([t for t in tasks if t["status"] != "Готово"])

    if not active:
        return await message.answer("Все задачи уже выполнены! 🎉")

    builder = InlineKeyboardBuilder()
    for t in active[:20]:
        prio = PRIORITY_LABEL.get(t["priority"], "")
        label = f"{t['title'][:42]} {prio}"
        builder.button(text=label, callback_data=f"complete:{t['id']}")
    builder.adjust(1)

    await message.answer("Какую задачу отметить выполненной?", reply_markup=builder.as_markup())


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
    await message.answer("Анализирую прогресс...")
    tasks = await notion.get_all_tasks()
    metrics = await notion.get_recent_metrics()
    analysis = await analyze_progress(tasks, metrics)
    await message.answer(f"<b>Анализ прогресса</b>\n\n{analysis}", parse_mode="HTML")


# ── /projects ─────────────────────────────────────────────────

@router.message(Command("projects"))
async def cmd_projects(message: Message):
    if not is_allowed(message.from_user.id):
        return
    projects = await notion.get_projects()
    if not projects:
        return await message.answer("📁 Проектов пока нет.")

    lines = ["<b>Проекты</b>\n"]
    for i, proj in enumerate(projects, 1):
        tasks = await notion.get_tasks_by_project(proj)
        done = len([t for t in tasks if t["status"] == "Готово"])
        active = len(tasks) - done
        lines.append(f"  {i}. <b>{proj}</b> — {active} актив. / {done} готово")

    builder = InlineKeyboardBuilder()
    for proj in projects:
        builder.button(text=proj, callback_data=f"proj_tasks:{proj}")
    builder.adjust(2)

    await message.answer("\n".join(lines), parse_mode="HTML", reply_markup=builder.as_markup())


# ── Subtask flow ─────────────────────────────────────────────────

@router.callback_query(F.data.startswith("subtask:"))
async def cb_subtask_start(callback: CallbackQuery):
    task_id = callback.data.split(":", 1)[1]
    _subtask_state[callback.from_user.id] = task_id
    await callback.answer()
    await callback.message.answer(
        "📝 Напиши текст подзадачи (или несколько через новую строку):"
    )


# ── Callbacks for deadline selection ──────────────────────────────

@router.callback_query(F.data.startswith("dl:"))
async def cb_deadline_select(callback: CallbackQuery):
    """Handle deadline button press for pending tasks."""
    user_id = callback.from_user.id
    parts = callback.data.split(":")
    # dl:N where N is days offset, or dl:custom
    offset = parts[1]
    await callback.answer()

    if user_id not in _pending_tasks or not _pending_tasks[user_id]:
        return await callback.message.answer("Нет задач, ожидающих дедлайн.")

    if offset == "custom":
        await callback.message.answer(
            "📅 Напиши дедлайн в формате ДД.ММ.ГГГГ\n"
            "Например: 15.04.2026"
        )
        return

    days = int(offset)
    deadline = (date.today() + timedelta(days=days)).isoformat()
    await _finalize_pending_tasks(callback.message, user_id, deadline=deadline)


@router.callback_query(F.data.startswith("proj:"))
async def cb_project_select(callback: CallbackQuery):
    """Handle project button press for pending tasks."""
    user_id = callback.from_user.id
    project = callback.data.split(":", 1)[1]
    await callback.answer()

    if user_id not in _pending_tasks or not _pending_tasks[user_id]:
        return await callback.message.answer("Нет задач, ожидающих проект.")

    await _finalize_pending_tasks(callback.message, user_id, project=project)


# ── Voice messages ────────────────────────────────────────────────

@router.message(F.voice)
async def handle_voice(message: Message, bot: Bot):
    if not is_allowed(message.from_user.id):
        return
    await message.answer("🎙 Слушаю...")
    file = await bot.get_file(message.voice.file_id)
    voice_bytes = await bot.download_file(file.file_path)
    voice_data = voice_bytes.read()
    text = await transcribe_voice(voice_data)
    await message.answer(f"📝 <i>Расшифровка:</i>\n{text}", parse_mode="HTML")
    _add_to_history(message.from_user.id, "user", text)
    await _process_text(message, text)


# ── Text messages ─────────────────────────────────────────────────

@router.message(F.text)
async def handle_text(message: Message):
    if not is_allowed(message.from_user.id):
        return
    if message.text.startswith("/"):
        return
    _add_to_history(message.from_user.id, "user", message.text)
    await _process_text(message, message.text)


# ── Core processing logic ────────────────────────────────────────

async def _process_text(message: Message, text: str):
    user_id = message.from_user.id

    # Check subtask flow
    if user_id in _subtask_state:
        task_id = _subtask_state.pop(user_id)
        subtasks = [s.strip() for s in text.split("\n") if s.strip()]
        count = 0
        for s in subtasks:
            try:
                await notion.add_subtask(task_id, s)
                count += 1
            except Exception as e:
                logger.error(f"Failed to add subtask: {e}")
        await message.answer(f"✅ Добавлено {count} подзадач(и)!")
        return

    # Check rename flow
    if user_id in _rename_state:
        old_name = _rename_state.pop(user_id)
        new_name = text.strip()
        await _do_rename(message, old_name, new_name)
        return

    # Check pending tasks waiting for deadline (date format)
    if user_id in _pending_tasks and _pending_tasks[user_id]:
        # Try to parse as date
        parsed_date = _try_parse_date(text.strip())
        if parsed_date:
            await _finalize_pending_tasks(message, user_id, deadline=parsed_date)
            return

    history = _get_history(user_id)
    intent = await classify_intent(text, history=history)
    logger.info(f"Intent: {intent} for text: {text[:60]}")

    if intent == "new_tasks":
        await _handle_new_tasks(message, text)
    elif intent == "add_subtask":
        await _handle_add_subtask(message, text)
    elif intent == "update_metrics":
        await _handle_metrics_update(message, text)
    elif intent == "check_progress":
        await cmd_progress(message)
    elif intent == "complete_task":
        await _smart_complete(message, text)
    elif intent == "list_tasks":
        await cmd_tasks(message)
    elif intent == "list_projects":
        await cmd_projects(message)
    elif intent == "add_project":
        # Extract project name from text
        name = await extract_project_name(text)
        if name and name not in KNOWN_PROJECTS:
            KNOWN_PROJECTS.append(name)
            await message.answer(f"✅ Проект <b>{name}</b> добавлен!", parse_mode="HTML")
        elif name:
            await message.answer(f"Проект <b>{name}</b> уже существует.", parse_mode="HTML")
        else:
            await message.answer("Не понял название проекта. Используй /addproject Название")
    elif intent == "rename_project":
        await _handle_rename_project(message, text)
    elif intent == "project_tasks":
        await _handle_project_tasks(message, text)
    elif intent == "help":
        await cmd_start(message)
    else:
        await message.answer(
            "💬 Понял тебя. Если хочешь добавить задачу — скажи конкретнее, "
            "или используй /help чтобы увидеть команды."
        )


def _try_parse_date(text: str) -> str | None:
    """Try to parse a date from common formats."""
    for fmt in ("%d.%m.%Y", "%d.%m.%y", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            d = datetime.strptime(text, fmt).date()
            return d.isoformat()
        except ValueError:
            continue
    return None


async def _handle_new_tasks(message: Message, text: str, pre_extracted=None):
    """Extract and create tasks, asking for missing deadline/project."""
    tasks = pre_extracted or await extract_tasks(text)

    if not tasks:
        return await message.answer("Не удалось разобрать задачи. Попробуй переформулировать.")

    user_id = message.from_user.id
    created = []
    needs_deadline = []
    needs_project = []

    for task in tasks:
        missing_deadline = not task.get("deadline")
        missing_project = not task.get("project")

        if missing_deadline or missing_project:
            # Store for follow-up
            if user_id not in _pending_tasks:
                _pending_tasks[user_id] = []
            _pending_tasks[user_id].append(task)
            if missing_deadline:
                needs_deadline.append(task)
            if missing_project:
                needs_project.append(task)
        else:
            try:
                await notion.create_task(task)
                created.append(task)
            except Exception as e:
                logger.error(f"Failed to create task '{task['title']}': {e}")

    # Report created tasks
    if created:
        lines = [f"<b>Создано {len(created)} задач(и):</b>\n"]
        for t in created:
            dl = _fmt_deadline_short(t.get("deadline"))
            lines.append(
                f"  <b>{t['title']}</b>\n"
                f"  {t.get('project', 'Общее')}  |  {dl}"
            )
        await message.answer("\n".join(lines), parse_mode="HTML")

    # Ask for missing fields
    if needs_project:
        task_names = ", ".join(f"«{t['title']}»" for t in needs_project)
        builder = InlineKeyboardBuilder()
        for proj in KNOWN_PROJECTS:
            if proj != "Общее":
                builder.button(text=proj, callback_data=f"proj:{proj}")
        builder.button(text="Общее", callback_data="proj:Общее")
        builder.adjust(2)
        await message.answer(
            f"📁 К какому проекту отнести: {task_names}?",
            reply_markup=builder.as_markup(),
            parse_mode="HTML",
        )
    elif needs_deadline:
        task_names = ", ".join(f"«{t['title']}»" for t in needs_deadline)
        builder = InlineKeyboardBuilder()
        builder.button(text="Завтра", callback_data="dl:1")
        builder.button(text="Через 3 дня", callback_data="dl:3")
        builder.button(text="Через неделю", callback_data="dl:7")
        builder.button(text="Свой дедлайн", callback_data="dl:custom")
        builder.adjust(2)
        await message.answer(
            f"⏰ Когда дедлайн для: {task_names}?",
            reply_markup=builder.as_markup(),
            parse_mode="HTML",
        )


async def _finalize_pending_tasks(message: Message, user_id: int, deadline: str = None, project: str = None):
    """Fill in missing fields and create pending tasks."""
    if user_id not in _pending_tasks:
        return

    tasks = _pending_tasks[user_id]

    # Fill in provided field
    for task in tasks:
        if deadline and not task.get("deadline"):
            task["deadline"] = deadline
        if project and not task.get("project"):
            task["project"] = project

    # Check if still missing something
    still_needs_deadline = [t for t in tasks if not t.get("deadline")]
    still_needs_project = [t for t in tasks if not t.get("project")]

    if still_needs_project:
        task_names = ", ".join(f"«{t['title']}»" for t in still_needs_project)
        builder = InlineKeyboardBuilder()
        for proj in KNOWN_PROJECTS:
            if proj != "Общее":
                builder.button(text=proj, callback_data=f"proj:{proj}")
        builder.button(text="Общее", callback_data="proj:Общее")
        builder.adjust(2)
        await message.answer(
            f"📁 К какому проекту: {task_names}?",
            reply_markup=builder.as_markup(),
            parse_mode="HTML",
        )
        return

    if still_needs_deadline:
        task_names = ", ".join(f"«{t['title']}»" for t in still_needs_deadline)
        builder = InlineKeyboardBuilder()
        builder.button(text="Завтра", callback_data="dl:1")
        builder.button(text="Через 3 дня", callback_data="dl:3")
        builder.button(text="Через неделю", callback_data="dl:7")
        builder.button(text="Свой дедлайн", callback_data="dl:custom")
        builder.adjust(2)
        await message.answer(
            f"⏰ Когда дедлайн: {task_names}?",
            reply_markup=builder.as_markup(),
            parse_mode="HTML",
        )
        return

    # All fields filled — create all tasks
    created = []
    for task in tasks:
        try:
            await notion.create_task(task)
            created.append(task)
        except Exception as e:
            logger.error(f"Failed to create task '{task['title']}': {e}")

    del _pending_tasks[user_id]

    if created:
        lines = [f"<b>Создано {len(created)} задач(и):</b>\n"]
        for t in created:
            dl = _fmt_deadline_short(t.get("deadline"))
            lines.append(
                f"  <b>{t['title']}</b>\n"
                f"  {t.get('project')}  |  {dl}"
            )
        lines.append("\nЗадачи добавлены на доску.")
        _add_to_history(message.chat.id, "bot", f"Создано {len(created)} задач")
        await message.answer("\n".join(lines), parse_mode="HTML")


async def _smart_complete(message: Message, text: str):
    """Try to match task from text, ask confirmation, fallback to list."""
    tasks = await notion.get_all_tasks()
    active = _sort_tasks([t for t in tasks if t["status"] != "Готово"])

    if not active:
        return await message.answer("Все задачи уже выполнены! 🎉")

    # Try GPT matching
    matched_id = await match_task_from_text(text, active)

    if matched_id:
        # Find the matched task
        matched = next((t for t in active if t["id"] == matched_id), None)
        if matched:
            dl = _fmt_deadline_short(matched.get("deadline"))
            builder = InlineKeyboardBuilder()
            builder.button(text="Да, закрыть", callback_data=f"complete:{matched['id']}")
            builder.button(text="Нет, другая", callback_data="complete_list")
            builder.adjust(2)

            await message.answer(
                f"Ты имеешь в виду эту задачу?\n\n"
                f"<b>{matched['title']}</b>\n"
                f"{matched['project']}  |  {dl}",
                parse_mode="HTML",
                reply_markup=builder.as_markup(),
            )
            return

    # Fallback: show full list
    await cmd_done(message)


@router.callback_query(F.data == "complete_list")
async def cb_complete_list(callback: CallbackQuery):
    """Fallback: show full task list for completion."""
    await callback.answer()
    tasks = await notion.get_all_tasks()
    active = _sort_tasks([t for t in tasks if t["status"] != "Готово"])

    if not active:
        return await callback.message.answer("Все задачи выполнены! 🎉")

    builder = InlineKeyboardBuilder()
    for t in active[:20]:
        prio = PRIORITY_LABEL.get(t["priority"], "")
        label = f"{t['title'][:42]} {prio}"
        builder.button(text=label, callback_data=f"complete:{t['id']}")
    builder.adjust(1)

    await callback.message.answer(
        "Выбери задачу из списка:",
        reply_markup=builder.as_markup(),
    )


async def _handle_add_subtask(message: Message, text: str):
    """Show active tasks to select for adding subtask."""
    tasks = await notion.get_all_tasks()
    active = _sort_tasks([t for t in tasks if t["status"] != "Готово"])

    if not active:
        return await message.answer("Нет активных задач для подзадач.")

    builder = InlineKeyboardBuilder()
    for t in active[:15]:
        builder.button(text=t['title'][:45], callback_data=f"subtask:{t['id']}")
    builder.adjust(1)

    await message.answer(
        "📝 К какой задаче добавить подзадачу?",
        reply_markup=builder.as_markup(),
    )


async def _handle_metrics_update(message: Message, text: str, pre_extracted=None):
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
        lines = [f"<b>Записано {len(saved)} метрик(и):</b>\n"]
        for m in saved:
            lines.append(
                f"  {m.get('project', 'Общее')}  |  "
                f"<b>{m['metric_name']}</b>: {m['value']} {m.get('unit', '')}"
            )
        _add_to_history(message.from_user.id, "bot", f"Записано {len(saved)} метрик")
        await message.answer("\n".join(lines), parse_mode="HTML")
    else:
        await message.answer("⚠️ Не удалось сохранить метрики.")


# ── Project management handlers ─────────────────────────────────

async def _handle_rename_project(message: Message, text: str):
    await message.answer("✏️ Определяю проект...")
    rename_info = await extract_rename(text)
    old_name = rename_info.get("old_name")
    new_name = rename_info.get("new_name")
    if not old_name or not new_name:
        return await message.answer(
            "Не смог определить. Напиши: <i>\"переименуй проект Старое в Новое\"</i>",
            parse_mode="HTML",
        )
    await _do_rename(message, old_name, new_name)


async def _do_rename(message: Message, old_name: str, new_name: str):
    await message.answer(f"🔄 Переименовываю <b>{old_name}</b> → <b>{new_name}</b>...", parse_mode="HTML")
    count = await notion.rename_project(old_name, new_name)
    if count > 0:
        # Update known projects
        if old_name in KNOWN_PROJECTS:
            KNOWN_PROJECTS.remove(old_name)
        if new_name not in KNOWN_PROJECTS:
            KNOWN_PROJECTS.append(new_name)
        await message.answer(f"✅ Проект <b>{new_name}</b> — обновлено {count} задач.", parse_mode="HTML")
    else:
        await message.answer(f"⚠️ Не нашёл задач с проектом <b>{old_name}</b>.", parse_mode="HTML")


async def _handle_project_tasks(message: Message, text: str):
    project_name = await extract_project_name(text)
    if not project_name:
        projects = await notion.get_projects()
        if not projects:
            return await message.answer("Проектов пока нет.")
        builder = InlineKeyboardBuilder()
        for proj in projects:
            builder.button(text=f"📁 {proj}", callback_data=f"proj_tasks:{proj}")
        builder.adjust(1)
        return await message.answer("Какой проект показать?", reply_markup=builder.as_markup())
    await _show_project_tasks(message, project_name)


async def _show_project_tasks(message: Message, project_name: str):
    tasks = await notion.get_tasks_by_project(project_name)
    if not tasks:
        return await message.answer(f"Задач по проекту <b>{project_name}</b> не найдено.", parse_mode="HTML")

    active = _sort_tasks([t for t in tasks if t["status"] != "Готово"])
    done = [t for t in tasks if t["status"] == "Готово"]

    lines = [f"<b>{project_name}</b>\n"]
    if active:
        lines.append(f"<b>Активные — {len(active)}</b>")
        lines.append(_fmt_task_table(active))
    if done:
        lines.append(f"\n<b>Выполнено — {len(done)}</b>")
        for i, t in enumerate(done, 1):
            lines.append(f"  {i}. <s>{t['title']}</s>")
    lines.append(f"\nВсего: {len(tasks)}  |  Активных: {len(active)}  |  Готово: {len(done)}")
    await message.answer("\n".join(lines), parse_mode="HTML")


@router.callback_query(F.data.startswith("proj_tasks:"))
async def cb_project_tasks(callback: CallbackQuery):
    project_name = callback.data.split(":", 1)[1]
    await callback.answer()
    await _show_project_tasks(callback.message, project_name)


@router.callback_query(F.data.startswith("rename_start:"))
async def cb_rename_start(callback: CallbackQuery):
    old_name = callback.data.split(":", 1)[1]
    _rename_state[callback.from_user.id] = old_name
    await callback.answer()
    await callback.message.answer(
        f"✏️ Переименовываю проект <b>{old_name}</b>.\nНапиши новое название:",
        parse_mode="HTML",
    )
