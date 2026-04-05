"""
AI module — parses free-form voice text into structured tasks,
analyses project progress, generates insights.
Uses OpenAI GPT-4o.
"""
import json
import logging
from datetime import datetime
from openai import AsyncOpenAI
from config import config

logger = logging.getLogger(__name__)
client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)

# Known projects — updated dynamically from Notion
KNOWN_PROJECTS = [
    "ИИвизация", "Томас Кралов", "София Кралов",
    "Hash Hedge", "Займы", "GMG", "Solmate", "Общее",
]

# ── Task extraction from free-form text ──────────────────────────

TASK_EXTRACTION_PROMPT = """Ты — AI-ассистент Олега. Тебе приходит текст (обычно расшифровка голосового сообщения).
Извлеки из него задачи. Для каждой задачи ОБЯЗАТЕЛЬНО определи:

1. title — краткое название задачи (до 80 символов)
2. description — подробное описание, если есть
3. project — название проекта. ОБЯЗАТЕЛЬНОЕ ПОЛЕ.
   Известные проекты: {projects}
   Выбери наиболее подходящий из списка. Если не упоминается — поставь null (бот спросит).
4. priority — "high" / "medium" / "low" (определи по контексту и срочности)
5. deadline — дедлайн в формате YYYY-MM-DD. ОБЯЗАТЕЛЬНОЕ ПОЛЕ.
   Если упоминается конкретная дата — используй её.
   Если говорит "завтра", "послезавтра", "через неделю" — вычисли дату.
   Если дедлайн не упоминается — поставь null (бот спросит).
6. tags — массив тегов (например: ["маркетинг", "дизайн"])

ВАЖНО: project и deadline — обязательны. Если не можешь определить — верни null, бот уточнит у пользователя.

Если в тексте нет задач (просто разговор), верни пустой массив.
Если задач несколько — верни все.

Сегодня: {today}

Верни ТОЛЬКО валидный JSON-массив, без markdown-блоков.
Пример:
[
  {{
    "title": "Подготовить презентацию для инвесторов",
    "description": "Нужна презентация на 10 слайдов с финансовыми показателями",
    "project": "Hash Hedge",
    "priority": "high",
    "deadline": "2026-04-10",
    "tags": ["презентация", "инвесторы"]
  }}
]
"""


async def extract_tasks(text: str) -> list[dict]:
    """Extract structured tasks from free-form text."""
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        projects_str = ", ".join(KNOWN_PROJECTS)
        response = await client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": TASK_EXTRACTION_PROMPT.format(
                        today=today, projects=projects_str
                    ),
                },
                {"role": "user", "content": text},
            ],
            temperature=0.1,
            max_tokens=2000,
        )
        raw = response.choices[0].message.content.strip()
        # Clean potential markdown code fences
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]
        tasks = json.loads(raw)
        logger.info(f"Extracted {len(tasks)} tasks from text")
        return tasks
    except Exception as e:
        logger.error(f"Task extraction failed: {e}")
        return []


# ── Metric update extraction ─────────────────────────────────────

METRIC_PROMPT = """Ты — AI-ассистент. Из текста извлеки обновления метрик/KPI проектов.
Для каждого обновления определи:

1. project — название проекта
2. metric_name — название метрики (например: "Конверсия", "MRR", "DAU", "Задач закрыто")
3. value — числовое значение
4. unit — единица измерения (%, $, шт, и т.д.)
5. comment — пояснение, если есть

Сегодня: {today}

Верни ТОЛЬКО валидный JSON-массив. Если метрик нет — верни [].
"""


async def extract_metrics(text: str) -> list[dict]:
    """Extract metric updates from text."""
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        response = await client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": METRIC_PROMPT.format(today=today)},
                {"role": "user", "content": text},
            ],
            temperature=0.1,
            max_tokens=1500,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]
        return json.loads(raw)
    except Exception as e:
        logger.error(f"Metric extraction failed: {e}")
        return []


# ── Progress analysis ────────────────────────────────────────────

ANALYSIS_PROMPT = """Ты — AI-ассистент для управления проектами. Проанализируй текущее состояние задач и метрик.

Задачи:
{tasks_json}

Метрики:
{metrics_json}

Дай краткий анализ на русском языке:
1. Общий прогресс: сколько задач выполнено / в работе / просрочено
2. Проблемные зоны: какие задачи застряли и ПОЧЕМУ (предположи причины)
3. Метрики: что растёт, что падает, на что обратить внимание
4. ТОП-3 рекомендации: что сделать прямо сейчас

Будь конкретным, говори по делу, без воды. Используй emoji для наглядности.
"""


async def analyze_progress(tasks: list[dict], metrics: list[dict]) -> str:
    """Generate progress analysis from tasks and metrics."""
    try:
        response = await client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": ANALYSIS_PROMPT.format(
                        tasks_json=json.dumps(tasks, ensure_ascii=False, indent=2),
                        metrics_json=json.dumps(metrics, ensure_ascii=False, indent=2),
                    ),
                },
            ],
            temperature=0.3,
            max_tokens=2000,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Progress analysis failed: {e}")
        return "Не удалось сгенерировать анализ. Попробуй позже."


# ── Intent classification ────────────────────────────────────────

INTENT_PROMPT = """Определи намерение пользователя по его сообщению. Учитывай контекст предыдущих сообщений, если они есть.

Категории:
- "new_tasks" — хочет добавить задачу(и)
- "add_subtask" — хочет добавить подзадачу к существующей задаче
- "update_metrics" — сообщает цифры, метрики, KPI
- "check_progress" — хочет узнать статус, прогресс, что происходит
- "complete_task" — хочет отметить задачу выполненной
- "list_tasks" — хочет увидеть список задач
- "list_projects" — хочет увидеть список проектов
- "add_project" — хочет добавить новый проект
- "rename_project" — хочет переименовать проект
- "project_tasks" — хочет увидеть задачи конкретного проекта
- "help" — спрашивает что умеет бот
- "chat" — просто разговор, не про задачи

Верни ТОЛЬКО одно слово — категорию.
"""


# ── Project rename extraction ──────────────────────────────────

RENAME_PROJECT_PROMPT = """Из текста пользователя извлеки:
1. old_name — текущее (старое) название проекта
2. new_name — новое название проекта

Верни ТОЛЬКО валидный JSON без markdown-блоков:
{{"old_name": "...", "new_name": "..."}}

Если не удалось определить оба названия, верни {{"old_name": null, "new_name": null}}.
"""


async def extract_rename(text: str) -> dict:
    """Extract project rename info from text."""
    try:
        response = await client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": RENAME_PROJECT_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0,
            max_tokens=200,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]
        return json.loads(raw)
    except Exception as e:
        logger.error(f"Rename extraction failed: {e}")
        return {"old_name": None, "new_name": None}


# ── Project name extraction from text ──────────────────────────

PROJECT_NAME_PROMPT = """Из текста пользователя определи название проекта, о котором он спрашивает.
Известные проекты: {projects}
Верни ТОЛЬКО название проекта — одну строку, без кавычек и JSON.
Если не удалось определить — верни слово "null".
"""


async def extract_project_name(text: str) -> str | None:
    """Extract project name from user text."""
    try:
        projects_str = ", ".join(KNOWN_PROJECTS)
        response = await client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": PROJECT_NAME_PROMPT.format(projects=projects_str),
                },
                {"role": "user", "content": text},
            ],
            temperature=0,
            max_tokens=50,
        )
        name = response.choices[0].message.content.strip().strip('"')
        return None if name.lower() == "null" else name
    except Exception as e:
        logger.error(f"Project name extraction failed: {e}")
        return None


MATCH_TASK_PROMPT = """У пользователя есть список активных задач. Он говорит что закончил задачу.
Определи, какую именно задачу он имеет в виду.

Активные задачи (id | название | проект):
{tasks_list}

Текст пользователя: {text}

Верни ТОЛЬКО id задачи (UUID) которая лучше всего подходит.
Если ни одна задача не подходит — верни "null".
"""


async def match_task_from_text(text: str, tasks: list[dict]) -> str | None:
    """Match user's text description to a specific task. Returns task ID or None."""
    if not tasks:
        return None
    try:
        tasks_list = "\n".join(
            f"{t['id']} | {t['title']} | {t.get('project', '')}"
            for t in tasks
        )
        response = await client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": MATCH_TASK_PROMPT.format(
                        tasks_list=tasks_list, text=text
                    ),
                },
            ],
            temperature=0,
            max_tokens=100,
        )
        result = response.choices[0].message.content.strip().strip('"')
        if result.lower() == "null" or len(result) < 10:
            return None
        return result
    except Exception as e:
        logger.error(f"Task matching failed: {e}")
        return None


async def classify_intent(text: str, history: list[dict] | None = None) -> str:
    """Classify user intent from message text, with optional conversation history."""
    try:
        messages = [{"role": "system", "content": INTENT_PROMPT}]

        if history:
            context_lines = []
            for h in history[-6:]:
                role_label = "Пользователь" if h["role"] == "user" else "Бот"
                context_lines.append(f"{role_label}: {h['text']}")
            if context_lines:
                messages.append({
                    "role": "user",
                    "content": f"Контекст предыдущих сообщений:\n"
                    + "\n".join(context_lines)
                    + f"\n\nТекущее сообщение:\n{text}",
                })
            else:
                messages.append({"role": "user", "content": text})
        else:
            messages.append({"role": "user", "content": text})

        response = await client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=messages,
            temperature=0,
            max_tokens=20,
        )
        intent = response.choices[0].message.content.strip().lower().strip('"')
        logger.info(f"Classified intent: {intent}")
        return intent
    except Exception as e:
        logger.error(f"Intent classification failed: {e}")
        return "chat"
