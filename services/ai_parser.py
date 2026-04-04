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

# ── Task extraction from free-form text ──────────────────────────

TASK_EXTRACTION_PROMPT = """Ты — AI-ассистент Олега. Тебе приходит текст (обычно расшифровка голосового сообщения).
Извлеки из него задачи. Для каждой задачи определи:

1. title — краткое название задачи (до 80 символов)
2. description — подробное описание, если есть
3. project — название проекта, если упоминается (иначе "Общее")
4. priority — "high" / "medium" / "low" (определи по контексту и срочности)
5. deadline — дедлайн в формате YYYY-MM-DD, если упоминается (иначе null)
6. tags — массив тегов (например: ["маркетинг", "дизайн"])

Если в тексте нет задач (просто разговор), верни пустой массив.
Если задач несколько — верни все.

Сегодня: {today}

Верни ТОЛЬКО валидный JSON-массив, без markdown-блоков.
Пример:
[
  {{
    "title": "Подготовить презентацию для инвесторов",
    "description": "Нужна презентация на 10 слайдов с финансовыми показателями",
    "project": "Fundraising",
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
        response = await client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": TASK_EXTRACTION_PROMPT.format(today=today)},
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

INTENT_PROMPT = """Определи намерение пользователя по его сообщению. Категории:
- "new_tasks" — хочет добавить задачу(и)
- "update_metrics" — сообщает цифры, метрики, KPI
- "check_progress" — хочет узнать статус, прогресс, что происходит
- "complete_task" — хочет отметить задачу выполненной
- "list_tasks" — хочет увидеть список задач
- "help" — спрашивает что умеет бот
- "chat" — просто разговор, не про задачи

Верни ТОЛЬКО одно слово — категорию.
"""


async def classify_intent(text: str) -> str:
    """Classify user intent from message text."""
    try:
        response = await client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": INTENT_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0,
            max_tokens=20,
        )
        intent = response.choices[0].message.content.strip().lower().strip('"')
        logger.info(f"Classified intent: {intent}")
        return intent
    except Exception as e:
        logger.error(f"Intent classification failed: {e}")
        return "chat"
