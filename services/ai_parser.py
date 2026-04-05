"""
AI module â parses free-form voice text into structured tasks,
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

# Known projects â updated dynamically from Notion
KNOWN_PROJECTS = [
    "ÐÐÐ²Ð¸Ð·Ð°ÑÐ¸Ñ", "Ð¢Ð¾Ð¼Ð°Ñ ÐÑÐ°Ð»Ð¾Ð²", "Ð¡Ð¾ÑÐ¸Ñ ÐÑÐ°Ð»Ð¾Ð²",
    "Hash Hedge", "ÐÐ°Ð¹Ð¼Ñ", "GMG", "Solmate", "ÐÐ±ÑÐµÐµ",
]

# ââ Task extraction from free-form text ââââââââââââââââââââââââââ

TASK_EXTRACTION_PROMPT = """Ð¢Ñ â AI-Ð°ÑÑÐ¸ÑÑÐµÐ½Ñ ÐÐ»ÐµÐ³Ð°. Ð¢ÐµÐ±Ðµ Ð¿ÑÐ¸ÑÐ¾Ð´Ð¸Ñ ÑÐµÐºÑÑ (Ð¾Ð±ÑÑÐ½Ð¾ ÑÐ°ÑÑÐ¸ÑÑÐ¾Ð²ÐºÐ° Ð³Ð¾Ð»Ð¾ÑÐ¾Ð²Ð¾Ð³Ð¾ ÑÐ¾Ð¾Ð±ÑÐµÐ½Ð¸Ñ).
ÐÐ·Ð²Ð»ÐµÐºÐ¸ Ð¸Ð· Ð½ÐµÐ³Ð¾ Ð·Ð°Ð´Ð°ÑÐ¸. ÐÐ»Ñ ÐºÐ°Ð¶Ð´Ð¾Ð¹ Ð·Ð°Ð´Ð°ÑÐ¸ ÐÐÐ¯ÐÐÐ¢ÐÐÐ¬ÐÐ Ð¾Ð¿ÑÐµÐ´ÐµÐ»Ð¸:

1. title â ÐºÑÐ°ÑÐºÐ¾Ðµ Ð½Ð°Ð·Ð²Ð°Ð½Ð¸Ðµ Ð·Ð°Ð´Ð°ÑÐ¸ (Ð´Ð¾ 80 ÑÐ¸Ð¼Ð²Ð¾Ð»Ð¾Ð²)
2. description â Ð¿Ð¾Ð´ÑÐ¾Ð±Ð½Ð¾Ðµ Ð¾Ð¿Ð¸ÑÐ°Ð½Ð¸Ðµ, ÐµÑÐ»Ð¸ ÐµÑÑÑ
3. project â Ð½Ð°Ð·Ð²Ð°Ð½Ð¸Ðµ Ð¿ÑÐ¾ÐµÐºÑÐ°. ÐÐÐ¯ÐÐÐ¢ÐÐÐ¬ÐÐÐ ÐÐÐÐ.
   ÐÐ·Ð²ÐµÑÑÐ½ÑÐµ Ð¿ÑÐ¾ÐµÐºÑÑ: {projects}
   ÐÑÐ±ÐµÑÐ¸ Ð½Ð°Ð¸Ð±Ð¾Ð»ÐµÐµ Ð¿Ð¾Ð´ÑÐ¾Ð´ÑÑÐ¸Ð¹ Ð¸Ð· ÑÐ¿Ð¸ÑÐºÐ°. ÐÑÐ»Ð¸ Ð½Ðµ ÑÐ¿Ð¾Ð¼Ð¸Ð½Ð°ÐµÑÑÑ â Ð¿Ð¾ÑÑÐ°Ð²Ñ null (Ð±Ð¾Ñ ÑÐ¿ÑÐ¾ÑÐ¸Ñ).
4. priority â "high" / "medium" / "low" (Ð¾Ð¿ÑÐµÐ´ÐµÐ»Ð¸ Ð¿Ð¾ ÐºÐ¾Ð½ÑÐµÐºÑÑÑ Ð¸ ÑÑÐ¾ÑÐ½Ð¾ÑÑÐ¸)
5. deadline â Ð´ÐµÐ´Ð»Ð°Ð¹Ð½ Ð² ÑÐ¾ÑÐ¼Ð°ÑÐµ YYYY-MM-DD. ÐÐÐ¯ÐÐÐ¢ÐÐÐ¬ÐÐÐ ÐÐÐÐ.
   ÐÑÐ»Ð¸ ÑÐ¿Ð¾Ð¼Ð¸Ð½Ð°ÐµÑÑÑ ÐºÐ¾Ð½ÐºÑÐµÑÐ½Ð°Ñ Ð´Ð°ÑÐ° â Ð¸ÑÐ¿Ð¾Ð»ÑÐ·ÑÐ¹ ÐµÑ.
   ÐÑÐ»Ð¸ Ð³Ð¾Ð²Ð¾ÑÐ¸Ñ "Ð·Ð°Ð²ÑÑÐ°", "Ð¿Ð¾ÑÐ»ÐµÐ·Ð°Ð²ÑÑÐ°", "ÑÐµÑÐµÐ· Ð½ÐµÐ´ÐµÐ»Ñ" â Ð²ÑÑÐ¸ÑÐ»Ð¸ Ð´Ð°ÑÑ.
   ÐÑÐ»Ð¸ Ð´ÐµÐ´Ð»Ð°Ð¹Ð½ Ð½Ðµ ÑÐ¿Ð¾Ð¼Ð¸Ð½Ð°ÐµÑÑÑ â Ð¿Ð¾ÑÑÐ°Ð²Ñ null (Ð±Ð¾Ñ ÑÐ¿ÑÐ¾ÑÐ¸Ñ).
6. tags â Ð¼Ð°ÑÑÐ¸Ð² ÑÐµÐ³Ð¾Ð² (Ð½Ð°Ð¿ÑÐ¸Ð¼ÐµÑ: ["Ð¼Ð°ÑÐºÐµÑÐ¸Ð½Ð³", "Ð´Ð¸Ð·Ð°Ð¹Ð½"])

ÐÐÐÐÐ: project Ð¸ deadline â Ð¾Ð±ÑÐ·Ð°ÑÐµÐ»ÑÐ½Ñ. ÐÑÐ»Ð¸ Ð½Ðµ Ð¼Ð¾Ð¶ÐµÑÑ Ð¾Ð¿ÑÐµÐ´ÐµÐ»Ð¸ÑÑ â Ð²ÐµÑÐ½Ð¸ null, Ð±Ð¾Ñ ÑÑÐ¾ÑÐ½Ð¸Ñ Ñ Ð¿Ð¾Ð»ÑÐ·Ð¾Ð²Ð°ÑÐµÐ»Ñ.

ÐÑÐ»Ð¸ Ð² ÑÐµÐºÑÑÐµ Ð½ÐµÑ Ð·Ð°Ð´Ð°Ñ (Ð¿ÑÐ¾ÑÑÐ¾ ÑÐ°Ð·Ð³Ð¾Ð²Ð¾Ñ), Ð²ÐµÑÐ½Ð¸ Ð¿ÑÑÑÐ¾Ð¹ Ð¼Ð°ÑÑÐ¸Ð².
ÐÑÐ»Ð¸ Ð·Ð°Ð´Ð°Ñ Ð½ÐµÑÐºÐ¾Ð»ÑÐºÐ¾ â Ð²ÐµÑÐ½Ð¸ Ð²ÑÐµ.

Ð¡ÐµÐ³Ð¾Ð´Ð½Ñ: {today}

ÐÐµÑÐ½Ð¸ Ð¢ÐÐÐ¬ÐÐ Ð²Ð°Ð»Ð¸Ð´Ð½ÑÐ¹ JSON-Ð¼Ð°ÑÑÐ¸Ð², Ð±ÐµÐ· markdown-Ð±Ð»Ð¾ÐºÐ¾Ð².
ÐÑÐ¸Ð¼ÐµÑ:
[
  {{
    "title": "ÐÐ¾Ð´Ð³Ð¾ÑÐ¾Ð²Ð¸ÑÑ Ð¿ÑÐµÐ·ÐµÐ½ÑÐ°ÑÐ¸Ñ Ð´Ð»Ñ Ð¸Ð½Ð²ÐµÑÑÐ¾ÑÐ¾Ð²",
    "description": "ÐÑÐ¶Ð½Ð° Ð¿ÑÐµÐ·ÐµÐ½ÑÐ°ÑÐ¸Ñ Ð½Ð° 10 ÑÐ»Ð°Ð¹Ð´Ð¾Ð² Ñ ÑÐ¸Ð½Ð°Ð½ÑÐ¾Ð²ÑÐ¼Ð¸ Ð¿Ð¾ÐºÐ°Ð·Ð°ÑÐµÐ»ÑÐ¼Ð¸",
    "project": "Hash Hedge",
    "priority": "high",
    "deadline": "2026-04-10",
    "tags": ["Ð¿ÑÐµÐ·ÐµÐ½ÑÐ°ÑÐ¸Ñ", "Ð¸Ð½Ð²ÐµÑÑÐ¾ÑÑ"]
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


# ââ Metric update extraction âââââââââââââââââââââââââââââââââââââ

METRIC_PROMPT = """Ð¢Ñ â AI-Ð°ÑÑÐ¸ÑÑÐµÐ½Ñ. ÐÐ· ÑÐµÐºÑÑÐ° Ð¸Ð·Ð²Ð»ÐµÐºÐ¸ Ð¾Ð±Ð½Ð¾Ð²Ð»ÐµÐ½Ð¸Ñ Ð¼ÐµÑÑÐ¸Ðº/KPI Ð¿ÑÐ¾ÐµÐºÑÐ¾Ð².
ÐÐ»Ñ ÐºÐ°Ð¶Ð´Ð¾Ð³Ð¾ Ð¾Ð±Ð½Ð¾Ð²Ð»ÐµÐ½Ð¸Ñ Ð¾Ð¿ÑÐµÐ´ÐµÐ»Ð¸:

1. project â Ð½Ð°Ð·Ð²Ð°Ð½Ð¸Ðµ Ð¿ÑÐ¾ÐµÐºÑÐ°
2. metric_name â Ð½Ð°Ð·Ð²Ð°Ð½Ð¸Ðµ Ð¼ÐµÑÑÐ¸ÐºÐ¸ (Ð½Ð°Ð¿ÑÐ¸Ð¼ÐµÑ: "ÐÐ¾Ð½Ð²ÐµÑÑÐ¸Ñ", "MRR", "DAU", "ÐÐ°Ð´Ð°Ñ Ð·Ð°ÐºÑÑÑÐ¾")
3. value â ÑÐ¸ÑÐ»Ð¾Ð²Ð¾Ðµ Ð·Ð½Ð°ÑÐµÐ½Ð¸Ðµ
4. unit â ÐµÐ´Ð¸Ð½Ð¸ÑÐ° Ð¸Ð·Ð¼ÐµÑÐµÐ½Ð¸Ñ (%, $, ÑÑ, Ð¸ Ñ.Ð´.)
5. comment â Ð¿Ð¾ÑÑÐ½ÐµÐ½Ð¸Ðµ, ÐµÑÐ»Ð¸ ÐµÑÑÑ

Ð¡ÐµÐ³Ð¾Ð´Ð½Ñ: {today}

ÐÐµÑÐ½Ð¸ Ð¢ÐÐÐ¬ÐÐ Ð²Ð°Ð»Ð¸Ð´Ð½ÑÐ¹ JSON-Ð¼Ð°ÑÑÐ¸Ð². ÐÑÐ»Ð¸ Ð¼ÐµÑÑÐ¸Ðº Ð½ÐµÑ â Ð²ÐµÑÐ½Ð¸ [].
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


# ââ Progress analysis ââââââââââââââââââââââââââââââââââââââââââââ

ANALYSIS_PROMPT = """Ð¢Ñ â AI-Ð°ÑÑÐ¸ÑÑÐµÐ½Ñ Ð´Ð»Ñ ÑÐ¿ÑÐ°Ð²Ð»ÐµÐ½Ð¸Ñ Ð¿ÑÐ¾ÐµÐºÑÐ°Ð¼Ð¸. ÐÑÐ¾Ð°Ð½Ð°Ð»Ð¸Ð·Ð¸ÑÑÐ¹ ÑÐµÐºÑÑÐµÐµ ÑÐ¾ÑÑÐ¾ÑÐ½Ð¸Ðµ Ð·Ð°Ð´Ð°Ñ Ð¸ Ð¼ÐµÑÑÐ¸Ðº.

ÐÐ°Ð´Ð°ÑÐ¸:
{tasks_json}

ÐÐµÑÑÐ¸ÐºÐ¸:
{metrics_json}

ÐÐ°Ð¹ ÐºÑÐ°ÑÐºÐ¸Ð¹ Ð°Ð½Ð°Ð»Ð¸Ð· Ð½Ð° ÑÑÑÑÐºÐ¾Ð¼ ÑÐ·ÑÐºÐµ:
1. ÐÐ±ÑÐ¸Ð¹ Ð¿ÑÐ¾Ð³ÑÐµÑÑ: ÑÐºÐ¾Ð»ÑÐºÐ¾ Ð·Ð°Ð´Ð°Ñ Ð²ÑÐ¿Ð¾Ð»Ð½ÐµÐ½Ð¾ / Ð² ÑÐ°Ð±Ð¾ÑÐµ / Ð¿ÑÐ¾ÑÑÐ¾ÑÐµÐ½Ð¾
2. ÐÑÐ¾Ð±Ð»ÐµÐ¼Ð½ÑÐµ Ð·Ð¾Ð½Ñ: ÐºÐ°ÐºÐ¸Ðµ Ð·Ð°Ð´Ð°ÑÐ¸ Ð·Ð°ÑÑÑÑÐ»Ð¸ Ð¸ ÐÐÐ§ÐÐÐ£ (Ð¿ÑÐµÐ´Ð¿Ð¾Ð»Ð¾Ð¶Ð¸ Ð¿ÑÐ¸ÑÐ¸Ð½Ñ)
3. ÐÐµÑÑÐ¸ÐºÐ¸: ÑÑÐ¾ ÑÐ°ÑÑÑÑ, ÑÑÐ¾ Ð¿Ð°Ð´Ð°ÐµÑ, Ð½Ð° ÑÑÐ¾ Ð¾Ð±ÑÐ°ÑÐ¸ÑÑ Ð²Ð½Ð¸Ð¼Ð°Ð½Ð¸Ðµ
4. Ð¢ÐÐ-3 ÑÐµÐºÐ¾Ð¼ÐµÐ½Ð´Ð°ÑÐ¸Ð¸: ÑÑÐ¾ ÑÐ´ÐµÐ»Ð°ÑÑ Ð¿ÑÑÐ¼Ð¾ ÑÐµÐ¹ÑÐ°Ñ

ÐÑÐ´Ñ ÐºÐ¾Ð½ÐºÑÐµÑÐ½ÑÐ¼, Ð³Ð¾Ð²Ð¾ÑÐ¸ Ð¿Ð¾ Ð´ÐµÐ»Ñ, Ð±ÐµÐ· Ð²Ð¾Ð´Ñ. ÐÑÐ¿Ð¾Ð»ÑÐ·ÑÐ¹ emoji Ð´Ð»Ñ Ð½Ð°Ð³Ð»ÑÐ´Ð½Ð¾ÑÑÐ¸.
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
        return "ÐÐµ ÑÐ´Ð°Ð»Ð¾ÑÑ ÑÐ³ÐµÐ½ÐµÑÐ¸ÑÐ¾Ð²Ð°ÑÑ Ð°Ð½Ð°Ð»Ð¸Ð·. ÐÐ¾Ð¿ÑÐ¾Ð±ÑÐ¹ Ð¿Ð¾Ð·Ð¶Ðµ."


# ââ Intent classification ââââââââââââââââââââââââââââââââââââââââ

INTENT_PROMPT = """ÐÐ¿ÑÐµÐ´ÐµÐ»Ð¸ Ð½Ð°Ð¼ÐµÑÐµÐ½Ð¸Ðµ Ð¿Ð¾Ð»ÑÐ·Ð¾Ð²Ð°ÑÐµÐ»Ñ Ð¿Ð¾ ÐµÐ³Ð¾ ÑÐ¾Ð¾Ð±ÑÐµÐ½Ð¸Ñ. Ð£ÑÐ¸ÑÑÐ²Ð°Ð¹ ÐºÐ¾Ð½ÑÐµÐºÑÑ Ð¿ÑÐµÐ´ÑÐ´ÑÑÐ¸Ñ ÑÐ¾Ð¾Ð±ÑÐµÐ½Ð¸Ð¹, ÐµÑÐ»Ð¸ Ð¾Ð½Ð¸ ÐµÑÑÑ.

ÐÐ°ÑÐµÐ³Ð¾ÑÐ¸Ð¸:
- "new_tasks" â ÑÐ¾ÑÐµÑ Ð´Ð¾Ð±Ð°Ð²Ð¸ÑÑ Ð·Ð°Ð´Ð°ÑÑ(Ð¸)
- "add_subtask" â ÑÐ¾ÑÐµÑ Ð´Ð¾Ð±Ð°Ð²Ð¸ÑÑ Ð¿Ð¾Ð´Ð·Ð°Ð´Ð°ÑÑ Ðº ÑÑÑÐµÑÑÐ²ÑÑÑÐµÐ¹ Ð·Ð°Ð´Ð°ÑÐµ
- "update_metrics" â ÑÐ¾Ð¾Ð±ÑÐ°ÐµÑ ÑÐ¸ÑÑÑ, Ð¼ÐµÑÑÐ¸ÐºÐ¸, KPI
- "check_progress" â ÑÐ¾ÑÐµÑ ÑÐ·Ð½Ð°ÑÑ ÑÑÐ°ÑÑÑ, Ð¿ÑÐ¾Ð³ÑÐµÑÑ, ÑÑÐ¾ Ð¿ÑÐ¾Ð¸ÑÑÐ¾Ð´Ð¸Ñ
- "complete_task" â ÑÐ¾ÑÐµÑ Ð¾ÑÐ¼ÐµÑÐ¸ÑÑ Ð·Ð°Ð´Ð°ÑÑ Ð²ÑÐ¿Ð¾Ð»Ð½ÐµÐ½Ð½Ð¾Ð¹
- "list_tasks" â ÑÐ¾ÑÐµÑ ÑÐ²Ð¸Ð´ÐµÑÑ ÑÐ¿Ð¸ÑÐ¾Ðº Ð·Ð°Ð´Ð°Ñ
- "list_projects" â ÑÐ¾ÑÐµÑ ÑÐ²Ð¸Ð´ÐµÑÑ ÑÐ¿Ð¸ÑÐ¾Ðº Ð¿ÑÐ¾ÐµÐºÑÐ¾Ð²
- "add_project" â ÑÐ¾ÑÐµÑ Ð´Ð¾Ð±Ð°Ð²Ð¸ÑÑ Ð½Ð¾Ð²ÑÐ¹ Ð¿ÑÐ¾ÐµÐºÑ
- "rename_project" â ÑÐ¾ÑÐµÑ Ð¿ÐµÑÐµÐ¸Ð¼ÐµÐ½Ð¾Ð²Ð°ÑÑ Ð¿ÑÐ¾ÐµÐºÑ
- "project_tasks" â ÑÐ¾ÑÐµÑ ÑÐ²Ð¸Ð´ÐµÑÑ Ð·Ð°Ð´Ð°ÑÐ¸ ÐºÐ¾Ð½ÐºÑÐµÑÐ½Ð¾Ð³Ð¾ Ð¿ÑÐ¾ÐµÐºÑÐ°
- "help" â ÑÐ¿ÑÐ°ÑÐ¸Ð²Ð°ÐµÑ ÑÑÐ¾ ÑÐ¼ÐµÐµÑ Ð±Ð¾Ñ
- "chat" â Ð¿ÑÐ¾ÑÑÐ¾ ÑÐ°Ð·Ð³Ð¾Ð²Ð¾Ñ, Ð½Ðµ Ð¿ÑÐ¾ Ð·Ð°Ð´Ð°ÑÐ¸

ÐÐµÑÐ½Ð¸ Ð¢ÐÐÐ¬ÐÐ Ð¾Ð´Ð½Ð¾ ÑÐ»Ð¾Ð²Ð¾ â ÐºÐ°ÑÐµÐ³Ð¾ÑÐ¸Ñ.
"""


# ââ Project rename extraction ââââââââââââââââââââââââââââââââââ

RENAME_PROJECT_PROMPT = """ÐÐ· ÑÐµÐºÑÑÐ° Ð¿Ð¾Ð»ÑÐ·Ð¾Ð²Ð°ÑÐµÐ»Ñ Ð¸Ð·Ð²Ð»ÐµÐºÐ¸:
1. old_name â ÑÐµÐºÑÑÐµÐµ (ÑÑÐ°ÑÐ¾Ðµ) Ð½Ð°Ð·Ð²Ð°Ð½Ð¸Ðµ Ð¿ÑÐ¾ÐµÐºÑÐ°
2. new_name â Ð½Ð¾Ð²Ð¾Ðµ Ð½Ð°Ð·Ð²Ð°Ð½Ð¸Ðµ Ð¿ÑÐ¾ÐµÐºÑÐ°

ÐÐµÑÐ½Ð¸ Ð¢ÐÐÐ¬ÐÐ Ð²Ð°Ð»Ð¸Ð´Ð½ÑÐ¹ JSON Ð±ÐµÐ· markdown-Ð±Ð»Ð¾ÐºÐ¾Ð²:
{{"old_name": "...", "new_name": "..."}}

ÐÑÐ»Ð¸ Ð½Ðµ ÑÐ´Ð°Ð»Ð¾ÑÑ Ð¾Ð¿ÑÐµÐ´ÐµÐ»Ð¸ÑÑ Ð¾Ð±Ð° Ð½Ð°Ð·Ð²Ð°Ð½Ð¸Ñ, Ð²ÐµÑÐ½Ð¸ {{"old_name": null, "new_name": null}}.
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


# ââ Project name extraction from text ââââââââââââââââââââââââââ

PROJECT_NAME_PROMPT = """ÐÐ· ÑÐµÐºÑÑÐ° Ð¿Ð¾Ð»ÑÐ·Ð¾Ð²Ð°ÑÐµÐ»Ñ Ð¾Ð¿ÑÐµÐ´ÐµÐ»Ð¸ Ð½Ð°Ð·Ð²Ð°Ð½Ð¸Ðµ Ð¿ÑÐ¾ÐµÐºÑÐ°, Ð¾ ÐºÐ¾ÑÐ¾ÑÐ¾Ð¼ Ð¾Ð½ ÑÐ¿ÑÐ°ÑÐ¸Ð²Ð°ÐµÑ.
ÐÐ·Ð²ÐµÑÑÐ½ÑÐµ Ð¿ÑÐ¾ÐµÐºÑÑ: {projects}
ÐÐµÑÐ½Ð¸ Ð¢ÐÐÐ¬ÐÐ Ð½Ð°Ð·Ð²Ð°Ð½Ð¸Ðµ Ð¿ÑÐ¾ÐµÐºÑÐ° â Ð¾Ð´Ð½Ñ ÑÑÑÐ¾ÐºÑ, Ð±ÐµÐ· ÐºÐ°Ð²ÑÑÐµÐº Ð¸ JSON.
ÐÑÐ»Ð¸ Ð½Ðµ ÑÐ´Ð°Ð»Ð¾ÑÑ Ð¾Ð¿ÑÐµÐ´ÐµÐ»Ð¸ÑÑ â Ð²ÐµÑÐ½Ð¸ ÑÐ»Ð¾Ð²Ð¾ "null".
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


MATCH_TASK_PROMPT = """Ð£ Ð¿Ð¾Ð»ÑÐ·Ð¾Ð²Ð°ÑÐµÐ»Ñ ÐµÑÑÑ ÑÐ¿Ð¸ÑÐ¾Ðº Ð°ÐºÑÐ¸Ð²Ð½ÑÑ Ð·Ð°Ð´Ð°Ñ. ÐÐ½ Ð³Ð¾Ð²Ð¾ÑÐ¸Ñ ÑÑÐ¾ Ð·Ð°ÐºÐ¾Ð½ÑÐ¸Ð» Ð·Ð°Ð´Ð°ÑÑ.
ÐÐ¿ÑÐµÐ´ÐµÐ»Ð¸, ÐºÐ°ÐºÑÑ Ð¸Ð¼ÐµÐ½Ð½Ð¾ Ð·Ð°Ð´Ð°ÑÑ Ð¾Ð½ Ð¸Ð¼ÐµÐµÑ Ð² Ð²Ð¸Ð´Ñ.

ÐÐºÑÐ¸Ð²Ð½ÑÐµ Ð·Ð°Ð´Ð°ÑÐ¸ (id | Ð½Ð°Ð·Ð²Ð°Ð½Ð¸Ðµ | Ð¿ÑÐ¾ÐµÐºÑ):
{tasks_list}

Ð¢ÐµÐºÑÑ Ð¿Ð¾Ð»ÑÐ·Ð¾Ð²Ð°ÑÐµÐ»Ñ: {text}

ÐÐµÑÐ½Ð¸ Ð¢ÐÐÐ¬ÐÐ id Ð·Ð°Ð´Ð°ÑÐ¸ (UUID) ÐºÐ¾ÑÐ¾ÑÐ°Ñ Ð»ÑÑÑÐµ Ð²ÑÐµÐ³Ð¾ Ð¿Ð¾Ð´ÑÐ¾Ð´Ð¸Ñ.
ÐÑÐ»Ð¸ Ð½Ð¸ Ð¾Ð´Ð½Ð° Ð·Ð°Ð´Ð°ÑÐ° Ð½Ðµ Ð¿Ð¾Ð´ÑÐ¾Ð´Ð¸Ñ â Ð²ÐµÑÐ½Ð¸ "null".
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
                role_label = "ÐÐ¾Ð»ÑÐ·Ð¾Ð²Ð°ÑÐµÐ»Ñ" if h["role"] == "user" else "ÐÐ¾Ñ"
                context_lines.append(f"{role_label}: {h['text']}")
            if context_lines:
                messages.append({
                    "role": "user",
                    "content": f"ÐÐ¾Ð½ÑÐµÐºÑÑ Ð¿ÑÐµÐ´ÑÐ´ÑÑÐ¸Ñ ÑÐ¾Ð¾Ð±ÑÐµÐ½Ð¸Ð¹:\n"
                    + "\n".join(context_lines)
                    + f"\n\nÐ¢ÐµÐºÑÑÐµÐµ ÑÐ¾Ð¾Ð±ÑÐµÐ½Ð¸Ðµ:\n{text}",
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
