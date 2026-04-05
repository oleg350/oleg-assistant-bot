"""
Notion integration — manages tasks board and metrics database.

Expected Notion databases:

TASKS DB properties:
  - Name (title)
  - Status (select: "Новая", "В работе", "Готово", "Заблокирована")
  - Priority (select: "high", "medium", "low")
  - Project (select)
  - Deadline (date)
  - Tags (multi_select)
  - Description (rich_text)
  - Created (date, auto)

METRICS DB properties:
  - Name (title) — metric name
  - Project (select)
  - Value (number)
  - Unit (rich_text)
  - Date (date)
  - Comment (rich_text)
"""
import logging
from datetime import datetime, date
from typing import Optional
import httpx
from config import config

logger = logging.getLogger(__name__)

NOTION_API = "https://api.notion.com/v1"
HEADERS = {
    "Authorization": f"Bearer {config.NOTION_TOKEN}",
    "Content-Type": "application/json",
    "Notion-Version": "2022-06-28",
}


class NotionTaskBoard:
    """Manages the tasks Kanban board in Notion."""

    def __init__(self):
        self.db_id = config.NOTION_DATABASE_ID
        self.metrics_db_id = config.NOTION_METRICS_DB_ID

    # ── Tasks ─────────────────────────────────────────────────

    async def create_task(self, task: dict) -> dict:
        """Create a new task page in the Notion database."""
        properties = {
            "Name": {"title": [{"text": {"content": task["title"]}}]},
            "Status": {"select": {"name": "Новая"}},
            "Priority": {"select": {"name": task.get("priority", "medium")}},
            "Project": {"select": {"name": task.get("project", "Общее")}},
        }

        if task.get("deadline"):
            properties["Deadline"] = {"date": {"start": task["deadline"]}}

        if task.get("tags"):
            properties["Tags"] = {
                "multi_select": [{"name": tag} for tag in task["tags"]]
            }

        if task.get("description"):
            properties["Description"] = {
                "rich_text": [{"text": {"content": task["description"][:2000]}}]
            }

        properties["Created"] = {"date": {"start": datetime.now().isoformat()}}

        body = {"parent": {"database_id": self.db_id}, "properties": properties}

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{NOTION_API}/pages", headers=HEADERS, json=body, timeout=30
            )
            resp.raise_for_status()
            result = resp.json()
            logger.info(f"Created Notion task: {task['title']} -> {result['id']}")
            return result

    async def get_all_tasks(self, status_filter: Optional[str] = None) -> list[dict]:
        """Fetch all tasks, optionally filtered by status."""
        body = {"page_size": 100}
        if status_filter:
            body["filter"] = {
                "property": "Status",
                "select": {"equals": status_filter},
            }

        # Sort by deadline ascending (urgent first)
        body["sorts"] = [{"property": "Deadline", "direction": "ascending"}]

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{NOTION_API}/databases/{self.db_id}/query",
                headers=HEADERS,
                json=body,
                timeout=30,
            )
            resp.raise_for_status()
            pages = resp.json()["results"]

        tasks = []
        for page in pages:
            props = page["properties"]
            task = {
                "id": page["id"],
                "title": self._get_title(props.get("Name", {})),
                "status": self._get_select(props.get("Status", {})),
                "priority": self._get_select(props.get("Priority", {})),
                "project": self._get_select(props.get("Project", {})),
                "deadline": self._get_date(props.get("Deadline", {})),
                "tags": self._get_multi_select(props.get("Tags", {})),
                "description": self._get_rich_text(props.get("Description", {})),
                "url": page.get("url", ""),
            }
            tasks.append(task)
        return tasks

    async def update_task_status(self, page_id: str, new_status: str) -> dict:
        """Update task status (e.g. mark as done)."""
        body = {"properties": {"Status": {"select": {"name": new_status}}}}
        async with httpx.AsyncClient() as client:
            resp = await client.patch(
                f"{NOTION_API}/pages/{page_id}",
                headers=HEADERS,
                json=body,
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()

    async def get_overdue_tasks(self) -> list[dict]:
        """Get tasks that are past their deadline and not done."""
        today = date.today().isoformat()
        body = {
            "filter": {
                "and": [
                    {"property": "Deadline", "date": {"before": today}},
                    {
                        "property": "Status",
                        "select": {"does_not_equal": "Готово"},
                    },
                ]
            },
            "sorts": [{"property": "Deadline", "direction": "ascending"}],
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{NOTION_API}/databases/{self.db_id}/query",
                headers=HEADERS,
                json=body,
                timeout=30,
            )
            resp.raise_for_status()
            pages = resp.json()["results"]

        return [self._parse_page(p) for p in pages]

    async def get_upcoming_deadlines(self, days: int = 3) -> list[dict]:
        """Get tasks with deadlines in the next N days."""
        from datetime import timedelta

        today = date.today()
        future = (today + timedelta(days=days)).isoformat()
        today_str = today.isoformat()

        body = {
            "filter": {
                "and": [
                    {"property": "Deadline", "date": {"on_or_after": today_str}},
                    {"property": "Deadline", "date": {"on_or_before": future}},
                    {"property": "Status", "select": {"does_not_equal": "Готово"}},
                ]
            },
            "sorts": [{"property": "Deadline", "direction": "ascending"}],
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{NOTION_API}/databases/{self.db_id}/query",
                headers=HEADERS,
                json=body,
                timeout=30,
            )
            resp.raise_for_status()
            pages = resp.json()["results"]

        return [self._parse_page(p) for p in pages]

    # ── Metrics ───────────────────────────────────────────────

    async def add_metric(self, metric: dict) -> dict:
        """Add a metric entry to the metrics database."""
        properties = {
            "Name": {"title": [{"text": {"content": metric["metric_name"]}}]},
            "Project": {"select": {"name": metric.get("project", "Общее")}},
            "Value": {"number": float(metric.get("value", 0))},
            "Date": {"date": {"start": datetime.now().date().isoformat()}},
        }
        if metric.get("unit"):
            properties["Unit"] = {
                "rich_text": [{"text": {"content": metric["unit"]}}]
            }
        if metric.get("comment"):
            properties["Comment"] = {
                "rich_text": [{"text": {"content": metric["comment"][:2000]}}]
            }

        body = {"parent": {"database_id": self.metrics_db_id}, "properties": properties}
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{NOTION_API}/pages", headers=HEADERS, json=body, timeout=30
            )
            resp.raise_for_status()
            return resp.json()

    async def get_recent_metrics(self, project: Optional[str] = None) -> list[dict]:
        """Get recent metrics, optionally for a specific project."""
        body = {
            "sorts": [{"property": "Date", "direction": "descending"}],
            "page_size": 50,
        }
        if project:
            body["filter"] = {"property": "Project", "select": {"equals": project}}

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{NOTION_API}/databases/{self.metrics_db_id}/query",
                headers=HEADERS,
                json=body,
                timeout=30,
            )
            resp.raise_for_status()
            pages = resp.json()["results"]

        metrics = []
        for page in pages:
            props = page["properties"]
            metrics.append({
                "metric_name": self._get_title(props.get("Name", {})),
                "project": self._get_select(props.get("Project", {})),
                "value": props.get("Value", {}).get("number"),
                "unit": self._get_rich_text(props.get("Unit", {})),
                "date": self._get_date(props.get("Date", {})),
                "comment": self._get_rich_text(props.get("Comment", {})),
            })
        return metrics

    # ── Projects ──────────────────────────────────────────────

    async def get_projects(self) -> list[str]:
        """Get a list of unique project names from tasks DB."""
        tasks = await self.get_all_tasks()
        projects = set()
        for t in tasks:
            if t["project"]:
                projects.add(t["project"])
        return sorted(projects)

    async def rename_project(self, old_name: str, new_name: str) -> int:
        """Rename a project across all tasks. Returns count of updated tasks."""
        tasks = await self.get_all_tasks()
        matching = [t for t in tasks if t["project"].lower() == old_name.lower()]

        count = 0
        async with httpx.AsyncClient() as client:
            for t in matching:
                body = {"properties": {"Project": {"select": {"name": new_name}}}}
                try:
                    resp = await client.patch(
                        f"{NOTION_API}/pages/{t['id']}",
                        headers=HEADERS,
                        json=body,
                        timeout=30,
                    )
                    resp.raise_for_status()
                    count += 1
                except Exception as e:
                    logger.error(f"Failed to rename project for task {t['id']}: {e}")
        logger.info(f"Renamed project '{old_name}' -> '{new_name}' in {count} tasks")
        return count

    async def get_tasks_by_project(self, project_name: str) -> list[dict]:
        """Get all tasks for a specific project."""
        body = {
            "filter": {
                "property": "Project",
                "select": {"equals": project_name},
            },
            "sorts": [{"property": "Deadline", "direction": "ascending"}],
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{NOTION_API}/databases/{self.db_id}/query",
                headers=HEADERS,
                json=body,
                timeout=30,
            )
            resp.raise_for_status()
            pages = resp.json()["results"]

        return [self._parse_page(p) for p in pages]

    # ── Subtasks (to-do blocks inside a task page) ──────────────

    async def add_subtask(self, page_id: str, subtask_text: str) -> bool:
        """Add a to-do item (subtask) as a block inside a task page."""
        body = {
            "children": [
                {
                    "object": "block",
                    "type": "to_do",
                    "to_do": {
                        "rich_text": [{"text": {"content": subtask_text}}],
                        "checked": False,
                    },
                }
            ]
        }
        async with httpx.AsyncClient() as client:
            resp = await client.patch(
                f"{NOTION_API}/blocks/{page_id}/children",
                headers=HEADERS,
                json=body,
                timeout=30,
            )
            resp.raise_for_status()
            logger.info(f"Added subtask to {page_id}: {subtask_text}")
            return True

    async def get_subtasks(self, page_id: str) -> list[dict]:
        """Get all to-do blocks (subtasks) from a task page."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{NOTION_API}/blocks/{page_id}/children",
                headers=HEADERS,
                timeout=30,
            )
            resp.raise_for_status()
            blocks = resp.json()["results"]

        subtasks = []
        for b in blocks:
            if b["type"] == "to_do":
                text = "".join(
                    t["text"]["content"] for t in b["to_do"]["rich_text"]
                )
                subtasks.append({
                    "id": b["id"],
                    "text": text,
                    "checked": b["to_do"]["checked"],
                })
        return subtasks

    # ── Task field updates ────────────────────────────────────

    async def update_task_deadline(self, page_id: str, deadline: str) -> dict:
        """Update task deadline."""
        body = {"properties": {"Deadline": {"date": {"start": deadline}}}}
        async with httpx.AsyncClient() as client:
            resp = await client.patch(
                f"{NOTION_API}/pages/{page_id}",
                headers=HEADERS,
                json=body,
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()

    async def update_task_project(self, page_id: str, project: str) -> dict:
        """Update task project."""
        body = {"properties": {"Project": {"select": {"name": project}}}}
        async with httpx.AsyncClient() as client:
            resp = await client.patch(
                f"{NOTION_API}/pages/{page_id}",
                headers=HEADERS,
                json=body,
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()

    async def get_active_tasks_summary(self) -> str:
        """Get a formatted summary of active tasks grouped by project for digest messages."""
        tasks = await self.get_all_tasks()
        active = [t for t in tasks if t["status"] != "Готово"]

        if not active:
            return "Все задачи выполнены!"

        # Group by project
        projects: dict[str, list] = {}
        for t in active:
            proj = t["project"] or "Общее"
            projects.setdefault(proj, []).append(t)

        overdue_count = 0
        lines = []
        priority_order = {"high": 0, "medium": 1, "low": 2}
        priority_label = {"high": "!!!", "medium": "!!", "low": "!"}

        for proj, proj_tasks in sorted(projects.items()):
            proj_tasks.sort(key=lambda t: (
                t.get("deadline") or "9999-99-99",
                priority_order.get(t.get("priority", ""), 2),
            ))
            lines.append(f"\n<b>{proj}</b>")
            for i, t in enumerate(proj_tasks, 1):
                prio = priority_label.get(t["priority"], "")
                dl = t.get("deadline") or ""
                if dl:
                    from datetime import date as d
                    try:
                        dl_date = datetime.fromisoformat(dl).date()
                        delta = (dl_date - d.today()).days
                        if delta < 0:
                            dl_str = f"-{abs(delta)}д!"
                            overdue_count += 1
                        elif delta == 0:
                            dl_str = "сегодня!"
                        elif delta == 1:
                            dl_str = "завтра"
                        elif delta <= 7:
                            dl_str = f"{delta}д"
                        else:
                            dl_str = dl_date.strftime("%d.%m")
                    except ValueError:
                        dl_str = dl
                else:
                    dl_str = "нет"
                lines.append(f"  {i}. {t['title']}  {dl_str}  {prio}")

        header = f"<b>Активные задачи — {len(active)}</b>"
        if overdue_count:
            header += f"\nПросрочено: {overdue_count}"

        return header + "\n".join(lines)

    # ── Helpers ────────────────────────────────────────────────

    def _parse_page(self, page: dict) -> dict:
        props = page["properties"]
        return {
            "id": page["id"],
            "title": self._get_title(props.get("Name", {})),
            "status": self._get_select(props.get("Status", {})),
            "priority": self._get_select(props.get("Priority", {})),
            "project": self._get_select(props.get("Project", {})),
            "deadline": self._get_date(props.get("Deadline", {})),
            "tags": self._get_multi_select(props.get("Tags", {})),
            "url": page.get("url", ""),
        }

    @staticmethod
    def _get_title(prop: dict) -> str:
        items = prop.get("title", [])
        return items[0]["text"]["content"] if items else ""

    @staticmethod
    def _get_select(prop: dict) -> str:
        sel = prop.get("select")
        return sel["name"] if sel else ""

    @staticmethod
    def _get_multi_select(prop: dict) -> list[str]:
        return [s["name"] for s in prop.get("multi_select", [])]

    @staticmethod
    def _get_date(prop: dict) -> Optional[str]:
        d = prop.get("date")
        return d["start"] if d else None

    @staticmethod
    def _get_rich_text(prop: dict) -> str:
        items = prop.get("rich_text", [])
        return items[0]["text"]["content"] if items else ""


# Singleton
notion = NotionTaskBoard()
