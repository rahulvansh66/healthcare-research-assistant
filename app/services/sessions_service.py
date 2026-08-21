"""Session metadata (title, timestamps) for the sidebar's session list.

Distinct from LangGraph's Postgres checkpointer (app/agents/graph.py), which
stores the actual per-turn conversation state. This is just a thin registry
so the UI can list and switch between threads without scanning checkpoints.
"""
import psycopg

from app.config import settings

_TITLE_MAX_CHARS = 80


def init_sessions_table() -> None:
    with psycopg.connect(settings.DATABASE_URL, autocommit=True) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                thread_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)


def record_turn(thread_id: str, question: str) -> None:
    """Upsert this thread: set the title on first turn, bump updated_at every turn."""
    title = question.strip()[:_TITLE_MAX_CHARS]
    with psycopg.connect(settings.DATABASE_URL, autocommit=True) as conn:
        conn.execute(
            """
            INSERT INTO sessions (thread_id, title)
            VALUES (%s, %s)
            ON CONFLICT (thread_id) DO UPDATE SET updated_at = now()
            """,
            (thread_id, title),
        )


def list_sessions() -> list[dict]:
    with psycopg.connect(settings.DATABASE_URL, autocommit=True) as conn:
        rows = conn.execute(
            "SELECT thread_id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
        ).fetchall()
    return [
        {
            "thread_id": r[0],
            "title": r[1],
            "created_at": r[2].isoformat(),
            "updated_at": r[3].isoformat(),
        }
        for r in rows
    ]
