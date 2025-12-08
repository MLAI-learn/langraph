# db_tools.py
import sqlite3
import datetime
from typing import Optional, List, Tuple

DEFAULT_DB = "tasks_langgraph.db"

def connect_db(path: str = DEFAULT_DB):
    conn = sqlite3.connect(path, check_same_thread=False)
    init_db(conn)
    return conn

def init_db(conn: sqlite3.Connection):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT,
            category TEXT,
            priority TEXT,
            due_date TEXT,
            completed INTEGER DEFAULT 0,
            created_at TEXT,
            updated_at TEXT
        )
        """
    )
    conn.commit()

def add_task(conn: sqlite3.Connection, title: str, description: str = "", category: str = "general",
             priority: str = "medium", due_date: Optional[str] = None) -> int:
    now = datetime.datetime.utcnow().isoformat()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO tasks (title, description, category, priority, due_date, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (title, description, category, priority, due_date, now, now),
    )
    conn.commit()
    return cur.lastrowid

def list_tasks(conn: sqlite3.Connection, include_completed: bool = False) -> List[Tuple]:
    q = "SELECT id, title, description, category, priority, due_date, completed FROM tasks"
    if not include_completed:
        q += " WHERE completed = 0"
    q += " ORDER BY created_at DESC"
    cur = conn.cursor()
    cur.execute(q)
    return cur.fetchall()

def complete_task(conn: sqlite3.Connection, task_id: int) -> int:
    cur = conn.cursor()
    cur.execute("UPDATE tasks SET completed = 1, updated_at = ? WHERE id = ?", (datetime.datetime.utcnow().isoformat(), task_id))
    conn.commit()
    return cur.rowcount

def delete_task(conn: sqlite3.Connection, task_id: int) -> int:
    cur = conn.cursor()
    cur.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
    conn.commit()
    return cur.rowcount

def search_tasks(conn: sqlite3.Connection, query: str):
    cur = conn.cursor()
    pattern = f"%{query}%"
    cur.execute("SELECT id, title, description, category, priority, due_date, completed FROM tasks WHERE title LIKE ? OR description LIKE ? ORDER BY created_at DESC", (pattern, pattern))
    return cur.fetchall()
