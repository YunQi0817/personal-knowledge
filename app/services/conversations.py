"""对话记录持久化（SQLite）。

为什么是 SQLite
---------------
界面原先把对话放在 ``st.session_state`` 里，**刷新页面就没了**。要持久化，
三种选择：

* **JSON 文件**：删除很简单，但"列出会话列表"必须读遍所有文件才能拿到标题和
  更新时间，还得额外维护一个索引并保证它与文件同步——一旦不一致就得写修复逻辑。
* **Milvus**：它是向量库，拿它存会话元数据既别扭，又会把"看历史对话"绑死在
  Milvus 在线这一件事上。Milvus 没启动就连历史都读不了，显然不合理。
* **SQLite**：标准库自带、单文件、事务安全，列表与删除各一条 SQL 就够，
  且不依赖任何外部服务。个人知识库的量级（几十到几千条会话）完全够用。

数据落在 ``data/conversations.db``，可以用任何 SQLite 客户端直接查看。
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import closing, contextmanager
from datetime import datetime
from typing import Any, Iterator

from app import config

#: 会话标题最多取多少个字
TITLE_LIMIT = 18

#: 新建会话的占位标题；首条用户消息到来时会被替换
DEFAULT_TITLE = "新对话"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id         TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    sources         TEXT NOT NULL DEFAULT '[]',
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation
    ON messages(conversation_id, id);
"""


# ------------------------------------------------------------------ 连接


@contextmanager
def _session() -> Iterator[sqlite3.Connection]:
    """每次操作开一个短连接：Streamlit 会跨线程重跑脚本，长连接不安全。"""
    connection = sqlite3.connect(str(config.CONVERSATIONS_DB), timeout=10)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def _init() -> None:
    """建表（幂等）。模块导入时执行一次。"""
    with closing(sqlite3.connect(str(config.CONVERSATIONS_DB))) as connection:
        connection.executescript(_SCHEMA)
        connection.commit()


def _now() -> str:
    """本地时间，形如 ``2026-10-18T23:20:11``——字符串排序即时间排序。"""
    return datetime.now().isoformat(timespec="seconds")


def _title_from(text: str) -> str:
    stripped = (text or "").strip()
    if not stripped:
        return DEFAULT_TITLE
    first_line = stripped.splitlines()[0]
    return first_line[:TITLE_LIMIT] + ("…" if len(first_line) > TITLE_LIMIT else "")


_init()


# ------------------------------------------------------------------ 会话


def create(title: str = DEFAULT_TITLE) -> str:
    """新建会话，返回其 id。"""
    conversation_id = uuid.uuid4().hex[:8]
    now = _now()
    with _session() as connection:
        connection.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at)"
            " VALUES (?, ?, ?, ?)",
            (conversation_id, title, now, now),
        )
    return conversation_id


def list_all() -> list[dict[str, Any]]:
    """按最近更新倒序列出全部会话，附带消息条数。"""
    with _session() as connection:
        rows = connection.execute(
            """
            SELECT c.id, c.title, c.created_at, c.updated_at,
                   (SELECT COUNT(*) FROM messages m
                     WHERE m.conversation_id = c.id) AS message_count
              FROM conversations c
             ORDER BY c.updated_at DESC, c.rowid DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def messages(conversation_id: str) -> list[dict[str, Any]]:
    """按时间正序取一个会话的全部消息。"""
    with _session() as connection:
        rows = connection.execute(
            "SELECT role, content, sources FROM messages"
            " WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        try:
            item["sources"] = json.loads(item["sources"] or "[]")
        except json.JSONDecodeError:
            item["sources"] = []
        result.append(item)
    return result


def append(
    conversation_id: str,
    role: str,
    content: str,
    sources: list[dict[str, Any]] | None = None,
) -> None:
    """追加一条消息；首条用户消息会顺带把会话标题定下来。"""
    now = _now()
    with _session() as connection:
        existing = connection.execute(
            "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()[0]
        connection.execute(
            "INSERT INTO messages (conversation_id, role, content, sources, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (
                conversation_id,
                role,
                content,
                json.dumps(sources or [], ensure_ascii=False),
                now,
            ),
        )
        connection.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conversation_id),
        )
        if role == "user" and existing == 0:
            connection.execute(
                "UPDATE conversations SET title = ? WHERE id = ?",
                (_title_from(content), conversation_id),
            )


def rename(conversation_id: str, title: str) -> None:
    with _session() as connection:
        connection.execute(
            "UPDATE conversations SET title = ? WHERE id = ?",
            (title.strip() or DEFAULT_TITLE, conversation_id),
        )


def delete(conversation_id: str) -> int:
    """删除会话及其全部消息，返回删掉的消息条数。"""
    with _session() as connection:
        removed = connection.execute(
            "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()[0]
        connection.execute(
            "DELETE FROM messages WHERE conversation_id = ?", (conversation_id,)
        )
        connection.execute(
            "DELETE FROM conversations WHERE id = ?", (conversation_id,)
        )
    return int(removed)


def delete_empty() -> int:
    """清掉一条消息都没有的空会话。

    用户可能反复点「新建对话」，不清理的话每次都会在库里留一个空壳。
    """
    with _session() as connection:
        cursor = connection.execute(
            "DELETE FROM conversations WHERE id NOT IN"
            " (SELECT DISTINCT conversation_id FROM messages)"
        )
    return cursor.rowcount or 0


def latest_id() -> str | None:
    """最近一次更新的会话 id；一条都没有时返回 ``None``。"""
    items = list_all()
    return items[0]["id"] if items else None
