"""章节摘要：生成与缓存。

设计取舍：**惰性生成 + 缓存**，而不是入库时预生成
--------------------------------------------------
这个结论是实测出来的。某 48 KB 的 docx（31 块 / 11 节）：

    解析     0.1s
    向量化   0.5s
    写入     3.5s      ← 入库本身很快
    摘要    21.8s      ← 独占 84% 的时间

摘要每个章节要一次 LLM 调用，把入库从 4 秒拖到 26 秒；更糟的是**整个入库被绑在
外部 API 的稳定性上**——连续两批（21 次调用）实测触发过限流重试，入库卡了十几分钟
没有任何输出。

改成惰性之后：

    入库         仍然 4 秒
    第一次总结   多花约 25 秒（现生成）
    之后         秒回（命中缓存）

**最终状态与"入库时预生成"完全一样**——摘要都落盘了，只是时机从"入库"挪到
"第一次需要"。附带好处是从不为从没被总结过的文档付费。

缓存为什么放 SQLite
-------------------
摘要需要**增量回填**（先生成缺的那几条）。而 Milvus 的 ``upsert`` 需要整行数据
（含 1024 维向量），为写一段摘要重新上传向量既慢又别扭。SQLite 一条 UPDATE 就够了。
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import closing, contextmanager
from datetime import datetime
from typing import Any, Iterator, Sequence

from app import config
from app.core import llm, milvus

logger = logging.getLogger(__name__)

#: 单节送进模型的字符上限（摘要只需知道"讲了什么"，不必读完整节）
SECTION_INPUT_MAX = 1500

#: 摘要输出的字符上限
SUMMARY_MAX_CHARS = 140

#: 并发调用数。实测单次 1.5~3 秒，并发 4 时 11 节约 22 秒。
SUMMARY_CONCURRENCY = 4

_SCHEMA = """
CREATE TABLE IF NOT EXISTS section_summaries (
    doc_id     TEXT NOT NULL,
    section    TEXT NOT NULL,
    summary    TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (doc_id, section)
);
"""

_SECTION_PROMPT = """下面是一份文档中「{title}」这一节的内容。

{body}

请用一句不超过 60 字的话概括这一节讲了什么。只输出那句话本身，不要任何前缀、标题或解释。"""

_LIBRARY_PROMPT = """下面是一个个人知识库中所有章节的摘要清单。

{catalog}

请写一份知识库总览，要求：
1. 先说明库里包含哪几份文档，各自大致讲什么；
2. 再概括整体覆盖了哪些主题；
3. 只依据上面的清单，不要编造清单里没有的内容；
4. **总长控制在 400 字以内**——这是速览，不是详细报告。"""

_DOCUMENT_PROMPT = """下面是一份文档《{name}》各章节的摘要。

{catalog}

请写一份这份文档的内容概要，要求：
1. 按章节顺序组织；
2. 尽量保留关键数字、日期、人名等具体信息；
3. 只依据上面的摘要，不要编造摘要里没有的内容；
4. **总长控制在 500 字以内**。"""


# ------------------------------------------------------------------ 缓存


@contextmanager
def _session() -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(str(config.SUMMARIES_DB), timeout=10)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def _init() -> None:
    with closing(sqlite3.connect(str(config.SUMMARIES_DB))) as connection:
        connection.executescript(_SCHEMA)
        connection.commit()


_init()


def _cached() -> dict[tuple[str, str], str]:
    with _session() as connection:
        rows = connection.execute(
            "SELECT doc_id, section, summary FROM section_summaries"
        ).fetchall()
    return {(row["doc_id"], row["section"]): row["summary"] for row in rows}


def _store(items: dict[tuple[str, str], str]) -> None:
    if not items:
        return
    now = datetime.now().isoformat(timespec="seconds")
    with _session() as connection:
        connection.executemany(
            "INSERT INTO section_summaries (doc_id, section, summary, created_at)"
            " VALUES (?, ?, ?, ?)"
            " ON CONFLICT(doc_id, section) DO UPDATE SET"
            " summary = excluded.summary, created_at = excluded.created_at",
            [(doc, sec, text, now) for (doc, sec), text in items.items()],
        )


def forget(doc_id: str) -> int:
    """丢掉某份文档的摘要缓存（文档被删除或重新入库时调用）。"""
    with _session() as connection:
        cursor = connection.execute(
            "DELETE FROM section_summaries WHERE doc_id = ?", (doc_id,)
        )
    return cursor.rowcount or 0


# ------------------------------------------------------------------ 工具


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + "…"


def _clean(text: Any) -> str:
    """把模型输出收敛成一行短摘要。"""
    if not isinstance(text, str):
        text = str(text or "")
    for line in text.splitlines():
        line = line.strip().lstrip("#-*· ").strip()
        if line:
            return _clip(line, SUMMARY_MAX_CHARS)
    return ""


def _catalog(items: Sequence[dict[str, Any]]) -> str:
    return "\n".join(
        f"- [{item.get('section') or '开篇'}] {item.get('summary', '')}" for item in items
    )


def _generate(missing: Sequence[dict[str, Any]]) -> dict[tuple[str, str], str]:
    """为缺摘要的章节并发生成，返回 ``{(doc_id, section): 摘要}``。"""
    prompts = [
        _SECTION_PROMPT.format(
            title=item.get("section") or "（开篇）",
            body=_clip(item.get("parent_text", ""), SECTION_INPUT_MAX),
        )
        for item in missing
    ]
    try:
        replies = llm.model.batch(
            prompts,
            config={"max_concurrency": SUMMARY_CONCURRENCY},
            return_exceptions=True,
        )
    except Exception as exc:  # noqa: BLE001 - 摘要失败不该影响总结流程本身
        logger.warning("章节摘要批量生成失败：%s", exc)
        return {}

    result: dict[tuple[str, str], str] = {}
    for item, reply in zip(missing, replies):
        if isinstance(reply, BaseException):
            logger.warning(
                "章节「%s」摘要失败：%s", item.get("section") or "(开篇)", reply
            )
            continue
        text = _clean(getattr(reply, "content", ""))
        if text:
            result[(item["doc_id"], item["section"])] = text
    return result


# ------------------------------------------------------------------ 对外


def ensure(doc_id: str | None = None) -> list[dict[str, Any]]:
    """确保章节摘要齐备（缺的现生成并缓存），返回按原文顺序的章节清单。

    第一次调用会有约 25 秒的生成时间，之后命中缓存即可秒回。
    """
    sections = milvus.list_sections(doc_id)
    if not sections:
        return []

    cache = _cached()
    missing = [item for item in sections if not cache.get((item["doc_id"], item["section"]))]
    if missing:
        logger.info("生成 %d 条章节摘要（首次总结需要一点时间）…", len(missing))
        fresh = _generate(missing)
        _store(fresh)
        cache.update(fresh)

    return [
        {**item, "summary": cache.get((item["doc_id"], item["section"]), "")}
        for item in sections
    ]


def build(chunks: Sequence[Any]) -> dict[tuple[str, ...], str]:
    """按章节分组并生成摘要（**不写缓存**）。

    保留给批量预生成入口（``python ingest.py --summaries``）——想在导入后
    一次性把摘要都算好时用它。
    """
    grouped: dict[tuple[str, ...], str] = {}
    for chunk in chunks:
        key = tuple(chunk.headings)
        if key not in grouped:
            grouped[key] = chunk.parent_text or chunk.text
    if not grouped:
        return {}

    keys = list(grouped)
    missing = [
        {
            "doc_id": "",
            "section": keys[i][-1] if keys[i] else "",
            "parent_text": grouped[keys[i]],
        }
        for i in range(len(keys))
    ]
    fresh = _generate(missing)
    return {
        keys[i]: fresh.get(("", missing[i]["section"]), "") for i in range(len(keys))
    }


def library_overview(sections: Sequence[dict[str, Any]], documents: int) -> str:
    """把各节摘要汇总成一份知识库总览（**一次**模型调用）。"""
    if not sections:
        return ""
    header = f"（共 {documents} 份文档、{len(sections)} 个章节）\n"
    return _invoke_long(_LIBRARY_PROMPT.format(catalog=header + _catalog(sections)))


def document_overview(name: str, sections: Sequence[dict[str, Any]]) -> str:
    """把某份文档各节摘要汇总成内容概要（**一次**模型调用）。"""
    if not sections:
        return ""
    return _invoke_long(_DOCUMENT_PROMPT.format(name=name, catalog=_catalog(sections)))


def _invoke_long(prompt: str) -> str:
    """汇总类调用：要成段的输出，不做单行收敛。"""
    try:
        message = llm.model.invoke(prompt)
    except Exception as exc:  # noqa: BLE001
        logger.warning("汇总生成失败：%s", exc)
        return ""
    content = getattr(message, "content", "")
    return (content or "").strip() if isinstance(content, str) else str(content or "")
