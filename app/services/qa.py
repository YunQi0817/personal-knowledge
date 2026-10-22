"""检索并生成回答。

写法完全参照教程 ``chapter10-RAG/05-案例`` 的
「7、检索」与「8、生产与回答生成」两段：

    def retrieve(query, limit=3):
        query_vector = embed_model.embed_query(str(query))
        results = client.search(
            collection_name=COLLECTION_NAME,
            data=[query_vector],
            limit=limit,
            output_fields=["text","chunk_id","source"],
        )
        return results[0]

    def generate_answer(query):
        hits = retrieve(query, limit=5)
        context_blocks = []
        for i, hit in enumerate(hits, 1):
            text = hit["entity"]["text"]
            source = hit["entity"].get("source", "unknown")
            chunk_id = hit["entity"].get("chunk_id", "unknown")
            context_blocks.append(f"[片段{i} | chunk_id={chunk_id} | source={source}]\\n{text}")
        context = "\\n\\n".join(context_blocks)
        user_prompt = f"问题：\\n{query}\\n\\n上下文：\\n{context}\\n"
        result = agent.invoke({"messages":[{"role":"user","content":user_prompt}]})

**两处差异**（均已确认）：

1. 教程有一行笔误 ``hits = retrieve(str, limit=5)``——传入了内置类型 ``str``
   而非 ``query``。这里按 ``query`` 写，不照搬笔误。
2. 教程用 ``agent.invoke()`` 一次拿完整答案；这里改用 ``agent.stream()``
   以支持 Streamlit 逐字输出。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Sequence

from app import config
from app.core import milvus
from app.core.embeddings import embed_model
from app.core.llm import agent
from app.services import summarize

#: 送进上下文的字符预算；某一节超预算时就退回命中到的小块，
#: 免得一个超大章节把其他章节全挤出去。
CONTEXT_BUDGET = 4000


def retrieve(
    query: str, limit: int | None = None, mode: str = milvus.MODE_HYBRID
) -> list[dict[str, Any]]:
    """检索：默认**混合检索**（稠密语义 + BM25 词法，RRF 融合）。

    BM25 那一路需要问题**原文**，所以除了向量还要把 query 传下去。
    """
    # 将此问题向量化
    query_vector = embed_model.embed_query(str(query))
    # 从向量数据库中检索数据
    return milvus.search(
        query_vector, str(query), limit=limit or config.RETRIEVE_LIMIT, mode=mode
    )


def build_user_prompt(query: str, hits: Sequence[dict[str, Any]]) -> str:
    """组装 user_prompt：片段块格式与教程一致。

    **Small-to-Big**：索引的是小块（``Chunk.text``，便于精确命中），但送进模型的
    是它**所属章节的完整文本**（``parent_text``）。这是实测出来的——某策划书只喂
    小块时 12 题集命中 9/12，改喂所属章节则是 11/12：小块让「活动时间」排到第 1 名，
    而完整章节让「活动流程」这类需要整段上下文的问题不至于被切碎。

    同一章节的多个命中只保留一份，否则一个章节会重复占满上下文。
    """
    context_blocks: list[str] = []
    seen: set[tuple[str, str]] = set()
    budget = CONTEXT_BUDGET

    for hit in hits:
        entity = hit["entity"]
        source = entity.get("source", "unknown")
        section = entity.get("section", "")
        key = (source, section)
        if key in seen:
            continue

        # 优先给完整章节；超出预算就退回这次真正命中的小块
        text = entity.get("parent_text") or entity.get("text") or ""
        if len(text) > budget:
            text = entity.get("text") or text
        if budget <= 0:
            break
        seen.add(key)
        budget -= len(text)

        name = Path(source).name or source
        chunk_id = entity.get("chunk_id", "unknown")
        context_blocks.append(
            f"[片段{len(context_blocks) + 1} | chunk_id={chunk_id} | source={name}]\n{text}"
        )

    # 将多个上下文片段用换行符连成一个大字符串
    context = "\n\n".join(context_blocks)

    # 构造 Prompt
    return f"""问题：
{query}

上下文：
{context}
"""


def format_sources(hits: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """整理成界面可展示的引用格式。"""
    return [
        {
            "source": hit["entity"].get("source", ""),
            "chunk_id": hit["entity"].get("chunk_id"),
            "section": hit["entity"].get("section", ""),
            "score": hit.get("distance"),
            "snippet": (hit["entity"].get("text", "") or "")[:120],
        }
        for hit in hits
    ]


def stream_answer(
    query: str,
    hits: list[dict[str, Any]] | None = None,
    history: list[dict[str, str]] | None = None,
    limit: int | None = None,
) -> Iterator[str]:
    """流式产出回答。

    ``hits`` 允许调用方预先传入，这样界面可以先拿到来源、再流式输出，
    不必为了展示引用而重复检索一次。

    教程用 ``agent.invoke`` 一次返回；这里用 ``agent.stream`` 逐段产出，
    事件形如 ``(AIMessageChunk, metadata)``。
    """
    if hits is None:
        hits = retrieve(query, limit)

    user_prompt = build_user_prompt(query, hits)

    messages: list[dict[str, str]] = list(history or [])
    messages.append({"role": "user", "content": user_prompt})

    for event in agent.stream({"messages": messages}, stream_mode="messages"):
        # stream_mode="messages" 产出 (消息块, 元数据) 元组
        message = event[0] if isinstance(event, tuple) else event
        text = getattr(message, "content", "") or ""
        if text:
            yield text


def answer(
    query: str,
    hits: list[dict[str, Any]] | None = None,
    history: list[dict[str, str]] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """非流式问答，便于脚本内自测与批量评估。"""
    if hits is None:
        hits = retrieve(query, limit)
    text = "".join(stream_answer(query, hits=hits, history=history))
    return {"answer": text, "sources": format_sources(hits), "hits": hits}


# ================================================================== 总结
#
# 与上面的问答走**完全不同的路径**：问答是"检索 top-K 再生成"，总结是
# "取全量（或全量摘要）再生成"。原因是需求不同——问答要的是相关性，
# 总结要的是完整性。实测把"总结这份文档"当成普通提问时，top-3 只覆盖约 10%，
# 模型会给出一个看起来很像总结、实则以偏概全的结果。


def summarize_document(doc_id: str) -> dict[str, Any]:
    """总结**一份文档**。

    章节摘要**惰性生成并缓存**：第一次调用会花约 25 秒生成，之后秒回。
    汇总本身只有 **1 次**模型调用，文档再长也不会撑爆上下文。
    """
    sections = summarize.ensure(doc_id)
    if not sections:
        return {"doc_id": doc_id, "title": "", "summary": "", "sections": []}
    source = sections[0].get("source") or ""
    name = Path(source).name or source
    return {
        "doc_id": doc_id,
        "title": name,
        "summary": summarize.document_overview(name, sections),
        "sections": sections,
    }


def summarize_library() -> dict[str, Any]:
    """总结**整个知识库**。

    读所有章节的摘要再汇总，同样只有 1 次模型调用——这正是"预生成摘要"的价值。
    """
    sections = summarize.ensure()
    documents = len({item.get("doc_id", "") for item in sections})
    return {
        "documents": documents,
        "sections": sections,
        "summary": summarize.library_overview(sections, documents),
    }
