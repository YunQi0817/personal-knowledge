"""暴露给 Agent 的工具集。

**为什么需要这一层**：固定管道（``qa.stream_answer``）无论用户问什么都先检索一次、
且只检索一次——问「知识库里有哪些文档」它也会去做向量检索。Agent 化之后，
要不要检索、检索几次、先列文档还是先检索、知识库答不了要不要联网，
都由模型自己决定。

工具函数上方的 docstring 就是模型看到的**工具说明**，所以写得越具体，
选错工具的概率越低。
"""

from __future__ import annotations

import os
from pathlib import Path

from langchain.tools import tool

from app.core import milvus


def _doc_label(doc: dict) -> str:
    """文档的展示名：优先文件名，退回 doc_id。"""
    source = str(doc.get("source", ""))
    return Path(source).name or str(doc.get("doc_id", ""))


#: 记录本次问答中 search_knowledge 命中的**原始**结果，供界面渲染引用。
#:
#: 说明：这是模块级状态，面向**单用户本地应用**。Streamlit 每个会话独立跑脚本，
#: 但模块级变量是跨会话共享的——将来要做多用户并发，需要改成按会话隔离，
#: 或改为从工具返回内容里解析引用。
_SEARCH_LOG: list[dict] = []


def take_search_log() -> list[dict]:
    """取出并清空检索日志（原始 milvus 命中）。"""
    hits = list(_SEARCH_LOG)
    _SEARCH_LOG.clear()
    return hits


@tool
def search_knowledge(query: str, top_k: int = 3) -> str:
    """在个人知识库中检索与问题相关的资料片段。

    任何涉及知识库内容的问题，都必须先调用本工具获取资料，不要凭自身记忆回答。
    返回若干片段，每段都标注了来源文件名与 chunk_id，引用时请使用这些标注。
    """
    # 延迟导入：qa → agent → tools 之间存在环，模块级导入会炸
    from app.services import qa

    hits = qa.retrieve(query, limit=top_k)
    _SEARCH_LOG.extend(hits)
    if not hits:
        return "知识库中没有检索到相关内容。"
    return qa.build_context(hits)


@tool
def list_documents() -> str:
    """列出知识库中已入库的全部文档及其片段数量。

    当用户问「知识库里有哪些文档」「都上传了什么资料」时调用本工具，
    不要用 search_knowledge 去猜。
    """
    docs = milvus.list_documents()
    if not docs:
        return "知识库当前是空的。"
    lines = [f"知识库共 {len(docs)} 篇文档："]
    for index, doc in enumerate(docs, start=1):
        lines.append(
            f"{index}. {_doc_label(doc)}"
            f"（doc_id={doc.get('doc_id')}，{doc.get('chunks')} 个片段）"
        )
    return "\n".join(lines)


@tool
def summarize_document(name: str) -> str:
    """总结知识库中的某一份文档。

    name 传文档名的一部分即可，例如「策划书」「knowledge」。
    当用户要求「总结 / 概括某份文档讲了什么」时调用。
    本工具按文档**全量**章节摘要汇总，不要用 search_knowledge 代替。
    """
    from app.services import qa

    docs = milvus.list_documents()
    if not docs:
        return "知识库当前是空的，无法总结。"

    keyword = (name or "").strip().lower()
    matched = [doc for doc in docs if keyword and keyword in _doc_label(doc).lower()]
    if not matched:
        available = "、".join(_doc_label(doc) for doc in docs)
        return f"没有找到名字包含「{name}」的文档。现有文档：{available}"

    doc = matched[0]
    result = qa.summarize_document(str(doc.get("doc_id", "")))
    title = result.get("title") or _doc_label(doc)
    return f"《{title}》章节摘要汇总：\n{result.get('summary', '')}"


@tool
def web_search(query: str) -> str:
    """联网搜索外部信息。

    仅在「知识库检索后确认没有相关资料」且「问题需要外部或最新信息」时调用。
    不要用它回答知识库里已有的内容。
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "未配置 TAVILY_API_KEY，无法联网搜索。"
    try:
        from langchain_tavily import TavilySearch

        raw = TavilySearch(max_results=3, api_key=api_key).invoke({"query": query})
    except Exception as exc:  # noqa: BLE001
        return f"联网搜索失败：{type(exc).__name__}: {exc}"

    items = raw.get("results") if isinstance(raw, dict) else raw
    if not items:
        return "联网没有搜到相关结果。"

    blocks = []
    for index, item in enumerate(items[:3], start=1):
        title = item.get("title", "")
        url = item.get("url", "")
        content = (item.get("content") or "")[:300]
        blocks.append(f"[网络{index}] {title}\n{url}\n{content}")
    return "\n\n".join(blocks)


#: 交给 Agent 的完整工具列表
ALL_TOOLS = [search_knowledge, list_documents, summarize_document, web_search]

#: 只依赖知识库、不联网的工具子集（离线或不想引入外部信息时用）
LOCAL_TOOLS = [search_knowledge, list_documents, summarize_document]
