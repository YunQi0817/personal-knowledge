from __future__ import annotations

import threading
from typing import Any

from app import config

# ------------------------------------------------------------------ 字段名
#: 主键（字符串，形如 "a3f2b1c9:0"）
PRIMARY_FIELD = "id"
#: 主键最大长度（Milvus 的 VARCHAR 按**字节**计）
ID_MAX_LENGTH = 128

#: 正文。**显式字段**（不是动态字段），因为 BM25 要拿它分词。
TEXT_FIELD = "text"
#: text 字段上限。中文一个字 3 字节，16384 字节约合 5400 个汉字。
TEXT_MAX_LENGTH = 16384
#: 稠密向量字段。快速建表时会自动命名成 ``vector``，显式 schema 下由我们定名。
DENSE_FIELD = "dense"
#: 稀疏向量字段。**不手写**——由下面的 BM25 Function 从 ``text`` 自动生成。
SPARSE_FIELD = "sparse"
#: BM25 函数名
BM25_FUNCTION = "bm25"
#: 中文分析器（Milvus 服务端内置，不需要额外装分词库）
ANALYZER = {"type": "chinese"}

#: 动态字段，与教程命名一致
SOURCE_FIELD = "source"
CHUNK_ID_FIELD = "chunk_id"
#: 章节路径。由解析层注入（见 ``app/services/parsers/base.py`` 的 ``Chunk.section``）。
SECTION_FIELD = "section"
#: 所属章节的**完整文本**。Small-to-Big 的载体：索引小块保证精确命中，
#: 送进模型的是这个字段（完整上下文）。
PARENT_FIELD = "parent_text"

#: 检索时需要回传的字段
OUTPUT_FIELDS = [TEXT_FIELD, CHUNK_ID_FIELD, SOURCE_FIELD, SECTION_FIELD, PARENT_FIELD]

#: 混合检索的两种模式
MODE_HYBRID = "hybrid"
MODE_DENSE = "dense"

_lock = threading.RLock()
_client: Any = None


class MilvusNotAvailableError(RuntimeError):
    """连不上 Milvus 时抛出，附带可操作的修复提示。"""


# ================================================================== 连接


def get_client() -> Any:
    """连接 Milvus；数据库不存在则创建，并切换过去。"""
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                from pymilvus import MilvusClient

                try:
                    client = MilvusClient(config.MILVUS_URI)
                    # 查询已有的数据库，如果不存在指定名的数据库，则进行创建
                    existed_databases = client.list_databases()
                    if config.DB_NAME not in existed_databases:
                        client.create_database(db_name=config.DB_NAME)
                    # 切换到指定的数据库
                    client.use_database(db_name=config.DB_NAME)
                    _client = client
                except Exception as exc:  # noqa: BLE001 - 统一转成带提示的错误
                    raise MilvusNotAvailableError(_hint(exc)) from exc
    return _client


def _hint(exc: Exception) -> str:
    return (
        f"无法连接 Milvus（{config.MILVUS_URI}）：{exc}\n"
        "请依次确认：\n"
        "  1) Docker Desktop 已启动；\n"
        "  2) 已在 E:\\Milvus 目录执行 standalone.bat start；\n"
        "  3) 端口 19530 可访问。"
    )


def reset_client() -> None:
    """丢弃缓存的客户端（连接异常后供 UI 重试）。"""
    global _client
    with _lock:
        _client = None


def ping() -> bool:
    """探测 Milvus 是否可用，不抛异常。"""
    try:
        get_client().list_collections()
        return True
    except Exception:  # noqa: BLE001 - 探测函数，任何异常都视为不可用
        return False


# ================================================================== 建集合


def ensure_collection() -> None:
    """建集合与索引（幂等）。

    .. note::
       **这里刻意不用** ``create_collection(dimension=…, metric_type=…)`` 那套
       快速建表简写。混合检索要求 ``text`` 是一个**显式的、开了分析器的 VARCHAR
       字段**——BM25 要拿它分词，而快速建表只会把 ``text`` 当成动态字段。
       稠密向量字段名也因此从自动生成的 ``vector`` 变成显式的 ``dense``。

    稀疏向量**不需要手写**：``BM25 Function`` 会在写入时自动从 ``text`` 生成。
    """
    client = get_client()
    if client.has_collection(collection_name=config.COLLECTION_NAME):
        return
    with _lock:
        if client.has_collection(collection_name=config.COLLECTION_NAME):
            return

        from pymilvus import DataType, Function, FunctionType, MilvusClient

        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field(
            PRIMARY_FIELD, DataType.VARCHAR, is_primary=True, max_length=ID_MAX_LENGTH
        )
        schema.add_field(
            TEXT_FIELD,
            DataType.VARCHAR,
            max_length=TEXT_MAX_LENGTH,
            enable_analyzer=True,      # BM25 分词的前提
            analyzer_params=ANALYZER,  # 中文
        )
        schema.add_field(DENSE_FIELD, DataType.FLOAT_VECTOR, dim=config.EMBED_DIM)
        schema.add_field(SPARSE_FIELD, DataType.SPARSE_FLOAT_VECTOR)
        schema.add_function(
            Function(
                name=BM25_FUNCTION,
                function_type=FunctionType.BM25,
                input_field_names=[TEXT_FIELD],
                output_field_names=[SPARSE_FIELD],
                params={},
            )
        )

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name=DENSE_FIELD, index_type="AUTOINDEX", metric_type="COSINE"
        )
        index_params.add_index(
            field_name=SPARSE_FIELD,
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
        )

        client.create_collection(
            collection_name=config.COLLECTION_NAME,
            schema=schema,
            index_params=index_params,
        )


def drop_collection() -> None:
    """删除集合（重建索引用）。"""
    client = get_client()
    if client.has_collection(collection_name=config.COLLECTION_NAME):
        client.drop_collection(collection_name=config.COLLECTION_NAME)


def describe() -> dict[str, Any]:
    """返回集合结构，便于排查。"""
    client = get_client()
    ensure_collection()
    return client.describe_collection(config.COLLECTION_NAME)


# ================================================================== 工具


def _safe_doc_id(doc_id: str) -> str:
    """校验 doc_id 为十六进制哈希，防止过滤表达式注入。"""
    text = (doc_id or "").strip()
    if not text or any(c not in "0123456789abcdefABCDEF" for c in text):
        raise ValueError(f"非法的 doc_id: {doc_id!r}")
    return text


def make_primary_key(doc_id: str, chunk_id: int) -> str:
    """主键 = ``"{doc_id}:{chunk_id}"``。"""
    return f"{doc_id}:{chunk_id}"


# ================================================================== 入库


def insert_data(data: list[dict[str, Any]]) -> dict[str, Any]:
    """写入数据：``client.upsert`` + 手动 flush。

    主键相同时为幂等覆盖，因此重复入库不会产生重复数据。
    """
    client = get_client()
    ensure_collection()
    result = client.upsert(collection_name=config.COLLECTION_NAME, data=data)
    # Milvus 不会第一时间将数据落盘，手动 flush 后才能立即检索到
    client.flush(collection_name=config.COLLECTION_NAME)
    return result


# ================================================================== 检索


def _dedupe_by_section(hits: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    """按章节去重：同一章节只占一个召回名额。

    为什么需要：检索按**小块**算相似度，同一章节的多个小块会一起排到前面
    （实测「哪些套餐不支持 OCR」的 top-3 全部来自「二、套餐说明」）。而下游
    ``qa.build_user_prompt`` 送进模型的是 ``parent_text``（整章节全文）并按
    ``(source, section)`` 去重——那第 2、3 名送进去的内容与第 1 名**完全相同**，
    属于纯粹的重复，白白占掉本该给其他章节的名额。

    在检索阶段就去重，等价于把召回粒度从「小块级 top-k」对齐成
    「章节级 top-k」，与 Small-to-Big 的设计意图一致。

    无章节的文档（纯文本且没有标题）退回按块去重，避免整篇只返回一条。
    """
    seen: set[tuple[str, str]] = set()
    picked: list[dict[str, Any]] = []
    for hit in hits:
        entity = hit.get("entity", {}) or {}
        section = str(entity.get(SECTION_FIELD) or "").strip()
        source = str(entity.get(SOURCE_FIELD, ""))
        if section:
            key = (source, section)
        else:
            # 没有章节信息时用 chunk_id 区分，等于不去重
            key = (source, f"#chunk:{entity.get(CHUNK_ID_FIELD)}")
        if key in seen:
            continue
        seen.add(key)
        picked.append(hit)
        if len(picked) >= top_k:
            break
    return picked


def search(
    query_vector: list[float],
    query_text: str = "",
    limit: int | None = None,
    mode: str = MODE_HYBRID,
) -> list[dict[str, Any]]:
    """检索。

    默认走**混合检索**：稠密向量负责语义，BM25 负责词法——日期、人名、金额、
    编号这类精确串是稠密向量的天然短板（实测 dense 把它们排在 0.3~0.6 分堆里
    毫无区分度，而 BM25 能给出唯一且干净的命中）。两路结果用 RRF 融合。

    Args:
        query_vector: 问题的稠密向量。
        query_text: 问题**原文**。BM25 那一路需要它，所以不能只传向量；
            留空时自动退化为纯稠密检索。
        limit: 最终返回条数。
        mode: ``"hybrid"``（默认）或 ``"dense"``（只走向量，便于 A/B 对比）。

    Returns:
        与教程一致的 ``results[0]``：``[{"distance": float, "entity": {...}}, ...]``

    .. warning::
       混合模式下 ``distance`` 是 **RRF 融合分**（实测量级 0.02~0.03），
        只表示排名先后，**不是相似度**，不能当阈值用。
    """
    client = get_client()
    ensure_collection()
    top_k = limit or config.RETRIEVE_LIMIT

    if mode == MODE_DENSE or not query_text.strip():
        results = client.search(
            collection_name=config.COLLECTION_NAME,
            data=[query_vector],
            # 集合里有两个向量字段（dense / sparse），不指明会报
            # "multiple anns_fields exist, please specify a anns_field"
            anns_field=DENSE_FIELD,
            search_params={"metric_type": "COSINE"},
            # 去重需要更大的候选池：只取 top_k 的话，同一章节占满名额后
            # 就没有别的章节可挑了
            limit=max(top_k * config.SECTION_DEDUPE_FETCH_FACTOR, top_k),
            output_fields=OUTPUT_FIELDS,
        )
        hits = results[0]
        if config.SECTION_DEDUPE:
            hits = _dedupe_by_section(hits, top_k)
        return hits

    from pymilvus import AnnSearchRequest, RRFRanker

    # 每路先多取一些再融合——只取 top_k 的话，某一路的正确答案若排在第 4 位
    # 就直接被丢掉了，融合无从谈起。
    fetch = max(top_k * config.HYBRID_FETCH_FACTOR, top_k)
    requests = [
        AnnSearchRequest(
            data=[query_vector],
            anns_field=DENSE_FIELD,
            param={"metric_type": "COSINE"},
            limit=fetch,
        ),
        AnnSearchRequest(
            data=[query_text],
            anns_field=SPARSE_FIELD,
            param={"metric_type": "BM25"},
            limit=fetch,
        ),
    ]
    results = client.hybrid_search(
        collection_name=config.COLLECTION_NAME,
        reqs=requests,
        ranker=RRFRanker(config.RRF_K),
        # 去重前先要回完整候选池（fetch 已放大），否则同一章节占满 top_k
        # 之后就没得挑了
        limit=fetch if config.SECTION_DEDUPE else top_k,
        output_fields=OUTPUT_FIELDS,
    )
    hits = results[0]
    if config.SECTION_DEDUPE:
        hits = _dedupe_by_section(hits, top_k)
    return hits


# ================================================================== 文档管理


def count_chunks(doc_id: str | None = None) -> int:
    """统计片段数。

    .. warning::
       **不要**用 ``get_collection_stats()["row_count"]``。Milvus 的 upsert 是
       「逻辑删除 + 插入」，被覆盖的旧行要等 compaction 后才从该计数中消失，
       因此重复入库后它会虚高（实测 11 → 22），真实行数其实没变。
       这里用 ``count(*)`` 聚合查询，只统计存活行。
    """
    client = get_client()
    ensure_collection()
    expr = f'id like "{_safe_doc_id(doc_id)}:%"' if doc_id else ""
    rows = client.query(
        collection_name=config.COLLECTION_NAME,
        filter=expr,
        output_fields=["count(*)"],
    )
    if rows and isinstance(rows[0], dict) and "count(*)" in rows[0]:
        return int(rows[0]["count(*)"])
    return 0


def list_documents() -> list[dict[str, Any]]:
    """列出已入库文档。

    主键形如 ``"{doc_id}:{chunk_id}"``，因此按 ``:`` 前缀聚合即可还原文档清单。
    """
    client = get_client()
    ensure_collection()

    agg: dict[str, dict[str, Any]] = {}
    iterator = client.query_iterator(
        collection_name=config.COLLECTION_NAME,
        filter="",
        output_fields=[PRIMARY_FIELD, SOURCE_FIELD, CHUNK_ID_FIELD],
        batch_size=1000,
    )
    try:
        while True:
            rows = iterator.next()
            if not rows:
                break
            for row in rows:
                key = str(row.get(PRIMARY_FIELD) or "")
                if ":" not in key:
                    continue
                doc_id = key.rsplit(":", 1)[0]
                current = agg.get(doc_id)
                if current is None:
                    agg[doc_id] = {
                        "doc_id": doc_id,
                        "source": row.get(SOURCE_FIELD, ""),
                        "chunks": 1,
                    }
                else:
                    current["chunks"] += 1
    finally:
        iterator.close()

    return sorted(agg.values(), key=lambda d: d["doc_id"])


def delete_document(doc_id: str) -> int:
    """删除一份文档的全部片段，返回删除条数。"""
    client = get_client()
    ensure_collection()
    safe_id = _safe_doc_id(doc_id)
    removed = count_chunks(safe_id)
    client.delete(
        collection_name=config.COLLECTION_NAME, filter=f'id like "{safe_id}:%"'
    )
    client.flush(collection_name=config.COLLECTION_NAME)
    return removed


# ================================================================== 遍历


def _iter_rows(output_fields: list[str], filter_: str = ""):
    """分页遍历全部命中行。"""
    client = get_client()
    ensure_collection()
    iterator = client.query_iterator(
        collection_name=config.COLLECTION_NAME,
        filter=filter_,
        output_fields=output_fields,
        batch_size=1000,
    )
    try:
        while True:
            rows = iterator.next()
            if not rows:
                break
            yield from rows
    finally:
        iterator.close()


def list_sections(doc_id: str | None = None) -> list[dict[str, Any]]:
    """按**章节去重**地取出 ``(doc_id, source, section, parent_text)``，按原文顺序。

    摘要模块靠它知道"有哪些章节需要摘要"。同一节的每个块存着同一份
    ``parent_text``，所以这里要去重；排序用该章节**第一块**的 ``chunk_id``，
    这样顺序和原文档一致。
    """
    expr = f'id like "{_safe_doc_id(doc_id)}:%"' if doc_id else ""
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    for row in _iter_rows(
        [PRIMARY_FIELD, SOURCE_FIELD, SECTION_FIELD, PARENT_FIELD, CHUNK_ID_FIELD],
        expr,
    ):
        key_text = str(row.get(PRIMARY_FIELD) or "")
        if ":" not in key_text:
            continue
        doc = key_text.rsplit(":", 1)[0]
        section = row.get(SECTION_FIELD) or ""
        key = (doc, section)
        order = int(row.get(CHUNK_ID_FIELD) or 0)
        current = seen.get(key)
        if current is None:
            seen[key] = {
                "doc_id": doc,
                "source": row.get(SOURCE_FIELD) or "",
                "section": section,
                "parent_text": row.get(PARENT_FIELD) or row.get(TEXT_FIELD) or "",
                "order": order,
            }
        else:
            current["order"] = min(current["order"], order)

    items = sorted(seen.values(), key=lambda item: (item["source"], item["order"]))
    for item in items:
        item.pop("order", None)
    return items


def list_chunks(doc_id: str) -> list[dict[str, Any]]:
    """按**原文顺序**取出某份文档的全部块。

    **刻意不走检索**：总结一份文档要的是"完整"，不是"与问题最相似"。
    实测走向量检索时 top-3 只覆盖约 10%，会输出一个以偏概全的假总结——
    而且它看起来很像总结，用户看不出问题。
    """
    rows = list(
        _iter_rows(
            [TEXT_FIELD, CHUNK_ID_FIELD, SOURCE_FIELD, SECTION_FIELD],
            f'id like "{_safe_doc_id(doc_id)}:%"',
        )
    )
    rows.sort(key=lambda row: int(row.get(CHUNK_ID_FIELD) or 0))
    return rows
