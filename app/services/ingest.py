from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Union

# 必须最先导入：由它引导 NLTK_DATA，否则 Unstructured 系列会回退到
# %APPDATA%\nltk_data 并尝试联网下载。
from app import config  # noqa: F401

from app.core import milvus
from app.core.embeddings import embed_model
from app.services import parsers, summarize
from app.services.parsers import Chunk

logger = logging.getLogger(__name__)

#: 支持的扩展名（供 Streamlit file_uploader 的 type 参数复用）
#:
#: 权威定义在 ``app.services.parsers``——每个扩展名对应一个解析器；
#: 这里只是转出，避免界面与入库两处各维护一份清单。
SUPPORTED_SUFFIXES: list[str] = parsers.SUPPORTED_SUFFIXES


# ================================================================== 工具


def compute_doc_id(raw: bytes) -> str:
    """用**原始字节**的哈希前 16 位作为文档的稳定标识。

    刻意基于原始字节而不是解析后的文本：解析器或切分参数变化时 ``doc_id``
    保持不变，重复入库仍是幂等覆盖。若基于解析结果，换一种切分策略就会算出
    新的 ``doc_id``，旧记录不再被覆盖，反而变成检索得到的"幽灵片段"。
    """
    return hashlib.sha1(raw).hexdigest()[:16]


# ================================================================== 入库


def ingest_documents(chunks: list[Chunk], *, raw: bytes) -> dict[str, Any]:
    """核心入库：向量化 → 写入 Milvus。

    Args:
        chunks: 解析器产出的块（已带 ``headings``）。
        raw: 原始文件字节，**仅**用于计算稳定 ``doc_id``。

    Returns:
        ``{"doc_id": str, "chunks": int}``；无可提取文本时 ``chunks`` 为 0 且不写库。
    """
    usable = [c for c in chunks if (c.text or "").strip()]
    if not usable:
        return {"doc_id": "", "chunks": 0}

    doc_id = compute_doc_id(raw)

    # 向量化过程
    vectors = embed_model.embed_documents([c.text for c in usable])

    # 构建数据。稀疏向量不用管：BM25 Function 会在写入时从 text 自动生成。
    # 章节摘要**不在这里生成**——它每个章节要一次 LLM 调用，会把入库从 4 秒
    # 拖到 26 秒，还把入库绑死在外部 API 的稳定性上。改成惰性生成 + 缓存，
    # 见 app/services/summarize.py。
    data = [
        {
            "id": milvus.make_primary_key(doc_id, i),
            milvus.DENSE_FIELD: vectors[i],
            "text": chunk.text,
            "source": chunk.source,
            "section": chunk.section,
            "parent_text": chunk.parent_text,
            "chunk_id": i,
        }
        for i, chunk in enumerate(usable)
    ]

    milvus.insert_data(data)
    # 内容变了，这份文档的旧章节摘要作废
    summarize.forget(doc_id)
    return {"doc_id": doc_id, "chunks": len(data)}


def _display_source(target: Path) -> str:
    """把本地路径转成相对项目根的写法，例如 ``docs/knowledge.txt``。

    统一语义：**界面上传记文件名，本地导入记相对项目根的路径**——两者都不是
    绝对路径，换台机器、换个盘符都还能读。界面再用 ``Path(source).name`` 取基名。
    """
    try:
        return str(target.resolve().relative_to(config.BASE_DIR))
    except ValueError:  # 项目目录之外的文件，只能原样记
        return str(target)


def ingest_file(path: Union[str, Path]) -> dict[str, Any]:
    """入库本地文件（不复制原件，``source`` 记相对项目根的路径）。"""
    target = Path(path)
    chunks = parsers.parse(target, source=_display_source(target))
    return ingest_documents(chunks, raw=target.read_bytes())


def ingest_upload(name: str, data: bytes) -> dict[str, Any]:
    """入库界面上传的文件，并把**原始字节**落盘到 ``data/uploads/{doc_id}{后缀}``。

    落盘保的是原始格式与"可重跑性"，正文本身已经存在 Milvus 的 ``text`` 字段里。
    """
    # parse_bytes 内部落临时文件；块的 source 会被设成用户看到的真实文件名
    chunks = parsers.parse_bytes(name, data)
    result = ingest_documents(chunks, raw=data)
    saved: Path | None = None
    if result["chunks"]:
        saved = _save_original(result["doc_id"], name, data)
    return {**result, "saved": saved}


def _save_original(doc_id: str, name: str, data: bytes) -> Path:
    """保存原件，并写一份**身份文件** ``{doc_id}.json``。

    为什么需要身份文件：原件按 ``doc_id``（内容哈希）命名，**原始文件名就丢了**。
    一旦 ``drop_collection()`` 清库，光有原件是恢复不出友好名字的——内容回来了，
    身份丢了。记下名字与出处之后，``data/uploads/`` 才是一个**自包含的可重灌仓库**
    （见 ``reingest_all()``），而不只是"一堆哈希命名的孤儿文件"。
    """
    suffix = Path(name).suffix.lower() or ".txt"
    path = config.UPLOADS_DIR / f"{doc_id}{suffix}"
    path.write_bytes(data)

    manifest = {
        "doc_id": doc_id,
        "name": name,
        "suffix": suffix,
        "size": len(data),
        "sha1": hashlib.sha1(data).hexdigest(),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    (config.UPLOADS_DIR / f"{doc_id}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def reingest_all() -> list[dict[str, Any]]:
    """从 ``data/uploads/`` 重灌全部原件，**保留原始文件名**。

    用途有二：``drop_collection()`` 清库后一键恢复；解析器升级后重新入库。

    幂等——``doc_id`` 由内容哈希得出，重复执行只是覆盖同一批主键。
    返回逐条结果，失败的记 ``error`` 而不中断其余文件。
    """
    results: list[dict[str, Any]] = []
    for manifest_path in sorted(config.UPLOADS_DIR.glob("*.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            results.append(
                {"name": manifest_path.name, "chunks": 0, "error": f"身份文件损坏：{exc}"}
            )
            continue

        doc_id = manifest.get("doc_id") or manifest_path.stem
        suffix = manifest.get("suffix") or ".txt"
        name = manifest.get("name") or f"{doc_id}{suffix}"
        original = config.UPLOADS_DIR / f"{doc_id}{suffix}"

        if not original.exists():
            results.append({"name": name, "chunks": 0, "error": f"原件缺失：{original.name}"})
            continue
        try:
            result = ingest_upload(name, original.read_bytes())
        except Exception as exc:  # noqa: BLE001 - 单个文件失败不中断其余
            results.append({"name": name, "chunks": 0, "error": f"{type(exc).__name__}: {exc}"})
            continue
        results.append({**result, "name": name})
    return results


# ================================================================== 文档管理


def list_documents() -> list[dict[str, Any]]:
    """列出已入库文档。"""
    return milvus.list_documents()


def delete_document(doc_id: str) -> dict[str, Any]:
    """删除一份文档：先删 Milvus 记录，再清理落盘文件。

    顺序很关键——Milvus 记录是检索依据，一旦残留就会出现"已删除的文档仍被检索到"。
    因此先删向量；落盘文件删除失败只记警告，最多留个孤儿文件。
    """
    deleted_chunks = milvus.delete_document(doc_id)
    summarize.forget(doc_id)  # 文档没了，它的章节摘要缓存也没意义了

    removed_files: list[str] = []
    for path in config.UPLOADS_DIR.glob(f"{doc_id}.*"):
        try:
            path.unlink()
            removed_files.append(path.name)
        except OSError as exc:
            logger.warning("删除落盘文件失败 %s: %s", path, exc)

    return {"deleted_chunks": deleted_chunks, "removed_files": removed_files}
