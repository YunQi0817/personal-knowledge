"""批量入库 CLI。

用法::

    python ingest.py                    # 导入 docs/ 下的全部支持文件
    python ingest.py path/to/file.txt   # 只导入指定文件
    python ingest.py path/to/folder     # 导入指定目录（递归）
    python ingest.py --uploads          # 重灌 data/uploads/ 里的上传件
    python ingest.py --summaries        # 预生成各章节摘要（可选，一次性）

幂等性：``doc_id`` 取自**原始字节**哈希，因此重复执行不会产生重复数据，
只会覆盖同内容的旧记录。

``--uploads`` 用于 ``drop_collection()`` 清库后一键恢复，或在解析器升级后
重新入库——它从 ``data/uploads/{doc_id}.json`` 读回**原始文件名**，
因此恢复出来的文档名不会变成一串哈希。

**摘要默认不在入库时生成**：每个章节一次模型调用，会把入库从 4 秒拖到 26 秒，
还把入库绑死在外部 API 的稳定性上。摘要改为首次"总结"时惰性生成并缓存
（见 ``app/services/summarize.py``）。想在导入后一次性算好，用 ``--summaries``。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from app import config
from app.core import milvus
from app.services import ingest


def collect_files(target: Path) -> list[Path]:
    """收集待入库文件：目录则递归，文件则单个。"""
    if target.is_file():
        return [target]
    if not target.exists():
        return []
    return sorted(
        path
        for path in target.rglob("*")
        if path.is_file() and path.suffix.lower() in ingest.SUPPORTED_SUFFIXES
    )


def reingest_uploads() -> int:
    """从 ``data/uploads/`` 重灌全部上传件，保留原始文件名。"""
    if not milvus.ping():
        print("无法连接 Milvus。请先启动 Docker Desktop，")
        print("然后在 E:\\Milvus 执行 standalone.bat start")
        return 2

    results = ingest.reingest_all()
    if not results:
        print(f"{config.UPLOADS_DIR} 下没有身份文件（*.json），无可重灌内容。")
        return 1

    print(f"从 {config.UPLOADS_DIR} 重灌 {len(results)} 份文档")
    print("-" * 60)
    failed = 0
    for item in results:
        if item.get("error"):
            failed += 1
            print(f"  [失败] {item['name']}  →  {item['error']}")
        else:
            print(f"  [OK]   {item['name']}  →  {item['chunks']} 块"
                  f"  (doc_id {item['doc_id']})")
    print("-" * 60)
    print(f"完成：成功 {len(results) - failed}，失败 {failed}")
    print(f"集合当前片段总数：{milvus.count_chunks()}")
    return 0 if failed == 0 else 3


def prebuild_summaries() -> int:
    """为知识库所有章节预生成摘要（一次性，之后"总结"即可秒回）。"""
    if not milvus.ping():
        print("无法连接 Milvus。请先启动 Docker Desktop，")
        print("然后在 E:\\Milvus 执行 standalone.bat start")
        return 2

    from app.services import summarize

    print("正在为各章节生成摘要（每个章节一次模型调用，请稍候）…")
    started = time.time()
    sections = summarize.ensure()
    elapsed = time.time() - started
    filled = sum(1 for item in sections if item.get("summary"))
    print("-" * 60)
    print(f"完成：{filled}/{len(sections)} 个章节有摘要，耗时 {elapsed:.1f}s")
    return 0 if filled == len(sections) else 3


def main(argv: list[str]) -> int:
    if len(argv) > 1 and argv[1] in {"--uploads", "-u"}:
        return reingest_uploads()
    if len(argv) > 1 and argv[1] == "--summaries":
        return prebuild_summaries()

    target = Path(argv[1]).resolve() if len(argv) > 1 else config.DOCS_DIR
    files = collect_files(target)

    if not files:
        print(f"没有找到可入库的文件：{target}")
        print(f"支持的扩展名：{', '.join(ingest.SUPPORTED_SUFFIXES)}")
        return 1

    if not milvus.ping():
        print("无法连接 Milvus。请先启动 Docker Desktop，")
        print("然后在 E:\\Milvus 执行 standalone.bat start")
        return 2

    print(f"待入库 {len(files)} 个文件，集合：{config.DB_NAME}.{config.COLLECTION_NAME}")
    print("-" * 60)

    total_chunks = 0
    failed = 0
    for path in files:
        try:
            result = ingest.ingest_file(path)
            total_chunks += result["chunks"]
            if result["chunks"]:
                print(
                    f"  [OK]   {path.name}  →  {result['chunks']} 块"
                    f"  (doc_id {result['doc_id']})"
                )
            else:
                print(f"  [空]   {path.name}  →  无可提取文本，已跳过")
        except Exception as exc:  # noqa: BLE001 - 单个文件失败继续处理其余
            failed += 1
            print(f"  [失败] {path.name}  →  {type(exc).__name__}: {exc}")

    print("-" * 60)
    print(f"完成：写入 {total_chunks} 个片段，失败 {failed} 个文件")
    try:
        print(f"集合当前片段总数：{milvus.count_chunks()}")
        print(f"文档篇数：{len(milvus.list_documents())}")
    except Exception as exc:  # noqa: BLE001
        print(f"统计失败：{exc}")
    return 0 if failed == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
