"""兜底解析器：沿用改造前的 LangChain Loader + 定长切分。

它不认识章节结构，产出的块 ``headings`` **恒为空**。存在的意义只有一个：
保证「逐个格式替换」的过程中，其余格式的上传能力一天都不中断。

改造进度（每写完一个新格式的解析器，就从这里划掉一个）::

    已移交给专用解析器：.txt .text .log .md .markdown  (plaintext)
                        .docx                          (docx)
                        .pptx                          (pptx)
                        .xlsx .xlsm                    (excel)
    仍在此处          ：.pdf  .html  .htm  .csv  .json

**这五个格式都不依赖 ``unstructured``** —— 也就是说，把 ``.pdf`` 也换掉之后，
就可以把 ``unstructured``（连同它那 95 个直接依赖）从 requirements 里彻底移除。
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from langchain_community.document_loaders import (
    BSHTMLLoader,
    CSVLoader,
    JSONLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.document_loaders import BaseLoader

from app.services.parsers.base import Chunk, split_overflow

#: 本解析器负责的扩展名
SUFFIXES = {
    ".pdf",
    ".html",
    ".htm",
    ".csv",
    ".json",
}


def loader_for(path: Union[str, Path]) -> BaseLoader:
    """按扩展名返回对应的 LangChain Loader 实例。"""
    target = Path(path)
    suffix = target.suffix.lower()
    location = str(target)

    if suffix == ".pdf":
        return PyPDFLoader(location)
    if suffix in {".html", ".htm"}:
        return BSHTMLLoader(location, open_encoding="utf-8")
    if suffix == ".csv":
        return CSVLoader(location, encoding="utf-8")
    if suffix == ".json":
        # jq_schema="." 取根节点，text_content=False 保留 JSON 结构
        return JSONLoader(location, jq_schema=".", text_content=False)

    # 其余纯文本：自动探测编码（可处理 GBK 等）
    return TextLoader(location, autodetect_encoding=True)


def parse(path: Union[str, Path]) -> list[Chunk]:
    """解析文件，产出 ``headings`` 为空的块（定长切分）。"""
    target = Path(path)
    chunks: list[Chunk] = []
    for document in loader_for(target).load():
        page = document.metadata.get("page")
        for piece in split_overflow(document.page_content or ""):
            chunks.append(
                Chunk(
                    text=piece,
                    headings=[],
                    source=target.name,
                    page=page if isinstance(page, int) else None,
                )
            )
    return chunks
