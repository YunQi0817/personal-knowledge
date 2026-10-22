"""文档解析层：把任意格式的文件解析成统一的 ``Chunk`` 列表。

这是「通用知识库」的**契约层**。每个格式的解析器都实现同一个签名::

    parse(path: Path) -> list[Chunk]

新增一种格式 = 写一个模块 + 在 ``_PARSERS`` 里登记一行。
未登记的格式自动走 ``legacy`` 兜底，因此逐个替换的过程中不会丢失已有能力。

契约的核心是 ``Chunk.headings``——每个块都必须知道自己属于文档的哪个章节。
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable, Union

from app.services.parsers import docx, excel, legacy, plaintext, pptx
from app.services.parsers.base import Chunk, split_overflow, with_heading

__all__ = [
    "Chunk",
    "SUPPORTED_SUFFIXES",
    "parse",
    "parse_bytes",
    "split_overflow",
    "with_heading",
]

#: 扩展名 -> 解析器函数。**新增格式只需在这里加一行。**
_PARSERS: dict[str, Callable[[Union[str, Path]], list[Chunk]]] = {}
for _module in (plaintext, docx, pptx, excel):
    for _suffix in _module.SUFFIXES:
        _PARSERS[_suffix] = _module.parse
for _suffix in legacy.SUFFIXES:  # setdefault：已登记的格式不被兜底覆盖
    _PARSERS.setdefault(_suffix, legacy.parse)

#: 全部支持的扩展名（供 Streamlit file_uploader 的 type 参数复用）
SUPPORTED_SUFFIXES: list[str] = sorted(_PARSERS)


def parse(path: Union[str, Path], *, source: str | None = None) -> list[Chunk]:
    """解析文件。

    Args:
        path: 待解析文件。
        source: 覆盖块的来源名；上传场景下用它记录用户看到的真实文件名。
    """
    target = Path(path)
    handler = _PARSERS.get(target.suffix.lower(), legacy.parse)
    chunks = handler(target)
    name = source or target.name
    for chunk in chunks:
        chunk.source = name
    return chunks


def parse_bytes(name: str, data: bytes) -> list[Chunk]:
    """解析字节流：按原后缀落一个临时文件，解析完立即删除。"""
    suffix = Path(name).suffix.lower() or ".txt"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(data)
        tmp_path = Path(handle.name)
    try:
        return parse(tmp_path, source=name)
    finally:
        tmp_path.unlink(missing_ok=True)
