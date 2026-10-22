"""纯文本 / Markdown 解析器（``.txt`` ``.log`` ``.text`` ``.md`` ``.markdown``）。

纯文本没有任何样式信号，章节只能靠**书面约定**识别。本解析器支持两种：

1. **围栏标题块**——本项目知识库使用的格式::

       ==============================
       二、套餐说明
       ==============================

   围栏三行会被归一化成一行 Markdown 标题 ``# 二、套餐说明``。
   这是修复"标题孤立"的关键：还原后标题与正文同属一节，不会被切散。

2. **Markdown ATX 标题**——``#`` 到 ``######``，按层级维护标题路径。

识别出标题后按标题边界分节，每节再按长度兜底切分，
**每个子块都会被注入自己的章节标题**。
"""

from __future__ import annotations

from pathlib import Path
from typing import Union
import re

from app import config
from app.services.parsers.base import Chunk, SectionBuilder

#: 围栏标题块：整行等号 / 标题行 / 整行等号（``$`` 在 re.M 下匹配行尾）
FENCED = re.compile(r"^={10,}$\n^(?P<title>[^\n]+?)$\n^={10,}$", re.M)

#: Markdown ATX 标题
ATX = re.compile(r"^(?P<hashes>#{1,6})[ \t]+(?P<title>\S.*?)[ \t]*$")

#: 本解析器负责的扩展名
SUFFIXES = {".txt", ".text", ".log", ".md", ".markdown"}


def _read_text(path: Path) -> str:
    """读文本，统一换行符，并依次尝试 UTF-8（含 BOM）与 GB18030。

    .. warning::
       **换行必须归一化。** Windows 上的文件是 CRLF，而正则的 ``$`` 只认 ``\\n``，
       ``^={10,}$`` 这类带锚点的模式会全部落空——实测同一份 knowledge.txt，
       原样匹配 **0** 处，CRLF→LF 后 **9** 处，章节识别直接失效。
    """
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="replace")  # 兜底，不抛异常
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):  # GB18030 兼容 GBK / GB2312
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _sections(text: str) -> list[tuple[list[str], str]]:
    """切成 ``[(标题路径, 正文), ...]``；标题路径为空表示文首前言。"""
    sections: list[tuple[list[str], str]] = []
    stack: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer)
        if body.strip():
            sections.append((list(stack), body))
        buffer.clear()

    for line in text.splitlines():
        match = ATX.match(line)
        if match:
            flush()
            # 同级标题替换、低级标题截断——维护出正确的层级路径
            level = len(match.group("hashes"))
            stack = stack[: level - 1] + [match.group("title").strip()]
        else:
            buffer.append(line)
    flush()
    return sections


def parse(path: Union[str, Path]) -> list[Chunk]:
    """解析文件，返回带章节标题的块列表。"""
    target = Path(path)

    # 围栏标题块 -> Markdown 标题（一级：同级标题互不嵌套）
    text = FENCED.sub(lambda m: f"# {m.group('title').strip()}", _read_text(target))

    # 交给 SectionBuilder：切分、超短合并、parent_text 回填都在那里统一处理，
    # 本模块只负责"把文本翻译成标题/正文事件"
    builder = SectionBuilder(source=target.name)
    for headings, body in _sections(text):
        if headings:
            builder.heading(len(headings), headings[-1])
        builder.body(body)
    return builder.chunks()
