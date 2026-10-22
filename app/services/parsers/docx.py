"""Word 解析器（``.docx``）。

理想情况下，Word 文档里每个段落都带 ``style.name``（``Heading 1`` / 中文版的
``标题 1``），这是**天然且 100% 准确的层级信号**。

但真实文档常常不理想。实测某份 WPS 出的策划书：

===========================  ======
``Normal``                   139 个段落
``WPSOffice手动目录 1``         11 个（目录条目）
``Heading 2``                  7 个
===========================  ======

11 个章节只有 7 个套了标题样式，而且**通篇没有 Heading 1**。于是本模块需要处理
三类脏数据，它们各自对应下面一个函数：

1. ``_is_toc`` —— **目录条目**。它罗列了全部章节名，因而"对任何章节类问题都
   完美匹配"，会把真正的内容块挤出 top-K。样式名里带"目录/TOC"，或形如
   ``标题\\t页码`` 的行，都按目录丢弃。
2. ``_normalize`` —— **层级归一化**。通篇只有 ``Heading 2`` 时，``stack[:level-1]``
   会把所有章节嵌套成父子链（``活动简介 > 二、活动背景 > 三、活动内容``）。
   必须先扫出全文实际用到的最小层级，把它重映射为 1。
3. ``_numbered_heading`` —— **无样式标题**。靠中文序号识别 ``四、活动的目的与意义``
   这类漏套样式的顶级章节。

另外两个必须绕开的坑：

* ``document.paragraphs`` 与 ``document.tables`` 是**两个独立列表**，分开遍历会
  丢失文档顺序（表格全跑到最后）。必须按底层 XML 的 body 子元素顺序遍历。
* ``paragraph.text`` **不包含超链接里的文字**（python-docx 的 ``runs`` 只取直接
  子节点 ``w:r``，而超链接文字在 ``w:hyperlink`` 里）。这里改为自己遍历 ``w:t``
  节点收集，顺带正确处理制表符与换行。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Union
import re

from docx import Document
from docx.oxml.ns import qn
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.services.parsers.base import Chunk, SectionBuilder

#: 本解析器负责的扩展名
SUFFIXES = {".docx"}

#: ``Heading 1`` / ``标题 1`` 这类带编号的标题样式
_HEADING_STYLE = re.compile(r"^\s*(?:heading|标题)\s*(\d+)\s*$", re.IGNORECASE)

#: 不带编号的标题类样式，一律按一级处理
_TITLE_STYLES = {"title", "标题", "subtitle", "副标题"}

#: 目录段落的样式名特征
_TOC_STYLE = re.compile(r"目录|\btoc\b", re.IGNORECASE)

#: 目录条目：一行文字 + 制表符 + 页码
_TOC_LINE = re.compile(r"^.{2,45}\t\s*\d+\s*$")

#: 无样式顶级章节标题：``一、活动目的和意义``
#:
#: **只认「一、」这一级。** 实测 ``（一）主办方：西南财经大学天府学院易班工作站``
#: 与 ``（一）指导思想`` 在文本上无法区分（都是短行 + 中文序号），
#: 对「（一）」做启发式会误判一大片列表项。
_NUMBERED = re.compile(r"^\s*[一二三四五六七八九十]+\s*[、.．]\s*\S")

#: 上面那条规则允许的最大长度；超过就不像标题了
_NUMBERED_MAX_CHARS = 30

#: 二级条目：``（一）`` ``（二）`` 这种中文序号
_SUB_ITEM = re.compile(r"^\s*[（(]\s*[一二三四五六七八九十]+\s*[）)]")

#: 收集文字时整棵跳过的子树：修订删除内容、域代码、批注锚点
_SKIP_TAGS = {qn("w:del"), qn("w:instrText"), qn("w:delInstrText")}

_TEXT, _TAB, _BREAK = qn("w:t"), qn("w:tab"), (qn("w:br"), qn("w:cr"))


# ------------------------------------------------------------------ 遍历


def _iter_blocks(document: Any) -> Iterator[Any]:
    """按文档真实顺序产出 ``Paragraph`` 与 ``Table``。"""
    for child in document.element.body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, document)
        elif isinstance(child, CT_Tbl):
            yield Table(child, document)


def _collect(node: Any, parts: list[str]) -> None:
    for child in node:
        if child.tag in _SKIP_TAGS:
            continue
        if child.tag == _TEXT:
            parts.append(child.text or "")
        elif child.tag == _TAB:
            parts.append("\t")
        elif child.tag in _BREAK:
            parts.append("\n")
        else:
            _collect(child, parts)


def _paragraph_text(paragraph: Paragraph) -> str:
    """段落全文，**含超链接文字**。"""
    parts: list[str] = []
    _collect(paragraph._element, parts)
    return "".join(parts).strip()


# ------------------------------------------------------------------ 标题判定


def _style_name(paragraph: Paragraph) -> str:
    style = paragraph.style
    return ((style.name or "") if style is not None else "").strip()


def _styled_level(paragraph: Paragraph) -> int | None:
    """**只看样式**给出的层级；没有则 ``None``。"""
    name = _style_name(paragraph)
    match = _HEADING_STYLE.match(name)
    if match:
        return int(match.group(1))
    if name.lower() in _TITLE_STYLES:
        return 1
    # 自定义样式名：退回大纲级别（OOXML 里 0 表示一级）
    try:
        outline = paragraph.style.paragraph_format.outline_level
    except Exception:  # noqa: BLE001 - 某些样式没有段落格式
        outline = None
    if outline is not None and 0 <= int(outline) <= 8:
        return int(outline) + 1
    return None


def _is_toc(paragraph: Paragraph, text: str) -> bool:
    """是不是目录条目。目录必须整条丢弃——见模块文档。"""
    if _TOC_STYLE.search(_style_name(paragraph)):
        return True
    return bool(_TOC_LINE.match(text))


def _numbered_heading(text: str) -> bool:
    """无样式段落里的**顶级**章节标题。"""
    if len(text) > _NUMBERED_MAX_CHARS:
        return False
    if text.endswith(("：", ":")):  # 标题不会以冒号收尾，这是"键：值"的样子
        return False
    return bool(_NUMBERED.match(text))


def _heading_level(
    paragraph: Paragraph, text: str, base_level: int
) -> int | None:
    """段落是标题则返回**原始**层级，否则 ``None``。

    启发式命中的标题返回 ``base_level``，也就是"文档里其他章节标题用的那一级"，
    归一化后自然成为一级。
    """
    styled = _styled_level(paragraph)
    if styled is not None:
        return styled
    if _numbered_heading(text):
        return base_level
    return None


# ------------------------------------------------------------------ 表格


def _table_to_text(table: Table) -> str:
    """两列表格转「键：值」句子，其余转 Markdown 表格。

    两列表格本质就是键值对，转成句子远比 Markdown 表利于检索——
    向量能看到「知识库数量上限：10 个」，而不是一串裸露的词。
    """
    rows = [
        [(cell.text or "").strip().replace("\n", " ") for cell in row.cells]
        for row in table.rows
    ]
    rows = [row for row in rows if any(row)]
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]

    if width == 2:
        return "\n".join(f"{key}：{value}".rstrip("：") for key, value in rows)

    header = "| " + " | ".join(rows[0]) + " |"
    rule = "| " + " | ".join("---" for _ in range(width)) + " |"
    return "\n".join([header, rule, *("| " + " | ".join(r) + " |" for r in rows[1:])])


# ------------------------------------------------------------------ 入口


def parse(path: Union[str, Path]) -> list[Chunk]:
    """解析 Word 文档，按 ``Heading`` 样式（辅以序号启发式）划分章节。"""
    target = Path(path)
    document = Document(str(target))
    blocks = list(_iter_blocks(document))

    # 第一遍：找出全文**实际用到**的最小标题层级，后面据此归一化
    styled_levels = [
        level
        for block in blocks
        if isinstance(block, Paragraph)
        for level in [_styled_level(block)]
        if level is not None
    ]
    base_level = min(styled_levels) if styled_levels else 1

    builder = SectionBuilder(source=target.name)
    for block in blocks:
        if isinstance(block, Table):
            text = _table_to_text(block)
            if text:
                builder.body(text, content_type="table")
            continue

        text = _paragraph_text(block)
        if not text or _is_toc(block, text):
            continue

        level = _heading_level(block, text, base_level)
        if level is not None:
            builder.heading(level - base_level + 1, text)
            continue

        if _SUB_ITEM.match(text):
            # 二级条目（``（二）活动时间：…``）**先切一刀再入正文**。
            #
            # 为什么是"先切"：这样每条 ``（N）`` 会和它**后面**那些没有编号的
            # 段落归到一块。反过来先 body 再 flush 的话，上一条会跟这一条粘住。
            #
            # 为什么需要切：``活动简介`` 的 8 个条目原本挤在一个 191 字的块里，
            # 查「活动时间」要跟主办方/地点/类型/级别/指导老师/负责人/参与者
            # 一起竞争，实测正确答案排到第 7 名。
            #
            # 过短的条目（如 ``（五）活动级别：校级活动``）会被 SectionBuilder
            # 挂起、并入下一条，不会产生碎片——那正是 MIN_SECTION_CHARS 的职责。
            builder.flush()

        builder.body(text)

    return builder.chunks()
