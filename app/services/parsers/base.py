"""块契约与切分工具。

这里只有 ``Chunk`` 这个数据结构和几个与格式无关的工具函数，
**不含任何 import 具体解析器**，因此可以被任何解析器安全导入（无循环依赖）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import config

#: 层级路径扁平化成字符串时使用的连接符
HEADING_JOINER = " > "


@dataclass(slots=True)
class Chunk:
    """统一的知识块——所有格式的解析器都必须产出这个结构。

    ``headings`` 是契约的核心：块一旦脱离章节上下文，语义就会被长正文
    稀释，检索时排不上号（见 DESIGN.md §6.5 与 §10.3）。
    """

    text: str
    headings: list[str] = field(default_factory=list)
    #: 所属章节的**完整文本**（Small-to-Big 用：小块检索、大块送模型）
    parent_text: str = ""
    source: str = ""
    page: int | None = None
    content_type: str = "text"  # text | table | code

    @property
    def section(self) -> str:
        """层级路径的扁平字符串，写入 Milvus 的 ``section`` 字段。"""
        return HEADING_JOINER.join(h for h in self.headings if h)


# ------------------------------------------------------------------ 长度兜底

_splitters: dict[int, RecursiveCharacterTextSplitter] = {}


def _splitter(budget: int) -> RecursiveCharacterTextSplitter:
    """按预算缓存切分器——标题前缀长度不同，可用预算也就不同。"""
    splitter = _splitters.get(budget)
    if splitter is None:
        splitter = _splitters[budget] = RecursiveCharacterTextSplitter(
            chunk_size=budget,
            chunk_overlap=min(config.CHUNK_OVERLAP, budget // 2),
            separators=["\n\n", "\n", "。", " ", ""],
        )
    return splitter


def split_overflow(text: str, budget: int | None = None) -> list[str]:
    """按长度兜底切分；已经够短的整块返回。

    ``budget`` 应扣掉标题前缀的长度，否则注入标题后总长会超出 ``CHUNK_SIZE``。
    """
    budget = budget or config.CHUNK_SIZE
    text = text.strip()
    if not text:
        return []
    if len(text) <= budget:
        return [text]
    return [piece for piece in _splitter(budget).split_text(text) if piece.strip()]


def with_heading(heading: str, body: str) -> str:
    """把章节标题注入块首。

    注入的字符串就是**写进向量库的正文本身**，因此标题会一起参与 embedding——
    这正是给被稀释的长块补回语义锚点的关键（DESIGN.md §10.3）。
    """
    body = body.strip()
    if not heading or not body:
        return body
    return f"【{heading}】{body}"


# ------------------------------------------------------------------ 事件组装


#: 正文短于这个字符数的节先**挂起**，并入下一节，而不是独立成块。
#:
#: 实测某份策划书里产生了正文只有 5 个字的碎片块（``【二 、活动背景】活动简介：``）。
#: 这种块没有任何检索价值，却因为"短而语义纯"，余弦分数天然偏高、反复挤进 top-3。
MIN_SECTION_CHARS = 20


def _content_len(text: str) -> int:
    """真实内容量：去掉全部空白后的字符数。"""
    return len("".join(text.split()))


class SectionBuilder:
    """按文档顺序接收「标题 / 正文」事件，产出带章节路径的 ``Chunk`` 列表。

    docx / pptx / xlsx 三个解析器要做的事完全相同：维护标题栈、把同属一节的
    正文聚起来、超长再兜底切分、每块注入标题前缀。把它收拢在这里，
    各解析器只需负责**把文件格式翻译成事件流**：

        builder = SectionBuilder(source=path.name)
        builder.heading(1, "二、套餐说明")
        builder.body("当前标准订阅套餐分为四档：…")
        builder.body("| 项目 | 数值 |\\n| --- | --- |", content_type="table")
        chunks = builder.chunks()

    ``merge_tiny`` 控制超短节的处理（见 ``MIN_SECTION_CHARS``）。默认开启；
    Excel 这类"每个行组都是刻意切开"的解析器应传 ``False``——否则一个只有
    一行的 sheet 会被挂起到下一张 sheet 里，把两张表混在一起。
    """

    def __init__(self, source: str = "", *, merge_tiny: bool = True) -> None:
        self.source = source
        self.merge_tiny = merge_tiny
        self._stack: list[str] = []
        self._buffer: list[str] = []
        self._kinds: list[str] = []
        self._carry: list[str] = []  # 挂起等待并入下一节的超短正文
        #: 章节路径 -> 该章节产出的全部正文。收尾时靠它回填 parent_text。
        #:
        #: 为什么不在 _emit 里直接算 parent：解析是**流式**的，产出第一块时
        #: 还不知道这一节后面有什么；而小节级切分又会多次调用 flush，
        #: 拿单次 payload 当 parent 会退化成"一小段"（实测踩过这个坑）。
        self._raw_by_section: dict[tuple[str, ...], list[str]] = {}
        self._page: int | None = None
        self._chunks: list[Chunk] = []

    # -- 事件 ---------------------------------------------------------------
    def heading(self, level: int, title: str, page: int | None = None) -> None:
        """开一个新章节。``level`` 从 1 起；同级替换、低级截断。"""
        self.flush()
        title = (title or "").strip()
        if page is not None:
            self._page = page
        if not title:
            return
        self._stack = self._stack[: max(level - 1, 0)] + [title]

    def body(self, text: str, content_type: str = "text", page: int | None = None) -> None:
        """往当前章节追加一段正文（聚合到下一次 flush）。"""
        text = (text or "").strip()
        if not text:
            return
        self._buffer.append(text)
        self._kinds.append(content_type)
        if page is not None:
            self._page = page

    def flush(self) -> None:
        """结束当前小节。

        正文过短时**不产出块**，而是挂起，等下一次 flush 并入下一节——
        这样既不丢内容，也不会留下无检索价值的碎片块。

        也用于「同一标题下需要切成多块」的场景，例如 Excel 每个行组都要
        独立成块并各自重复表头。
        """
        if not self._buffer:
            return

        payload = "\n\n".join(self._buffer)
        kinds = list(self._kinds)
        self._buffer.clear()
        self._kinds.clear()

        # 非空白字符数才是真实内容量
        if self.merge_tiny and _content_len(payload) < MIN_SECTION_CHARS:
            self._carry.append(payload)
            return

        if self._carry:
            payload = "\n\n".join([*self._carry, payload])
            self._carry.clear()

        self._emit(payload, kinds)

    def chunks(self) -> list[Chunk]:
        """收尾、回填 ``parent_text``，并返回全部块。"""
        self.flush()
        if self._carry:
            # 全文末尾仍有挂起内容（最后一节太短，后面没有可并入的了）：单独输出，不丢
            self._emit("\n\n".join(self._carry), ["text"] * len(self._carry))
            self._carry.clear()

        # Small-to-Big 的回填：到这一刻才知道每一节到底有多长，
        # 于是把同一节的**所有块**一律指向该节的完整文本。
        # 不能在建块时就算——解析是流式的，产出第一块时还不知道本节后面有什么。
        for chunk in self._chunks:
            parts = self._raw_by_section.get(tuple(chunk.headings))
            if parts:
                heading = chunk.headings[-1] if chunk.headings else ""
                chunk.parent_text = with_heading(heading, "\n\n".join(parts))
        return self._chunks

    # -- 内部 ---------------------------------------------------------------
    def _emit(self, payload: str, kinds: list[str]) -> None:
        heading = self._stack[-1] if self._stack else ""
        content_type = "table" if kinds and all(k == "table" for k in kinds) else "text"
        # 前缀会占用正文预算，先扣掉，避免注入后超出 CHUNK_SIZE
        prefix = len(f"【{heading}】") if heading else 0
        pieces = self._merge_tiny(
            split_overflow(payload, max(50, config.CHUNK_SIZE - prefix))
        )
        # 记下本节正文；parent_text 留到 chunks() 收尾时统一回填
        self._raw_by_section.setdefault(tuple(self._stack), []).append(payload)
        for piece in pieces:
            self._chunks.append(
                Chunk(
                    text=with_heading(heading, piece),
                    headings=list(self._stack),
                    source=self.source,
                    page=self._page,
                    content_type=content_type,
                )
            )

    def _merge_tiny(self, pieces: list[str]) -> list[str]:
        """把过短的碎片并入**相邻**的块。

        为什么要在这一层再做一次：``split_overflow`` 内部会把"短片段"攒着等
        合并，可一旦遇到超长片段，它会先把攒下的短片段吐出来再递归切分——
        于是 ``活动简介：`` 这种 5 个字的碎片照样能单独成块（与当初"标题孤立"
        是同一个机制）。这里做最后一道兜底：短块向后并，末尾的短块向前并。
        """
        if not self.merge_tiny or len(pieces) < 2:
            return pieces

        merged: list[str] = []
        for piece in pieces:
            if merged and _content_len(merged[-1]) < MIN_SECTION_CHARS:
                merged[-1] = f"{merged[-1]}\n\n{piece}"
            else:
                merged.append(piece)
        if len(merged) > 1 and _content_len(merged[-1]) < MIN_SECTION_CHARS:
            merged[-2] = f"{merged[-2]}\n\n{merged.pop()}"
        return merged
