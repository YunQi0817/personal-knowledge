"""个人知识库 · AI 对话界面（Streamlit）。

启动（**必须**用 streamlit run，不要用 python 直接执行）::

    streamlit run app/ui/streamlit_app.py

架构说明：Streamlit 与后端**同进程**，直接调用 ``app.services``，
不经过 HTTP / SSE，因此项目里没有 api 层。

界面结构：

    侧边栏  新建对话 / 对话记录（可删除）/ 知识库管理 / 设置 / 连接状态
    主区域  欢迎页（空对话时）或对话流；输入框由 Streamlit 固定在底部

**对话记录持久化在 SQLite**（``data/conversations.db``），因此重启项目后
历史对话仍在。原先放在 ``st.session_state`` 里，刷新页面就丢。

命名约定：本文件**不能**命名为 ``app.py``。它所在目录会被加入 ``sys.path``，
若其中存在 ``app.py``，``import app`` 就会解析到这个文件本身而不是 ``app/`` 包。
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 把项目根目录置于 sys.path 最前，否则 `from app.xxx import ...` 会失败。
#
# 这里必须**无条件**插到位置 0，而不能写成"不存在才插入"：
# IDE（如 PyCharm）会把项目根注入 PYTHONPATH，此时根目录确实已在 sys.path 中，
# 但位置排在脚本目录**之后**；一旦脚本目录里存在与顶层包同名的模块，
# `import app` 就会命中那个文件而不是 app/ 包，报错 "app is not a package"。
# ---------------------------------------------------------------------------
_ROOT = str(Path(__file__).resolve().parents[2])
while _ROOT in sys.path:
    sys.path.remove(_ROOT)
sys.path.insert(0, _ROOT)

import streamlit as st  # noqa: E402

from app import config  # noqa: E402
from app.core import llm, milvus  # noqa: E402
from app.services import conversations as chat_store  # noqa: E402
from app.services import ingest, qa  # noqa: E402
from app.ui import theme  # noqa: E402

st.set_page_config(
    page_title="个人知识库",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="expanded",
)

theme.apply()

#: 角色 -> 头像
AVATAR = {"user": "🧑", "assistant": "✨"}

#: 欢迎页的建议问题
SUGGESTIONS = (
    "订阅套餐有哪些？",
    "四档的区别是什么？",
    "基础版多少钱？",
    "试用版到期后数据会立刻删除吗？",
)


# ============================================================== 状态与回调


def _bootstrap() -> None:
    """保证会话状态完整，并指向一个真实存在的会话。"""
    state = st.session_state
    state.setdefault("pending", None)  # 点了建议问题、等待本轮处理
    state.setdefault("pending_delete", None)  # 已点过一次删除、等待确认的会话 id
    state.setdefault("summary_request", None)  # ("library" | "document", doc_id)

    existing = {item["id"] for item in chat_store.list_all()}
    if state.get("active") not in existing:
        # 首次进入，或当前会话已被删除
        state.active = chat_store.latest_id() or chat_store.create()


# 下列回调交给 ``on_click``：回调在脚本重跑**之前**执行，因此本轮渲染时
# 状态已经是最新的，既不用手动 st.rerun()，也不会出现"点了按钮画面还是旧的"。


def _start_new() -> None:
    """新建会话。先清掉空壳，免得反复点「新建对话」在库里堆垃圾记录。"""
    chat_store.delete_empty()
    st.session_state.active = chat_store.create()
    st.session_state.pending_delete = None


def _switch_to(conversation_id: str) -> None:
    st.session_state.active = conversation_id
    st.session_state.pending_delete = None


def _ask(question: str) -> None:
    st.session_state.pending = question


def _request_summary(kind: str, doc_id: str | None = None) -> None:
    """请求生成总结：``library`` 整库总览，``document`` 单份文档概要。

    总结**不走检索**——它要的是完整而不是相似（见 ``qa.summarize_document``
    的说明）。所以这里只是记下意图，由主区域去调对应的汇总函数。
    """
    st.session_state.summary_request = (kind, doc_id)


def _doc_name(document: dict) -> str:
    """文档在界面上显示的名字。"""
    source = document.get("source") or ""
    return Path(source).name or source


def _delete_clicked(conversation_id: str) -> None:
    """第一次点击只是"上膛"，再点一次才真正删除。"""
    state = st.session_state
    if state.get("pending_delete") != conversation_id:
        state.pending_delete = conversation_id
        return

    chat_store.delete(conversation_id)
    state.pending_delete = None
    if state.get("active") == conversation_id:
        state.active = chat_store.latest_id() or chat_store.create()


@st.cache_data(ttl=30, show_spinner=False)
def kb_snapshot() -> dict:
    """一次拿齐知识库状态。

    缓存 30 秒是为了避免每一轮重渲染都去打 Milvus；入库/删除后会显式
    ``st.cache_data.clear()`` 让它立即失效。返回字典而**不抛异常**——
    Milvus 没启动时界面应当给出提示，而不是整页崩掉。
    """
    try:
        if not milvus.ping():
            return {"ok": False, "chunks": 0, "documents": [], "error": "Milvus 未连接"}
        return {
            "ok": True,
            "chunks": milvus.count_chunks(),
            "documents": milvus.list_documents(),
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001 - 状态查询失败不该拖垮界面
        return {"ok": False, "chunks": 0, "documents": [], "error": str(exc)}


# ============================================================== 渲染


def render_sources(sources: list[dict] | None) -> None:
    """把引用来源渲染成卡片列表。"""
    if not sources:
        return
    with st.expander(f"📎　引用来源 · {len(sources)} 条"):
        for index, item in enumerate(sources, start=1):
            name = Path(item.get("source") or "").name or "未知"
            section = item.get("section") or ""
            score = item.get("score")
            score_text = f"{score:.3f}" if isinstance(score, (int, float)) else "—"
            section_html = (
                f'<span class="kb-src-sec">{html.escape(section)}</span>' if section else ""
            )
            st.markdown(
                f"""<div class="kb-src">
  <div class="kb-src-head">
    <span class="kb-src-idx">{index}</span>
    <span class="kb-src-name">{html.escape(name)}</span>
    {section_html}
    <span class="kb-src-score">{score_text}</span>
  </div>
  <div class="kb-src-body">{html.escape(item.get("snippet", ""))}</div>
</div>""",
                unsafe_allow_html=True,
            )


def render_hero(snapshot: dict) -> None:
    """空对话时的欢迎页。"""
    if snapshot["ok"]:
        subtitle = (
            f"已收录 {len(snapshot['documents'])} 篇文档 · "
            f"{snapshot['chunks']} 个片段"
        )
    else:
        subtitle = "知识库尚未连接"

    st.markdown(
        f"""<div class="kb-hero">
  <div class="kb-hero-icon">✨</div>
  <div class="kb-hero-title">个人知识库</div>
  <div class="kb-hero-sub">{html.escape(subtitle)}</div>
</div>""",
        unsafe_allow_html=True,
    )

    st.write("")
    left, right = st.columns(2)
    for index, question in enumerate(SUGGESTIONS):
        column = left if index % 2 == 0 else right
        column.button(
            question,
            key=f"suggest_{index}",
            use_container_width=True,
            on_click=_ask,
            args=(question,),
        )


# ============================================================== 取输入

_bootstrap()
conversation_id = st.session_state.active

# 先取输入。st.chat_input 由 Streamlit 固定在页面底部，因此可以提前调用——
# 这样侧栏的会话标题、主区域的欢迎页在**同一轮**内就能反映最新状态，
# 不会出现"刚提完问、侧栏还写着『新对话』"的滞后。
prompt = st.chat_input("就知识库内容提问…")
if st.session_state.pending:
    prompt = st.session_state.pending
    st.session_state.pending = None

# 历史必须在追加本轮提问**之前**取，否则当前问题会被重复拼一次
history: list[dict[str, str]] = []
if prompt:
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in chat_store.messages(conversation_id)
    ]
    chat_store.append(conversation_id, "user", prompt)

snapshot = kb_snapshot()
sidebar_items = chat_store.list_all()


# ============================================================== 侧边栏

with st.sidebar:
    st.button(
        "＋　新建对话",
        type="primary",
        use_container_width=True,
        on_click=_start_new,
    )

    st.markdown('<div class="kb-label">对话记录</div>', unsafe_allow_html=True)
    for item in sidebar_items:
        active = item["id"] == conversation_id
        armed = st.session_state.pending_delete == item["id"]
        title_col, action_col = st.columns([5, 1], gap="small")

        title_col.button(
            ("● " if active else "") + item["title"],
            key=f"conv_{item['id']}",
            use_container_width=True,
            on_click=_switch_to,
            args=(item["id"],),
        )
        action_col.button(
            "确认" if armed else "✕",
            # 上膛后换 key，好让 CSS 把这一颗单独标红
            key=(f"convdel_ok_{item['id']}" if armed else f"convdel_{item['id']}"),
            on_click=_delete_clicked,
            args=(item["id"],),
            help="再点一次就删除这条对话" if armed else "删除这条对话",
        )

    # ---------------------------------------------------------- 知识库
    with st.expander(
        f"📚　知识库 · {len(snapshot['documents'])} 篇 / {snapshot['chunks']} 块",
        expanded=False,
    ):
        # ---------- 总结入口 ----------
        # 总结不是"提问"，而是一个动作，所以放在这里而不是输入框
        if snapshot["ok"] and snapshot["documents"]:
            st.button(
                "📋　知识库总览",
                use_container_width=True,
                on_click=_request_summary,
                args=("library",),
                help="读全部章节摘要，汇总出整个知识库讲了什么",
            )
            doc_names = {_doc_name(doc): doc["doc_id"] for doc in snapshot["documents"]}
            picked = st.selectbox(
                "总结单份文档",
                ["（选择一份文档…）", *doc_names],
                label_visibility="collapsed",
                key="summary_pick",
            )
            if picked != "（选择一份文档…）":
                st.button(
                    "📋　总结这份文档",
                    use_container_width=True,
                    on_click=_request_summary,
                    args=("document", doc_names[picked]),
                )
            st.divider()

        uploaded = st.file_uploader(
            "上传文档",
            type=[suffix.lstrip(".") for suffix in ingest.SUPPORTED_SUFFIXES],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if st.button("入库", disabled=not uploaded, use_container_width=True):
            for file in uploaded or []:
                try:
                    with st.spinner(f"处理 {file.name} …"):
                        result = ingest.ingest_upload(file.name, file.getvalue())
                    if result["chunks"]:
                        st.success(f"{file.name} → {result['chunks']} 个片段")
                    else:
                        st.warning(f"{file.name}：未提取到有效文本")
                except Exception as exc:  # noqa: BLE001 - 单个文件失败不影响其余
                    st.error(f"{file.name} 入库失败：{exc}")
            st.cache_data.clear()  # 让 kb_snapshot 立即反映新数据
            st.rerun()

        if snapshot["documents"]:
            st.divider()
            for document in snapshot["documents"]:
                label = _doc_name(document)
                name_col, action_col = st.columns([5, 2], gap="small")
                name_col.markdown(
                    f"<div style='font-size:.82rem;line-height:1.9'>{html.escape(label)}"
                    f"<span style='color:rgba(128,128,128,1)'> · {document['chunks']} 块</span></div>",
                    unsafe_allow_html=True,
                )
                if action_col.button("删除", key=f"docdel_{document['doc_id']}"):
                    try:
                        ingest.delete_document(document["doc_id"])
                        st.cache_data.clear()
                        st.rerun()
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"删除失败：{exc}")

    # ------------------------------------------------------------ 设置
    with st.expander("⚙️　设置", expanded=False):
        limit = st.slider(
            "检索片段数", min_value=1, max_value=10, value=config.RETRIEVE_LIMIT
        )
        st.caption(f"对话模型　`{llm.model.model_name}`")
        st.caption(f"嵌入模型　`{config.EMBED_MODEL_NAME}`")

    # ------------------------------------------------------------ 状态
    st.divider()
    if snapshot["ok"]:
        st.caption(
            f"🟢 Milvus 已连接　`{config.DB_NAME}.{config.COLLECTION_NAME}`"
        )
    else:
        st.caption(f"🔴 {snapshot['error']}")
        st.caption(f"地址：`{config.MILVUS_URI}`")
        st.caption("启动 Docker Desktop 后，在 `E:\\Milvus` 执行 `standalone.bat start`")


# ============================================================== 主区域

# ---------------------------------------------------------- 总结请求
# 侧栏点了"总览/总结"后在这里执行。结果作为一条助手消息写进对话记录，
# 于是它和普通问答一样会被持久化、可以回看。
request = st.session_state.get("summary_request")
if request:
    st.session_state.summary_request = None
    kind, target = request
    with st.spinner("正在汇总…"):
        try:
            if kind == "library":
                result = qa.summarize_library()
                heading = "知识库总览"
            else:
                result = qa.summarize_document(str(target))
                heading = f"《{result.get('title') or target}》内容概要"
            body = (result.get("summary") or "").strip()
            content = (
                f"## {heading}\n\n{body}"
                if body
                else f"## {heading}\n\n（汇总失败或知识库为空，请检查模型配置与入库情况）"
            )
        except Exception as exc:  # noqa: BLE001 - 汇总失败也要落成一条消息，别静默
            content = f"## 汇总失败\n\n{exc}"
    chat_store.append(conversation_id, "assistant", content)

# 本轮提问已在上方写入数据库，这里统一从库里读，天然包含新消息
current_messages = chat_store.messages(conversation_id)
if not current_messages:
    render_hero(snapshot)

for message in current_messages:
    with st.chat_message(message["role"], avatar=AVATAR[message["role"]]):
        if message["role"] == "user":
            # 自己套一层容器做气泡：Streamlit 不提供按角色定位消息的钩子
            # （见 theme.py 中 .kb-user 的说明），只能在这里区分
            st.markdown(
                f'<div class="kb-user">{html.escape(message["content"])}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(message["content"])
        render_sources(message.get("sources"))

if prompt:
    with st.chat_message("assistant", avatar=AVATAR["assistant"]):
        # 不用 st.write_stream：Agent 会先花几秒调工具，这期间要能更新状态提示，
        # 所以手动累积文本并逐步重绘。
        status = st.status("正在理解问题…", expanded=False)
        holder = st.empty()
        buffer = ""
        used: list[str] = []
        try:
            for kind, payload in qa.stream_events(prompt, history=history):
                if kind == "tool":
                    label = qa.TOOL_LABELS.get(payload["name"], payload["name"])
                    if payload["phase"] == "start":
                        used.append(label)
                        status.update(label=f"正在{label}…", state="running")
                elif kind == "text":
                    buffer += payload
                    holder.markdown(buffer + "▌")
            holder.markdown(buffer)
            # 折叠成一行"用过的工具"，让用户看到 Agent 做了什么
            done = "、".join(dict.fromkeys(used)) or "直接回答（未使用工具）"
            status.update(label=done, state="complete", expanded=False)

            sources = qa.format_sources(qa.take_search_log())
            render_sources(sources)
            chat_store.append(conversation_id, "assistant", buffer, sources)
        except Exception as exc:  # noqa: BLE001 - 把后端错误显示到界面
            status.update(label=f"生成失败：{exc}", state="error")
            st.error(f"生成失败：{exc}")
