"""界面主题：自定义 CSS。

单独一个模块，是为了让 ``streamlit_app.py`` 只关心流程，样式改动不必翻逻辑代码。

选择器策略
----------
Streamlit 的自动生成类名（``css-xxxxxx``）**每次发版都会变**，不能依赖。
这里只使用 ``data-testid`` 属性——它是 Streamlit 明确的测试契约，跨版本稳定得多。

即便如此，个别选择器（尤其是用 ``:has()`` 区分用户/助手消息的）在版本变化后
可能失配；失配的后果只是"少一层样式"，不会报错，界面依然可用。
"""

from __future__ import annotations

import streamlit as st

CSS = """
<style>
/* ================================================================ 排版 */
/* 字体只设在容器上、靠继承生效。
   **不要**用 ``[class*="st-"]`` 这类宽选择器：Streamlit 的自动类名形如
   ``st-emotion-cache-xxxx``，图标的 <span> 也带一个，于是会直接命中图标元素、
   覆盖掉它的 Material 图标字体 —— 结果是 ``keyboard_arrow_right`` 这类
   连字被当成普通文字渲染出来（侧栏里会看到 "arr⬛w_r" 之类的乱码）。 */
html, body, [data-testid="stAppViewContainer"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
                 "Microsoft YaHei", "Hiragino Sans GB", "Source Han Sans SC",
                 sans-serif;
    -webkit-font-smoothing: antialiased;
}

/* 保险：图标字体一律还原，避免将来又被别的规则波及 */
[data-testid="stIconMaterial"] {
    font-family: "Material Symbols Rounded" !important;
}

/* 隐去 Streamlit 的装饰件（部署按钮、菜单、页脚、状态浮标）
 *
 * **不要**把 [data-testid="stToolbar"] 加进下面这行。Streamlit 1.66 把侧栏
 * 折叠后的**展开按钮**放在工具栏内部：
 *
 *     stToolbar > stExpandSidebarButton          ← 展开按钮
 *     stSidebar > stSidebarHeader > stSidebarCollapseButton   ← 收起按钮
 *
 * 隐藏整个 stToolbar 会连带藏掉展开按钮，表现为「侧栏能收起、再也展不开」。
 * 工具栏本身保持可见但透明即可，里面的部署按钮/状态浮标已单独隐藏。 */
#MainMenu, footer, [data-testid="stDecoration"],
[data-testid="stStatusWidget"], [data-testid="stAppDeployButton"] {
    display: none !important;
}
[data-testid="stToolbar"] { background: transparent !important; }
[data-testid="stHeader"] { background: transparent; }

/* ============================================================ 主区域 */
/* ChatGPT 式窄栏：正文约 780px 居中 */
[data-testid="stMainBlockContainer"], .block-container {
    max-width: 780px;
    padding-top: 2.6rem;
    padding-bottom: 8rem;
}

/* ============================================================ 侧边栏 */
[data-testid="stSidebar"] {
    border-right: 1px solid rgba(128, 128, 128, .16);
}
[data-testid="stSidebarUserContent"], [data-testid="stSidebar"] .block-container {
    padding-top: 1.4rem;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: .3rem; }

/* 会话列表项：左对齐、无边框、悬停高亮 —— 像列表项而不是按钮。
   只作用于 ``st-key-conv_`` / ``st-key-convactive_`` 开头的按钮：Streamlit 会给
   带 key 的元素加上 ``st-key-<key>`` 类，用它可以精确圈定范围，
   不会波及「入库」「删除」以及侧栏折叠按钮。 */
[data-testid="stSidebar"] [class*="st-key-conv_"] button,
[data-testid="stSidebar"] [class*="st-key-convactive_"] button {
    justify-content: flex-start !important;
    border: none;
    background: transparent;
    box-shadow: none;
    padding: 0 .55rem;
    border-radius: .5rem;
    font-size: .875rem;
    font-weight: 400;
    line-height: 1.2;
    height: 2.05rem;
    min-height: 2.05rem;
    color: inherit;
    overflow: hidden;
}

/* **关键**：Streamlit 按钮内部还有**两层 flex 容器**，默认 justify-content: center。
   只把 <button> 改成 flex-start 是没用的——button 的 flex-start 只作用于它的
   直接子元素，而那层子元素本来就满宽；真正把文字挤到中间的是里面那两层。

   实测（CDP 量的 x 坐标）：短标题 x = 29 + (197 - 宽度) / 2，正是居中公式。

   这里对按钮内部**所有后代**统一左对齐，不写 emotion 生成的哈希类名
   （st-emotion-cache-xxxx），否则 Streamlit 一升级就失效。 */
[data-testid="stSidebar"] [class*="st-key-conv_"] button *,
[data-testid="stSidebar"] [class*="st-key-convactive_"] button * {
    justify-content: flex-start !important;
    text-align: left !important;
}
[data-testid="stSidebar"] [class*="st-key-conv_"] button p,
[data-testid="stSidebar"] [class*="st-key-convactive_"] button p,
[data-testid="stSidebar"] [class*="st-key-conv_"] button [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [class*="st-key-convactive_"] button [data-testid="stMarkdownContainer"] {
    width: 100% !important;
    margin: 0 !important;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

[data-testid="stSidebar"] [class*="st-key-conv_"] button:hover {
    background: rgba(128, 128, 128, .13);
    border: none;
}

/* 当前对话：浅紫底 + 左侧竖条。
   用 inset 阴影画竖条，能和圆角共存；不再往标题里塞 "●"。 */
[data-testid="stSidebar"] [class*="st-key-convactive_"] button {
    background: rgba(79, 70, 229, .10) !important;
    color: #4f46e5 !important;
    box-shadow: inset 2px 0 0 0 #4f46e5 !important;
}
[data-testid="stSidebar"] [class*="st-key-convactive_"] button p {
    font-weight: 550 !important;
}
[data-testid="stSidebar"] [class*="st-key-convactive_"] button:hover {
    background: rgba(79, 70, 229, .16) !important;
}

/* 会话行：标题与删除按钮垂直居中对齐，列间距收紧 */
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
    align-items: center;
    gap: .1rem;
}

/* 会话的删除按钮：平时低调，悬停变红。
   注意选择器前缀 ``st-key-convdel_`` 同时也会匹配 ``st-key-convdel_ok_``，
   所以"已上膛"的规则必须写在**后面**，靠同优先级后者覆盖前者生效。 */
[data-testid="stSidebar"] [class*="st-key-convdel_"] button {
    border: none;
    background: transparent;
    box-shadow: none;
    color: rgba(128, 128, 128, .5);
    padding: 0;
    height: 1.7rem;
    min-height: 1.7rem;
    width: 1.7rem;
    font-size: .78rem;
    line-height: 1;
    border-radius: .45rem;
}
[data-testid="stSidebar"] [class*="st-key-convdel_"] button:hover {
    background: rgba(239, 68, 68, .13);
    color: #dc2626;
    border: none;
}
[data-testid="stSidebar"] [class*="st-key-convdel_ok_"] button {
    border: none;
    background: #dc2626;
    color: #fff;
    font-size: .72rem;
    font-weight: 600;
    width: auto;
    padding: 0 .4rem;
}
[data-testid="stSidebar"] [class*="st-key-convdel_ok_"] button:hover {
    background: #b91c1c;
    color: #fff;
}

/* 「新建对话」主按钮。
   **必须显式写 background**：侧栏里若有任何规则把它刷成 transparent，
   就只剩白字 → 白底白字，按钮直接"消失"（实测踩过这个坑）。 */
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
    justify-content: center;
    font-weight: 550;
    color: #fff;
    background: #4f46e5;
    border: none;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"]:hover {
    background: #4338ca;
    color: #fff;
}

/* 侧栏小节标题 */
.kb-label {
    font-size: .72rem;
    font-weight: 600;
    letter-spacing: .06em;
    text-transform: uppercase;
    color: rgba(128, 128, 128, .95);
    margin: 1.1rem 0 .35rem .15rem;
}

/* ============================================================== 对话 */
[data-testid="stChatMessage"] {
    background: transparent;
    border: none;
    padding: .45rem 0;
}
[data-testid="stChatMessageContent"] .stMarkdown p { line-height: 1.72; }

/* 用户消息气泡。
   刻意**不**用 ``:has([data-testid*="AvatarUser"])`` 这类选择器区分角色：
   实测 Streamlit 1.66 的消息节点里**没有任何头像 testid**
   （``stChatMessage`` 下只有 ``stChatMessageContent``），那种写法永远不会命中。
   改为在 Python 侧给用户消息套一个 ``.kb-user`` 容器，简单且不依赖内部结构。 */
.kb-user {
    display: inline-block;      /* 收缩贴合文字，而不是铺满整行 */
    max-width: 100%;
    background: rgba(99, 102, 241, .10);
    border-radius: .9rem;
    padding: .62rem .95rem;
    white-space: pre-wrap;
    word-break: break-word;
    font-size: .95rem;
}

/* ============================================================ 引用卡片 */
.kb-src {
    border: 1px solid rgba(128, 128, 128, .2);
    border-radius: .65rem;
    padding: .55rem .78rem;
    margin: .35rem 0;
    background: rgba(128, 128, 128, .045);
}
.kb-src-head {
    display: flex;
    align-items: center;
    gap: .5rem;
    font-size: .82rem;
    margin-bottom: .3rem;
}
.kb-src-idx {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 1.15rem;
    height: 1.15rem;
    border-radius: .35rem;
    background: rgba(99, 102, 241, .16);
    font-size: .7rem;
    font-weight: 600;
}
.kb-src-name { font-weight: 600; }
.kb-src-sec { color: rgba(128, 128, 128, 1); font-size: .78rem; }
.kb-src-score {
    margin-left: auto;
    color: rgba(128, 128, 128, 1);
    font-variant-numeric: tabular-nums;
    font-size: .78rem;
}
.kb-src-body {
    font-size: .8rem;
    line-height: 1.55;
    color: rgba(115, 115, 115, 1);
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
}

/* ============================================================ 欢迎页 */
.kb-hero { text-align: center; padding: 3.2rem 0 1.4rem; }
.kb-hero-icon { font-size: 2.5rem; line-height: 1; }
.kb-hero-title {
    font-size: 1.65rem;
    font-weight: 650;
    letter-spacing: -.015em;
    margin-top: .55rem;
}
.kb-hero-sub {
    color: rgba(128, 128, 128, 1);
    font-size: .875rem;
    margin-top: .45rem;
}

/* ============================================================ 输入框 */
[data-testid="stChatInput"] {
    border-radius: 1.05rem;
    border: 1px solid rgba(128, 128, 128, .3);
    box-shadow: 0 6px 22px rgba(0, 0, 0, .07);
}
[data-testid="stBottomBlockContainer"] {
    max-width: 780px;
    padding-bottom: 1.3rem;
}

/* ============================================================== 组件 */
[data-testid="stExpander"] {
    border: 1px solid rgba(128, 128, 128, .19);
    border-radius: .7rem;
    background: rgba(128, 128, 128, .035);
}
[data-testid="stExpander"] summary { font-size: .855rem; }
[data-testid="stExpander"] summary:hover { color: inherit; }

[data-testid="stSidebar"] [data-testid="stExpander"] { margin-bottom: .4rem; }
[data-testid="stSidebar"] [data-testid="stExpander"] summary p {
    font-size: .875rem;
    font-weight: 500;
}

hr { margin: 1rem 0; opacity: .5; }

/* 建议问题按钮：卡片感 */
.kb-suggest .stButton > button,
[data-testid="stMainBlockContainer"] .stButton > button {
    border-radius: .7rem;
    border: 1px solid rgba(128, 128, 128, .22);
    font-size: .875rem;
    font-weight: 400;
    padding: .7rem .9rem;
    text-align: left;
    justify-content: flex-start;
    min-height: 0;
}
</style>
"""


def apply() -> None:
    """注入主题样式。必须在 ``set_page_config`` 之后调用。"""
    st.markdown(CSS, unsafe_allow_html=True)
