"""用 Streamlit 官方 AppTest 端到端跑一遍界面（不需要浏览器）。

能抓到 st.status / st.empty / 流式渲染之类的运行时错误——
这些光靠语法检查是发现不了的。

用法（项目根目录）：
    .venv\\Scripts\\python.exe eval\\agent\\test_ui_apptest.py
"""

from __future__ import annotations

import sys
from pathlib import Path


def find_root(start: Path) -> Path:
    """向上找到含 app/ui/streamlit_app.py 的目录。"""
    for candidate in [start, *start.parents]:
        if (candidate / "app" / "ui" / "streamlit_app.py").exists():
            return candidate
    raise SystemExit("找不到项目根目录（app/ui/streamlit_app.py）")


ROOT = find_root(Path(__file__).resolve().parent)
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = ROOT / "app" / "ui" / "streamlit_app.py"

QUESTION = "企业版是否一定支持私有化部署？"


def main() -> int:
    print(f"AppTest 加载 {APP.name} …")
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()

    ok = True
    if at.exception:
        ok = False
        print(f"  ✗ 首次加载：{len(at.exception)} 个异常")
        for exc in at.exception:
            print(f"      {exc.type}: {exc.message}")
    else:
        print("  ✓ 首次加载：无异常")

    if not at.chat_input:
        print("  ! 没找到 chat_input")
        return 1

    print(f"\n发送消息：{QUESTION}（会真的调 Agent，约 10~40 秒）…")
    at.chat_input[0].set_value(QUESTION).run()
    if at.exception:
        ok = False
        print(f"  ✗ 发送消息：{len(at.exception)} 个异常")
        for exc in at.exception:
            print(f"      {exc.type}: {exc.message}")
    else:
        print("  ✓ 发送消息：无异常")

    # 状态条：应显示用过的工具，或「直接回答（未使用工具）」
    statuses = at.get("status")
    labels = [getattr(element, "label", "") for element in statuses]
    print(f"  status：{labels or '（无）'}")
    if not statuses:
        ok = False
        print("  ✗ 没有渲染出工具状态条")
    elif labels and labels[-1] == "正在理解问题…":
        ok = False
        print("  ✗ 状态条停留在初始文案，说明流式循环没跑完")

    # 回答正文
    joined = "\n".join((markdown.value or "") for markdown in at.markdown)
    has_answer = "私有化部署" in joined
    print(f"  {'✓' if has_answer else '✗'} 回答正文已渲染")
    ok = ok and has_answer

    # 工具返回的原文不能出现在界面上
    leaked = "chunk_id" in joined or "[片段" in joined
    print(f"  {'✗' if leaked else '✓'} 工具原文未泄漏到界面")
    ok = ok and not leaked

    print("\n" + ("全部通过" if ok else "存在失败项"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
