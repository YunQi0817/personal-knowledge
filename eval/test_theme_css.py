"""静态回归测试：确保 theme.py 不会把侧栏展开按钮藏掉。

**为什么需要这个测试**

Streamlit 1.66 把侧栏折叠后的展开按钮放在工具栏内部：

    stToolbar > stExpandSidebarButton                        ← 展开按钮
    stSidebar > stSidebarHeader > stSidebarCollapseButton    ← 收起按钮

如果 CSS 里写了 ``[data-testid="stToolbar"] { display: none }``，
收起按钮照常工作，但展开按钮被一起藏掉 —— 表现为「侧栏能收起、再也展不开」。
这个坑已经踩过一次（2026-10-23），所以留个测试盯着。

用法（项目根目录）：
    .venv\\Scripts\\python.exe eval\\test_theme_css.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
THEME = ROOT / "app" / "ui" / "theme.py"

#: 这些 testid 一旦被 display:none 就会破坏交互
PROTECTED = ["stToolbar", "stExpandSidebarButton", "stSidebar"]

HIDING = re.compile(r"display\s*:\s*none", re.IGNORECASE)


def selectors_of(rule_body: str) -> list[str]:
    return [part.strip() for part in rule_body.split(",")]


def main() -> int:
    if not THEME.exists():
        print(f"[ERROR] 找不到 {THEME}")
        return 1

    css = THEME.read_text(encoding="utf-8")

    # 去掉注释，避免注释里提到的规则被误判
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)

    problems: list[str] = []
    for match in re.finditer(r"([^{}]+)\{([^{}]*)\}", css):
        selectors, body = match.group(1), match.group(2)
        if not HIDING.search(body):
            continue
        for selector in selectors_of(selectors):
            for protected in PROTECTED:
                if protected in selector:
                    problems.append(f"{selector.strip()} → {body.strip()[:60]}")

    print(f"检查 {THEME.relative_to(ROOT)}：")
    if problems:
        for problem in problems:
            print(f"  ✗ 受保护元素被 display:none：{problem}")
        print("\n  stToolbar 里含侧栏展开按钮，隐藏它会导致侧栏收起后展不开。")
        print("  要隐藏工具栏里的部署按钮/状态浮标，请单独针对那些 testid 写规则。")
        return 1

    print("  ✓ 未发现隐藏受保护元素的规则")
    for protected in PROTECTED:
        if protected in css:
            print(f"  · {protected} 在样式中出现（未被 display:none）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
