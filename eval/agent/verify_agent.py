"""Agent 工具化验收：工具选择是否正确 + 畸形调用率 + 与固定管道的延迟对比。

用法（项目根目录）：
    .venv\\Scripts\\python.exe eval\\agent\\verify_agent.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402,F401  导入即加载 .env
from app.services import qa  # noqa: E402
from app.services.agent import get_agent  # noqa: E402

VALID_TOOLS = {"search_knowledge", "list_documents", "summarize_document", "web_search"}

#: (问题, 期望工具, 说明)。期望为 None 表示「只观察，不判分」。
CASES: list[tuple[str, str | None, str]] = [
    ("基础版支持多少个正式成员？", "search_knowledge", "知识库有答案 → 应检索"),
    ("知识库里现在有哪些文档？", "list_documents", "元问题 → 应列文档，而不是做向量检索"),
    ("帮我总结一下策划书讲了什么", "summarize_document", "要求总结 → 应走摘要路径"),
    ("1+1 等于几？", None, "与知识库无关 → 允许不调任何工具"),
]

LATENCY_QUESTIONS = [
    "基础版支持多少个正式成员？",
    "试用版到期后数据会立刻删除吗？",
]


def calls_of(messages) -> list[str]:
    names: list[str] = []
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            names.append(str(call.get("name", "")))
    return names


def main() -> int:
    print(f"AGENT_MODE={config.AGENT_MODE}  AGENT_MAX_STEPS={config.AGENT_MAX_STEPS}")
    print("=" * 78)

    passed = valid = malformed = 0
    for question, expected, note in CASES:
        print(f"\n问：{question}\n    （{note}）")
        started = time.perf_counter()
        try:
            result = get_agent().invoke({"messages": [{"role": "user", "content": question}]})
        except Exception as exc:  # noqa: BLE001
            print(f"    ✗ 失败：{type(exc).__name__}: {str(exc)[:200]}")
            continue
        elapsed = (time.perf_counter() - started) * 1000

        names = calls_of(result["messages"])
        bad = [n for n in names if n not in VALID_TOOLS]
        valid += len(names) - len(bad)
        malformed += len(bad)

        if expected is None:
            verdict = "✓ 未调用工具（符合预期）" if not names else f"△ 调用了 {names}（可接受）"
            passed += 1
        elif expected in names:
            verdict = f"✓ 调用了 {expected}"
            passed += 1
        else:
            verdict = f"✗ 期望 {expected}，实际 {names or '未调用'}"

        answer = str(result["messages"][-1].content).replace("\n", " ")[:80]
        print(f"    {verdict}  序列={names}  耗时={elapsed:.0f}ms")
        print(f"    答：{answer}")

    total = valid + malformed
    rate = malformed / total if total else 0
    print("\n" + "=" * 78)
    print(f"工具选择通过 {passed}/{len(CASES)}（第 4 题只观察）")
    print(f"工具调用：有效 {valid}  畸形 {malformed}  畸形率 {rate:.0%}")

    # ---- 延迟对比 ----
    print("\n" + "=" * 78)
    print("延迟对比（预热后）")
    print(f"{'问题':<26}{'固定管道':>12}{'Agent':>12}{'倍数':>9}")
    print("-" * 78)
    qa.answer("预热")
    get_agent().invoke({"messages": [{"role": "user", "content": "预热"}]})
    for question in LATENCY_QUESTIONS:
        t0 = time.perf_counter()
        qa.answer(question)
        fixed_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        get_agent().invoke({"messages": [{"role": "user", "content": question}]})
        agent_ms = (time.perf_counter() - t0) * 1000

        ratio = agent_ms / fixed_ms if fixed_ms else 0
        print(f"{question:<26}{fixed_ms:>10.0f}ms{agent_ms:>10.0f}ms{ratio:>8.2f}x")

    print("\n注：固定管道只用 1 次模型调用；Agent 至少 2 次（决定工具 + 生成回答）。")
    return 0 if passed >= 3 else 1


if __name__ == "__main__":
    sys.exit(main())
