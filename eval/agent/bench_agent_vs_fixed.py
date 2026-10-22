"""对比固定管道与 Agent 模式的延迟（预热后）。

Agent 的代价是至少多一次模型往返（先决定调不调工具），必须量化。
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\27173\Desktop\Personal knowledge")
sys.path.insert(0, str(PROJECT_ROOT))

from app import config  # noqa: E402,F401
from app.services import qa  # noqa: E402
from app.services.agent import get_agent  # noqa: E402

QUESTIONS = [
    "基础版支持多少个正式成员？",
    "试用版到期后数据会立刻删除吗？",
]


def time_fixed(question: str) -> float:
    started = time.perf_counter()
    qa.answer(question)
    return (time.perf_counter() - started) * 1000


def time_agent(question: str) -> tuple[float, int]:
    started = time.perf_counter()
    result = get_agent().invoke({"messages": [{"role": "user", "content": question}]})
    elapsed = (time.perf_counter() - started) * 1000
    calls = 0
    for message in result["messages"]:
        calls += len(getattr(message, "tool_calls", None) or [])
    return elapsed, calls


print("预热中（首次调用含模型连接开销）...")
qa.answer("预热")
get_agent().invoke({"messages": [{"role": "user", "content": "预热"}]})
print()

print(f"{'问题':<26} {'固定管道':>12} {'Agent':>12} {'倍数':>8}  工具调用")
print("-" * 78)
for question in QUESTIONS:
    fixed_ms = time_fixed(question)
    agent_ms, calls = time_agent(question)
    ratio = agent_ms / fixed_ms if fixed_ms else 0
    print(f"{question:<26} {fixed_ms:>10.0f}ms {agent_ms:>10.0f}ms {ratio:>7.2f}x  {calls}")

print()
print("注：固定管道只用 1 次模型调用；Agent 至少 2 次（决定工具 + 生成回答）。")
