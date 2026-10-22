"""验证当前配置的模型是否支持工具调用（function calling）。

这是 Agent 工具化的前置条件：如果模型不会调用工具，方案必须换。

用法（项目根目录）：
    .venv\\Scripts\\python.exe -B <本脚本绝对路径>

注意：密钥从项目的 .env 读取，所以必须先导入 app.config
（它用 BASE_DIR 定位 .env，比 load_dotenv() 靠调用栈向上找更可靠）。
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\27173\Desktop\Personal knowledge")
sys.path.insert(0, str(PROJECT_ROOT))

from app import config  # noqa: E402  导入即完成 load_dotenv

print("=== 环境变量（只报是否配置，不打印密钥）===")
for key in ("MIMO_API_KEY", "SILICONFLOW_API_KEY", "TAVILY_API_KEY"):
    value = os.getenv(key)
    state = "已配置" if value else "缺失"
    print(f"  {key:22} {state} (长度 {len(value) if value else 0})")
for key in ("MIMO_BASE_URL", "MIMO_MODEL"):
    print(f"  {key:22} {os.getenv(key)}")

print()
print("=== 工具调用测试 ===")

from langchain.agents import create_agent  # noqa: E402
from langchain.chat_models import init_chat_model  # noqa: E402
from langchain.tools import tool  # noqa: E402


@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气。"""
    return f"{city}：晴，25 度"


@tool
def add_numbers(a: int, b: int) -> int:
    """计算两个整数之和。"""
    return a + b


model = init_chat_model(
    model=os.getenv("MIMO_MODEL"),
    model_provider="openai",
    api_key=os.getenv("MIMO_API_KEY"),
    base_url=os.getenv("MIMO_BASE_URL"),
)

agent = create_agent(
    model=model,
    tools=[get_weather, add_numbers],
    system_prompt="你是助手。需要天气或算术时，必须调用工具，不要凭记忆回答。",
)

cases = [
    ("杭州今天天气怎么样？", "get_weather"),
    ("帮我算一下 1234 加 5678 等于多少", "add_numbers"),
]

any_call = False
for question, expected in cases:
    print(f"\n问题：{question}")
    try:
        result = agent.invoke({"messages": [{"role": "user", "content": question}]})
        messages = result["messages"]
        calls = []
        for message in messages:
            for call in getattr(message, "tool_calls", None) or []:
                calls.append((call.get("name"), call.get("args")))
        print(f"  消息数：{len(messages)}")
        if calls:
            any_call = True
            for name, args in calls:
                hit = "✓" if name == expected else "✗"
                print(f"  {hit} 调用工具：{name}  参数：{args}")
        else:
            print("  ✗ 没有发起任何工具调用")
        print(f"  最终回答：{str(messages[-1].content)[:100]}")
    except Exception as exc:  # noqa: BLE001
        print(f"  失败：{type(exc).__name__}: {str(exc)[:300]}")

print()
print("=== 结论 ===")
print("两题都出现「调用工具」→ 模型支持 function calling，可以进入 Agent 工具化。")
if not any_call:
    print("若始终没有工具调用，需要换模型或改用「自己解析 JSON 决定工具」的降级方案。")
