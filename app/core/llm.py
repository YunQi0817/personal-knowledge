"""模型连接。

只负责「怎么连模型」，不含任何业务能力——检索、总结、联网都在 ``app/services``。
工具定义在 ``app/services/tools.py``，Agent 组装在 ``app/services/agent.py``。

**分层约定**：``ui → services → core``。core 不能反过来依赖 services：
工具若写在这里，就需要 ``import app.services.qa``，而 qa 又要 ``import app.core.llm``，
立刻形成循环导入。所以工具只能待在 services 层。
"""

import os

from langchain.chat_models import init_chat_model

from app import config  # noqa: F401  导入即按项目根目录加载 .env，比 load_dotenv() 可靠

#: 模型名从环境变量读，避免换模型时漏改这里。
DEFAULT_MODEL = "mimo-v2.6-flash"

model = init_chat_model(
    model=os.getenv("MIMO_MODEL", DEFAULT_MODEL),
    model_provider="openai",
    api_key=os.getenv("MIMO_API_KEY"),
    base_url=os.getenv("MIMO_BASE_URL"),
)
