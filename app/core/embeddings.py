from langchain.embeddings import init_embeddings
import os
from dotenv import load_dotenv

from app import config

load_dotenv(override=True)

# 初始化嵌入模型
#
# .. warning::
#    ``check_embedding_ctx_length=False`` **绝对不能删**。
#
#    LangChain 的 OpenAIEmbeddings 默认打开这个开关：它会先用 tiktoken
#    （cl100k_base，**OpenAI 的词表**）把文本切成 token，再把 **token ID 列表**
#    当作 ``input`` 发给服务端。而 OpenAI 接口规范里 ``List[int]`` 的含义是
#    "一段用 token ID 表示的文本"，服务端于是拿**它自己的词表**（BGE-M3 属
#    XLM-RoBERTa 系）去解码这串 cl100k 的 ID —— 解出来是乱码。
#
#    后果不是报错而是**静默降级**：向量算的是乱码，与原文无关。实测同一段文本
#    两条路径的余弦只有 0.42，四个测试问题的正确片段从"第 1 名"掉到"第 11~23 名"，
#    并且「基础版多少钱」这类问题怎么调切分参数都救不回来。关掉开关后 LangChain
#    直接发送原始文本，与直连 API 的余弦为 1.000000。
embed_model = init_embeddings(
    model="openai:" + config.EMBED_MODEL_NAME,
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url=os.getenv("SILICONFLOW_BASE_URL"),
    check_embedding_ctx_length=False,
)
