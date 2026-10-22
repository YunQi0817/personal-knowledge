from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# app/config.py -> app/ -> 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(override=True)

# ---------------------------------------------------------------------------
# NLTK 数据目录引导（必须先于 nltk / unstructured 的导入执行）
#
# langchain_community 的 Unstructured* Loader（.md / .pptx / .xlsx）依赖
# unstructured，它内部会调用 nltk.download() 取两个分词与词性标注资源：
#   - averaged_perceptron_tagger_eng
#   - punkt_tab
# nltk 只会写入「已存在且可写」的候选目录；若该目录不存在，它会回退到
# %APPDATA%\nltk_data 并尝试联网下载，在受限环境下会以 PermissionError
# 或长时间超时告终（实测 72 秒后失败）。
#
# 因此这里把 NLTK_DATA 指向项目内的 .nltk_data，并**先建好目录**。
# ---------------------------------------------------------------------------
NLTK_DATA_DIR = BASE_DIR / ".nltk_data"
NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NLTK_DATA", str(NLTK_DATA_DIR))


# =========================
# 基本配置（与教程一致）
# =========================
MILVUS_URI = "http://localhost:19530"  # Milvus 服务的连接地址
DB_NAME = "rag_tutorial"               # 自定义数据库名称
COLLECTION_NAME = "docs"               # 向量集合名称（类似于传统数据库的表）

# BGE-M3 在 SiliconFlow / Milvus 文档中都是 1024 维
EMBED_MODEL_NAME = "Pro/BAAI/bge-m3"   # 嵌入模型名称
EMBED_DIM = 1024                       # BGE-M3 模型输出的向量维度固定为 1024


# =========================
# 文档切分
# =========================
CHUNK_SIZE = 200   # 单块字符上限（章节标题前缀会占用其中一部分）
CHUNK_OVERLAP = 80  # 超长块兜底切分时的重叠字符数


# =========================
# 检索
# =========================
RETRIEVE_LIMIT = 3  # 检索返回的片段数

# ---- 混合检索（稠密 + BM25，RRF 融合）----
#: 每一路先取 ``limit × 该倍数`` 再融合。只取 limit 的话，某一路的正确答案
#: 若排在第 4 位就被丢掉了，融合无从谈起。
HYBRID_FETCH_FACTOR = 4
#: RRF 的平滑常数：越大越弱化头部排名。60 是原论文的推荐值。
RRF_K = 60

# ---- 章节级去重 ----
#: 同一章节只占一个召回名额。
#:
#: 背景：检索按**小块**算相似度，同一章节的多个小块会一起排到前面
#: （实测「哪些套餐不支持 OCR」的 top-3 全部来自「二、套餐说明」）。
#: 而送进模型的是 parent_text（整章节全文），那些重复小块的内容与第 1 名
#: 完全相同——纯属浪费名额。开启后召回粒度从「小块级 top-k」对齐为
#: 「章节级 top-k」，与 Small-to-Big 的设计意图一致。
#:
#: 设环境变量 RAG_SECTION_DEDUPE=0 可关闭，用于 A/B 对比（见 eval/README.md）。
SECTION_DEDUPE = os.getenv("RAG_SECTION_DEDUPE", "1").lower() not in ("0", "false", "no", "off")

#: 去重前把候选池放大多少倍。太小会导致挑不满 top_k 个不同章节。
SECTION_DEDUPE_FETCH_FACTOR = 3


# =========================
# 路径
# =========================
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = BASE_DIR / "docs"            # 批量入库的原始文件目录
UPLOADS_DIR = DATA_DIR / "uploads"      # 界面上传的原始文件落盘目录
CONVERSATIONS_DB = DATA_DIR / "conversations.db"   # 对话记录（SQLite）
SUMMARIES_DB = DATA_DIR / "summaries.db"           # 章节摘要缓存（SQLite）

for _path in (DATA_DIR, DOCS_DIR, UPLOADS_DIR):
    _path.mkdir(parents=True, exist_ok=True)
