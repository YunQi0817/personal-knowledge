# 个人知识库 RAG 问答服务 — 设计文档

> 版本：v1（待评审）
> 范围：个人知识库的入库、检索、问答，单机运行

---

## 1. 目标与范围

### 1.1 要做什么

一个本地运行的个人知识库问答服务：

1. 把本地文档（txt / md / pdf / docx 等）切块、向量化，存进 Milvus
2. 用户提问时检索相关片段，拼成上下文交给大模型
3. 流式返回答案，并展示引用来源
4. 支持在界面上传文档、查看已入库文档、删除文档

### 1.2 不做什么（本期明确排除）

| 排除项 | 原因 |
|---|---|
| 对外提供 HTTP API | 单机自用，无第三方调用需求 |
| 多用户 / 权限 / 登录 | 个人知识库 |
| 联网检索（Tavily） | 已有密钥但本期不接入 |
| Rerank 重排 | 本地无重排模型，且需先有 baseline 才能判断是否有用 |
| 生产部署（Docker 化、反向代理） | 本机 `streamlit run` 即可 |

### 1.3 成功标准

- 问"权限生效时间"这类知识库内的问题，能答对且带来源
- 重复导入同一文件，数据不翻倍
- 删除某文档后，检索不到它的任何片段

---

## 2. 已确认的环境事实

以下均为**实测确认**，不是推测：

| 项 | 事实 | 确认方式 |
|---|---|---|
| Milvus 服务端 | **v3.0.2**，`E:\Milvus\standalone.bat` 管理 | 读 `standalone.bat`；`get_server_version()` 实测返回 |
| Milvus 部署形态 | 单容器 + **嵌入式 etcd** + 本地存储，无需 MinIO | 读启动参数 |
| Milvus 数据目录 | `E:\Milvus\volumes\milvus`。原 147 MB 教程实验数据**已删除**，当前 `collections = []` | 字符串扫描 + `list_collections()` |
| Milvus 服务状态 | ✅ **运行中**，`127.0.0.1:19530` 可达；容器 `milvus-standalone` 已重建 | `Test-NetConnection` + pymilvus 连接 |
| pymilvus 客户端 | **2.6.12** | `pymilvus.__version__` |
| 客户端↔服务端兼容 | ✅ **实测互通**（2.6.12 → 3.0.2），跨版本无问题 | `list_databases()` 正常返回 |
| BM25 全文检索 | ✅ **可用**（中文分词器 + `FunctionType.BM25` + 稀疏索引均建成） | 建测试集合后即删 |
| Docker daemon | ✅ 运行中，但**沙箱内禁止访问命名管道**，`docker` CLI 需一次性提权 | `docker version` 报 `Access is denied` |
| Embedding | ✅ `BAAI/bge-m3` → **1024 维**，实测通过 | 实际调用 API |
| LLM — MiMo | ✅ `mimo-v2.6-flash` 实测通过 | 实际调用 API |
| LLM — DeepSeek | ⚠️ `deepseek-v4.1-flash` 实测返回 **400** | 实际调用 API，见 §10 R2 |
| Streamlit | ✅ **1.66.0** 已安装 | 清华镜像装成功 |
| `milvus-lite` | ❌ Windows 无发行轮子，**不可用** | pip 报 `No matching distribution` |
| 本地重排模型 | ❌ 无 `sentence_transformers` / `FlagEmbedding` | 包目录检查 |
| pip 网络 | ⚠️ 默认走代理 `127.0.0.1:7890` 会 SSL 超时；**清华镜像 + 关代理可用** | 实测对比 |

**关键推论**：

1. `milvus-lite` 不可用 ⇒ **必须依赖 Docker 里的 Milvus Standalone**，没有"免 Docker"退路。
2. 客户端/服务端跨版本**实测互通**，且 **BM25 可用** ⇒ 阶段三（混合检索）无技术障碍。

---

## 3. 技术选型

| 层 | 选型 | 理由 |
|---|---|---|
| 前端 | **Streamlit** | 同进程直调，省掉 SSE + HTTP + 手写前端三套代码；上传/列表/多轮对话内置 |
| 文档加载 | LangChain `TextLoader` 等 | 格式适配现成 |
| 切块 | LangChain `RecursiveCharacterTextSplitter` | 分隔符策略可配，适合中文 |
| 向量化 | LangChain `init_embeddings` → SiliconFlow | OpenAI 兼容，一行接入 |
| 对话模型 | LangChain `init_chat_model` → DeepSeek / MiMo | 同上 |
| 向量库 | **原生 `pymilvus.MilvusClient`** | 用户明确要求；避免双抽象 |
| 配置 | `pydantic-settings` | 类型校验 + `.env` 映射 |

### 3.1 为什么向量库用原生 pymilvus 而非 `langchain_milvus`

两者都能用，取舍如下：

| | 原生 pymilvus | `langchain_milvus.Milvus` |
|---|---|---|
| 混合检索 | 自己写 `AnnSearchRequest` ×2 + `RRFRanker`，约 20 行 | `builtin_function=BM25BuiltInFunction(...)` 一个参数 |
| 重排 | 自己调 API | `reranker=Function(...)` 声明式 |
| 可控性 | 完全透明，出错好查 | 封装较深，出问题要读源码 |
| 与 LangChain 链衔接 | 需自己包 Retriever | 天生是 `VectorStore` |

**结论**：按你的要求选**原生 pymilvus**。代价是阶段三的混合检索要手写，但代码量可接受。

> 备注：`langchain_milvus` 0.3.3 已在 `.venv` 中，其 `BM25BuiltInFunction`、`MilvusCollectionHybridSearchRetriever` 可作阶段三的实现参考。

---

## 4. 整体架构

```
┌──────────────────────────────────────────────────────┐
│  Streamlit UI         app/ui/streamlit_app.py        │
│  问答页 / 上传 / 文档列表 / 参数设置                   │
└───────────────────────┬──────────────────────────────┘
                        │ 直接函数调用（同进程，无 HTTP）
┌───────────────────────▼──────────────────────────────┐
│  Service 层                                           │
│   services/qa.py       检索 → 拼上下文 → 流式生成      │
│   services/ingest.py   加载 → 切块 → 向量化 → 入库     │
└───────────────────────┬──────────────────────────────┘
                        │
┌───────────────────────▼──────────────────────────────┐
│  Core 层                                              │
│   core/milvus.py       原生 MilvusClient 封装         │
│   core/embeddings.py   LangChain init_embeddings      │
│   core/llm.py          LangChain init_chat_model      │
│   config.py            pydantic-settings 配置          │
└───────────────────────┬──────────────────────────────┘
                        │
        ┌───────────────┼────────────────┐
        ▼               ▼                ▼
  Milvus v3.0.2    SiliconFlow      DeepSeek / MiMo
  (本地 Docker)     bge-m3 (1024维)   Chat API
```

**分层原则**：Service 层不感知传输格式，Core 层不感知业务语义。这样以后加 FastAPI，Service 层可原样复用。

---

## 5. 目录结构

```
Personal knowledge/
├── .env                        # 密钥与模型配置
├── .nltk_data/                 # NLTK 资源（Unstructured Loader 依赖，约 11 MB）
├── DESIGN.md                   # 本文档
├── app/
│   ├── config.py               # get_settings() + NLTK_DATA 引导
│   ├── core/
│   │   ├── embeddings.py       # init_embeddings 封装
│   │   ├── llm.py              # init_chat_model + 供应商降级
│   │   ├── milvus.py           # 原生 MilvusClient 封装，返回 Document
│   │   └── retriever.py        # 【新增】MilvusRetriever(BaseRetriever)
│   ├── services/
│   │   ├── ingest.py           # LangChain Loader → 切块 → 入库
│   │   └── qa.py               # 检索 + LCEL 链问答
│   └── ui/
│       └── streamlit_app.py    # Streamlit 入口（注意命名，见 §5.2）
├── docs/                       # 手动批量入库的原始文件
├── ingest.py                   # 批量入库 CLI
└── data/
    └── uploads/                # 界面上传的原始文件，以 {doc_id}{后缀} 命名
```

### 5.0 与教程的对应关系

代码**完全参照教程** ``chapter10-RAG`` 编写，对应关系如下：

| 教程位置 | 教程写法 | 本项目位置 |
|---|---|---|
| 05「1、全局配置」 | `MILVUS_URI` / `DB_NAME` / `COLLECTION_NAME` / `EMBED_MODEL_NAME` / `EMBED_DIM` | `app/config.py` |
| 05「2、初始化Milvus」 | `MilvusClient` → `create_database` → `use_database` | `core/milvus.get_client()` |
| 05「2、创建collection」 | `create_collection(dimension=EMBED_DIM, metric_type="COSINE")` | `core/milvus.ensure_collection()` |
| 05「3、初始化 Embedding」 | `init_embeddings(model="openai:"+EMBED_MODEL_NAME, ...)` | `core/embeddings.py` |
| 05「4、读取文档并切分」 | `TextLoader` + `RecursiveCharacterTextSplitter(200/80, separators)` | `services/ingest.py` |
| 05「5、生成向量并写入」 | `embed_model.embed_documents` → `client.upsert` → `flush` | `services/ingest.ingest_documents()` |
| 05「6、创建Agent」 | `init_chat_model(model="mimo-v2.6-flash", model_provider="openai")` + `create_agent(model, tools=[], system_prompt)` | `core/llm.py` |
| 05「7、检索」 | `embed_model.embed_query` → `client.search(output_fields=["text","chunk_id","source"])` | `services/qa.retrieve()` |
| 05「8、生产与回答生成」 | 拼 `[片段{i} \| chunk_id=.. \| source=..]` + `问题：/上下文：` prompt | `services/qa.build_user_prompt()` |
| 01 文档加载器 | `TextLoader` / `PyPDFLoader` / `UnstructuredWordDocumentLoader` / … | `services/ingest.loader_for()` |

模型初始化、`.env` 加载、`os.getenv` 取值均写在各自模块内，与教程单元格**逐字一致**；
密钥不集中在 `config.py`（教程的「全局配置」段本来也不含密钥）。

#### 四处必要的偏离

| # | 偏离 | 原因 |
|---|---|---|
| 1 | 主键 `id` 用字符串 `"{doc_id}:{chunk_id}"`，而非顺序整数 `i` | 顺序整数在多文件场景下主键相撞，`upsert` 会**静默覆盖**先入库的文件（§6.2） |
| 2 | 用 `agent.stream()` 而非 `agent.invoke()` | 界面需要逐字输出；语义相同 |
| 3 | 教程 `retrieve(str, limit=5)` 的笔误未照搬 | 传入了内置类型 `str` 而非 `query`，照搬则检索必然错误 |
| 4 | 模型名用 `mimo-v2.6-flash`（教程写 `mimo-v2.5`） | 这是当前 MiMo 可用模型，与 `.env` 的 `MIMO_MODEL` 一致 |

#### LangChain 1.x 的版本提醒

`langchain.chains` 模块在 1.x 中**已被移除**，教程里若出现 `RetrievalQA` /
`create_retrieval_chain` 等写法在本环境不可用。实测：

```
FAIL  langchain.chains.create_retrieval_chain -> No module named 'langchain.chains'
FAIL  langchain.chains.RetrievalQA            -> No module named 'langchain.chains'
OK    langchain.agents.create_agent
OK    langchain_core.runnables.RunnablePassthrough
```

本项目按教程 05 的方式使用 `create_agent`，因此不受影响。

### 5.1 与初始结构的差异

| 路径 | 动作 | 说明 |
|---|---|---|
| `app/api/` | **删除**（已删） | Streamlit 无需 HTTP 层 |
| `app/main.py` | **删除**（已删） | 同上 |
| `app/core/retriever.py` | **删除**（已删） | 空壳；改用 `milvus.py`，叫 retriever 会误导 |
| `app/models/` | **删除**（已删，原定「待定」） | 无 HTTP 层后 Pydantic 请求/响应模型无用武之地 |
| `data/chroma/` | **删除**（已删） | 旧 Chroma 方案残留空目录 |
| 根 `main.py` / `run.py` | **删除**（已删） | 0 字节残留 |
| `_reference_code/` | **删除**（已删） | Chroma 方案，与 Milvus 路线不符 |

**关于 `app/models/`**：已删除。检索结果用普通 `dict` 描述并在 core 层收口，不单开类型层。

### 5.2 启动方式与导入约定（踩坑记录）

**必须用 `streamlit run`，不能直接用 `python` 执行：**

```powershell
cd "C:\Users\27173\Desktop\Personal knowledge"
.venv\Scripts\python.exe -m streamlit run app/ui/streamlit_app.py
```

用 `python app/ui/streamlit_app.py` 只会进 Streamlit 的 bare mode，脚本能跑但不提供服务。

**入口文件不能命名为 `app.py`。** 这是实际踩到的坑，故障链条如下：

```
1. 入口文件曾叫 app/ui/app.py
2. 运行脚本时，Python 把脚本所在目录 app/ui 放进 sys.path[0]
3. app/ui/app.py 这个「文件」于是与项目根的「app/ 包」同名
4. import app 命中 app/ui/app.py  →  'app' is not a package  →  报错退出
```

**PyCharm 会放大这个问题**：它把内容根（项目根）注入 `PYTHONPATH`，此时项目根**确实已在** `sys.path` 中，但位置排在脚本目录**之后**。若引导代码写成"根目录不存在才插入"，就会被这个判断跳过，于是 `app/ui` 仍留在 `sys.path[0]`，必然触发上述冲突。

因此入口文件里采用**无条件置顶**：

```python
_ROOT = str(Path(__file__).resolve().parents[2])
while _ROOT in sys.path:      # 先移除，避免留在靠后的位置
    sys.path.remove(_ROOT)
sys.path.insert(0, _ROOT)     # 再无条件插到最前
```

> 复现命令（验证修复时可重跑）：
> ```powershell
> $env:PYTHONPATH = "C:\Users\27173\Desktop\Personal knowledge"
> .venv\Scripts\python.exe app/ui/streamlit_app.py
> ```
> 修复前会 `ModuleNotFoundError: No module named 'app.config'; 'app' is not a package`；修复后仅输出 bare mode 警告。

> **PyCharm 用户注意**：原有 Run Configuration 指向已删除的 `app/ui/app.py`，需改为 `streamlit run app/ui/streamlit_app.py`（或改用 Terminal 启动）。

---

## 6. 数据模型

### 6.1 Milvus 集合

```
集合名: personal_knowledge
```

| 字段 | 类型 | 约束 | 说明 |
|---|---|---|---|
| `id` | VARCHAR | 主键, max_length=128 | `"{doc_id}:{chunk_id}"` |
| `vector` | FLOAT_VECTOR | dim=1024 | bge-m3 输出 |
| `text` | VARCHAR | 动态字段 | 切块正文 |
| `source` | VARCHAR | 动态字段 | 文件路径或原始文件名 |
| `chunk_id` | INT64 | 动态字段 | 块在文档内的序号 |

> **建表方式**：采用教程的**快速建表**——`create_collection(dimension=1024, metric_type="COSINE", primary_field_name="id", id_type="string")`。
> 除主键与向量外，`text` / `source` / `chunk_id` 均为**动态字段**（快速建表默认 `enable_dynamic_field=True`），
> 无需预先声明，写入时直接带上即可。
>
> 索引由 Milvus 自动创建（快速建表的默认行为），未手工指定 HNSW 参数。

> **与早期设计的差异**：早期版本用显式 schema（`pk` / `doc_id` / `chunk_index` / `created_at` 共 7 字段 + HNSW 索引），
> 现已改为教程的快速建表。相应地，`created_at` 字段取消，文档清单改为从主键前缀 `"{doc_id}:"` 聚合得出。

#### Milvus 存的不只是向量

Milvus 是一张**带 schema 的表**，向量只是它支持的字段类型之一。实测支持的类型：

```
标量: BOOL  INT8/16/32/64  FLOAT  DOUBLE  VARCHAR  STRING
      ARRAY  JSON  GEOMETRY  TIMESTAMPTZ  STRUCT
向量: FLOAT_VECTOR  SPARSE_FLOAT_VECTOR  BINARY_VECTOR
      FLOAT16_VECTOR  BFLOAT16_VECTOR  INT8_VECTOR
```

因此本集合中 `text` / `source` / `chunk_id` 都是与向量同表存储的标量字段。检索时用
`output_fields=["text","chunk_id","source"]` 即可直接取回正文，**不需要另建关系库或文件清单**。

能力边界：

| 能存 | 说明 |
|---|---|
| 标量 + 元数据 | 用 `JSON` 字段可塞任意结构，无需改 schema |
| 正文 | 单字段上限 **65535 字节**（`pymilvus.settings.MaxVarCharLength`）≈ 2.1 万汉字；本设计每块 200 字，远低于上限 |


| 不适合 | 原因 |
|---|---|
| **原始二进制文件** | 没有 BLOB 类型，PDF / 图片的字节流存不了 |
| 整本大文档（单字段） | 超 65535 字节装不下 —— 这正是要切块的原因 |
| `JOIN` / 复杂关系查询 | 不是关系型数据库 |
| 便宜的 `DISTINCT` / 聚合 | 列文档需去重扫描，无原生 DISTINCT |

### 6.2 主键设计（关键决策）

**主键 = `"{doc_id}:{chunk_index}"`，`doc_id = sha1(文件内容)[:16]`**

对比你 demo 里的 `id = 0,1,2,...`：

| 场景 | 自增 int（demo） | 确定性字符串（本设计） |
|---|---|---|
| 两个文件入库 | 主键相撞，后入库的**覆盖**先入库的，且**不报错** | 永不冲突 |
| 同一文件重复入库 | 产生重复数据 | 幂等覆盖 |
| 文件内容修改后重入库 | 新旧混杂，无法区分 | 新 `doc_id`，可识别清理 |
| 删除整份文档 | 需逐条找 | `delete(filter="doc_id == 'xxx'")` 一次删净 |

**为什么 `doc_id` 用内容哈希而非文件路径**：

| 方案 | 缺点 |
|---|---|
| 文件路径 | 文件改名后被视为新文档，产生重复 |
| 修改时间 | `touch` 一下就变，不稳定 |
| **内容哈希** | 内容不变则恒定，内容变则换新 —— 语义正确 |

### 6.3 实现中踩到的坑（重要）

统计片段数时**不能**用 `get_collection_stats()["row_count"]`。

Milvus 的 `upsert` 语义是「**逻辑删除 + 插入**」：同主键写入时，旧行被标记删除，新行插入。被标记删除的行要等 **compaction** 之后才会从 `row_count` 中消失，因此重复入库后该计数会虚高。

实测数据（同一份 `knowledge.txt` 入库 3 次）：

| 指标 | 值 | 说明 |
|---|---|---|
| `get_collection_stats()["row_count"]` | **22** | ❌ 虚高，含未 compaction 的逻辑删除行 |
| `query(filter='doc_id == "..."')` | **11** | ✅ 真实存活行 |
| `query_iterator` 全量扫描 | **11** | ✅ 真实存活行 |
| `count(*)` 聚合查询 | **11** | ✅ 真实存活行 |

**结论**：数据层面 upsert 是幂等的（无重复主键）；只有 `row_count` 这个统计口径会误导。

`count_chunks()` 因此改用 `count(*)` 聚合查询。

### 6.4 已知缺陷：改切块参数会污染集合

**现象**：`doc_id` 只由**文件原始字节**哈希得出（`sha1(字节)[:16]`），**不含切块参数**。
于是同一文件用不同 `CHUNK_SIZE` 重复入库时，两次的块会共用同一段主键空间：

```
CHUNK_SIZE=600 入库 → pk = "983b7af3:0" .. "983b7af3:12"   （13 块）
CHUNK_SIZE=200 入库 → pk = "983b7af3:0" .. "983b7af3:44"   （45 块）
                            ↑ 前 13 个主键与上一次完全重叠
```

`upsert` 会把重叠部分标记为逻辑删除并写入新行，但**被删除的旧行在 compaction 之前依然存在**，
导致 `client.get(ids=[...])` 可能取回**旧版本的向量与文本**，`query` 结果也可能不一致。

**实测证据**（诊断「基础版多少钱」召回问题时发现）：

```
用 client.get 取回 pk="983b7af34cd663bd:4" 的向量
  → 与按当前文本重算的向量 cos = 0.326   （正常应为 1.000）
而同一时刻 Milvus 检索返回该块的 distance = 0.7138，与手工余弦 0.7131 吻合
```

两个数字互相矛盾 —— 说明取回的是**上一版切块留下的残留行**。

**影响**：仅在使用同一 `doc_id` 且**改变切块参数**重灌时出现。日常增量入库（参数不变）不受影响。

**规避**：

| 方案 | 说明 |
|---|---|
| **改切块参数后先 `drop_collection()`** | 最简单可靠，本项目的推荐做法 |
| 把切块参数并入 `doc_id` | `sha1(f"{内容}\|{chunk_size}\|{chunk_overlap}")`，参数变则主键空间整体换新 |
| 定期 compaction | 治标，且不便触发 |

---

### 6.5 重大缺陷：向量算的是乱码（已修复）

**这是本项目迄今最严重的问题，也是「基础版多少钱答不出来」「atguigu 是什么答不出来」
等一系列检索问题的真正根因——此前被误判为「长块稀释」，见 §10.3 的更正。**

#### 现象

同一段文本，`embed_model.embed_query(t)` 与直连 API `client.create(input=[t])`
返回的向量**余弦只有 0.42**；同一批文本，批量调用与单条调用对约 3% 的块
给出互不相同的向量。

#### 根因

LangChain 的 `OpenAIEmbeddings` 默认 `check_embedding_ctx_length=True`，该开关会：

1. 用 **tiktoken / cl100k_base**（**OpenAI 的词表**）把文本切成 token；
2. 把 **token ID 列表**当作 `input` 发给服务端。

而 OpenAI 接口规范中 `List[int]` 的含义是「一段用 token ID 表示的文本」——
服务端于是拿**自己的词表**去解码这串 cl100k 的 ID。BGE-M3 属 XLM-RoBERTa 系，
词表完全不同，解出来是乱码。

拦截到的真实请求：

```
>>> LangChain 实际发出的请求：
    kwargs 键: ['input', 'model']
    input      : 1 条, 首条=[11144, 41920, 5486, 28194, 245, 165, 97, 238, ...]
    首条与本块文本相同? False          ← 发的是 token ID，不是文本
    model      : 'Pro/BAAI/bge-m3'
```

**它不报错，只是静默降级**——所以极难发现。

#### 影响（实测，四个问题的正确答案片段排名）

| 问题 | 修复前 | 修复后 |
|---|---|---|
| 订阅套餐有哪些？ | 第 11 名 (0.371) | **第 1 名 (0.736)** |
| 四档的区别是什么？ | 第 13~23 名 | **第 1 名 (0.493)** |
| 基础版多少钱？ | 第 21 名 (0.301) | **第 1 名 (0.713)** |
| 试用版到期后数据会立刻删除吗？ | 第 11 名 (0.397) | **第 1 名 (0.765)** |

#### 修复

`app/core/embeddings.py` 加一个参数：

```python
embed_model = init_embeddings(
    model="openai:" + config.EMBED_MODEL_NAME,
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url=os.getenv("SILICONFLOW_BASE_URL"),
    check_embedding_ctx_length=False,   # 不能删，原因见上
)
```

关掉后 LangChain 直接发原始文本，与直连 API 的余弦为 **1.000000**，
批量与单条的一致性为 **0.999947**。

#### 换任何 OpenAI 兼容的 embedding 服务，都该先跑这个自检

```python
import numpy as np
from app.core.embeddings import embed_model

t = "任意一段中文文本，务必含全角标点和换行\n第二行"
lc  = embed_model.embed_query(t)                                          # LangChain 路径
raw = embed_model.client.create(model=..., input=[t]).data[0].embedding   # 直连路径
cos = float(np.dot(lc, raw) / (np.linalg.norm(lc) * np.linalg.norm(raw)))
assert cos > 0.999, f"两条路径不一致（cos={cos:.4f}），embedding 配置有问题"
```

---

### 6.6 纯文本解析的 CRLF 陷阱（已修复）

Windows 上的文本文件是 **CRLF** 换行，而正则的 `$` 只认 `\n`——
`^={10,}$` 这类带锚点的模式会**全部落空**：

```
文件前 120 字节: b'atguigu\xe5\x8a\xa9\xe6\x89\x8b...\xe7\x89\x88\xef\xbc\x89\r\n\r\n...'
含 CRLF: True

^={10,}$\n^(?P<title>[^\n]+?)$\n^={10,}$   （re.M）
   原样匹配   : 0 处
   CRLF→LF 后 : 9 处
```

后果是**章节识别静默失效**：所有块都落在「前言」里，标题注入完全没生效，
而表面上不报任何错。

**规避**：解析器读文件后统一归一化，见 `app/services/parsers/plaintext.py`：

```python
return text.replace("\r\n", "\n").replace("\r", "\n")
```

---

## 7. 核心流程

### 7.1 解析与入库流程（parsers → ingest）

#### 为什么要单独一层解析器

知识库要吃 14 种格式，而「怎么切」完全取决于格式：纯文本只能靠正则猜章节，
Word 有 `Heading` 样式，PPT 有幻灯片边界，Excel 有工作表。把共同部分抽成契约，
各格式只负责**把文件翻译成事件流**。

```
                      ┌───────────────────────────────┐
   任意文件  ────────▶ │  parsers.parse(path)          │
                      │  按扩展名分派到具体解析器       │
                      └───────────────┬───────────────┘
                                      │
      ┌──────────────┬────────────────┼────────────────┬──────────────┐
      ▼              ▼                ▼                ▼              ▼
 plaintext.py    docx.py          pptx.py         excel.py       legacy.py
 .txt .md        .docx            .pptx           .xlsx .xlsm    .pdf .html
 .log .text                                                      .csv .json
      │              │                │                │              │
      └──────────────┴────────────────┴────────────────┴──────────────┘
                                      │
                                      ▼
                      ┌───────────────────────────────┐
                      │  SectionBuilder               │
                      │   · 维护标题栈 → headings 路径  │
                      │   · 聚合同一节的正文            │
                      │   · 超长兜底切分                │
                      │   · 注入 【标题】前缀           │
                      └───────────────┬───────────────┘
                                      ▼
                             list[Chunk]  ──▶  ingest.py
```

新增一种格式 = 写一个模块（实现 `parse(path)`）+ 在 `_PARSERS` 登记一行。

#### 契约

```python
@dataclass(slots=True)
class Chunk:
    text: str                     # 已注入标题前缀的正文，直接参与 embedding
    headings: list[str]           # 层级路径，如 ["二、套餐说明", "2. 基础版"]
    source: str
    page: int | None = None       # PPT 幻灯片号 / PDF 页码
    content_type: str = "text"    # text | table | code
```

`headings` 是契约的核心：块一旦脱离章节上下文，语义就会被长正文稀释
（§6.5 与 §10.3 有实测数据）。

#### 各格式的结构信号

| 格式 | 信号来源 | 可靠性 |
|---|---|---|
| `.txt` `.log` | 围栏标题块 `====\n标题\n====`，或 Markdown `#` | 靠约定 |
| `.md` `.markdown` | Markdown ATX 标题 | 高 |
| `.docx` | 段落样式 `Heading N` / `标题 N`，辅以中文序号启发式 | 样式可靠；启发式见下 |
| `.pptx` | 幻灯片边界 + 标题占位符 | **100%** |
| `.xlsx` `.xlsm` | 工作表边界 + 首行表头 | ~95% |
| `.pdf` `.html` `.csv` `.json` | 尚无结构信号，走 `legacy` 兜底 | 低 |

#### docx 的三类脏数据（实测某份 WPS 策划书）

理想情况下 Word 的 `style.name` 是 100% 可靠的层级信号，但真实文档常常不理想。
那份策划书的样式统计：

```
Normal                139 个段落
WPSOffice手动目录 1     11 个   ← 目录条目
Heading 2               7 个   ← 只有这么点，且通篇没有 Heading 1
```

11 个章节只有 7 个套了样式。`docx.py` 因此要处理三类脏数据：

| # | 问题 | 处理 |
|---|---|---|
| 1 | **目录条目**罗列了全部章节名，因而"对任何章节类问题都完美匹配"，会把真正的内容块挤出 top-K（实测「活动有哪些形式？」目录块排第 1） | 样式名含"目录/TOC"，或形如 `标题\t页码` 的行，**整条丢弃** |
| 2 | 通篇只有 `Heading 2` 时，`stack[:level-1]` 会把所有章节**嵌套成父子链**（`活动简介 > 二、活动背景 > 三、活动内容`），6/7 个章节归属错位 | 先扫出全文实际用到的最小层级，**重映射为 1** |
| 3 | 4 个章节漏套样式，内容被算进前一节 | `Normal` + 中文序号 + 短行 → 视为标题 |

第 3 条**只认「一、」这一级**。实测 `（一）主办方：西南财经大学天府学院易班工作站`
与 `（一）指导思想` 在文本上无法区分（都是短行 + 中文序号），对「（一）」做启发式
会误判一大片列表项。

**效果**（12 题集 top-3 命中）：修复前 **9/12** → 修复后 **10/12**，其中
「活动有哪些形式？」由第 5 名升到第 3 名，「活动流程」「活动目的」升到第 1 名。

> **目录必须在段落级过滤，不能按块丢。** 那份文档的封面与目录恰好被合并进同一块，
> 块级过滤会把封面一起丢掉。

#### 超短块的产生与消除

同一份文档里出现过正文只有 5 个字的碎片块（`【二 、活动背景】活动简介：`）。
它没有检索价值，却因为"短而语义纯"，余弦分数天然偏高、反复挤进 top-3。

两道防线都在 `SectionBuilder` 里（`MIN_SECTION_CHARS = 20`）：

1. **节级挂起**：整节正文过短时不产出块，挂起并入下一节。
2. **块级兜底**：`split_overflow` 内部会把短片段攒着等合并，可一旦遇到超长片段，
   它会先把攒下的短片段吐出来再递归切分——碎片照样能单独成块（与「标题孤立」
   是同一个机制）。因此在 `_emit` 里再合并一次：短块向后并，末尾的短块向前并。

Excel 传 `merge_tiny=False`：它的每个行组都是**刻意切开**的，若把"只有一行"的
工作表挂起，会被并进下一张表里，两张表就混了。

#### Small-to-Big：小块检索、大块送模型

`Chunk` 有**两个文本字段**，用途不同：

| 字段 | 内容 | 用途 |
|---|---|---|
| `text` | 小块（≤ `CHUNK_SIZE`） | **建向量、参与检索**——块越小越精确 |
| `parent_text` | 它所属**章节的完整文本** | **送进模型**——上下文才完整 |

这个设计是被数据逼出来的。某策划书「活动简介」一节有 8 个独立条目
（主办方 / 时间 / 地点 / 类型 / 级别 / 指导老师 / 负责人 / 参与者）挤在
一个 191 字的块里，查「活动时间」要跟另外 7 条竞争，正确答案排到第 7 名。

于是做了**小节级切分**（按 `（一）（二）` 二次切分）。结果**好坏参半**：

| 问题 | 切分前 | 切分后 |
|---|---|---|
| 活动时间是什么时候？ | 第 7 名 | **第 1 名** ✅ |
| 活动在哪里举办？ | 第 2 名 | **第 1 名** ✅ |
| 活动的流程是怎样的？ | 第 1 名 | **第 6 名** ❌ |
| 活动有哪些形式？ | 第 3 名 | **第 5 名** ❌ |
| **top-3 命中** | **10/12** | **9/12** ❌ |

短事实类问题变准了，但「流程」这类需要**整节上下文**的问题被切碎后反而变差。
**这正是 Small-to-Big 存在的理由**——索引小块、送模型大块，两者兼得：

| | 只用小块 | **Small-to-Big** |
|---|---|---|
| 策划书 12 题集（上下文命中） | 9/12 | **11/12** |
| knowledge.txt 5 题集 | — | **5/5** |

唯一未命中是「经费预算是多少？」——那份文档里真的没有预算内容（只写了
「另附表」），属于**正确拒答**。

`qa.build_user_prompt()` 里的三条规则：

1. 优先用 `parent_text`；超过 `CONTEXT_BUDGET`（4000 字）就退回命中的小块。
2. 同一章节的多个命中**只保留一份**，否则一个章节会重复占满上下文。
3. 片段编号按去重后的顺序重排。

> **坑**：`parent_text` 必须在 `chunks()` 收尾时**统一回填**，不能在 `_emit`
> 里算。解析是流式的——产出第一块时还不知道本节后面有什么，而小节级切分又会
> 多次调用 `flush()`。实测直接拿单次 payload 当 parent，会让
> 「九、突发状况的预计及应对措施」的 parent 退化成只有「（一）指导思想」那一小段，
> 结果那一节的命中反而从 ✅ 变成 ❌。

#### 混合检索：稠密 + BM25

`text` 是**显式字段**（不是动态字段）且开了中文分析器，一个 `BM25 Function`
从它自动生成稀疏向量——**不用手写**，写入时只给 `text` 和 `dense`。

```python
schema.add_field("text", DataType.VARCHAR, max_length=16384,
                 enable_analyzer=True, analyzer_params={"type": "chinese"})
schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
schema.add_function(Function(name="bm25", function_type=FunctionType.BM25,
                             input_field_names=["text"],
                             output_field_names=["sparse"]))
index_params.add_index(field_name="sparse",
                       index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
```

**这就是为什么不能用快速建表简写**：`create_collection(dimension=…)` 只会把
`text` 当成动态字段，BM25 无从分词；稠密向量字段名也因此从自动生成的 `vector`
变成显式的 `dense`。

##### 实测结论：当前规模下没有可测量的收益

| 评测 | 纯稠密 | 混合 BM25+RRF |
|---|---|---|
| 策划书 12 题 | 11/12 | **11/12** |
| 客服知识库 5 题 | 5/5 | **5/5** |
| 精确串专项 10 题（日期/人名/编号/金额） | 10/10 | **10/10** |

**两套题全部打平。** 原因是语料太小（67 块），稠密检索本来就够用；加上
Small-to-Big 送进模型的是整节，排序稍有偏差也被兜住了。

孤立测试里 BM25 的优势是真实的——「商业街」「200 篇」「6月13」这类精确串，
稠密返回满 4 条、分数挤在 0.29~0.78 之间几乎没区分度，而 BM25 给出唯一且干净的
命中（分数 1.6~2.57）。只是这份语料里稠密恰好也没错。

**留着的理由**：它是规模变大后的保险。查询延迟实测混合 106 ms / 稠密 167 ms
（差异在噪声内）。``qa.retrieve(query, mode="dense")`` 保留了对比开关。

##### 三个踩过的坑

1. 集合里有两个向量字段后，`client.search()` **必须显式指定 `anns_field`**，
   否则报 `multiple anns_fields exist`。
2. `MilvusClient.search()` 的参数名是 **`search_params`**，写成 `param=` 会和
   内部转发冲突，报 `got multiple values for keyword argument 'param'`。
3. 稀疏字段**不可 query 回读**（`not allowed to retrieve raw data of field sparse`）。
4. RRF 的 `distance` 只有排名含义（实测量级 0.02~0.03），**不是相似度**，
   不能当阈值用。

#### 章节摘要：惰性生成 + 缓存

侧栏的「📋 知识库总览」「总结这份文档」走的是**与问答完全不同的路径**：

| | 问答 | 总结 |
|---|---|---|
| 取什么 | 检索 top-K（相关） | 全量摘要（完整） |
| 为什么 | 要的是相关性 | 要的是完整性 |

实测把「总结这份文档」当成普通提问时，top-3 只覆盖约 10%，模型会给出一个
**看起来很像总结、实则以偏概全**的结果，而用户看不出问题。所以总结单独走一条路。

##### 为什么不在入库时生成

实测某 48 KB 的 docx（31 块 / 11 节）：

```
解析     0.1s
向量化   0.5s
写入     3.5s      ← 入库本身很快
摘要    21.8s      ← 独占 84%
```

更糟的是它把入库**绑死在外部 API 的稳定性上**：连续两批 21 次调用实测触发过
限流重试，入库卡了十几分钟没有任何输出。

改成惰性之后：

```
入库         仍然 4 秒
第一次总结   多花约 70 秒
之后         秒回（命中缓存）
```

**最终状态与"入库时预生成"完全一样**，只是时机从"入库"挪到"第一次需要"。
想在导入后一次性算好，用 `python ingest.py --summaries`。

##### 缓存为什么放 SQLite 而不是 Milvus

摘要需要**增量回填**（先生成缺的那几条）。Milvus 的 `upsert` 要整行数据
（含 1024 维向量），为写一段摘要重新上传向量既慢又别扭；SQLite 一条
`INSERT ... ON CONFLICT DO UPDATE` 就够。落点 `data/summaries.db`。

##### 汇总调用要限制输出长度

第一版没写长度限制，模型输出 900+ 字，一次汇总要 **36 秒**。加上
「总长控制在 400 字以内」后降到 **7.3 秒**。

#### 为什么不用 unstructured

实测 `unstructured==0.20.6`（`strategy="fast"`）：

| 格式 | 结果 |
|---|---|
| `.docx` | ✅ 分类准确（Title / UncategorizedText / Table） |
| `.pptx` | ❌ **正文项目符号被误标成 Title**；备注页完全丢失 |
| `.xlsx` | ❌ 拍平成 `试用版 0 1`，**列名与值的绑定关系断掉** |

三个里两个不能用，而唯一能用的 `.docx` 恰好最容易自研（读 `style.name` 即可）。
为一个格式长期背 95 个直接依赖不划算。**实测导入整个 app 后
`unstructured` 已不再被加载**，可从环境中移除。

#### 入库步骤

```
任意文件
   │
   ├─ 1. 解析        parsers.parse(path) → list[Chunk]（带 headings）
   │
   ├─ 2. 算 doc_id   sha1(原始字节)[:16]     ← 只由字节决定，与解析/切分方式无关
   │
   ├─ 3. 向量化      embed_documents([c.text]) → N × 1024
   │
   ├─ 4. 组装写入    client.upsert([
   │                    {id: "{doc_id}:{i}", dense, text, source, section,
   │                     parent_text, chunk_id}
   │                 ])
   │                  ← sparse 不用给：BM25 Function 从 text 自动生成
   │                  ← section / parent_text 落在 Milvus 动态字段
   │                  ← 摘要**不在这里**生成，见下面「章节摘要」
   │
   └─ 5. 落盘        client.flush(collection)  ← 不 flush 立即检索不到
```

**幂等性**：`doc_id` 只由**原始字节**决定，因此换解析器或改切分参数后重复入库
仍是覆盖同一批主键，既不会产生新 `doc_id`，也不会留下幽灵记录（见 §6.4）。

**表格的处理原则**：绝不把二维表格原样塞进向量。Excel 的每一行转成
`列名：值；列名：值`，Word 两列表格转成 `键：值`——要让向量看到**语义**，
而不是一串裸露的数字。

### 7.2 检索流程（retrieve）

**阶段一（本期实现）**：稠密单路

```
query
  → embed_query(query) → 1024 维
  → client.search(
        collection_name,
        data=[vector],
        limit=retrieve_top_k,
        output_fields=["text", "source", "doc_id", "chunk_index"],
        search_params={"metric_type": "COSINE", "params": {"ef": 64}},
    )
  → [{text, source, doc_id, chunk_index, score}, ...]
```

**阶段三（后续）**：BM25 混合

```
稠密路: embed_query(query) → AnnSearchRequest(anns_field="vector", param=COSINE)
稀疏路: 原始查询文本     → AnnSearchRequest(anns_field="sparse", param=IP)
                                    ↑ 注意传文本，不是向量（服务端分词）
  → client.hybrid_search(reqs=[稠密, 稀疏], ranker=RRFRanker(), limit=fused_top_k)
```

需要在集合上增加：`text` 字段 `enable_analyzer=True`、`Function(BM25)` 生成 `sparse` 字段、`sparse` 建 `SPARSE_INVERTED_INDEX`。**中文必须显式配分词器**，否则 BM25 失效。

### 7.3 问答流程（qa）

```
用户问题
  → retrieve(question)              # 检索片段
  → build_context(hits)             # 拼成 "[片段1] 来源:xxx\n内容"
  → build_messages(question, context, history)
  → llm.stream(messages)            # 逐 token
  → yield 裸文本 token              # ← 只产出文本，不关心传输格式
  → UI 层: st.write_stream(...)     # Streamlit 负责渲染
  → 展示来源列表
```

**层次修正**：当前 `app/services/qa.py` 的 `sse_generator` 直接产出 `data: {...}\n\n` 这种 SSE 字符串，属于层次错位——Service 层不该知道传输格式。改为输出裸 token 后：

```python
# 现在（Streamlit）
st.write_stream(qa.stream_answer(q, history))

# 将来若要加 FastAPI，只在这一层包装
yield f"data: {json.dumps({'delta': t})}\n\n"
```

同时 `stream_answer` 由 `async def` 改为 `def`（同步生成器）——Streamlit 的 `st.write_stream` 消费同步生成器，且 LangChain 的 `.stream()` 本就是同步的，异步只在 FastAPI 场景需要。

### 7.4 上下文组装

```
[片段1] 来源：E:\...\knowledge.txt
<正文>

[片段2] 来源：E:\...\knowledge.txt
<正文>
```

带来源标注有两个作用：一是便于 UI 展示引用，二是让模型能区分不同片段，减少跨片段混淆。

**长度控制**：设总字符上限（如 6000），超出则截断，避免超出模型上下文窗口。

### 7.5 上传流程与文件落盘

界面上传（Streamlit）与 CLI 入库是**两个入口，共用同一套 ingest 逻辑**，区别只在文件从哪来。

```
Streamlit st.file_uploader
   │
   ├─ 1. 读入内存        UploadedFile.read() → bytes
   │
   ├─ 2. 解码为文本      load_bytes(name, data)  ── 按后缀分派
   │
   ├─ 3. 算 doc_id       sha1(文本内容)[:16]
   │
   ├─ 4. 落盘            data/uploads/{doc_id}{后缀}
   │                     ↑ 用 doc_id 命名，同内容天然只存一份，不会重名冲突
   │
   └─ 5. 交给 ingest     切块 → 向量化 → Milvus upsert（同 §7.1）
```

**落盘决策（已确认）**：保留原始文件，命名为 `data/uploads/{doc_id}{后缀}`。

理由与代价：

| 保留原文件的价值 | 不保留的代价 |
|---|---|
| 改 `chunk_size` 后能干净重跑入库 | 需从 Milvus 拼回全文，还要处理切块 overlap 去重 |
| PDF / docx 的原始保真度（表格、排版） | 只剩抽取出的纯文本 |
| 能下载回原文件 | 不能 |

> **注意**：正文本身**已经存在 Milvus 的 `text` 字段**里（见 §6.1）。落盘保的不是"内容"，而是**原始格式**与**可重跑性**。

**一致性要求**：删除文档时**必须同时删除** Milvus 记录与落盘文件，否则产生孤儿文件。删除顺序：先删 Milvus（可回滚失败），再删文件；若文件删除失败仅记录警告，不影响主流程。

**去重行为（已确认）**：`doc_id = sha1(内容)[:16]`，因此**内容相同、文件名不同的两个文件视为同一份文档**。第二次上传会覆盖 `source` 字段为后者的文件名，集合中仍只有一份。这是知识库场景期望的去重行为，但需在 UI 上提示，避免误判为丢失。

---

## 8. 配置设计

统一走 `pydantic-settings`，从项目根 `.env` 读取，**禁止硬编码密钥**。

| 分组 | 配置项 | 默认值 |
|---|---|---|
| 对话模型 | `deepseek_api_key` / `deepseek_base_url` / `deepseek_model` | 从 `.env` |
| 对话模型 | `mimo_api_key` / `mimo_base_url` / `mimo_model` | 从 `.env` |
| 生成参数 | `temperature` / `max_tokens` | 0.3 / 2048 |
| 向量化 | `siliconflow_api_key` / `siliconflow_base_url` / `siliconflow_embedding_model` | 从 `.env` |
| 向量化 | `embedding_dim` / `embedding_batch_size` | 1024 / 32 |
| Milvus | `milvus_uri` / `milvus_collection_name` | `http://127.0.0.1:19530` / `personal_knowledge` |
| 切块 | `chunk_size` / `chunk_overlap` | 600 / 80 |
| 检索 | `retrieve_top_k` | 5 |
| 检索（预留） | `vector_top_k` / `bm25_top_k` / `fused_top_k` / `rerank_top_k` | 20 / 20 / 20 / 5 |
| 路径 | `data_dir` / `docs_dir` / `uploads_dir` | 派生属性 |

**设计要点**：

- 用 `get_settings()` + `@lru_cache` 保证 `.env` 只解析一次
- 派生路径用 `@property`，不写进 `.env`
- 预留混合检索参数，阶段三直接用，不改配置结构

---

## 9. 关键决策记录

| 编号 | 决策 | 理由 | 备选与否决原因 |
|---|---|---|---|
| D1 | 向量库用**原生 pymilvus** | 用户要求；透明可控；避免双抽象 | `langchain_milvus` 更省事，但封装较深 |
| D2 | 前端只用 **Streamlit**，不引入 FastAPI | 同进程直调，省掉 SSE + HTTP + 前端三套代码 | FastAPI 可对外供 API，但本场景无此需求 |
| D3 | 主键用 `doc_id:chunk_index` | 多文件不冲突；重复入库幂等；可整份删除 | 自增 int 会覆盖 + 残留 |
| D4 | `doc_id` = 内容 sha1 前 16 位 | 内容不变则稳定，内容变则换新 | 路径（改名重复）、mtime（touch 就变） |
| D5 | Service 层输出**裸 token**，不输出 SSE | 层次正确，传输格式归 UI 层 | 原实现直接拼 SSE 字符串 |
| D6 | 流式用**同步**生成器 | Streamlit 消费同步生成器；异步仅 FastAPI 需要 | `async def` 在 Streamlit 下要多套 `asyncio` |
| D7 | 切块 `600/80` | 中文 600 字语义较完整 | demo 的 200 偏碎 |
| D8 | 检索能力**分阶段**上 | 先有 dense baseline，才能判断混合检索是否真的提升 | 一次全上无法归因 |
| D9 | 集合名用**独立**的 `personal_knowledge` | 不动 `E:\Milvus` 里已有的 147 MB 数据 | 复用旧集合可能撞 schema |

---

## 10. 风险与待验证项

> **阶段 0 已于实施前完成验证**，结论见下表「状态」列。

| 编号 | 风险 | 状态 | 结论 / 缓解措施 |
|---|---|---|---|
| **R1** | pymilvus **2.6.12** 客户端 vs Milvus **3.0.2** 服务端，跨大版本 | ✅ **已解除** | 实测连通：`get_server_version()` 正确返回 `3.0.2`，`list_databases()` 正常。**版本不匹配不是问题** |
| **R2** | `.env` 的 `DEEPSEEK_MODEL=deepseek-v4.1-flash` 实测 **400**，服务端只认 `deepseek-flash`/`deepseek-v4-pro` | ✅ **已缓解**（未改 `.env`） | LLM 层已实现**供应商自动降级**：首选失败且未产出内容时，自动切到下一个可用供应商。配置类错误（4xx）会被记入 `_broken_providers`，本次会话内不再重试，避免每次提问都白跑一次失败请求。实测：首次调用自动降级成功，第二次直接走备选（1.94s） |
| **R3** | BM25 可用性未知 | ✅ **已解除** | 实测建集合成功：`analyzer_params={"type":"chinese"}` + `FunctionType.BM25` + `SPARSE_INVERTED_INDEX` 全部通过。**阶段三可行** |
| **R4** | `E:\Milvus\volumes` 已有 147 MB 数据，内容未知 | ✅ **已处理** | 扫描确认是教程实验数据（`knowledge.txt` 正文 + notebook 04 的 4 句测试数据）。已删除，当前 `collections = []` |
| **R5** | Docker daemon 需手动启动；沙箱禁止访问命名管道 | ⚠️ **保留** | 沙箱内 `docker` CLI 报 `Access is denied`（命名管道受限），需一次性提权。启动步骤：Docker Desktop → `E:\Milvus\standalone.bat start` |
| **R6** | 本地无重排模型，pip 受代理限制装不了 | ⚠️ **保留** | 跳过重排；若需要则调 SiliconFlow rerank API |
| **R7** | `docs/` 目录当前不存在 | ✅ **已处理** | 已创建并放入 `knowledge.txt` 作为测试数据 |

### 10.1 LangChain Loader 化带来的新依赖（实测踩坑）

改用 `langchain_community` 官方 Loader 后，暴露出三个必须记录的依赖问题：

| # | 问题 | 现象 | 处理 |
|---|---|---|---|
| **L1** | `Unstructured*` Loader 需要 NLTK 资源 | 首次使用报 `PermissionError: %APPDATA%\nltk_data`，**耗时 72 秒后失败** | 在 `app/config.py` 导入期把 `NLTK_DATA` 指向项目内 `.nltk_data` 并**预先建目录**；资源已预置（约 11 MB） |
| **L2** | `TextLoader(autodetect_encoding=True)` 需要 `chardet` | 未安装时报 `ModuleNotFoundError: No module named 'chardet'` | 从清华镜像安装 |
| **L3** | **`chardet` 7.x 与 LangChain 不兼容** | 报 `TypeError: FileEncoding.__new__() got an unexpected keyword argument 'mime_type'` | **必须固定 `chardet==5.2.0`** |

**L1 的根因值得记下**：`unstructured/nlp/tokenize.py` 会调用
`nltk.download("averaged_perceptron_tagger_eng")` 与 `nltk.download("punkt_tab")`。
`nltk` 只会写入**已存在且可写**的候选目录；目录不存在时它回退到 `%APPDATA%\nltk_data`
并尝试联网下载。仅设置 `NLTK_DATA` 环境变量**不够**——必须先把目录建出来。

**L3 的教训**：直接 `pip install chardet` 会装到 7.x 并立即炸掉编码探测。
复现命令：

```powershell
.venv\Scripts\python.exe -c "from langchain_community.document_loaders import TextLoader; TextLoader(r'docs\knowledge.txt', autodetect_encoding=True).load()"
```

### 10.2 阶段 0 附带处理的事项

| 事项 | 说明 |
|---|---|
| 容器重建 | 删除旧卷时 Docker Desktop 已在运行，导致容器挂载句柄失效并以 `ExitCode 134` 崩溃（`panic: failed to mkdir`）。已删除旧容器并按 `standalone.bat` 的等价参数**重建** `milvus-standalone`，现运行正常 |
| 卷目录 | `E:\Milvus\volumes\milvus` 已清空重建，Milvus 全新初始化 |
| 保留文件 | `E:\Milvus\embedEtcd.yaml`、`user.yaml`、`standalone.bat` **未改动**，容器重建后仍可正常使用 |

### 10.3 ~~已知检索局限：长块稀释~~ → 已更正为 §6.5 的 embedding 缺陷

> **⚠️ 本节结论已被推翻，保留仅作记录。**
>
> 当时观测到的现象（价格块排 30/45、换任何切块尺寸都进不了 top-3）**全部属实**，
> 但归因错了。那不是「长块稀释」，而是 **embedding 一直在给乱码算向量**（见 §6.5）。
> 修复 `check_embedding_ctx_length=False` 之后，**同样的切分、同样的 `CHUNK_SIZE=200`**，
> 「基础版多少钱」对应的片段直接排到 **第 1 名（cos 0.713）**。
>
> **教训**：当「怎么调参数都救不回来」时，应先怀疑数据链路，而不是先怀疑算法。
> 下面那张「换切块尺寸实测表」其实是最该触发怀疑的证据——分数随切块尺寸
> 无规律地跳（0.58 → 0.53 → 0.27 → 0.23 → 0.29），本身就说明相似度不可靠。
> 我当时却把它当成了「dense-only 的结构性局限」。

按教程设置（`CHUNK_SIZE=200`、`RETRIEVE_LIMIT=3`）实测发现：**问「基础版多少钱」答不出来**，
而知识库里明确写着「基础版 - 价格：99 元 / 用户 / 月」。

**根因不是切块丢内容，而是「长块稀释」** —— 含价格的块被埋在 11 条无关属性中，
其向量被稀释；而一句 61 字的引言反而成为最强匹配。

| 文本 | 长度 | 与「基础版多少钱」的余弦 |
|---|---|---|
| chunk 4：`当前标准订阅套餐分为四档：试用版、基础版、专业版、企业版。` | **61 字** | **0.7131** |
| `基础版价格：99 元 / 用户 / 月`（单独成块时） | 短 | 0.6950 |
| chunk 6：`2. 基础版 - 价格：99 元 … - 成员人数上限 … - 月度额度 …`（含价格） | **171 字** | **0.2298** |

**换切块尺寸无法解决**（实测价格块的排名）：

| CHUNK_SIZE | 块数 | 含价格块排名 | 分数 | top-3 是否含价格 |
|---|---|---|---|---|
| 60 | 137 | 58 / 137 | 0.5845 | ❌ |
| 100 | 81 | 40 / 81 | 0.5321 | ❌ |
| 150 | 62 | 39 / 62 | 0.2722 | ❌ |
| **200（教程值）** | **45** | **30 / 45** | **0.2298** | ❌ |
| 400 | 24 | 13 / 24 | 0.2896 | ❌ |
| 600 | 13 | 7 / 13 | 0.2898 | ❌ |
| 800 | 9 | 5 / 9 | 0.2987 | ❌ |

缩小切块确实把价格块的分数从 0.23 提到 0.58，但引言块始终稳定在 ~0.71，因此**任何切块尺寸都进不了 top-3**。

**这是 dense-only 检索的结构性局限**，对应阶段 3（BM25 混合检索）要解决的问题：
`价格` / `99 元` 这类词能靠词法匹配直接命中，不必依赖语义相似度。

**当前表现**：Agent 会如实回答「无法回答」而不是编造 —— 这是期望行为（系统提示词要求不幻觉），
但知识库明明有答案，属于**召回不足**而非生成问题。

---

## 11. 分阶段实施计划

```
阶段 0  环境验证（不写业务代码）                        ✅ 已完成
        ├─ 启动 Docker Desktop + E:\Milvus\standalone.bat start
        ├─ 验证 pymilvus 2.6.12 能否连上 Milvus 3.0.2          → 解 R1 ✅
        ├─ list_collections() 查明现有 147 MB 内容              → 解 R4 ✅
        └─ 试建带 BM25 的测试集合，验证后即删                    → 解 R3 ✅
        ✔ 产出：R1/R3/R4 全部解除，阶段三可行

阶段 1  dense-only 端到端（本期主体）                  ✅ 已完成
        ├─ config.py / embeddings.py / llm.py
        ├─ core/milvus.py   建集合 / 入库 / 检索 / 列文档 / 删除
        ├─ services/ingest.py（+ 根目录 ingest.py CLI）
        ├─ services/qa.py   真检索 + 裸 token 生成器
        └─ ui/streamlit_app.py   Streamlit 界面
        ✔ 产出：能问答、能带来源 → 验收 13 项全部通过（§12）

阶段 2  灌数据实测                                     ← 当前
        ├─ 导入 knowledge.txt                                  （已完成，11 块）
        ├─ 验证：重复导入不翻倍、删除后检索不到                  （已完成）
        └─ 用真实问题评估召回质量                              （待做）
        ✔ 产出：baseline 数据

阶段 3  混合检索（前提已满足：R3 验证通过）
        └─ BM25 + RRF
        ✔ 产出：与 baseline 对比，确认是否真的变好

阶段 4  查询侧优化（可选，按需）
        ├─ 多查询改写（用 llm.generate）
        ├─ 元数据过滤（按 source / 时间）
        └─ 父子块（小块检索、大块喂上下文）
```

**阶段 0 结论**：R1 与 R3 均已实测通过，无阻塞项。可按阶段 1 继续推进。

---

## 12. 验收标准

### 阶段 1 验收 — ✅ 全部通过（按教程重写后复验）

| # | 检查项 | 通过标准 | 实测结果 |
|---|---|---|---|
| 1 | 依赖自检 | 全部模块可导入 | ✅ 6 个模块无报错 |
| 2 | Milvus 连接 | 连接不抛异常 | ✅ 2.6.12 客户端 ↔ 3.0.2 服务端 |
| 3 | 建集合 | 快速建表 + 字符串主键 | ✅ `id` VARCHAR(128) / `vector` dim=1024，其余动态字段 |
| 4 | 入库 | 记录数 > 0 | ✅ `knowledge.txt` → **45 块**（教程设置 `chunk_size=200`） |
| 5 | 幂等 | 重复导入记录数不变 | ✅ 连续 2 次入库均为 **45 块**（doc_id `983b7af34cd663bd`） |
| 6 | 检索 | 返回教程格式结果 | ✅ `hit["entity"]["text"]` / `hit["distance"]` 取值正常 |
| 7 | 问答 | Agent 流式产出 | ✅ `create_agent` + `agent.stream` 正常输出 |
| 8 | 来源 | UI 显示引用来源 | ✅ 展示来源 + 片段号 + 相似度 |
| 9 | 删除 | 按主键前缀删净 | ✅ `id like "{doc_id}:%"` 删除生效 |
| 10 | 上传落盘 | `data/uploads/{doc_id}{后缀}` 存在 | ✅ `8ebdaa089b1e0481.docx` |
| 11 | 内容去重 | 同内容异名 → 仍 1 篇 | ✅ 两次上传 doc_id 相同，uploads 仅 1 份 |
| 12 | 删除同步 | 落盘文件同时删除 | ✅ 文件被清理 |
| 13 | UI | 可打开并完成操作 | ✅ `AppTest` 无头自检 0 异常 |
| 14 | **Loader 分派** | 各格式都有对应 LangChain Loader | ✅ txt(utf-8/GBK) / md / pdf / docx / html / csv / json 全部通过 |
| 15 | **嵌入/存储正确性** | 向量无损 | ✅ 接口确定性 cos=1.000000，存储往返 cos=1.000000 |
| 16 | 召回质量 | 能答出库内问题 | ❌ **未通过** —— 「基础版多少钱」召回不到价格块，见 §10.3 |

> **第 16 项未通过是已知局限**，不是新引入的缺陷：按教程设置（`chunk_size=200` + `RETRIEVE_LIMIT=3`）
> 时，含「99 元」的块排在第 30/45 位。该问题的解法是阶段 3 的 BM25 混合检索。


### 阶段 3 验收

| # | 检查项 | 通过标准 |
|---|---|---|
| 1 | BM25 生效 | 含专有名词/编号的查询，召回优于 dense-only |
| 2 | 不退化 | 常规语义查询质量不低于 baseline |

---

## 13. 已确认决策

| # | 事项 | 决策 |
|---|---|---|
| 1 | 上传文件是否落盘 | **落盘**，命名 `data/uploads/{doc_id}{后缀}`（§7.5） |
| 2 | 内容相同、文件名不同 | **按内容去重**，视为一份文档（§7.5） |
| 3 | 主键方案 | `{doc_id}:{chunk_index}`，`doc_id = sha1(内容)[:16]`（§6.2） |
| 4 | 前端选型 | **仅 Streamlit**，不引入 FastAPI（§9 D2） |
| 5 | 向量库客户端 | **原生 pymilvus**，不用 `langchain_milvus`（§9 D1） |
| 6 | `.env` 的 DeepSeek 模型名 | **不改**，仅做降级兜底（§10 R2） |
| 7 | `requirements.txt` | 本期不动 |

## 14. 待确认事项

1. **上一轮已写的 3 个文件**（`config.py` / `embeddings.py` / `llm.py`）是否保留？可回退。
2. **`app/models/schemas.py`** 是否删除？（Streamlit 无 HTTP 层，Pydantic 请求/响应模型已无用）
3. **`E:\Milvus\volumes` 里的 147 MB 数据**：阶段 0 查明后再决定是否清理。
4. **`data/chroma/`** 旧方案残留空目录，建议一并删除。
