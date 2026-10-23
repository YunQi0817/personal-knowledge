# 个人知识库 RAG 问答 Demo

基于 **LangChain 1.2 + Milvus + Streamlit** 的最小可用 RAG 问答服务。
上传文档 → 切分 → 向量化 → 存入 Milvus → 检索 → Agent 流式作答。

设计说明见 [DESIGN.md](DESIGN.md)。

---

## 一、前置条件

| 依赖 | 版本 | 说明 |
|---|---|---|
| Python | 3.11 | 虚拟环境 `.venv` 已建好 |
| Docker Desktop | 28.x | Milvus 跑在容器里，**必须先启动** |
| Milvus | v3.0.2 | 镜像 `milvusdb/milvus:v3.0.2`，安装目录 `E:\Milvus` |
| `.env` | — | 已存在，含 `MIMO_API_KEY`、`SILICONFLOW_API_KEY` |

> 三个服务都必须就位：**Docker Desktop → Milvus → Streamlit**。少一个都跑不起来。

---

## 二、启动（每次开机后三步）

### 第 1 步：启动 Docker Desktop

```powershell
Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
```

等托盘图标变绿，确认引擎可用：

```powershell
docker info
```

> 报 `open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified`
> 就是 Docker Desktop 还没起来，继续等。

### 第 2 步：启动 Milvus

```powershell
cd E:\Milvus
.\standalone.bat start
```

- 容器 `milvus-standalone` 已存在 → 直接 `docker start`
- 容器不存在 → 自动 `docker run` 新建（内嵌 etcd，数据落在 `E:\Milvus\volumes\milvus`）
- 脚本会轮询健康检查，看到 `Start successfully.` 即可

确认端口就绪（通常 10 秒内）：

```powershell
Test-NetConnection localhost -Port 19530 -InformationLevel Quiet
```

> 返回 `True` 才算好。返回 `False` 再等几秒重试。

### 第 3 步：启动 Streamlit

```powershell
cd "C:\Users\27173\Desktop\Personal knowledge"
.venv\Scripts\python.exe -m streamlit run app/ui/streamlit_app.py
```

浏览器打开 **http://localhost:8501**

> ⚠️ 必须用 `streamlit run`，直接 `python app/ui/streamlit_app.py` 不行。
> 工作目录必须是项目根目录，否则 `import app` 会失败。

---

## 三、首次安装（换机器 / 重建环境时才需要）

```powershell
cd "C:\Users\27173\Desktop\Personal knowledge"

# 1. 建虚拟环境
python -m venv .venv

# 2. 装依赖（约 5 分钟）
.venv\Scripts\python.exe -m pip install -r requirements.txt
#
# requirements.txt 就是本项目运行所需的最小集合（19 个包）。
# 教程学习环境的全量清单在 docs/tutorial-requirements*.txt，与本项目运行无关。

# 3. 配密钥：复制下面的模板为 .env 并填入真实值
```

`.env` 需要这些键：

```ini
MIMO_API_KEY=...            # 对话模型
MIMO_BASE_URL=https://token-plan-cn.xiaomimimo.com/v1
MIMO_MODEL=mimo-v2.6-flash

SILICONFLOW_API_KEY=...     # 向量模型
SILICONFLOW_BASE_URL=...

TAVILY_API_KEY=...          # 可选，当前未启用
```

> `chardet` 必须是 `5.2.0`。7.x 会让 `UnstructuredExcelLoader` 等加载器报
> `FileEncoding.__new__() got an unexpected keyword argument 'mime_type'`。

---

## 四、导入文档

### 方式 A：界面上传（推荐）

打开 http://localhost:8501 → 左侧栏 **📚 知识库** → 展开 → 上传文件 → 点「入库」。

支持：`txt` `text` `log` `md` `markdown` `docx` `pptx` `xlsx` `xlsm` `pdf` `html` `htm` `csv` `json`

### 方式 B：命令行批量导入

```powershell
# 导入 docs/ 下全部支持的文件
.venv\Scripts\python.exe ingest.py

# 只导入指定文件 / 目录（递归）
.venv\Scripts\python.exe ingest.py path\to\file.pdf
.venv\Scripts\python.exe ingest.py path\to\folder

# 重灌 data/uploads/ 里的全部上传件（清库后恢复用，保留原始文件名）
.venv\Scripts\python.exe ingest.py --uploads

# 预生成各章节摘要（可选；不做的话第一次"总览"时现场生成）
.venv\Scripts\python.exe ingest.py --summaries
```

**幂等**：`doc_id` 取自文件**原始字节**的哈希，重复导入不会产生重复数据，只覆盖同内容的旧记录。

### 上传件存在哪、怎么恢复

界面上传的文件会落两份到 `data/uploads/`：

```
data/uploads/
├── 87061e12bdcb68f6.docx     # 原件（按内容哈希命名）
└── 87061e12bdcb68f6.json     # 身份文件：原始文件名、大小、哈希、上传时间
```

**为什么要有身份文件**：原件按内容哈希命名，原始文件名就丢了。一旦清库，光有原件是恢复不出友好名字的——**内容回来了，身份丢了**。有了 `.json`，`data/uploads/` 才是一个**自包含的可重灌仓库**：

```powershell
# 清库
.venv\Scripts\python.exe -c "from app.core import milvus; milvus.drop_collection()"

# 一条命令全恢复（含原始文件名）
.venv\Scripts\python.exe ingest.py --uploads
```

所以上传的文档**不需要再往 `docs/` 里放一份**。

---

## 五、界面说明

界面是 AI 对话式的，分左右两栏：

| 位置 | 内容 |
|---|---|
| **左侧栏** | 「＋ 新建对话」、对话记录列表、📚 知识库管理（上传/删除）、⚙️ 设置（检索片段数、模型信息）、Milvus 连接状态 |
| **主区域** | 空对话时是欢迎页 + 4 个建议问题；有对话后是消息流。输入框由 Streamlit 固定在底部 |

几个使用要点：

- **多会话**：点「＋ 新建对话」开新会话，点侧栏的会话标题可切回，点行尾的 `✕` 删除（会先变成红色的「确认」，**再点一次才真删**，避免误删）。反复点「新建对话」不会堆积空会话——建新的之前会先把一条消息都没有的清掉。
- **对话记录会持久化**：存在 `data/conversations.db`（SQLite），**重启项目、刷新页面都还在**。见下面的说明。
- **引用来源**：每条回答下方有「📎 引用来源 · N 条」折叠区，展开是卡片列表，显示文件名、章节、相似度分数和片段摘要。
- **检索片段数**：在 ⚙️ 设置里，默认 3。问「四档的区别是什么」这类需要横向对比的问题时，调到 6~10 效果更好。

### 检索是怎么工作的：Small-to-Big

每个片段在库里存**两份文本**：

| 字段 | 内容 | 作用 |
|---|---|---|
| `text` | 小块（≤ 200 字） | **建向量、参与检索**——块越小越精确 |
| `parent_text` | 该块所属**章节的完整文本** | **送进模型**——上下文才完整 |

举例：某策划书「活动简介」一节有 8 个独立条目（主办方 / 时间 / 地点 / 类型 / 级别 / 指导老师 / 负责人 / 参与者）。如果整节挤成一块，查「活动时间」要跟另外 7 条竞争，正确答案会掉到第 7 名。所以解析时按 `（一）（二）` 把小节切开 —— 但切太碎又会让「活动的流程是怎样的？」这类需要**整段上下文**的问题变差（实测 10/12 → 9/12）。

**Small-to-Big 让两者兼得**：检索命中精确的小块，送进模型的是完整章节。

| | 只喂小块 | Small-to-Big |
|---|---|---|
| 策划书 12 题集 | 9/12 | **11/12** |
| 客服知识库 5 题集 | 5/5 | **5/5** |

代价是存储翻倍（纯文本冗余，不占向量维度）。同一章节的多个命中只保留一份，避免重复占满上下文。

### 检索用的是混合检索（稠密 + BM25）

Milvus 3.0 原生支持：`text` 字段开了**中文分析器**，一个 `BM25 Function` 从它自动生成稀疏向量（**不用手写**），检索时两路各取一批、用 **RRF** 融合。

**说句实话：在当前语料规模下，它没有带来可测量的收益。**

| 评测 | 纯稠密 | 混合 |
|---|---|---|
| 策划书 12 题 | 11/12 | **11/12** |
| 客服知识库 5 题 | 5/5 | **5/5** |
| 精确串专项 10 题（日期/人名/编号/金额） | 10/10 | **10/10** |

两套题都打平。原因是语料太小（67 块），稠密检索本来就够用；加上 Small-to-Big 送进去的是整节，排序稍有偏差也能被兜住。

**那为什么还留着**：它是**规模变大后的保险**。文档上百份、出现生僻编号或术语时，词法匹配的价值才会显现。查询延迟实测 106 ms（稠密 167 ms，差异在噪声范围内），代价可忽略。

想对比两种模式，代码里留了开关：`qa.retrieve(query, mode="dense")`。

### 只有一个 Agent，工具由它自己选

问答路径上**没有固定管道**。检索能力被包装成工具交给模型，由它决定要不要检索、检索几次、用哪个：

| 工具 | 何时用 |
|---|---|
| `search_knowledge` | 知识库相关问题（**强制先检索**） |
| `list_documents` | 问「有哪些文档」 |
| `summarize_document` | 要求总结某份文档 |
| `web_search` | 知识库确实没有、且需要外部信息 |

代码分工：`services/tools.py` 定义工具，`services/agent.py` 组装 Agent，
`core/llm.py` 只管连模型。**工具不能写在 `core/`**——它要调 `services.qa`，
而 `qa` 又要 `core.llm`，会形成循环导入。

界面上能看到 Agent 做了什么：处理中显示「正在检索知识库…」，完成后折叠成一行工具名。

> ⚠️ **系统提示词必须保持精简。** 实测把提示词写成 300 字长段落时，
> 模型有 **40%** 概率把工具调用写成 XML 而不是 JSON，产生无效往返、延迟翻 3 倍。
> 改成要点式后降到 0%。改动前请跑 `eval/agent/diag_prompt_length.py`。

**代价**：Agent 比固定管道慢 2.6~3.6 倍（至少多一次模型往返）。这个数据是在合并前
用双路径 A/B 测的，记录在 `eval/README.md`「改动记录 2」。

### 摘要与总结

侧栏「📚 知识库」里有**「📋 知识库总览」**和**「总结这份文档」**两个入口。结果会作为一条助手消息写进对话记录，和普通问答一样被持久化。

**摘要不在入库时生成，而是惰性生成 + 缓存。** 这是实测后的决定：

| 步骤 | 耗时 |
|---|---|
| 解析 + 向量化 + 写入 | **约 4 秒** |
| 每个章节的摘要（一次 LLM 调用） | 约 2 秒 × 章节数 |

某 48 KB 的 docx 有 11 个章节，摘要要占 **21.8 秒 —— 全部入库时间的 84%**。更糟的是它把入库**绑死在外部 API 的稳定性上**：连续两批 21 次调用实测触发过限流重试，入库卡了十几分钟没有任何输出。

改成惰性之后：

```
入库        仍然 4 秒
第一次总结  多花约 70 秒（生成全部章节摘要）
之后        秒回（命中缓存）
```

**最终状态和"入库时预生成"完全一样**——摘要都落盘了，只是时机从"入库"挪到"第一次需要"，附带好处是不为从没总结过的文档付费。

摘要缓存在 `data/summaries.db`（SQLite）。为什么不用 Milvus：摘要需要**增量回填**，而 Milvus 的 `upsert` 要整行数据（含 1024 维向量），为写一段摘要重新上传向量既慢又别扭。

想在导入后一次性算好摘要（比如睡前挂着跑）：

```powershell
.venv\Scripts\python.exe ingest.py --summaries
```

### 对话记录存在哪

`data/conversations.db`，用的是 **SQLite**（Python 标准库自带，零新依赖）。两张表：

```
conversations(id, title, created_at, updated_at)
messages(id, conversation_id, role, content, sources, created_at)
```

`messages.sources` 存的是引用来源的 JSON。用 DB Browser for SQLite 之类的工具可以直接打开查看、导出。

**为什么不用 MySQL**：这个项目的定位是个人本地知识库，SQLite 的能力远超需求，而 MySQL 会立刻带来三样代价——启动步骤从 2 步变 3 步、多一个「没起来就连历史都读不到」的依赖、外加建库建表配账号。

真要做成多人服务时，那时要加的不只是数据库（还有用户体系、鉴权、API 层）。而现在**存储访问全部收在 `app/services/conversations.py` 一个文件里**（8 个函数），届时替换只需改那一个文件，界面层一行都不用动。

> 换 MySQL 时注意 SQL 方言差异：占位符 `?` → `%s`、`AUTOINCREMENT` → `AUTO_INCREMENT`、`ORDER BY rowid` 在 MySQL 里不存在、索引列不能是无限长 `TEXT`（要写 `VARCHAR(64)`）。

### 改样式时注意（都是实测踩过的坑）

样式全部集中在 `app/ui/theme.py`，只使用 `data-testid` 选择器。三条经验：

1. **不要用 `[class*="st-"]` 这类宽选择器设 `font-family`**。Streamlit 的自动类名形如 `st-emotion-cache-xxxx`，图标 `<span>` 也带一个，会被一起命中、覆盖掉 Material 图标字体，结果是 `keyboard_arrow_right` 这类连字被当普通文字渲染出来（侧栏里出现 `arr⬛w_r` 乱码）。
2. **给侧栏按钮设 `background: transparent` 时要小心主按钮**。否则「新建对话」会变成白字 + 透明底 = 看不见。会话列表项请用 `[class*="st-key-conv_"]` 精确圈定。
3. **Streamlit 1.66 的消息节点里没有头像 testid**（`stChatMessage` 下只有 `stChatMessageContent`），所以无法用 `:has(...)` 区分用户/助手。用户气泡是在 Python 侧套 `.kb-user` 容器实现的。

> 改完 `theme.py` 必须**重启** Streamlit——它是模块级常量，热重载不会重新导入。

---

## 六、目录结构

```
Personal knowledge/
├── app/
│   ├── config.py                  # 全局配置：Milvus 地址、切分参数、路径
│   ├── core/
│   │   ├── llm.py                 # 模型连接（MiMo）——只负责连模型，不含业务
│   │   ├── embeddings.py          # 向量模型（SiliconFlow bge-m3）
│   │   └── milvus.py              # Milvus 客户端：建集合/写入/检索/删除
│   ├── services/
│   │   ├── parsers/               # 解析层：任意格式 → 带章节的 Chunk
│   │   │   ├── base.py            #   Chunk 契约 + SectionBuilder 组装器
│   │   │   ├── plaintext.py       #   .txt .md .log   —— 识别围栏标题
│   │   │   ├── docx.py            #   .docx          —— 读 Heading 样式
│   │   │   ├── pptx.py            #   .pptx          —— 按幻灯片（含备注页）
│   │   │   ├── excel.py           #   .xlsx .xlsm    —— 按工作表，转「键：值」
│   │   │   └── legacy.py          #   .pdf .html .csv .json 的兜底
│   │   ├── tools.py               # Agent 工具：检索 / 列文档 / 总结 / 联网
│   │   ├── agent.py               # Agent 组装（模型 + 工具 + 提示词）
│   │   ├── ingest.py              # 解析 → 向量化 → 入库
│   │   ├── conversations.py       # 对话记录持久化（SQLite）
│   │   ├── summarize.py           # 章节摘要：惰性生成 + SQLite 缓存
│   │   └── qa.py                  # 检索/上下文组装 + 流式事件（供界面）
│   └── ui/
│       ├── theme.py               # 界面样式（自定义 CSS）
│       └── streamlit_app.py       # 对话界面入口
├── data/uploads/                  # 上传件归档：{doc_id}.{后缀} 原件 + {doc_id}.json 身份
├── data/conversations.db          # 对话记录（SQLite，可用任意 SQLite 客户端查看）
├── data/summaries.db              # 章节摘要缓存（SQLite）
├── docs/                          # 批量导入的源文件目录
│   └── tutorial-requirements*.txt # 教程环境的全量清单（与运行无关）
├── eval/                          # 评测 harness（见 eval/README.md）
│   ├── run_eval.py                #   检索评测主程序
│   ├── debug_question.py          #   单题召回诊断
│   ├── questions/                 #   题集（客服 5 / 策划书 12 / 难题 8）
│   ├── results/                   #   历史结果存档
│   ├── agent/                     #   Agent 验收与提示词诊断
│   └── ui/                        #   界面端到端测试（AppTest）与 CSS 回归
├── ingest.py                      # 批量入库 CLI
├── requirements.txt               # 本项目运行依赖（装这个）
├── .gitattributes                 # 统一 LF 换行符
├── DESIGN.md                      # 设计文档
└── .env                           # 密钥（勿提交）
```

---

## 七、常用维护命令

```powershell
# 停止 Milvus
cd E:\Milvus; .\standalone.bat stop

# 重启 Milvus
cd E:\Milvus; .\standalone.bat restart

# 查看 Milvus 日志
docker logs -f milvus-standalone

# 停止 Streamlit
# 在运行它的窗口按 Ctrl+C
```

### ⚠️ 改了切分参数（CHUNK_SIZE / CHUNK_OVERLAP）之后

Milvus 的 `upsert` 是**逻辑删除 + 插入**，旧向量在被压缩前仍然参与检索，
会导致同一份文档以两种切分粒度同时命中。改切分参数后必须清空集合：

```powershell
.venv\Scripts\python.exe -c "from app.core import milvus; milvus.drop_collection(); print('done')"
```

然后重新导入。上传过的文件用 `--uploads` 一条命令就能全恢复：

```powershell
.venv\Scripts\python.exe ingest.py --uploads   # data/uploads/ 里的上传件
.venv\Scripts\python.exe ingest.py             # docs/ 里的本地文件
```

---

## 八、常见问题

| 现象 | 原因 / 解决 |
|---|---|
| `无法连接 Milvus` | Docker Desktop 没启动，或 Milvus 容器没起（回到第 1、2 步） |
| `ModuleNotFoundError: No module named 'app'` | 工作目录不对，必须在项目根目录下执行 |
| 上传 Excel/PDF 报编码错误 | `chardet` 版本不对，`pip install chardet==5.2.0` |
| `nltk_data` 权限报错 | 项目已把 `NLTK_DATA` 指到 `.nltk_data/`，确认该目录存在且可写 |
| 检索结果答非所问 | **先检查 `app/core/embeddings.py` 里的 `check_embedding_ctx_length=False` 还在不在**——这是本项目踩过最大的坑（向量一度算的是乱码），见 DESIGN.md §6.5 |
| 片段总数虚高 | 用 `count(*)` 查询而非 `get_collection_stats()`，代码已修正 |

---

## 九、验证服务是否正常

```powershell
# Milvus 端口
Test-NetConnection localhost -Port 19530 -InformationLevel Quiet

# Streamlit 健康检查
(Invoke-WebRequest http://localhost:8501/_stcore/health -UseBasicParsing).Content
# 期望输出：ok

# 库内数据量
.venv\Scripts\python.exe -c "from app.core import milvus; print('片段总数：', milvus.count_chunks()); print('文档篇数：', len(milvus.list_documents()))"
```
