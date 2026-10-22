# 评测 harness

一套可复跑的检索/问答评测，用来回答一个问题：**这次改动让系统变好了还是变坏了？**

没有评测，任何检索策略调整（换切分参数、加 rerank、改 top-k）都只能靠"感觉好像好一点"来判断。

---

## 快速开始

在**项目根目录**执行：

```powershell
# 1) 只校验题集（不连服务，改完题目先跑这个）
.venv\Scripts\python.exe eval\run_eval.py --check

# 2) 检索评测（需要 Milvus + embedding API）
.venv\Scripts\python.exe eval\run_eval.py

# 3) 对比纯稠密 vs 混合检索
.venv\Scripts\python.exe eval\run_eval.py --mode dense
.venv\Scripts\python.exe eval\run_eval.py --mode hybrid

# 4) 把召回提到 5 条再测
.venv\Scripts\python.exe eval\run_eval.py --limit 5

# 5) 连生成答案一起评测（额外需要 LLM API）
.venv\Scripts\python.exe eval\run_eval.py --with-answer

# 6) 存档结果，便于和下次对比
.venv\Scripts\python.exe eval\run_eval.py --json eval\results\before.json
```

> 评测前先确认 Milvus 在跑：`Test-NetConnection localhost -Port 19530 -InformationLevel Quiet`

---

## 指标定义

| 指标 | 含义 | 怎么算 |
|---|---|---|
| **召回命中率** | 正确事实是否出现在送进模型的上下文里 | 该题全部 `expected_facts` 都出现在召回内容的并集中，记 1 次命中 |
| **MRR** | 第一个"完全命中"的片段排在第几名的倒数 | 排第 1 → 1.0，排第 3 → 0.333，没命中 → 0 |
| **P50 / P95 延迟** | 单题检索耗时 | 只计检索，不含生成 |
| **章节多样性** | top-k 覆盖了多少个不同章节 | 不同章节数 ÷ 召回条数；1.0 表示每条都来自不同章节 |
| **答案命中率** | 生成答案里是否包含全部正确事实 | 仅 `--with-answer` 时统计 |

### 为什么用「事实包含」而不是「片段 ID 匹配」

因为段落的 `section` 命名会随解析器调整而变（改一次 heading 识别，所有 ID 全失效）。事实字符串是**语义锚点**，只要文档没改、答案没变，评测就稳定。

### 归一化

比较前对文本做 NFKC + 去空白处理，所以 `10 人` 与 `10人`、全角 `：` 与半角 `:` 视为相同。这避免了空格和全角符号造成的假阴性。

---

## 题集

| 题集 | 文件 | 题数 | 语料 |
|---|---|---|---|
| 客服知识库 | `questions/cs-5.json` | 5 | `docs/knowledge.txt` |
| 活动策划书 | `questions/drawing-12.json` | 12 | `data/uploads/87061e12bdcb68f6.docx` |

每题结构：

```json
{
  "id": "cs-01",
  "question": "基础版支持多少个正式成员？",
  "expected_facts": ["10 人", "已激活"],
  "expect_section": "四、成员与权限规则",
  "notes": "为什么出这道题"
}
```

- `expected_facts`：**必须全部出现**才算命中。多于一个事实的题，考查的是"能不能一次召回齐"，比单词匹配更严格。
- `expect_section`：目前只用于诊断输出，不参与打分。
- `notes`：给人看的，说明这题的坑在哪。

### 加新题

1. 在对应 JSON 的 `questions` 数组里追加一条
2. 跑 `--check`，它会检查每个 `expected_facts` 是否**真的出现在原文里**——写错一个字都会被抓出来
3. 跑一次完整评测

---

## `--check` 为什么重要

它会读取题集声明的语料原文，逐条核对 `expected_facts` 是否存在。

如果某个事实在原文里根本找不到，评测会**永远判错**——而且看起来像是"检索能力不行"，实际是题目写错了。这类假阴性极难排查，所以在写题阶段就拦掉。

---

## 当前基线（2026-10-22 实测）

语料：`knowledge.txt` + 策划书 docx。limit=3。

### 基础题集（回归保护）

| 题集 | 模式 | 召回 | MRR | P50 | P95 |
|---|---|---|---|---|---|
| cs-5 + drawing-12（17 题） | hybrid | **17/17** | 1.000 | 99.4ms | 131.0ms |
| cs-5 + drawing-12（17 题） | dense | **17/17** | 1.000 | 99.1ms | 109.3ms |

> ⚠️ **两份基础题集已饱和**（两种模式都满分）。它们现在只能当**回归护栏**用——防止改动把本来能答对的搞坏，但**测不出改进**。

### 难题集（区分度来源）

| 模式 | 召回 | MRR | 章节多样性 | P50 |
|---|---|---|---|---|
| hybrid | 8/8 | **0.938** | **0.71** | 98.9ms |
| dense | 8/8 | 0.917 | **0.75** | 97.7ms |

命中率同样饱和，但 **MRR 与章节多样性仍有区分度**，是当前做 A/B 对比主要看的两项。

### 已发现的可优化点

**top-3 的章节多样性只有 0.71 ~ 0.75**，意味着每次召回里约有 25% 是同一章节的重复小块。

后果：Small-to-Big 按 `(source, section)` 去重后，送进模型的章节数被压缩——本该覆盖 3 个章节的上下文实际只有 2 个左右，横跨多节的问题（如 hd-04）容易漏。

可能的改法：
1. 检索阶段就做章节级去重，凑够 `k` 个**不同章节**再返回
2. 引入 MMR / 多样性惩罚
3. 在候选集上做 rerank，但保证章节分散

> 改完务必重跑 `hard8`，比较 MRR 与多样性，别只看命中率。

---

## 已知局限

1. **语料太小**（67 个片段）。召回率接近饱和后，指标对改动不敏感——这是真实的，不必为了好看去凑大语料。
2. **事实匹配是字符串级的**。答对了但换了说法（"30 天" vs "一个月"）会被判错。所以在写 `expected_facts` 时优先选**数字、专有名词**这类不会换说法的词。
3. **不评测推理质量**。答案是否通顺、有没有幻觉，这套 harness 不管；它只管"正确事实有没有出现"。
4. **题集规模小**（17 题）。单题权重高，随机波动明显；做 A/B 对比时结论要看差值而不是绝对值。

---

## 建议用法

改检索策略前后各跑一次，用 `--json` 存档，对比：

```powershell
.venv\Scripts\python.exe eval\run_eval.py --json eval\results\01-baseline.json
# ... 改代码 ...
.venv\Scripts\python.exe eval\run_eval.py --json eval\results\02-after.json
```

两次的 `overall` 段直接对比即可。**如果改动没有带来可测量提升，就诚实记录"无显著差异"，而不是强行归因。**


