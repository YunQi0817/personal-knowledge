#!/usr/bin/env python
"""检索与问答评测 harness。

在**项目根目录**执行（需要能 import app）：

    python eval/run_eval.py --check                 # 只校验题集，不连 Milvus / 不调 API
    python eval/run_eval.py                         # 检索评测（需 Milvus + embedding API）
    python eval/run_eval.py --mode dense            # 对比纯稠密检索
    python eval/run_eval.py --limit 5               # 改召回条数
    python eval/run_eval.py --with-answer           # 同时评测生成答案（需 LLM API）
    python eval/run_eval.py --json eval/results/1.json

指标定义见 eval/README.md。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

# app 包在项目根目录，脚本位置是 <root>/eval/run_eval.py
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

QUESTIONS_DIR = Path(__file__).resolve().parent / "questions"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

QUESTION_SETS = {
    "cs5": QUESTIONS_DIR / "cs-5.json",
    "drawing12": QUESTIONS_DIR / "drawing-12.json",
    "hard8": QUESTIONS_DIR / "hard-8.json",
}


# ---------------------------------------------------------------- 文本归一
def norm(text: str) -> str:
    """归一化后再比较，避免全角/半角、空格差异造成误判。

    NFKC 会把全角数字、全角冒号、全角括号统一成半角；
    再去掉所有空白，这样 "10 人" 与 "10人" 视为相同。
    """
    return "".join(unicodedata.normalize("NFKC", text).split())


def missing_facts(text: str, facts: list[str]) -> list[str]:
    """返回 text 里没有出现的事实。"""
    haystack = norm(text)
    return [fact for fact in facts if norm(fact) not in haystack]


# ---------------------------------------------------------------- 加载
def load_set(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("name", "questions"):
        if key not in data:
            raise ValueError(f"{path.name}: 缺少字段 {key}")
    ids = [q["id"] for q in data["questions"]]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path.name}: 存在重复的题目 id")
    for q in data["questions"]:
        for key in ("id", "question", "expected_facts"):
            if key not in q:
                raise ValueError(f"{path.name}: 题目 {q.get('id', '?')} 缺少字段 {key}")
        if not isinstance(q["expected_facts"], list) or not q["expected_facts"]:
            raise ValueError(f"{path.name}: 题目 {q['id']} 的 expected_facts 必须是非空列表")
    return data


def read_corpus_text(rel: str) -> str | None:
    """读取题目集对应的语料纯文本，用于 --check 校验事实是否存在。

    读不到就返回 None（只是少一层校验，不阻断）。
    """
    path = ROOT / rel
    if not path.exists():
        return None
    if path.suffix == ".docx":
        try:
            import docx  # python-docx
        except ImportError:
            return None
        try:
            doc = docx.Document(str(path))
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception:
            return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


# ---------------------------------------------------------------- 评测
def eval_retrieval(qa_module, questions: list[dict], limit: int, mode: str) -> list[dict]:
    rows = []
    for item in questions:
        started = time.perf_counter()
        try:
            hits = qa_module.retrieve(item["question"], limit=limit, mode=mode)
        except Exception as exc:  # Milvus 未启动 / API 不可用等
            raise RuntimeError(f"{item['id']} 检索失败：{exc}") from exc
        elapsed_ms = (time.perf_counter() - started) * 1000

        # 召回上下文：Small-to-Big 下送进模型的是 parent_text
        context_parts: list[str] = []
        rank = None
        for index, hit in enumerate(hits, start=1):
            entity = hit.get("entity", {})
            chunk_text = entity.get("parent_text") or entity.get("text") or ""
            context_parts.append(chunk_text)
            if rank is None and not missing_facts(chunk_text, item["expected_facts"]):
                rank = index

        context = "\n".join(context_parts)
        miss = missing_facts(context, item["expected_facts"])
        top_section = (hits[0].get("entity", {}) or {}).get("section", "") if hits else ""
        # 章节多样性：top-k 覆盖了多少个不同章节。全部落在同一章节时，
        # Small-to-Big 去重后上下文只剩一节，信息面会明显变窄。
        sections = {
            (h.get("entity", {}) or {}).get("section", "") for h in hits
        }
        diversity = len(sections) / len(hits) if hits else 0.0

        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "hit": not miss,
                "rank": rank,
                "mrr": 0.0 if rank is None else 1.0 / rank,
                "missing": miss,
                "top_section": top_section,
                "latency_ms": round(elapsed_ms, 1),
                "expect_section": item.get("expect_section", ""),
                "section_diversity": round(diversity, 2),
            }
        )
    return rows


def eval_answer(qa_module, questions: list[dict], limit: int) -> list[dict]:
    rows = []
    for item in questions:
        started = time.perf_counter()
        result = qa_module.answer(item["question"], limit=limit)
        elapsed_ms = (time.perf_counter() - started) * 1000
        answer = result.get("answer", "")
        miss = missing_facts(answer, item["expected_facts"])
        rows.append(
            {
                "id": item["id"],
                "answer_hit": not miss,
                "missing": miss,
                "answer_chars": len(answer),
                "latency_ms": round(elapsed_ms, 1),
                "answer_preview": answer[:160].replace("\n", " "),
            }
        )
    return rows


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def summarize(rows: list[dict], total: int) -> dict:
    hits = sum(1 for r in rows if r.get("hit"))
    latencies = sorted(r["latency_ms"] for r in rows)
    return {
        "total": total,
        "retrieval_hit": hits,
        "retrieval_hit_rate": hits / total if total else 0.0,
        "mrr": sum(r.get("mrr", 0.0) for r in rows) / total if total else 0.0,
        "latency_p50_ms": statistics.median(latencies) if latencies else 0.0,
        "latency_p95_ms": (latencies[int(len(latencies) * 0.95) - 1] if latencies else 0.0),
        "section_diversity": (
            sum(r.get("section_diversity", 0.0) for r in rows) / total if total else 0.0
        ),
    }


# ---------------------------------------------------------------- 输出
def print_rows(rows: list[dict]) -> None:
    print(f"  {'题目':<8} {'召回':<4} {'排名':<5} {'耗时':>8}  说明")
    for row in rows:
        mark = "✓" if row.get("hit") else "✗"
        rank = str(row["rank"]) if row.get("rank") else "-"
        note = "" if row.get("hit") else "缺：" + "、".join(row.get("missing", []))
        print(f"  {row['id']:<8} {mark:<4} {rank:<5} {row['latency_ms']:>7.1f}ms  {note}")


def main() -> int:
    parser = argparse.ArgumentParser(description="检索与问答评测 harness")
    parser.add_argument("--set", dest="sets", action="append", choices=[*QUESTION_SETS, "all"],
                        help="要评测的题集，可重复；默认 all")
    parser.add_argument("--limit", type=int, default=None, help="召回条数，默认取 app.config.RETRIEVE_LIMIT")
    parser.add_argument("--mode", choices=["hybrid", "dense"], default="hybrid", help="检索模式")
    parser.add_argument("--with-answer", action="store_true", help="同时评测生成答案（需 LLM API）")
    parser.add_argument("--check", action="store_true", help="只校验题集，不连服务")
    parser.add_argument("--json", dest="json_out", default=None, help="把完整结果写入 JSON 文件")
    args = parser.parse_args()

    selected = args.sets or ["all"]
    if "all" in selected:
        selected = list(QUESTION_SETS)

    # ---- 加载并校验 ----
    loaded: dict[str, dict] = {}
    problems: list[str] = []
    for name in selected:
        path = QUESTION_SETS[name]
        if not path.exists():
            problems.append(f"{name}: 题集文件不存在 {path}")
            continue
        try:
            loaded[name] = load_set(path)
        except Exception as exc:
            problems.append(f"{name}: {exc}")

    if problems:
        for p in problems:
            print(f"[ERROR] {p}")
        return 1

    total = sum(len(v["questions"]) for v in loaded.values())
    print(f"题集：{', '.join(loaded)}（共 {total} 题）")

    # ---- --check：校验题集本身 ----
    if args.check:
        ok = True
        for name, data in loaded.items():
            corpus = read_corpus_text(data.get("corpus", ""))
            print(f"\n[{data['name']}] {len(data['questions'])} 题")
            if corpus is None:
                print("  ! 读不到语料原文，跳过「事实是否存在于原文」校验")
                continue
            for item in data["questions"]:
                miss = missing_facts(corpus, item["expected_facts"])
                if miss:
                    ok = False
                    print(f"  ✗ {item['id']} 事实在原文中找不到：{'、'.join(miss)}")
                else:
                    print(f"  ✓ {item['id']}")
        print("\n题集校验" + ("通过" if ok else "未通过：请修正上面标注的事实，它们会让评测出现假阴性"))
        return 0 if ok else 1

    # ---- 需要服务，延迟导入 ----
    try:
        from app import config
        from app.services import qa
    except Exception as exc:
        print(f"[ERROR] 无法导入 app 包（请在项目根目录执行）：{exc}")
        return 1

    limit = args.limit or config.RETRIEVE_LIMIT
    print(f"检索模式：{args.mode}　召回条数：{limit}")

    report: dict[str, Any] = {
        "mode": args.mode,
        "limit": limit,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sets": {},
    }
    all_rows: list[dict] = []

    try:
        for name, data in loaded.items():
            print(f"\n=== {data['name']} ===")
            rows = eval_retrieval(qa, data["questions"], limit, args.mode)
            print_rows(rows)

            if args.with_answer:
                print(f"\n  -- 生成答案评测（{data['name']}）--")
                ans_rows = eval_answer(qa, data["questions"], limit)
                merged = {r["id"]: r for r in rows}
                for r in ans_rows:
                    merged[r["id"]].update(r)
                for r in ans_rows:
                    mark = "✓" if r["answer_hit"] else "✗"
                    note = "" if r["answer_hit"] else "缺：" + "、".join(r["missing"])
                    print(f"  {r['id']:<8} {mark:<4} {r['latency_ms']:>7.1f}ms  {note}")
                rows = list(merged.values())

            summary = summarize(rows, len(rows))
            if args.with_answer:
                ans_hits = sum(1 for r in rows if r.get("answer_hit"))
                summary["answer_hit"] = ans_hits
                summary["answer_hit_rate"] = ans_hits / len(rows) if rows else 0.0

            print(
                f"\n  召回 {summary['retrieval_hit']}/{summary['total']}"
                f"（{pct(summary['retrieval_hit_rate'])}）"
                f"　MRR {summary['mrr']:.3f}"
                f"　P50 {summary['latency_p50_ms']:.1f}ms"
                f"　P95 {summary['latency_p95_ms']:.1f}ms"
                f"　章节多样性 {summary['section_diversity']:.2f}"
            )
            if args.with_answer:
                print(f"  答案 {summary['answer_hit']}/{summary['total']}（{pct(summary['answer_hit_rate'])}）")

            report["sets"][name] = {"summary": summary, "rows": rows}
            all_rows.extend(rows)

    except RuntimeError as exc:
        print(f"\n[ERROR] {exc}")
        print("  提示：检索需要 Milvus 与 embedding API。启动顺序：")
        print("    1) 启动 Docker Desktop")
        print(r"    2) cd E:\Milvus; .\standalone.bat start")
        print("    3) 确认 19530 端口可连后重试")
        return 1

    overall = summarize(all_rows, len(all_rows))
    report["overall"] = overall
    print("\n=== 汇总 ===")
    print(
        f"  共 {overall['total']} 题　召回 {overall['retrieval_hit']}/{overall['total']}"
        f"（{pct(overall['retrieval_hit_rate'])}）　MRR {overall['mrr']:.3f}"
        f"　P50 {overall['latency_p50_ms']:.1f}ms　P95 {overall['latency_p95_ms']:.1f}ms"
        f"　多样性 {overall['section_diversity']:.2f}"
    )
    if args.with_answer:
        ans_hits = sum(1 for r in all_rows if r.get("answer_hit"))
        print(f"  答案 {ans_hits}/{overall['total']}（{pct(ans_hits / overall['total'])}）")

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结果已写入 {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())


