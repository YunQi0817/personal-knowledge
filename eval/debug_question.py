"""诊断单题的召回情况：对比 hybrid 与 dense 的 top-k，看正确答案落在第几名。

用法（项目根目录）：
    .venv\\Scripts\\python.exe -B eval\\debug_question.py --set hard8 --id hd-03
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

QUESTIONS_DIR = Path(__file__).resolve().parent / "questions"
SETS = {
    "cs5": "cs-5.json",
    "drawing12": "drawing-12.json",
    "hard8": "hard-8.json",
}


def norm(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).split())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", dest="set_name", required=True, choices=list(SETS))
    parser.add_argument("--id", dest="qid", required=True)
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    data = json.loads((QUESTIONS_DIR / SETS[args.set_name]).read_text(encoding="utf-8"))
    item = next((q for q in data["questions"] if q["id"] == args.qid), None)
    if item is None:
        print(f"[ERROR] 题集 {args.set_name} 里没有 {args.qid}")
        return 1

    from app.services import qa

    print(f"问题：{item['question']}")
    print(f"期望事实：{'、'.join(item['expected_facts'])}\n")

    for mode in ("hybrid", "dense"):
        print(f"===== {mode} =====")
        hits = qa.retrieve(item["question"], limit=args.limit, mode=mode)
        for index, hit in enumerate(hits, start=1):
            entity = hit.get("entity", {}) or {}
            section = entity.get("section", "(无章节)")
            text = (entity.get("parent_text") or entity.get("text") or "").replace("\n", " ")
            distance = hit.get("distance")
            score = f"{distance:.4f}" if isinstance(distance, float) else str(distance)
            missing = [f for f in item["expected_facts"] if norm(f) not in norm(text)]
            flag = "✓ 含答案" if not missing else "✗ 缺 " + "、".join(missing)
            print(f"  #{index} 分数={score}  {flag}")
            print(f"      章节：{section}")
            print(f"      片段：{text[:100]}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
