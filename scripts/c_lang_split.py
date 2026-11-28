"""C 模块：验证集按文本语言分组，检查 v3 的提升是否只集中在英文/混排。

用法：
  python scripts/c_lang_split.py
  python scripts/c_lang_split.py --scores outputs/C_week6/val.pure_c.pure_classifier.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "outputs/A/team_dataset_v1/manifest.validation.jsonl"
DEFAULT_SCORES = ROOT / "outputs/C_week6/val.pure_c.pure_classifier.jsonl"


def bucket(text: str) -> str:
    has_ascii = bool(re.search(r"[A-Za-z]", text))
    has_cjk = bool(re.search(r"[\u4e00-\u9fff]", text))
    if has_cjk and not has_ascii:
        return "纯中文"
    if has_ascii and not has_cjk:
        return "纯英文"
    if has_ascii and has_cjk:
        return "中英混"
    return "其他"


def metrics(pairs: list[tuple[int, float]], threshold: float = 0.5) -> tuple:
    tp = sum(1 for y, s in pairs if y == 1 and s >= threshold)
    fp = sum(1 for y, s in pairs if y == 0 and s >= threshold)
    fn = sum(1 for y, s in pairs if y == 1 and s < threshold)
    tn = sum(1 for y, s in pairs if y == 0 and s < threshold)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return len(pairs), tp, fp, fn, tn, precision, recall, f1, fpr


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--scores", default=str(DEFAULT_SCORES))
    args = parser.parse_args()

    manifest = {r["sample_id"]: r for r in map(json.loads, Path(args.manifest).open(encoding="utf-8"))}
    scores = {r["sample_id"]: r["user_text_score"] for r in map(json.loads, Path(args.scores).open(encoding="utf-8"))}

    groups: dict[str, list[tuple[int, float]]] = {}
    skipped = 0
    for sample_id, row in manifest.items():
        score = scores.get(sample_id)
        if score is None:
            skipped += 1
            continue
        groups.setdefault(bucket(row["user_text"]), []).append((int(row["label"]), score))
    if skipped:
        print(f"（跳过 {skipped} 条无 user_text_score 的样本）")

    header = ["分组", "n", "tp", "fp", "fn", "tn", "P", "R", "F1", "FPR"]
    print("".join(f"{h:>8}" for h in header))
    for name in ("纯中文", "纯英文", "中英混", "其他"):
        if name not in groups:
            continue
        n, tp, fp, fn, tn, p, r, f1, fpr = metrics(groups[name])
        print(f"{name:>8}{n:>8}{tp:>8}{fp:>8}{fn:>8}{tn:>8}{p:>8.4f}{r:>8.4f}{f1:>8.4f}{fpr:>8.4f}")
    n, tp, fp, fn, tn, p, r, f1, fpr = metrics([(int(r["label"]), scores[s]) for s, r in manifest.items()])
    print(f"{'全部':>8}{n:>8}{tp:>8}{fp:>8}{fn:>8}{tn:>8}{p:>8.4f}{r:>8.4f}{f1:>8.4f}{fpr:>8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
