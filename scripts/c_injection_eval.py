"""C 模块：在开源提示注入数据集上单独评测「规则分支」与「分类器分支」。

用途：HoliSafe 是图文安全风险数据、不含注入句式，605 条验证集上规则零命中，
      不能证明规则有效。本脚本用第三方开源注入数据补这一块证据。

用法：
  python scripts/c_injection_eval.py --jsonl data/injection_eval/deepset_all.jsonl
  python scripts/c_injection_eval.py --jsonl data/injection_eval/cheezone_test.jsonl --lang zh

JSONL 每行至少含 text（或 prompt/content）与 label（1=注入，0=正常）。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mm_jailbreak_guard.contracts import DetectionRequest, ImageEvidence  # noqa: E402
from mm_jailbreak_guard.providers import make_text_provider  # noqa: E402
from mm_jailbreak_guard.rule_baseline import scan_with_rules  # noqa: E402

SPEC = "mm_jailbreak_guard.rule_classifier_provider:create_provider"


def load_rows(path: Path, lang: str | None) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            text = item.get("text") or item.get("prompt") or item.get("content") or ""
            if not text.strip():
                continue
            if lang and item.get("lang") and item["lang"] != lang:
                continue
            rows.append({"text": text, "label": int(item.get("label", 0)),
                         "lang": item.get("lang", ""), "type": item.get("type", "")})
    return rows


def score_with(mode: str, texts: list[str]) -> list[float]:
    os.environ["C_PROVIDER_MODE"] = mode
    provider = make_text_provider(SPEC, None)
    out = []
    for index, text in enumerate(texts):
        request = DetectionRequest(sample_id=f"INJ{index:05d}", user_text=text)
        image = ImageEvidence.unavailable(request.sample_id, "无 OCR")
        out.append(float(provider.analyze(request, image).user_text_score or 0.0))
    return out


def binary_metrics(labels: list[int], scores: list[float], threshold: float) -> dict:
    tp = sum(1 for y, s in zip(labels, scores) if y == 1 and s >= threshold)
    fp = sum(1 for y, s in zip(labels, scores) if y == 0 and s >= threshold)
    fn = sum(1 for y, s in zip(labels, scores) if y == 1 and s < threshold)
    tn = sum(1 for y, s in zip(labels, scores) if y == 0 and s < threshold)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    acc = (tp + tn) / max(len(labels), 1)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "P": round(precision, 4),
            "R": round(recall, 4), "F1": round(f1, 4), "FPR": round(fpr, 4),
            "ACC": round(acc, 4)}


def auc(labels: list[int], scores: list[float]) -> float:
    pos = [s for y, s in zip(labels, scores) if y == 1]
    neg = [s for y, s in zip(labels, scores) if y == 0]
    if not pos or not neg:
        return 0.0
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 4)


def rule_metrics(texts: list[str], labels: list[int]) -> dict:
    hits = [bool(scan_with_rules(text)[1]) for text in texts]
    tp = sum(1 for h, y in zip(hits, labels) if h and y == 1)
    fp = sum(1 for h, y in zip(hits, labels) if h and y == 0)
    fn = sum(1 for h, y in zip(hits, labels) if not h and y == 1)
    tn = sum(1 for h, y in zip(hits, labels) if not h and y == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"hit_pos": tp, "hit_neg": fp, "miss_pos": fn,
            "P": round(precision, 4), "R": round(recall, 4), "F1": round(f1, 4),
            "FPR": round(fp / (fp + tn), 4) if fp + tn else 0.0}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--lang", default=None, help="只评某一语言，如 zh / en")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out-csv", default=None)
    args = parser.parse_args()

    path = Path(args.jsonl)
    rows = load_rows(path, args.lang)
    texts = [r["text"] for r in rows]
    labels = [r["label"] for r in rows]
    print(f"样本 {len(rows)} 条：注入 {sum(labels)} / 正常 {len(labels) - sum(labels)}  |  文件 {path.name}")

    scores = score_with("pure_classifier", texts)
    combined = score_with("combined", texts)
    clf = binary_metrics(labels, scores, args.threshold)
    com = binary_metrics(labels, combined, args.threshold)
    rule = rule_metrics(texts, labels)

    print("\n分支                 P       R       F1      FPR     其他")
    print(f"规则（正则+shield）  {rule['P']:.4f}  {rule['R']:.4f}  {rule['F1']:.4f}  {rule['FPR']:.4f}  命中正/负={rule['hit_pos']}/{rule['hit_neg']} 漏={rule['miss_pos']}")
    print(f"TF-IDF+NB@{args.threshold}      {clf['P']:.4f}  {clf['R']:.4f}  {clf['F1']:.4f}  {clf['FPR']:.4f}  AUC={auc(labels, scores)} tp/fp/fn/tn={clf['tp']}/{clf['fp']}/{clf['fn']}/{clf['tn']}")
    print(f"combined（取 max）  {com['P']:.4f}  {com['R']:.4f}  {com['F1']:.4f}  {com['FPR']:.4f}  AUC={auc(labels, combined)}")

    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.out_csv).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["idx", "label", "lang", "type", "rule_hit", "classifier", "combined", "text"])
            for i, row in enumerate(rows):
                hit = bool(scan_with_rules(row["text"])[1])
                writer.writerow([i, row["label"], row["lang"], row["type"], int(hit),
                                 round(scores[i], 6), round(combined[i], 6),
                                 row["text"][:200].replace("\n", " ")])
        print(f"\n逐条结果：{args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
