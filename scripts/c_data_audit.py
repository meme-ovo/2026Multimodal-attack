"""C 模块数据/产物稽核：复算指标、检查 null 契约、test CSV、注入 CSV、可复现性、数据泄漏。

用法：
  python scripts/c_data_audit.py
  python scripts/c_data_audit.py --skip-overlap   # 跳过与训练集的重叠检查（需读 HF 缓存）
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OK, BAD = "[PASS]", "[FAIL]"
failures = 0


def check(ok: bool, text: str, detail: str = "") -> None:
    global failures
    if not ok:
        failures += 1
    print(f"{OK if ok else BAD} {text}" + (f"  ({detail})" if detail else ""))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def metrics(labels: list[int], scores: list[float], threshold: float = 0.5) -> dict:
    tp = sum(1 for y, s in zip(labels, scores) if y == 1 and s >= threshold)
    fp = sum(1 for y, s in zip(labels, scores) if y == 0 and s >= threshold)
    fn = sum(1 for y, s in zip(labels, scores) if y == 1 and s < threshold)
    tn = sum(1 for y, s in zip(labels, scores) if y == 0 and s < threshold)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    acc = (tp + tn) / max(len(labels), 1)
    ba = (recall + (tn / (tn + fp) if tn + fp else 0.0)) / 2
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "P": precision, "R": recall,
            "F1": f1, "FPR": fpr, "ACC": acc, "BA": ba}


def close(a: float, b: float, tol: float = 5e-4) -> bool:
    return abs(a - b) <= tol


def audit_val() -> tuple[list[dict], dict[str, dict]]:
    manifest = {r["sample_id"]: r for r in load_jsonl(ROOT / "outputs/A/team_dataset_v1/manifest.validation.jsonl")}
    scores = {r["sample_id"]: r for r in load_jsonl(ROOT / "outputs/C_week6/val.pure_c.combined.jsonl")}
    images = {r["sample_id"]: r for r in load_jsonl(ROOT / "artifacts/v2/image_evidence.validation.jsonl")}

    check(len(manifest) == 605, "验证集 605 条", str(len(manifest)))
    check(len(scores) == 605, "C 分数 605 条且无重复 sample_id", str(len(scores)))
    check(set(manifest) == set(scores), "C 分数与 manifest 的 sample_id 一一对应")

    bad_range = [sid for sid, r in scores.items()
                 if not (isinstance(r.get("user_text_score"), (int, float)) and 0 <= r["user_text_score"] <= 1)]
    check(not bad_range, "user_text_score 全在 [0,1] 且非空", f"异常 {len(bad_range)}")

    ocr_expected = 0
    violations = []
    for sid, r in scores.items():
        img = images.get(sid, {})
        status = img.get("ocr_status")
        text = (img.get("ocr_text") or "").strip()
        ready = status == "success" and bool(text)
        if ready:
            ocr_expected += 1
        ocr, comb = r.get("ocr_text_score"), r.get("combined_text_score")
        if ready:
            if ocr is None or comb is None:
                violations.append(f"{sid}: 有 OCR 但分数为 null")
        else:
            if ocr is not None or comb is not None:
                violations.append(f"{sid}: 无 OCR 却给了分数")
    check(not violations, "null 契约：无 OCR 必须为 null", f"违规 {len(violations)}")
    check(ocr_expected == 263, "OCR 可用样本 263 条", str(ocr_expected))

    labels = [int(manifest[sid]["label"]) for sid in manifest]
    vals = [scores[sid]["user_text_score"] for sid in manifest]
    m = metrics(labels, vals)
    check(m["tp"] == 303 and m["fp"] == 31 and m["fn"] == 201 and m["tn"] == 70,
          "混淆矩阵 tp303/fp31/fn201/tn70", f"{m['tp']}/{m['fp']}/{m['fn']}/{m['tn']}")
    check(close(m["F1"], 0.7232), "F1 = 0.7232", f"{m['F1']:.4f}")
    check(close(m["P"], 0.9072) and close(m["R"], 0.6012), "P 0.9072 / R 0.6012", f"{m['P']:.4f}/{m['R']:.4f}")
    check(close(m["FPR"], 0.3069) and close(m["BA"], 0.6471), "FPR 0.3069 / BA 0.6471", f"{m['FPR']:.4f}/{m['BA']:.4f}")
    return list(manifest.values()), images


def audit_test_csv() -> None:
    path = ROOT / "outputs/C_week6/C_prediction.test.csv"
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    manifest = [r for r in load_jsonl(ROOT / "outputs/A/team_dataset_v1/manifest.test.jsonl")]
    images = {r["sample_id"]: r for r in load_jsonl(ROOT / "artifacts/v2/image_evidence.test.jsonl")}
    ids = [r["sample_id"] for r in manifest]
    check(len(rows) == 609, "test CSV 609 行", str(len(rows)))
    check([r["sample_id"] for r in rows] == ids, "test CSV 顺序与 manifest 一致")
    expected_cols = ["sample_id", "user_text_score", "ocr_text_score", "combined_text_score", "text_pred", "ocr_status"]
    check(list(rows[0].keys()) == expected_cols, "test CSV 表头正确", ",".join(rows[0].keys()))

    bad_pred, bad_null, zero_as_null, bad_status = [], [], [], []
    for r in rows:
        score = float(r["user_text_score"])
        if int(r["text_pred"]) != int(score >= 0.5):
            bad_pred.append(r["sample_id"])
        status = r["ocr_status"]
        ocr, comb = r["ocr_text_score"], r["combined_text_score"]
        if status != "success":
            if ocr != "" or comb != "":
                bad_null.append(r["sample_id"])
            if ocr == "0" or comb == "0":
                zero_as_null.append(r["sample_id"])
        if r["sample_id"] in images and images[r["sample_id"]].get("ocr_status") != status:
            bad_status.append(r["sample_id"])
    check(not bad_pred, "text_pred = (user_score >= 0.5)", f"异常 {len(bad_pred)}")
    check(not bad_null, "无 OCR 行两列留空", f"异常 {len(bad_null)}")
    check(not zero_as_null, "没有把 null 写成 0", f"异常 {len(zero_as_null)}")
    check(not bad_status, "ocr_status 与 B 证据一致", f"异常 {len(bad_status)}")


def audit_injection() -> None:
    path = ROOT / "outputs/C_week7/injection_deepset.csv"
    if not path.is_file():
        print("[SKIP] 注入 CSV 不在本地（未跑 c_injection_eval.py），跳过")
        return
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    check(len(rows) == 662, "deepset CSV 662 行", str(len(rows)))
    labels = [int(r["label"]) for r in rows]
    rule = [1.0 if int(r["rule_hit"]) else 0.0 for r in rows]
    clf = [float(r["classifier"]) for r in rows]
    mr = metrics(labels, rule)
    mc = metrics(labels, clf)
    check(close(mr["P"], 0.8636, 2e-3) and close(mr["R"], 0.0722, 2e-3),
          "规则 P0.8636 / R0.0722", f"{mr['P']:.4f}/{mr['R']:.4f}")
    check(close(mc["P"], 0.0631, 2e-3) and close(mc["F1"], 0.0374, 2e-3),
          "分类器 F1 0.0374", f"P{mc['P']:.4f}/F1{mc['F1']:.4f}")
    check(close(auc(labels, clf), 0.2554, 2e-3), "分类器 AUC 0.2554", f"{auc(labels, clf):.4f}")

    zh_path = ROOT / "outputs/C_week7/injection_cheezone_zh.csv"
    if zh_path.is_file():
        with zh_path.open(encoding="utf-8-sig", newline="") as f:
            zh = list(csv.DictReader(f))
        zh_labels = [int(r["label"]) for r in zh]
        check(len(zh) == 836 and sum(zh_labels) == 367,
              "CheezOne zh CSV 836 行 / 367 注入", f"{len(zh)}/{sum(zh_labels)}")

    campus = ROOT / "outputs/C_week7/campus12.C_prediction.csv"
    if campus.is_file():
        with campus.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        flagged = sum(1 for r in rows if r["text_pred"] == "1")
        blank = sum(1 for r in rows if r["ocr_text_score"] == "")
        check(len(rows) == 12 and flagged == 3 and blank == 3,
              "校园 12 条：12 行 / 3 判风险 / 3 条无 OCR", f"{len(rows)}/{flagged}/{blank}")


def auc(labels: list[int], scores: list[float]) -> float:
    pos = [s for y, s in zip(labels, scores) if y == 1]
    neg = [s for y, s in zip(labels, scores) if y == 0]
    if not pos or not neg:
        return 0.0
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def audit_repro() -> None:
    pairs = [
        ("outputs/C_week6/val.pure_c.combined.jsonl", "outputs/C_week7/val.pure_c.combined.jsonl"),
        ("outputs/C_week6/val.text_evidence.combined.jsonl", "outputs/C_week7/val.text_evidence.combined.jsonl"),
    ]
    for a, b in pairs:
        pa, pb = ROOT / a, ROOT / b
        if not pb.is_file():
            check(False, f"可复现：缺少 {b}")
            continue
        check(sha256(pa) == sha256(pb), f"重跑结果与 Week6 逐字节一致：{pb.name}",
              "全同" if sha256(pa) == sha256(pb) else "不同")


def audit_duplicates(manifest: list[dict]) -> None:
    counter = Counter(r["user_text"] for r in manifest)
    groups = {t: c for t, c in counter.items() if c > 1}
    dup_rows = sum(c for c in groups.values())
    conflicts = 0
    for text in groups:
        labels = {int(r["label"]) for r in manifest if r["user_text"] == text}
        if len(labels) > 1:
            conflicts += 1
    check(len(groups) == 51 and dup_rows == 148 and conflicts == 2,
          "重复文本 51 组 / 148 行 / 2 组标签冲突",
          f"{len(groups)}/{dup_rows}/{conflicts}")


def audit_overlap() -> None:
    cache = Path.home() / ".cache/huggingface/hub/datasets--ynyg--Unified-Prompt-Guard"
    files = list(cache.rglob("*.parquet"))
    if not files:
        print(f"{BAD} 训练集缓存未找到，跳过重叠检查（{cache}）")
        return
    try:
        import pandas as pd
    except Exception as exc:
        print(f"{BAD} 缺少 pandas，跳过重叠检查：{exc}")
        return

    def norm(t: str) -> str:
        return " ".join(str(t).lower().split())

    train = set()
    for p in files:
        frame = pd.read_parquet(p, columns=None)
        col = "text" if "text" in frame.columns else frame.columns[1]
        train.update(norm(x) for x in frame[col].tolist())
    rows = load_jsonl(ROOT / "data/injection_eval/deepset_all.jsonl")
    overlap_idx = {i for i, r in enumerate(rows) if norm(r["text"]) in train}
    check(len(overlap_idx) < 20, "deepset 与训练集无实质重叠",
          f"重合 {len(overlap_idx)}/{len(rows)}")

    if not overlap_idx:
        return
    with (ROOT / "outputs/C_week7/injection_deepset.csv").open(encoding="utf-8-sig", newline="") as f:
        pred = {int(r["idx"]): r for r in csv.DictReader(f)}
    labels, scores = [], []
    for i in range(len(rows)):
        if i in overlap_idx:
            continue
        labels.append(int(rows[i]["label"]))
        scores.append(float(pred[i]["classifier"]))
    check(auc(labels, scores) < 0.30, "剔除重叠后分类器 AUC 仍 < 0.30",
          f"AUC {auc(labels, scores):.4f}（样本 {len(labels)}）")


def audit_package() -> None:
    root = ROOT / "handoff/C_week7_final"
    sums = root / "SHA256SUMS.txt"
    if not sums.is_file():
        print("[SKIP] 交付包不在本地（handoff/ 不入库），跳过包校验")
        return
    bad = 0
    lines = [line for line in sums.read_text(encoding="utf-8").splitlines() if line.strip()]
    for line in lines:
        digest, rel = line.split("  ", 1)
        p = root / rel
        if not p.is_file() or sha256(p) != digest:
            bad += 1
    check(bad == 0, f"交付包 {len(lines)} 项校验和", f"坏 {bad}")
    files = [p for p in root.rglob("*") if p.is_file()]
    check(len(files) == 39, "交付包 39 个文件", str(len(files)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-overlap", action="store_true")
    args = parser.parse_args()

    needed = [
        "outputs/A/team_dataset_v1/manifest.validation.jsonl",
        "outputs/A/team_dataset_v1/manifest.test.jsonl",
        "outputs/C_week6/val.pure_c.combined.jsonl",
        "outputs/C_week6/C_prediction.test.csv",
        "artifacts/v2/image_evidence.validation.jsonl",
        "artifacts/v2/image_evidence.test.jsonl",
    ]
    missing = [rel for rel in needed if not (ROOT / rel).is_file()]
    if missing:
        print("需要本地数据与产物，缺少：")
        for rel in missing:
            print("  -", rel)
        print("这些文件不在 git 里；从 A 的数据包和 B 的 artifacts/v2 取回后再跑。")
        return 2

    print("=" * 64)
    print("C 模块数据稽核")
    print("=" * 64)
    manifest, images = audit_val()
    print("-" * 64)
    audit_test_csv()
    print("-" * 64)
    audit_injection()
    print("-" * 64)
    audit_repro()
    print("-" * 64)
    audit_duplicates(manifest)
    print("-" * 64)
    audit_package()
    if not args.skip_overlap:
        print("-" * 64)
        audit_overlap()
    print("=" * 64)
    print(f"汇总：{failures} 项失败")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
