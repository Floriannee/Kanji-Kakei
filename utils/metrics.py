"""Evaluation metrics, one function per pipeline stage.

These define exactly what each target number means:

  Receipt detection (target 95%+):
      detection_iou_accuracy — fraction of images whose predicted receipt
      region has IoU >= 0.5 with the ground-truth region.

  OCR character accuracy (target 90-95%):
      character_accuracy — 1 - (normalized edit distance) averaged over lines,
      i.e. character-level accuracy via Levenshtein.

  Structured item extraction (target 80-90%):
      item_extraction_f1 — F1 over (name, price) item tuples, matching on
      normalized name + exact price.

  End-to-end (target 75-85%):
      end_to_end_score — fraction of receipts where store, total, and the item
      set are all correct (a strict all-or-mostly-right per-receipt score).
"""

from __future__ import annotations

from dataclasses import dataclass


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def _iou(box_a, box_b) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def detection_iou_accuracy(preds: list, gts: list, threshold: float = 0.5) -> float:
    """Fraction of images where predicted box IoU >= threshold."""
    if not gts:
        return 0.0
    hits = sum(1 for p, g in zip(preds, gts) if p is not None and _iou(p, g) >= threshold)
    return hits / len(gts)


def character_accuracy(pred_lines: list[str], gt_lines: list[str]) -> float:
    """1 - mean normalized edit distance across aligned lines."""
    if not gt_lines:
        return 0.0
    total = 0.0
    for pred, gt in zip(pred_lines, gt_lines):
        denom = max(len(gt), 1)
        total += 1.0 - (_levenshtein(pred, gt) / denom)
    return max(0.0, total / len(gt_lines))


@dataclass
class PRF:
    precision: float
    recall: float
    f1: float


def item_extraction_f1(pred_items: list[tuple], gt_items: list[tuple]) -> PRF:
    """F1 over (name, price) tuples. Name compared case-folded/stripped."""
    def norm(t):
        name, price = t
        return (str(name).strip().casefold(), None if price is None else round(float(price)))

    pred = [norm(t) for t in pred_items]
    gt = [norm(t) for t in gt_items]
    gt_pool = list(gt)
    tp = 0
    for p in pred:
        if p in gt_pool:
            gt_pool.remove(p)
            tp += 1
    fp = len(pred) - tp
    fn = len(gt) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return PRF(precision, recall, f1)


def end_to_end_score(pred_receipts: list[dict], gt_receipts: list[dict]) -> float:
    """Strict per-receipt correctness over store + total + item set."""
    if not gt_receipts:
        return 0.0
    correct = 0
    for pred, gt in zip(pred_receipts, gt_receipts):
        store_ok = (pred.get("store") or "").strip() == (gt.get("store") or "").strip()
        total_ok = pred.get("total") == gt.get("total")
        items_f1 = item_extraction_f1(
            pred.get("items", []), gt.get("items", [])
        ).f1
        if store_ok and total_ok and items_f1 >= 0.8:
            correct += 1
    return correct / len(gt_receipts)
