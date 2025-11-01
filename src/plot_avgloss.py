#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot average loss per epoch from JSONL logs.

- 依 epoch 分組，每個 epoch 畫一個點（無平滑）
- 自動偵測常見 loss 欄位：avg_loss, avg_pairwise_loss, avg_listwise_loss, loss
- 優先使用聚合鍵（avg_*），若同一 epoch 有多筆，取該 epoch 平均值
- x 軸固定為 epoch（整數）

Usage
  python plot_avgloss.py --input logs/reranker_listwise.jsonl --output logs/avgloss_epoch.png
"""
from __future__ import annotations
import argparse
import json
import os
from typing import Any, Dict, List, Iterable, Optional
import statistics

import matplotlib.pyplot as plt


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
            except Exception:
                continue
    return rows


def pick_num(d: Dict[str, Any], keys: Iterable[str]) -> Optional[float]:
    """Pick first numeric value found by keys. Supports one-level nesting via 'a.b' syntax."""
    for k in keys:
        cur = d
        parts = k.split('.')
        ok = True
        for p in parts:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                ok = False
                break
        if ok and isinstance(cur, (int, float)):
            return float(cur)
    return None


def extract_epoch_points(
    rows: List[Dict[str, Any]],
    epoch_keys: List[str],
    y_keys: List[str],
) -> tuple[list[int], list[float], str]:
    """Return (epochs_sorted, values_per_epoch, y_key_used)."""
    # 決定要用哪個 y_key（全檔至少有一次）
    y_key_used = None
    for k in y_keys:
        if any(pick_num(r, [k]) is not None for r in rows):
            y_key_used = k
            break
    if y_key_used is None:
        raise ValueError(f"No loss keys found. Tried: {y_keys}")

    # 以 epoch 分組
    buckets: dict[int, list[float]] = {}
    for r in rows:
        # 找 epoch
        e = None
        for ek in epoch_keys:
            v = pick_num(r, [ek])
            if v is not None:
                e = int(v)
                break
        if e is None:
            continue
        # 找 y
        y = pick_num(r, [y_key_used])
        if y is None:
            continue
        buckets.setdefault(e, []).append(float(y))

    if not buckets:
        raise ValueError("No (epoch, loss) pairs extracted from logs.")

    epochs = sorted(buckets.keys())
    values = [statistics.fmean(buckets[e]) for e in epochs]
    return epochs, values, y_key_used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=str, default='logs/reranker_listwise.jsonl', help='JSONL log path')
    ap.add_argument('--output', type=str, default=None, help='PNG path; default next to input')
    ap.add_argument('--title', type=str, default='Average Loss per Epoch')
    ap.add_argument('--dpi', type=int, default=150)
    args = ap.parse_args()

    rows = read_jsonl(args.input)
    if not rows:
        raise SystemExit(f'No valid JSON lines found in: {args.input}')

    # 常見 epoch 與 loss 鍵
    epoch_keys = ['epoch', 'trainer/epoch']
    y_keys = [
        'avg_loss',
        'avg_pairwise_loss',
        'avg_listwise_loss',
        'loss',
        'metrics.avg_loss',
        'metrics.avg_pairwise_loss',
    ]

    xs, ys, yk = extract_epoch_points(rows, epoch_keys=epoch_keys, y_keys=y_keys)

    plt.figure(figsize=(8, 4.5))
    plt.plot(xs, ys, marker='o', linewidth=1.8)
    plt.xlabel('epoch')
    plt.ylabel(yk)
    plt.title(args.title)
    plt.grid(True, alpha=0.3)

    out_path = args.output
    if not out_path:
        base, _ = os.path.splitext(os.path.basename(args.input))
        out_dir = os.path.dirname(args.input) or '.'
        out_path = os.path.join(out_dir, f'{base}_epoch_plot.png')

    plt.tight_layout()
    plt.savefig(out_path, dpi=args.dpi)
    print(f'Saved plot to: {out_path}')


if __name__ == '__main__':
    main()
