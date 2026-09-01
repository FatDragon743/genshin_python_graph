"""四星十抽保底概率与历史间隔分析。"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from .constants import FOUR_STAR_RULES, LUCK_POOLS


def four_star_rule(pool_key: str) -> dict[str, Any]:
    return dict(FOUR_STAR_RULES.get(pool_key) or FOUR_STAR_RULES["character"])


def four_star_conditional_rate(pull_index: int, pool_key: str = "character") -> float:
    """
    pull_index: 距上次四星及以上之后的第几次抽（1..hard）。
    返回该次抽获得「四星及以上」的条件概率。
    """
    rule = four_star_rule(pool_key)
    hard = int(rule["hard_pity"])
    soft = int(rule["soft_pity"])
    if pull_index >= hard:
        return 1.0
    if pull_index == soft:
        return float(rule["soft_rate"])
    if pull_index < soft:
        return float(rule["base_rate"])
    return 1.0


def calculate_four_star_curve(current_pity: int, pool_key: str = "character") -> dict[str, Any]:
    """从当前四星垫刀起，后续每次抽的单抽/累积出紫(及以上)概率。"""
    rule = four_star_rule(pool_key)
    hard = int(rule["hard_pity"])
    soft = int(rule["soft_pity"])
    pity = max(0, min(int(current_pity), hard - 1))

    xs = []
    singles = []
    for n in range(pity + 1, hard + 1):
        xs.append(n)  # 即将进行的是「第 n 抽未出四星+」
        singles.append(four_star_conditional_rate(n, pool_key))

    cums = []
    survive = 1.0
    for p in singles:
        survive *= 1 - p
        cums.append(1 - survive)

    expected = 0.0
    survive = 1.0
    for i, p in enumerate(singles):
        expected += (i + 1) * survive * p
        survive *= 1 - p

    next_rate = singles[0] if singles else 1.0
    pulls_to_soft = max(0, soft - pity)
    pulls_to_hard = max(0, hard - pity)

    milestones = {}
    for target in (0.5, 0.75, 0.9, 0.99):
        milestones[f"pulls_to_{int(target * 100)}%"] = None
        for i, c in enumerate(cums):
            if c >= target:
                milestones[f"pulls_to_{int(target * 100)}%"] = i + 1
                break

    return {
        "current_pity": pity,
        "next_rate": round(next_rate, 6),
        "expected_pulls": round(expected, 3),
        "pulls_to_soft": pulls_to_soft,
        "pulls_to_hard": pulls_to_hard,
        "soft_pity": soft,
        "hard_pity": hard,
        "base_rate": rule["base_rate"],
        "soft_rate": rule["soft_rate"],
        "note": rule.get("note", ""),
        "curve": {
            "x": xs,  # 第几次未出紫抽
            "single": [round(p, 6) for p in singles],
            "cumulative": [round(c, 6) for c in cums],
        },
        **milestones,
    }


def extract_four_star_state(
    raw: pd.DataFrame,
    pool_key: str,
) -> dict[str, Any]:
    """从 raw 计算该池当前四星垫刀与历史间隔（四星及以上重置）。"""
    pool = next((p for p in LUCK_POOLS if p["key"] == pool_key), None)
    types = tuple(pool["types"]) if pool else ()
    if raw is None or len(raw) == 0 or not types:
        empty = calculate_four_star_curve(0, pool_key)
        empty.update({"intervals": [], "names": [], "count": 0, "avg_interval": None})
        return empty

    df = raw[raw["gacha_type"].isin(types)].copy()
    df = df.sort_values(["time", "id"], kind="mergesort").reset_index(drop=True)

    intervals: list[int] = []
    names: list[str] = []
    pity = 0
    for _, row in df.iterrows():
        pity += 1
        try:
            rank = int(row.get("rank_type") or 0)
        except Exception:
            rank = 0
        if rank >= 4:
            intervals.append(pity)
            names.append(str(row.get("name") or ""))
            pity = 0

    curve = calculate_four_star_curve(pity, pool_key)
    avg = float(np.mean(intervals)) if intervals else None
    # 间隔分布 1..10
    hist = [0] * 10
    for v in intervals:
        idx = min(max(int(v), 1), 10) - 1
        hist[idx] += 1

    curve.update(
        {
            "intervals": intervals,
            "names": names,
            "count": len(intervals),
            "avg_interval": round(avg, 2) if avg is not None else None,
            "median_interval": float(np.median(intervals)) if intervals else None,
            "hist": {
                "labels": [str(i) for i in range(1, 11)],
                "counts": hist,
            },
            "recent": list(
                reversed(
                    [
                        {"name": n, "pity": p}
                        for n, p in zip(names[-16:], intervals[-16:])
                    ]
                )
            ),
        }
    )
    return curve
