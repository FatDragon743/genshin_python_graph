"""抽卡规律 / 分布洞察（供网页概率分析页）。"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .analysis import GenshinWishProbability
from .constants import POOL_PITY_RULES


def _safe_pct(n: int, d: int) -> Optional[float]:
    if d <= 0:
        return None
    return round(n / d, 4)


def _percentile(sorted_arr: list[int], q: float) -> Optional[float]:
    if not sorted_arr:
        return None
    return float(np.percentile(sorted_arr, q))


def theoretical_first_five_star_pmf(pool_key: str = "character") -> dict[str, Any]:
    """从 0 垫开始，第 k 抽出金的理论概率质量（按池规则）。"""
    calc = GenshinWishProbability.for_pool(pool_key)
    rule = POOL_PITY_RULES.get(pool_key) or POOL_PITY_RULES["character"]
    cond = []
    for k in range(1, calc.HARD_PITY + 1):
        i = k
        if i < calc.SOFT_PITY_START:
            p = calc.BASE_RATE
        elif i < calc.HARD_PITY:
            p = min(calc.BASE_RATE + calc.RATE_INCREASE * (i - calc.SOFT_PITY_START + 1), 1.0)
        else:
            p = 1.0
        cond.append(p)

    survival = 1.0
    pmf = []
    for p in cond:
        pmf.append(survival * p)
        survival *= 1 - p

    xs = list(range(1, calc.HARD_PITY + 1))
    cdf = []
    acc = 0.0
    for v in pmf:
        acc += v
        cdf.append(acc)

    expected = sum(x * p for x, p in zip(xs, pmf))
    return {
        "x": xs,
        "pmf": [round(v, 8) for v in pmf],
        "cdf": [round(v, 8) for v in cdf],
        "expected": round(expected, 3),
        "soft_pity": calc.SOFT_PITY_START,
        "hard_pity": calc.HARD_PITY,
        "base_rate": calc.BASE_RATE,
        "rate_increase": calc.RATE_INCREASE,
        "pool_key": pool_key,
        "pool_title": rule.get("title", pool_key),
        "featured_note": rule.get("featured_note", ""),
    }


def build_history_insights(
    wish_counts: list[int],
    characters: list[str] | None = None,
    *,
    fifty_fifty_flags: list[Optional[bool]] | None = None,
    pool_key: str = "character",
) -> dict[str, Any]:
    """
    fifty_fifty_flags: 与五星序列对齐；True=小保底不歪, False=歪, None=大保底/不追踪。
    """
    rule = POOL_PITY_RULES.get(pool_key) or POOL_PITY_RULES["character"]
    soft = int(rule["soft_pity"])
    hard = int(rule["hard_pity"])
    early_cut = int(rule.get("early_cutoff") or max(1, soft - 14))

    counts = [int(c) for c in (wish_counts or []) if c is not None]
    n = len(counts)
    characters = list(characters or [])
    while len(characters) < n:
        characters.append("")

    early_n = sum(1 for c in counts if c < early_cut)
    mid_n = sum(1 for c in counts if early_cut <= c < soft)
    soft_n = sum(1 for c in counts if soft <= c < hard)
    hard_n = sum(1 for c in counts if c >= hard)

    sorted_c = sorted(counts)
    avg = float(np.mean(counts)) if n else None
    median = float(np.median(counts)) if n else None
    std = float(np.std(counts, ddof=1)) if n >= 2 else (0.0 if n == 1 else None)

    step = 5
    fine_bins = list(range(0, hard + step, step))
    if fine_bins[-1] < hard:
        fine_bins.append(hard)
    fine_counts = [0] * (len(fine_bins) - 1)
    for c in counts:
        placed = False
        for i in range(len(fine_bins) - 1):
            lo, hi = fine_bins[i], fine_bins[i + 1]
            if lo < c <= hi or (c == 0 and lo == 0):
                fine_counts[i] += 1
                placed = True
                break
        if not placed and fine_counts:
            fine_counts[-1] += 1

    theo = theoretical_first_five_star_pmf(pool_key)
    emp_pmf = [0.0] * hard
    for c in counts:
        idx = min(max(int(c), 1), hard) - 1
        emp_pmf[idx] += 1
    if n:
        emp_pmf = [v / n for v in emp_pmf]

    ecdf_x = sorted_c
    ecdf_y = [((i + 1) / n) for i in range(n)] if n else []

    window = 5
    rolling = []
    for i in range(n):
        chunk = counts[max(0, i - window + 1) : i + 1]
        rolling.append(round(sum(chunk) / len(chunk), 2))

    ff = fifty_fifty_flags or [None] * n
    roll_5050_x = []
    roll_5050_rate = []
    wins = 0
    trials = 0
    for i, flag in enumerate(ff[:n]):
        if flag is None:
            continue
        trials += 1
        if flag:
            wins += 1
        roll_5050_x.append(i + 1)
        roll_5050_rate.append(round(wins / trials, 4))

    zones = {
        "early": {
            "count": early_n,
            "rate": _safe_pct(early_n, n),
            "label": f"早出(<{early_cut})",
        },
        "mid": {
            "count": mid_n,
            "rate": _safe_pct(mid_n, n),
            "label": f"中段({early_cut}-{soft - 1})",
        },
        "soft": {
            "count": soft_n,
            "rate": _safe_pct(soft_n, n),
            "label": f"软保底({soft}-{hard - 1})",
        },
        "hard": {
            "count": hard_n,
            "rate": _safe_pct(hard_n, n),
            "label": f"硬保底(≥{hard})",
        },
    }

    return {
        "n": n,
        "avg": round(avg, 2) if avg is not None else None,
        "median": round(median, 2) if median is not None else None,
        "std": round(std, 2) if std is not None else None,
        "min": min(counts) if counts else None,
        "max": max(counts) if counts else None,
        "p25": round(_percentile(sorted_c, 25), 1) if sorted_c else None,
        "p75": round(_percentile(sorted_c, 75), 1) if sorted_c else None,
        "theo_expected": theo["expected"],
        "avg_vs_theo": round(avg - theo["expected"], 2) if avg is not None else None,
        "zones": zones,
        "soft_pity_rate": _safe_pct(soft_n + hard_n, n),
        "hard_pity_rate": _safe_pct(hard_n, n),
        "early_rate": _safe_pct(early_n, n),
        "pool_key": pool_key,
        "soft_pity": soft,
        "hard_pity": hard,
        "fine_hist": {
            "labels": [f"{fine_bins[i] + 1}-{fine_bins[i + 1]}" for i in range(len(fine_bins) - 1)],
            "counts": fine_counts,
        },
        "compare_pmf": {
            "x": theo["x"],
            "theoretical": theo["pmf"],
            "empirical": [round(v, 6) for v in emp_pmf],
            "theo_cdf": theo["cdf"],
        },
        "ecdf": {"x": ecdf_x, "y": [round(v, 4) for v in ecdf_y]},
        "sequence": {
            "index": list(range(1, n + 1)),
            "pity": counts,
            "names": characters[:n],
            "rolling_avg": rolling,
        },
        "fifty_fifty_roll": {
            "index": roll_5050_x,
            "win_rate": roll_5050_rate,
            "final_rate": roll_5050_rate[-1] if roll_5050_rate else None,
            "trials": trials,
        },
    }


def fifty_fifty_flags_from_hits(hits: list[dict[str, Any]]) -> list[Optional[bool]]:
    """从 luck all_hits 生成与时间升序一致的 50/50 标记。"""
    flags: list[Optional[bool]] = []
    for h in hits:
        if h.get("is_guaranteed"):
            flags.append(None)
        elif h.get("is_off"):
            flags.append(False)
        else:
            flags.append(True)
    return flags
