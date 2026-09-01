"""欧非分析：限定/常驻（歪不歪）、长期与近期运势。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from .constants import (
    LUCK_POOLS,
    RECENT_FIVE_STAR_WINDOW,
    STANDARD_CHARS,
    STANDARD_WEAPONS,
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False


@dataclass
class FiveStarHit:
    name: str
    pity: int
    time: str
    gacha_type: int
    item_type: str
    is_off: bool  # 歪 / 常驻
    is_guaranteed: bool = False  # 大保底（上一发歪后）


@dataclass
class PoolSliceStats:
    total_pulls: int = 0
    five_count: int = 0
    avg_pity: float = 0.0
    win_rate: Optional[float] = None  # 小保底不歪率（不含大保底）
    avg_per_up: Optional[float] = None  # 平均每 UP（仅角色/可追踪池）
    off_count: int = 0  # 小保底歪次数
    on_count: int = 0  # 小保底不歪次数
    fifty_fifty_count: int = 0
    lose_streak: int = 0  # 结尾连续小保底歪
    luck_label: str = "—"
    luck_color: str = "#888888"
    hits: list[FiveStarHit] = field(default_factory=list)
    current_pity: int = 0
    awaiting_guaranteed: bool = False


@dataclass
class PoolLuckReport:
    key: str
    title: str
    track_5050: bool
    long_term: PoolSliceStats
    recent: PoolSliceStats
    all_hits: list[FiveStarHit] = field(default_factory=list)
    current_pity: int = 0
    awaiting_guaranteed: bool = False


@dataclass
class LuckReport:
    pools: list[PoolLuckReport]
    source: str = ""
    uid: str = ""


def is_standard_five_star(name: str, item_type: str = "") -> bool:
    """是否常驻五星（角色池歪 / 武器池歪）。"""
    name = str(name or "").strip()
    item_type = str(item_type or "")
    if name in STANDARD_CHARS:
        return True
    if name in STANDARD_WEAPONS:
        return True
    if "武器" in item_type and name in STANDARD_WEAPONS:
        return True
    if "角色" in item_type and name in STANDARD_CHARS:
        return True
    return False


def _fifty_fifty_attempts(hits: list[FiveStarHit]) -> list[FiveStarHit]:
    """只保留小保底（50/50）尝试，排除大保底必出。"""
    return [h for h in hits if not h.is_guaranteed]


def _lose_streak(attempts: list[FiveStarHit]) -> int:
    streak = 0
    for h in reversed(attempts):
        if h.is_off:
            streak += 1
        else:
            break
    return streak


def _luck_from_metrics(
    avg_pity: float,
    win_rate: Optional[float],
    five_count: int,
    track_5050: bool,
    lose_streak: int = 0,
    *,
    recent_mode: bool = False,
) -> tuple[str, str]:
    """根据垫数、小保底不歪率、连歪打标签。"""
    if five_count <= 0:
        return "无数据", "#9e9e9e"
    if five_count < 3:
        if avg_pity <= 50:
            return "样本少·偏欧", "#43a047"
        if avg_pity >= 75:
            return "样本少·偏非", "#e53935"
        return "样本少", "#757575"

    score = 0.0
    if avg_pity <= 50:
        score += 2.2
    elif avg_pity <= 58:
        score += 1.4
    elif avg_pity <= 65:
        score += 0.4
    elif avg_pity <= 72:
        score -= 0.6
    elif avg_pity <= 78:
        score -= 1.4
    else:
        score -= 2.2

    if track_5050 and win_rate is not None:
        # 小保底理论约 50%
        if win_rate >= 0.70:
            score += 1.6
        elif win_rate >= 0.58:
            score += 0.9
        elif win_rate >= 0.48:
            score += 0.15
        elif win_rate >= 0.38:
            score -= 0.7
        elif win_rate >= 0.28:
            score -= 1.5
        else:
            score -= 2.2

    # 连歪：近期手感强惩罚；长期也计入但略轻
    if track_5050 and lose_streak >= 1:
        weight = 1.15 if recent_mode else 0.55
        if lose_streak >= 3:
            score -= 2.8 * weight
        elif lose_streak == 2:
            score -= 1.4 * weight
        else:
            score -= 0.45 * weight

    if score >= 2.5:
        return "极欧", "#2e7d32"
    if score >= 1.0:
        return "小欧", "#66bb6a"
    if score >= -0.5:
        return "中平", "#f9a825"
    if score >= -1.8:
        return "小非", "#ef6c00"
    return "极非", "#c62828"


def _slice_stats(
    hits: list[FiveStarHit],
    total_pulls: int,
    current_pity: int,
    awaiting_guaranteed: bool,
    track_5050: bool,
    *,
    recent_mode: bool = False,
) -> PoolSliceStats:
    if not hits and total_pulls == 0:
        return PoolSliceStats(current_pity=current_pity, awaiting_guaranteed=awaiting_guaranteed)

    pities = [h.pity for h in hits]
    avg_pity = float(np.mean(pities)) if pities else 0.0
    win_rate = None
    avg_per_up = None
    off_count = 0
    on_count = 0
    fifty_n = 0
    streak = 0

    if track_5050 and hits:
        attempts = _fifty_fifty_attempts(hits)
        fifty_n = len(attempts)
        off_count = sum(1 for h in attempts if h.is_off)
        on_count = sum(1 for h in attempts if not h.is_off)
        if attempts:
            win_rate = on_count / fifty_n
        streak = _lose_streak(attempts)
        limited = sum(1 for h in hits if not h.is_off)
        if limited > 0:
            avg_per_up = total_pulls / limited

    label, color = _luck_from_metrics(
        avg_pity,
        win_rate,
        len(hits),
        track_5050,
        lose_streak=streak,
        recent_mode=recent_mode,
    )
    return PoolSliceStats(
        total_pulls=total_pulls,
        five_count=len(hits),
        avg_pity=avg_pity,
        win_rate=win_rate,
        avg_per_up=avg_per_up,
        off_count=off_count,
        on_count=on_count,
        fifty_fifty_count=fifty_n,
        lose_streak=streak,
        luck_label=label,
        luck_color=color,
        hits=list(hits),
        current_pity=current_pity,
        awaiting_guaranteed=awaiting_guaranteed,
    )


def analyze_pool(raw: pd.DataFrame, pool: dict[str, Any]) -> PoolLuckReport:
    types = tuple(pool["types"])
    track = bool(pool["track_5050"])
    df = raw[raw["gacha_type"].isin(types)].copy()
    if len(df) == 0:
        empty = PoolSliceStats()
        return PoolLuckReport(
            key=pool["key"],
            title=pool["title"],
            track_5050=track,
            long_term=empty,
            recent=empty,
        )

    df = df.sort_values(["time", "id"], kind="mergesort").reset_index(drop=True)
    hits: list[FiveStarHit] = []
    pity = 0
    need_guaranteed = False  # 角色/武器池：歪之后下一金为大保底

    for _, row in df.iterrows():
        pity += 1
        try:
            rank = int(row.get("rank_type") or 0)
        except Exception:
            rank = 0
        if rank != 5:
            continue
        name = str(row.get("name") or "")
        item_type = str(row.get("item_type") or "")
        off = is_standard_five_star(name, item_type) if track else False
        # 常驻池不做歪判断
        if not track:
            off = False
        guaranteed = bool(track and need_guaranteed)
        # 大保底理论上必出限定；若仍出常驻（数据异常）仍标歪
        hit = FiveStarHit(
            name=name,
            pity=pity,
            time=str(row.get("time") or ""),
            gacha_type=int(row.get("gacha_type") or 0),
            item_type=item_type,
            is_off=off,
            is_guaranteed=guaranteed,
        )
        hits.append(hit)
        if track:
            if off:
                need_guaranteed = True
            else:
                need_guaranteed = False
        pity = 0

    current_pity = pity
    awaiting = need_guaranteed if track else False
    total_pulls = len(df)

    # 近期：最近 N 个五星；抽数按「从这 N 个的起点到池末」估算
    window = RECENT_FIVE_STAR_WINDOW
    recent_hits = hits[-window:] if hits else []
    if recent_hits:
        start_time = recent_hits[0].time
        recent_df = df[df["time"] >= start_time]
        # 含该五星之前的垫刀：用 pity 反推更准
        recent_pulls = int(sum(h.pity for h in recent_hits) + current_pity)
        # 若时间切片更大，取 max 避免偏小
        recent_pulls = max(recent_pulls, len(recent_df))
    else:
        recent_pulls = current_pity

    long_term = _slice_stats(hits, total_pulls, current_pity, awaiting, track, recent_mode=False)
    recent = _slice_stats(recent_hits, recent_pulls, current_pity, awaiting, track, recent_mode=True)

    return PoolLuckReport(
        key=pool["key"],
        title=pool["title"],
        track_5050=track,
        long_term=long_term,
        recent=recent,
        all_hits=hits,
        current_pity=current_pity,
        awaiting_guaranteed=awaiting,
    )


def analyze_luck(raw: pd.DataFrame, source: str = "", uid: str = "") -> LuckReport:
    work = raw.copy()
    work["gacha_type"] = pd.to_numeric(work["gacha_type"], errors="coerce").fillna(0).astype(int)
    work["rank_type"] = pd.to_numeric(work["rank_type"], errors="coerce").fillna(0).astype(int)
    work["time"] = work["time"].fillna("").astype(str)
    work["name"] = work["name"].fillna("").astype(str)
    work["item_type"] = work["item_type"].fillna("").astype(str)
    work["id"] = work["id"].fillna("").astype(str)

    pools = [analyze_pool(work, p) for p in LUCK_POOLS]
    return LuckReport(pools=pools, source=source, uid=str(uid or ""))


def luck_to_dict(report: LuckReport) -> dict[str, Any]:
    """供网页 / API 使用的结构化结果。"""

    def hit_dict(h: FiveStarHit) -> dict[str, Any]:
        return {
            "name": h.name,
            "pity": h.pity,
            "time": h.time,
            "gacha_type": h.gacha_type,
            "item_type": h.item_type,
            "is_off": h.is_off,
            "is_guaranteed": h.is_guaranteed,
            "tag": "歪" if h.is_off else ("保" if h.is_guaranteed else "限定"),
        }

    def slice_dict(s: PoolSliceStats) -> dict[str, Any]:
        return {
            "total_pulls": s.total_pulls,
            "five_count": s.five_count,
            "avg_pity": round(s.avg_pity, 1) if s.five_count else None,
            "win_rate": round(s.win_rate, 4) if s.win_rate is not None else None,
            "avg_per_up": round(s.avg_per_up, 1) if s.avg_per_up is not None else None,
            "off_count": s.off_count,
            "on_count": s.on_count,
            "fifty_fifty_count": s.fifty_fifty_count,
            "lose_streak": s.lose_streak,
            "luck_label": s.luck_label,
            "luck_color": s.luck_color,
            "current_pity": s.current_pity,
            "awaiting_guaranteed": s.awaiting_guaranteed,
        }

    pools = []
    for p in report.pools:
        recent_hits = list(reversed(p.all_hits[-16:]))
        pools.append(
            {
                "key": p.key,
                "title": p.title,
                "track_5050": p.track_5050,
                "current_pity": p.current_pity,
                "awaiting_guaranteed": p.awaiting_guaranteed,
                "long_term": slice_dict(p.long_term),
                "recent": slice_dict(p.recent),
                "timeline": [hit_dict(h) for h in recent_hits],
                "all_hits": [hit_dict(h) for h in p.all_hits],
            }
        )
    return {
        "source": report.source,
        "uid": report.uid,
        "recent_window": RECENT_FIVE_STAR_WINDOW,
        "pools": pools,
    }


def format_luck_text(report: LuckReport) -> str:
    lines = [f"欧非分析 ← {report.source or '本地库'}"]
    if report.uid:
        lines[0] += f" | UID {report.uid}"
    for pool in report.pools:
        lt, rc = pool.long_term, pool.recent
        lines.append(f"\n【{pool.title}】当前垫 {pool.current_pity}" + (" · 大保底" if pool.awaiting_guaranteed else ""))
        lines.append(
            f"  长期 {lt.luck_label}｜总抽 {lt.total_pulls}｜出金 {lt.five_count}｜均垫 {lt.avg_pity:.1f}"
            + (
                f"｜小保底不歪:歪 {lt.on_count}:{lt.off_count}"
                f"（{lt.win_rate:.1%}，{lt.fifty_fifty_count}次博弈）｜均UP {lt.avg_per_up:.0f}"
                if lt.win_rate is not None
                else ""
            )
            + (f"｜连歪{lt.lose_streak}" if lt.lose_streak >= 2 else "")
        )
        lines.append(
            f"  近期 {rc.luck_label}｜近{len(rc.hits)}金均垫 {rc.avg_pity:.1f}"
            + (
                f"｜小保底不歪:歪 {rc.on_count}:{rc.off_count}"
                f"（{rc.win_rate:.1%}，{rc.fifty_fifty_count}次博弈）"
                if rc.win_rate is not None
                else ""
            )
            + (f"｜连歪{rc.lose_streak}" if rc.lose_streak else "")
        )
        # 最近若干金
        show = pool.all_hits[-12:]
        if show:
            bits = []
            for h in reversed(show):
                tag = "歪" if h.is_off else ("保" if h.is_guaranteed else "限定")
                if not pool.track_5050:
                    tag = "金"
                bits.append(f"{h.name}({h.pity}/{tag})")
            lines.append("  " + " → ".join(bits))
    return "\n".join(lines)


def _draw_pool_card(ax, pool: PoolLuckReport) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # 背景卡片
    card = FancyBboxPatch(
        (0.02, 0.04),
        0.96,
        0.92,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.2,
        edgecolor="#cfd8dc",
        facecolor="#fafafa",
    )
    ax.add_patch(card)

    lt, rc = pool.long_term, pool.recent
    ax.text(0.05, 0.90, pool.title, fontsize=15, fontweight="bold", va="center", color="#263238")

    # 长期/近期标签
    for x, stats, title in ((0.42, lt, "长期"), (0.72, rc, "近期")):
        ax.text(x, 0.90, title, fontsize=9, color="#78909c", va="center")
        badge = FancyBboxPatch(
            (x + 0.06, 0.84),
            0.18,
            0.10,
            boxstyle="round,pad=0.008,rounding_size=0.015",
            linewidth=0,
            facecolor=stats.luck_color,
        )
        ax.add_patch(badge)
        ax.text(
            x + 0.15,
            0.89,
            stats.luck_label,
            fontsize=11,
            fontweight="bold",
            color="white",
            ha="center",
            va="center",
        )

    # 统计行
    parts = [
        f"总抽数 {lt.total_pulls}",
        f"出金 {lt.five_count}",
        f"均垫 {lt.avg_pity:.1f}" if lt.five_count else "均垫 —",
    ]
    if pool.track_5050 and lt.win_rate is not None:
        parts.append(f"小保底不歪:歪 {lt.on_count}:{lt.off_count}")
        parts.append(f"不歪率 {lt.win_rate:.1%}")
        if lt.avg_per_up is not None:
            parts.append(f"均UP {lt.avg_per_up:.0f}")
    if pool.track_5050:
        parts.append("大保底中" if pool.awaiting_guaranteed else "小保底中")
    ax.text(0.05, 0.74, "　".join(parts), fontsize=9.5, color="#37474f", va="center")

    recent_line = f"近期(近{len(rc.hits)}金): 均垫 {rc.avg_pity:.1f}" if rc.hits else "近期: 暂无出金"
    if pool.track_5050 and rc.win_rate is not None:
        recent_line += f"　不歪:歪 {rc.on_count}:{rc.off_count}（{rc.win_rate:.1%}）"
    ax.text(0.05, 0.66, recent_line, fontsize=9, color="#546e7a", va="center")

    # 出金时间线（新→旧），含当前垫刀
    show = list(reversed(pool.all_hits[-14:]))
    cells: list[tuple[str, str, str, bool]] = []
    # (?, current pity)
    pity_note = f"{pool.current_pity}"
    cells.append(("?", pity_note, "#90a4ae", False))
    for h in show:
        if pool.track_5050:
            color = "#e53935" if h.is_off else "#1e88e5"
        else:
            color = "#6a1b9a"
        cells.append((h.name, str(h.pity), color, h.is_off and pool.track_5050))

    n = len(cells)
    if n == 0:
        return
    left, right = 0.05, 0.95
    width = right - left
    box_w = min(0.065, width / max(n, 1) * 0.92)
    gap = (width - box_w * n) / max(n, 1)
    y0 = 0.18
    for i, (name, pity_s, color, off) in enumerate(cells):
        x = left + i * (box_w + gap)
        rect = FancyBboxPatch(
            (x, y0),
            box_w,
            0.36,
            boxstyle="round,pad=0.006,rounding_size=0.01",
            linewidth=1.0,
            edgecolor=color,
            facecolor="#ffffff",
        )
        ax.add_patch(rect)
        # 名字（过长截断）
        label = name if len(name) <= 3 else name[:3]
        ax.text(
            x + box_w / 2,
            y0 + 0.22,
            label,
            fontsize=8 if len(label) <= 2 else 7,
            ha="center",
            va="center",
            color="#263238",
            fontweight="bold",
        )
        ax.text(
            x + box_w / 2,
            y0 + 0.08,
            pity_s,
            fontsize=8,
            ha="center",
            va="center",
            color="#546e7a",
        )
        if off:
            ax.text(
                x + box_w * 0.78,
                y0 + 0.30,
                "歪",
                fontsize=7,
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
                bbox=dict(boxstyle="circle,pad=0.15", facecolor="#e53935", edgecolor="none"),
            )


def plot_luck_dashboard(report: LuckReport, figsize: tuple[float, float] = (14, 16)):
    """绘制四池欧非总结图。"""
    fig, axes = plt.subplots(4, 1, figsize=figsize, constrained_layout=True)
    title = "抽卡总结 · 欧非分析"
    if report.uid:
        title += f"（UID {report.uid}）"
    fig.suptitle(title, fontsize=18, fontweight="bold", color="#1a237e")
    if report.source:
        fig.text(0.5, 0.965, f"数据: {report.source}", ha="center", fontsize=9, color="#78909c")

    for ax, pool in zip(axes, report.pools):
        _draw_pool_card(ax, pool)

    # 图例说明
    fig.text(
        0.5,
        0.01,
        "蓝框=限定　「保」=大保底(不计入50/50)　红章「歪」=小保底歪　"
        f"不歪率仅统计小保底博弈　? =当前垫刀　近期=最近{RECENT_FIVE_STAR_WINDOW}金",
        ha="center",
        fontsize=9,
        color="#607d8b",
    )
    return fig
