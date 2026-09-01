"""抽卡记录合并规则与保底计数重算。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .constants import RAW_COLUMNS


@dataclass
class MergeReport:
    added: int = 0
    kept_local_history: int = 0
    updated: int = 0
    skipped_dup: int = 0
    total_after: int = 0
    online_window_start: str = ""
    online_window_end: str = ""
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [
            f"新增 {self.added}",
            f"保留本地历史 {self.kept_local_history}",
            f"窗口内补缺 {self.updated}",
            f"窗口内对齐 {self.skipped_dup}",
            f"合计 {self.total_after}",
        ]
        if self.online_window_start:
            parts.append(f"线上窗口 {self.online_window_start} ~ {self.online_window_end}")
        return "；".join(parts)


def fingerprint(row: pd.Series | dict) -> str:
    time_ = str(row.get("time", "") or "")
    name = str(row.get("name", "") or "")
    rank = str(row.get("rank_type", "") or "")
    item_type = str(row.get("item_type", "") or "")
    gacha = str(row.get("gacha_type", "") or "")
    return f"{time_}|{name}|{rank}|{item_type}|{gacha}"


def recompute_pity_counters(df: pd.DataFrame) -> pd.DataFrame:
    """按时间升序重算总次数与保底内（遇 5 星重置保底内）。"""
    if df is None or len(df) == 0:
        return df.copy() if df is not None else pd.DataFrame()

    out = df.copy()
    if "时间" in out.columns:
        out = out.sort_values("时间", kind="mergesort").reset_index(drop=True)

    total = 0
    pity = 0
    totals = []
    pities = []
    for _, row in out.iterrows():
        total += 1
        pity += 1
        totals.append(total)
        pities.append(pity)
        rank = row.get("星级")
        try:
            is5 = int(rank) == 5
        except Exception:
            is5 = False
        if is5:
            pity = 0
    out["总次数"] = totals
    out["保底内"] = pities
    if "备注" not in out.columns:
        out["备注"] = ""
    return out


def _normalize_raw(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=RAW_COLUMNS)
    out = df.copy()
    for col in RAW_COLUMNS:
        if col not in out.columns:
            out[col] = "" if col not in ("gacha_type", "rank_type") else 0
    out["id"] = out["id"].fillna("").astype(str).replace({"nan": "", "None": ""})
    out["uid"] = out["uid"].fillna("").astype(str)
    out["gacha_type"] = pd.to_numeric(out["gacha_type"], errors="coerce").fillna(0).astype(int)
    out["rank_type"] = pd.to_numeric(out["rank_type"], errors="coerce").fillna(0).astype(int)
    out["time"] = out["time"].fillna("").astype(str)
    out["name"] = out["name"].fillna("").astype(str)
    out["item_type"] = out["item_type"].fillna("").astype(str)
    out["lang"] = out["lang"].fillna("zh-cn").astype(str)
    return out[RAW_COLUMNS]


def _finalize(result: pd.DataFrame) -> pd.DataFrame:
    result = _normalize_raw(result)
    with_id = result[result["id"] != ""]
    without_id = result[result["id"] == ""]
    if len(with_id):
        with_id = with_id.drop_duplicates(subset=["id"], keep="last")
    result = pd.concat([with_id, without_id], ignore_index=True)
    result = result.sort_values(["time", "id"], kind="mergesort").reset_index(drop=True)
    return _normalize_raw(result)


def union_local_raws(*raws: pd.DataFrame) -> pd.DataFrame:
    """
    合并多份本地历史：全部保留；指纹相同时优先带官方 id 的行。
    用于把根目录旧表 / local.xlsx 与 UID 库拼回完整时间线。
    """
    frames = [_normalize_raw(r) for r in raws if r is not None and len(r) > 0]
    if not frames:
        return pd.DataFrame(columns=RAW_COLUMNS)
    if len(frames) == 1:
        return frames[0]

    # 先按指纹分组：有 id 的覆盖无 id
    best: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    fp_counts: Counter = Counter()
    extras: list[dict[str, Any]] = []  # 同指纹多抽

    for frame in frames:
        for _, row in frame.iterrows():
            d = row.to_dict()
            fp = fingerprint(d)
            rid = str(d.get("id") or "")
            n = fp_counts[fp]
            fp_counts[fp] += 1
            if n == 0:
                best[fp] = d
                order.append(fp)
                continue
            # 已有同指纹：优先升级为带 id；若都有/都无 id，保留为合法多抽
            prev = best[fp]
            prev_id = str(prev.get("id") or "")
            if rid and not prev_id:
                best[fp] = d
            elif rid and prev_id and rid != prev_id:
                extras.append(d)
            elif not rid and prev_id:
                pass
            else:
                extras.append(d)

    rows = [best[fp] for fp in order] + extras
    return _finalize(pd.DataFrame(rows, columns=RAW_COLUMNS))


def merge_raw(
    local_raw: pd.DataFrame,
    online_raw: pd.DataFrame,
    *,
    trust_local: bool = True,
) -> tuple[pd.DataFrame, MergeReport]:
    """
    合并规则（默认信任本地历史）：
    1. 早于线上窗口的本地记录全部保留
    2. 窗口内：线上为准（带官方 id）；本地多出的补缺保留
    3. 线上新于本地的记录全部追加
    """
    report = MergeReport()
    local = _normalize_raw(local_raw)
    online = _normalize_raw(online_raw)

    if len(online) == 0:
        report.kept_local_history = len(local)
        report.total_after = len(local)
        report.notes.append("无线上新数据，保留本地全部")
        return local, report

    online_dedup = online.copy()
    if (online_dedup["id"] != "").any():
        online_dedup = online_dedup.sort_values(["time", "id"], kind="mergesort")
        online_dedup = online_dedup.drop_duplicates(subset=["id"], keep="last")

    times = [t for t in online_dedup["time"].tolist() if t]
    window_start = min(times) if times else ""
    window_end = max(times) if times else ""
    report.online_window_start = window_start
    report.online_window_end = window_end

    if len(local) == 0:
        report.added = len(online_dedup)
        report.total_after = len(online_dedup)
        return online_dedup.reset_index(drop=True), report

    if not trust_local:
        report.notes.append("trust_local=False 仍按信任本地历史处理")

    if window_start:
        history = local[local["time"] < window_start].copy()
        in_window = local[local["time"] >= window_start].copy()
    else:
        history = local.iloc[0:0].copy()
        in_window = local.copy()
    report.kept_local_history = len(history)

    online_ids = {i for i in online_dedup["id"].tolist() if i}
    online_fp_budget = Counter(fingerprint(r) for _, r in online_dedup.iterrows())

    gap_rows: list[dict[str, Any]] = []
    for _, row in in_window.iterrows():
        rid = str(row.get("id") or "")
        fp = fingerprint(row)
        if rid and rid in online_ids:
            report.skipped_dup += 1
            continue
        if online_fp_budget[fp] > 0:
            online_fp_budget[fp] -= 1
            report.skipped_dup += 1
            continue
        gap_rows.append(row.to_dict())
        report.updated += 1

    # 线上相对「历史+窗口本地」的真正新增数
    local_ids = {i for i in local["id"].tolist() if i}
    local_fp = Counter(fingerprint(r) for _, r in local.iterrows())
    added = 0
    for _, row in online_dedup.iterrows():
        rid = str(row.get("id") or "")
        fp = fingerprint(row)
        if rid and rid in local_ids:
            continue
        if local_fp[fp] > 0:
            local_fp[fp] -= 1
            continue
        added += 1
    report.added = added

    parts = [history]
    if gap_rows:
        parts.append(pd.DataFrame(gap_rows, columns=RAW_COLUMNS))
    parts.append(online_dedup)
    result = _finalize(pd.concat(parts, ignore_index=True))
    report.total_after = len(result)
    return result, report


def local_known_keys(local_raw: pd.DataFrame) -> tuple[set[str], Counter]:
    """本地已知官方 id 与指纹预算（增量拉取停点）。"""
    local = _normalize_raw(local_raw)
    ids = {i for i in local["id"].tolist() if i}
    fps = Counter(fingerprint(r) for _, r in local.iterrows())
    return ids, fps


def pool_cursors_from_raw(local_raw: pd.DataFrame) -> dict[str, dict[str, str]]:
    """按卡池标记本地最新一条，写入 meta 供下次对照。"""
    local = _normalize_raw(local_raw)
    cursors: dict[str, dict[str, str]] = {}
    if len(local) == 0:
        return cursors
    for gt, group in local.groupby("gacha_type", sort=False):
        g = group.sort_values(["time", "id"], ascending=[False, False], kind="mergesort")
        top = g.iloc[0]
        cursors[str(int(gt))] = {
            "newest_id": str(top.get("id") or ""),
            "newest_time": str(top.get("time") or ""),
        }
    return cursors
