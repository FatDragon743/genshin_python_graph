"""标准 xlsx 本地库读写。"""

from __future__ import annotations

import glob
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .constants import (
    DATA_DIR,
    GACHA_TYPE_TO_SHEET,
    MAIN_COLUMNS,
    META_SHEET,
    RAW_COLUMNS,
    RAW_SHEET,
    ROOT_DIR,
    SHEET_ORDER,
)


def ensure_data_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def current_db_path(uid: str | int) -> Path:
    ensure_data_dir()
    return DATA_DIR / f"原神祈愿记录_{uid}.xlsx"


def timestamped_db_path(uid: str | int, when: datetime | None = None) -> Path:
    ensure_data_dir()
    when = when or datetime.now()
    return DATA_DIR / f"原神祈愿记录_{uid}_{when.strftime('%Y%m%d_%H%M%S')}.xlsx"


def find_latest_xlsx(search_dirs: list[Path | str] | None = None) -> Optional[Path]:
    """查找最新的祈愿记录 xlsx（优先 data/ 下 UID 主库，其次按修改时间）。"""
    dirs = search_dirs or [DATA_DIR, ROOT_DIR]
    candidates: list[Path] = []
    for d in dirs:
        pattern = os.path.join(str(d), "原神祈愿记录*.xlsx")
        for p in glob.glob(pattern):
            candidates.append(Path(p))
    if not candidates:
        return None
    # 优先选行数更多、时间跨度更完整的 UID 主库（非纯时间戳备份）
    def score(p: Path) -> tuple:
        name = p.name
        is_backup = bool(re.search(r"_\d{8}_\d{6}\.xlsx$", name))
        return (0 if is_backup else 1, p.stat().st_mtime, p.stat().st_size)

    return max(candidates, key=score)


def find_legacy_xlsx(exclude: Path | None = None) -> list[Path]:
    """收集可用的本地历史表（根目录旧导出、data/local 等）。"""
    exclude_res = exclude.resolve() if exclude else None
    seen: set[Path] = set()
    out: list[Path] = []
    patterns = [
        DATA_DIR / "原神祈愿记录_local.xlsx",
        *sorted(ROOT_DIR.glob("原神祈愿记录*.xlsx")),
        *sorted(DATA_DIR.glob("原神祈愿记录*.xlsx")),
    ]
    for p in patterns:
        if not p.is_file():
            continue
        key = p.resolve()
        if exclude_res and key == exclude_res:
            continue
        if key in seen:
            continue
        # 跳过时间戳备份，避免重复合并
        if re.search(r"_\d{8}_\d{6}\.xlsx$", p.name):
            continue
        seen.add(key)
        out.append(p)
    return out


def load_seed_raw(uid: str | int, primary: Path | None = None) -> tuple[pd.DataFrame, list[str]]:
    """
    加载 UID 主库，并并入根目录/本地旧历史，避免线上一年窗口覆盖掉更早记录。
    返回 (raw, 来源文件名列表)。
    """
    from .merge import union_local_raws

    primary = primary or current_db_path(uid)
    frames: list[pd.DataFrame] = []
    sources: list[str] = []

    if primary.exists():
        book = load_workbook(primary)
        frames.append(book["raw"])
        sources.append(primary.name)

    for path in find_legacy_xlsx(exclude=primary):
        try:
            book = load_workbook(path)
            raw = book["raw"]
            if raw is None or len(raw) == 0:
                continue
            # 只并入能拉长历史的表（有更早时间，或主库为空）
            if frames:
                cur_min = str(frames[0]["time"].min() or "") if len(frames[0]) else ""
                other_min = str(raw["time"].min() or "")
                other_max = str(raw["time"].max() or "")
                cur_max = str(frames[0]["time"].max() or "") if len(frames[0]) else ""
                extends = (other_min and cur_min and other_min < cur_min) or (
                    other_max and cur_max and other_max > cur_max
                )
                if not extends and len(frames[0]) > 0:
                    # 仍并入：体量大很多的完整历史
                    if len(raw) <= len(frames[0]) * 1.05:
                        continue
            frames.append(raw)
            sources.append(path.name)
        except Exception:
            continue

    if not frames:
        return empty_raw_df(), []
    return union_local_raws(*frames), sources


def _normalize_time(val: Any) -> str:
    if pd.isna(val):
        return ""
    if isinstance(val, datetime):
        return val.strftime("%Y-%m-%d %H:%M:%S")
    s = str(val).strip()
    # pandas Timestamp
    try:
        ts = pd.to_datetime(s)
        if pd.notna(ts):
            return ts.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        pass
    return s


def empty_raw_df() -> pd.DataFrame:
    return pd.DataFrame(columns=RAW_COLUMNS)


def api_records_to_raw(records: list[dict], uid: str | int) -> pd.DataFrame:
    """将 getGachaLog 条目转为 __raw DataFrame。"""
    rows = []
    for r in records:
        rows.append(
            {
                "id": str(r.get("id", "")),
                "uid": str(uid),
                "gacha_type": int(r.get("gacha_type", 0)),
                "time": _normalize_time(r.get("time")),
                "name": str(r.get("name", "")),
                "item_type": str(r.get("item_type", "")),
                "rank_type": int(r.get("rank_type", 0) or 0),
                "lang": str(r.get("lang", "zh-cn")),
            }
        )
    if not rows:
        return empty_raw_df()
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def load_workbook(path: Path | str) -> dict[str, Any]:
    """
    加载标准库。
    返回 {
      'path': Path,
      'sheets': {sheet_name: DataFrame(主表列)},
      'raw': DataFrame,
      'meta': dict,
    }
    """
    path = Path(path)
    xl = pd.ExcelFile(path, engine="openpyxl")
    sheets: dict[str, pd.DataFrame] = {}
    for name in SHEET_ORDER:
        if name in xl.sheet_names:
            df = pd.read_excel(path, sheet_name=name, engine="openpyxl")
            for col in MAIN_COLUMNS:
                if col not in df.columns:
                    df[col] = None
            df = df[MAIN_COLUMNS].copy()
            sheets[name] = df
        else:
            sheets[name] = pd.DataFrame(columns=MAIN_COLUMNS)

    raw = empty_raw_df()
    if RAW_SHEET in xl.sheet_names:
        raw = pd.read_excel(path, sheet_name=RAW_SHEET, engine="openpyxl")
        for col in RAW_COLUMNS:
            if col not in raw.columns:
                raw[col] = None
        raw = raw[RAW_COLUMNS].copy()
        raw["id"] = raw["id"].astype(str)
        raw["uid"] = raw["uid"].astype(str)
        raw["gacha_type"] = pd.to_numeric(raw["gacha_type"], errors="coerce").fillna(0).astype(int)
        raw["rank_type"] = pd.to_numeric(raw["rank_type"], errors="coerce").fillna(0).astype(int)
        raw["time"] = raw["time"].map(_normalize_time)
    else:
        # 从主表反向构造无 id 的 raw（兼容旧文件）
        raw = sheets_to_raw(sheets, uid="")

    meta: dict[str, Any] = {}
    if META_SHEET in xl.sheet_names:
        mdf = pd.read_excel(path, sheet_name=META_SHEET, engine="openpyxl")
        if {"key", "value"}.issubset(set(mdf.columns)):
            meta = {str(k): v for k, v in zip(mdf["key"], mdf["value"])}

    return {"path": path, "sheets": sheets, "raw": raw, "meta": meta}


def sheets_to_raw(sheets: dict[str, pd.DataFrame], uid: str = "") -> pd.DataFrame:
    """旧主表 -> raw（无官方 id）。"""
    rows = []
    for sheet_name, df in sheets.items():
        if df is None or len(df) == 0:
            continue
        # 反查默认 gacha_type（取该 sheet 第一个 type）
        from .constants import SHEET_TO_GACHA_TYPES

        gtypes = SHEET_TO_GACHA_TYPES.get(sheet_name, (0,))
        default_type = gtypes[0]
        for _, row in df.iterrows():
            rows.append(
                {
                    "id": "",
                    "uid": str(uid),
                    "gacha_type": default_type,
                    "time": _normalize_time(row.get("时间")),
                    "name": str(row.get("名称", "") if pd.notna(row.get("名称")) else ""),
                    "item_type": str(row.get("类别", "") if pd.notna(row.get("类别")) else ""),
                    "rank_type": int(row.get("星级", 0) or 0) if pd.notna(row.get("星级")) else 0,
                    "lang": "zh-cn",
                }
            )
    if not rows:
        return empty_raw_df()
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def raw_to_sheets(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """raw -> 各 Sheet 主表（含总次数/保底内重算）。"""
    from .merge import recompute_pity_counters

    sheets: dict[str, pd.DataFrame] = {name: pd.DataFrame(columns=MAIN_COLUMNS) for name in SHEET_ORDER}
    if raw is None or len(raw) == 0:
        return sheets

    work = raw.copy()
    work["time"] = work["time"].map(_normalize_time)
    work["sheet"] = work["gacha_type"].map(lambda t: GACHA_TYPE_TO_SHEET.get(int(t), None))
    work = work[work["sheet"].notna()]

    for sheet_name in SHEET_ORDER:
        part = work[work["sheet"] == sheet_name].copy()
        if len(part) == 0:
            continue
        # 角色活动：301/400 合并后按时间排序
        part = part.sort_values(["time", "id"], kind="mergesort")
        main = pd.DataFrame(
            {
                "时间": part["time"].values,
                "名称": part["name"].values,
                "类别": part["item_type"].values,
                "星级": part["rank_type"].values,
                "总次数": 0,
                "保底内": 0,
                "备注": "",
            }
        )
        sheets[sheet_name] = recompute_pity_counters(main)

    return sheets


def save_workbook(
    path: Path | str,
    raw: pd.DataFrame,
    meta: dict[str, Any] | None = None,
    also_write_timestamped: bool = True,
) -> Path:
    """按标准格式写入 xlsx，并可选再写一份带时间戳备份。"""
    path = Path(path)
    ensure_data_dir()
    sheets = raw_to_sheets(raw)
    meta = dict(meta or {})
    meta.setdefault("updated_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    if "uid" in meta:
        # 保证 raw uid 一致
        raw = raw.copy()
        raw["uid"] = str(meta["uid"])

    def _write(target: Path) -> None:
        with pd.ExcelWriter(target, engine="openpyxl") as writer:
            for name in SHEET_ORDER:
                sheets[name].to_excel(writer, sheet_name=name, index=False)
            raw_out = raw.copy() if raw is not None else empty_raw_df()
            for col in RAW_COLUMNS:
                if col not in raw_out.columns:
                    raw_out[col] = None
            raw_out[RAW_COLUMNS].to_excel(writer, sheet_name=RAW_SHEET, index=False)
            mdf = pd.DataFrame({"key": list(meta.keys()), "value": list(meta.values())})
            mdf.to_excel(writer, sheet_name=META_SHEET, index=False)

    _write(path)
    if also_write_timestamped:
        uid = meta.get("uid") or "unknown"
        backup = timestamped_db_path(uid)
        if backup.resolve() != path.resolve():
            _write(backup)
    return path


def extract_five_star_stats(
    sheets: dict[str, pd.DataFrame],
    sheet_names: list[str] | None = None,
) -> tuple[list[int], list[str], int]:
    """
    从主表提取 5 星间隔抽数与角色名。
    返回 (wish_counts, characters, pulls_since_last)。
    默认合并「角色活动祈愿」用于分析；若指定 sheet_names 则只看这些表。
    """
    names = sheet_names or ["角色活动祈愿"]
    frames = []
    for n in names:
        df = sheets.get(n)
        if df is not None and len(df) > 0:
            frames.append(df)
    if not frames:
        return [], [], 0

    df = pd.concat(frames, ignore_index=True)
    df = df.copy()
    df["时间"] = df["时间"].map(_normalize_time)
    df = df.sort_values("时间", kind="mergesort")

    wish_counts: list[int] = []
    characters: list[str] = []
    count = 0
    for _, row in df.iterrows():
        count += 1
        rank = row.get("星级")
        is5 = False
        try:
            is5 = int(rank) == 5
        except Exception:
            is5 = bool(re.search(r"5\s*星|五星", str(rank)))
        if is5:
            wish_counts.append(count)
            name = row.get("名称")
            characters.append(str(name) if pd.notna(name) else "未知")
            count = 0
    pulls_since_last = count
    return wish_counts, characters, pulls_since_last


def find_pulls_from_excel(search_dir: str | None = None) -> tuple[Optional[int], str]:
    """兼容旧接口：返回距离上一 5 星抽数。"""
    path = find_latest_xlsx([search_dir] if search_dir else None)
    if path is None:
        return None, "未找到匹配的祈愿记录文件"
    try:
        book = load_workbook(path)
        _, _, pulls = extract_five_star_stats(book["sheets"])
        return pulls, path.name
    except Exception as e:
        return None, f"读取失败: {e}"


def read_full_wish_records_from_excel(
    search_dir: str | None = None,
) -> tuple[Optional[list], Optional[list], str]:
    """兼容旧接口：返回 5 星间隔序列。"""
    path = find_latest_xlsx([search_dir] if search_dir else None)
    if path is None:
        return None, None, "未找到匹配的祈愿记录文件"
    try:
        book = load_workbook(path)
        wish_counts, characters, _ = extract_five_star_stats(book["sheets"])
        if not wish_counts:
            return None, None, f"未在文件中检测到5星记录（文件: {path.name})"
        return wish_counts, characters, path.name
    except Exception as e:
        return None, None, f"读取失败: {e}"
