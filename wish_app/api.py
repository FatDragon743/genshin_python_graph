"""官方祈愿记录 API 拉取（默认增量：碰到本地已有记录即停）。"""

from __future__ import annotations

from collections import Counter
import json
import time
from typing import Callable

import pandas as pd
import requests

from .auth import GameRole, Session, gen_authkey
from .constants import FETCH_GACHA_TYPES, GACHA_API
from .merge import fingerprint, local_known_keys, merge_raw, pool_cursors_from_raw
from .xlsx_store import api_records_to_raw, empty_raw_df


class GachaApiError(RuntimeError):
    def __init__(self, retcode: int, message: str):
        super().__init__(f"[{retcode}] {message}")
        self.retcode = retcode
        self.message = message


def fetch_gacha_page(
    authkey: str,
    gacha_type: int,
    page: int = 1,
    size: int = 20,
    end_id: str = "0",
    region_host: str | None = None,
) -> list[dict]:
    """拉取单页抽卡记录（官方按时间从新到旧）。"""
    base = region_host or GACHA_API
    params = {
        "authkey_ver": "1",
        "sign_type": "2",
        "auth_appid": "webview_gacha",
        "lang": "zh-cn",
        "authkey": authkey,
        "gacha_type": str(gacha_type),
        "page": str(page),
        "size": str(size),
        "end_id": str(end_id or "0"),
    }
    r = requests.get(base, params=params, timeout=30)
    r.raise_for_status()
    body = r.json()
    retcode = body.get("retcode")
    if retcode != 0:
        raise GachaApiError(int(retcode), str(body.get("message") or body))
    return (body.get("data") or {}).get("list") or []


def _row_known(row: dict, known_ids: set[str], known_fps) -> bool:
    rid = str(row.get("id") or "")
    if rid and rid in known_ids:
        return True
    fp = fingerprint(
        {
            "time": row.get("time"),
            "name": row.get("name"),
            "rank_type": row.get("rank_type"),
            "item_type": row.get("item_type"),
            "gacha_type": row.get("gacha_type"),
        }
    )
    return known_fps.get(fp, 0) > 0


def fetch_gacha_type_incremental(
    authkey: str,
    gacha_type: int,
    known_ids: set[str] | None = None,
    known_fps=None,
    sleep_sec: float = 0.4,
    on_progress: Callable[[str], None] | None = None,
    full: bool = False,
) -> tuple[list[dict], bool]:
    """
    分页拉取某一卡池。默认从最新往旧拉，碰到本地已有记录即停。
    返回 (records, stopped_on_existing)。
    """
    known_ids = known_ids or set()
    fps = Counter(known_fps or {})
    all_rows: list[dict] = []
    page = 1
    end_id = "0"
    stopped = False

    while True:
        if on_progress:
            mode = "全量" if full or not (known_ids or fps) else "增量"
            on_progress(f"拉取卡池 {gacha_type} 第 {page} 页（{mode}）…")
        rows = fetch_gacha_page(authkey, gacha_type, page=page, size=20, end_id=end_id)
        if not rows:
            break

        if full or (not known_ids and not fps):
            all_rows.extend(rows)
        else:
            batch_new: list[dict] = []
            for row in rows:
                row.setdefault("gacha_type", str(gacha_type))
                if _row_known(row, known_ids, fps):
                    stopped = True
                    break
                batch_new.append(row)
            all_rows.extend(batch_new)
            if stopped:
                if on_progress:
                    on_progress(f"卡池 {gacha_type} 碰到本地已有记录，停止（本池新增 {len(all_rows)}）")
                break

        end_id = str(rows[-1].get("id") or "0")
        if len(rows) < 20:
            break
        page += 1
        time.sleep(sleep_sec)

    return all_rows, stopped


def fetch_all_wishes(
    session: Session,
    role: GameRole | None = None,
    gacha_types: list[int] | None = None,
    sleep_sec: float = 0.4,
    on_progress: Callable[[str], None] | None = None,
    known_ids: set[str] | None = None,
    known_fps=None,
    full: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """
    拉取卡池记录。默认增量（known_* 来自本地）。
    返回 (raw_df, fetch_meta)。
    """
    role = role or session.selected_role()
    if role is None:
        raise ValueError("未选择原神账号")
    if on_progress:
        on_progress("正在申请 authkey…")
    authkey = gen_authkey(session, role)
    types = gacha_types or list(FETCH_GACHA_TYPES)
    all_records: list[dict] = []
    any_stopped = False
    per_pool: dict[str, dict] = {}

    use_full = full or not (known_ids or known_fps)
    if on_progress:
        if use_full:
            on_progress("本地无历史或强制全量，开始全量拉取…")
        else:
            on_progress(f"增量同步：本地已知 {len(known_ids or [])} 条 id，碰到已有即停")

    for gt in types:
        try:
            rows, stopped = fetch_gacha_type_incremental(
                authkey,
                gt,
                known_ids=known_ids,
                known_fps=known_fps,
                sleep_sec=sleep_sec,
                on_progress=on_progress,
                full=use_full,
            )
        except GachaApiError as e:
            if e.retcode in (-1, -100, 100010):
                if on_progress:
                    on_progress(f"卡池 {gt} 跳过: {e.message}")
                continue
            if e.retcode in (-101, -10001) or "authkey" in e.message.lower():
                raise
            if on_progress:
                on_progress(f"卡池 {gt} 错误: {e}")
            continue

        any_stopped = any_stopped or stopped
        for row in rows:
            row.setdefault("gacha_type", str(gt))
            row["gacha_type"] = int(row.get("gacha_type") or gt)
        per_pool[str(gt)] = {
            "fetched": len(rows),
            "stopped_on_existing": stopped,
            "newest_time": str(rows[0].get("time") or "") if rows else "",
            "oldest_time": str(rows[-1].get("time") or "") if rows else "",
            "newest_id": str(rows[0].get("id") or "") if rows else "",
        }
        all_records.extend(rows)
        time.sleep(sleep_sec)

    fetch_meta = {
        "mode": "full" if use_full else "incremental",
        "stopped_on_existing": any_stopped,
        "fetched_total": len(all_records),
        "pools": per_pool,
    }
    if not all_records:
        return empty_raw_df(), fetch_meta
    return api_records_to_raw(all_records, uid=role.game_uid), fetch_meta


def sync_wishes_to_xlsx(
    session: Session,
    local_path=None,
    on_progress: Callable[[str], None] | None = None,
    full: bool = False,
):
    """
    增量拉取线上新记录，信任本地历史，合并后保存。
    返回 (saved_path, merge_report, book_dict)。
    """
    from .xlsx_store import (
        current_db_path,
        load_seed_raw,
        load_workbook,
        save_workbook,
    )

    role = session.selected_role()
    if role is None:
        raise ValueError("未选择原神账号")

    path = local_path or current_db_path(role.game_uid)
    local_raw, seed_sources = load_seed_raw(role.game_uid, primary=path)
    meta: dict = {}
    if path.exists():
        try:
            meta = dict(load_workbook(path).get("meta") or {})
        except Exception:
            meta = {}
    if on_progress:
        if seed_sources:
            on_progress(f"本地种子库: {', '.join(seed_sources)}（共 {len(local_raw)} 条）")
        else:
            on_progress("本地尚无历史，将全量拉取线上窗口")

    known_ids, known_fps = local_known_keys(local_raw)
    online_raw, fetch_meta = fetch_all_wishes(
        session,
        role=role,
        on_progress=on_progress,
        known_ids=known_ids,
        known_fps=known_fps,
        full=full,
    )

    if on_progress:
        on_progress("合并：保留窗口前本地历史，窗口内对齐线上，只追加新记录…")
    merged, report = merge_raw(local_raw, online_raw, trust_local=True)

    cursors = pool_cursors_from_raw(merged)
    meta.update(
        {
            "uid": role.game_uid,
            "nickname": role.nickname,
            "region": role.region,
            "last_sync": time.strftime("%Y-%m-%d %H:%M:%S"),
            "last_sync_mode": fetch_meta.get("mode"),
            "last_sync_fetched": fetch_meta.get("fetched_total"),
            "last_sync_stopped_on_existing": fetch_meta.get("stopped_on_existing"),
            "last_sync_range": json.dumps(fetch_meta.get("pools") or {}, ensure_ascii=False),
            "seed_sources": ",".join(seed_sources),
            "pool_cursors": json.dumps(cursors, ensure_ascii=False),
            "merge_summary": report.summary(),
        }
    )
    if on_progress:
        on_progress(f"写入 {path.name} …")
    also_ts = report.added > 0 or fetch_meta.get("mode") == "full"
    saved = save_workbook(path, merged, meta=meta, also_write_timestamped=also_ts)
    book = load_workbook(saved)
    return saved, report, book
