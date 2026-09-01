"""抽卡分析结果缓存：按文件指纹命中，避免重复读表与全量重算。"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from .xlsx_store import ensure_data_dir, load_workbook

_lock = threading.RLock()
_book_by_fp: dict[str, dict[str, Any]] = {}
_luck_by_fp: dict[str, dict[str, Any]] = {}
_prob_by_key: dict[str, dict[str, Any]] = {}
_status_db_by_fp: dict[str, dict[str, Any]] = {}


def cache_dir() -> Path:
    d = ensure_data_dir() / ".cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def file_fingerprint(path: Path | str) -> str:
    path = Path(path)
    st = path.stat()
    return f"{path.resolve()}|{st.st_mtime_ns}|{st.st_size}"


def _fp_hash(fp: str) -> str:
    return hashlib.sha1(fp.encode("utf-8")).hexdigest()[:20]


def invalidate_all() -> None:
    with _lock:
        _book_by_fp.clear()
        _luck_by_fp.clear()
        _prob_by_key.clear()
        _status_db_by_fp.clear()
        d = cache_dir()
        for p in d.glob("luck_*.json"):
            try:
                p.unlink()
            except OSError:
                pass
        for p in d.glob("prob_*.json"):
            try:
                p.unlink()
            except OSError:
                pass


def get_workbook_cached(path: Path | str) -> tuple[dict[str, Any], str, bool]:
    """返回 (book, fingerprint, from_cache)。"""
    path = Path(path)
    fp = file_fingerprint(path)
    with _lock:
        hit = _book_by_fp.get(fp)
        if hit is not None:
            return hit, fp, True
        # 文件已变：丢掉旧簿
        _book_by_fp.clear()
    book = load_workbook(path)
    with _lock:
        _book_by_fp[fp] = book
        # 轻量 status 用
        raw = book["raw"]
        raw_n = len(raw)
        _status_db_by_fp[fp] = {
            "file": path.name,
            "rows": raw_n,
            "time_start": str(raw["time"].min()) if raw_n else "",
            "time_end": str(raw["time"].max()) if raw_n else "",
        }
    return book, fp, False


def light_db_info(path: Path | str) -> dict[str, Any]:
    """优先用缓存；未命中时只返回文件级信息，不强制读 Excel。"""
    path = Path(path)
    if not path.exists():
        return {"file": path.name, "error": "missing"}
    fp = file_fingerprint(path)
    with _lock:
        cached = _status_db_by_fp.get(fp)
        if cached:
            return dict(cached)
    try:
        st = path.stat()
        return {
            "file": path.name,
            "rows": None,
            "bytes": st.st_size,
            "mtime": int(st.st_mtime),
            "time_start": "",
            "time_end": "",
        }
    except Exception as e:
        return {"file": path.name, "error": str(e)}


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def get_or_build_luck(
    path: Path | str,
    builder: Callable[[dict[str, Any], str], dict[str, Any]],
) -> tuple[dict[str, Any], bool, float]:
    """
    命中指纹则直接返回缓存；否则 builder(book, fp) 全量计算并落盘。
    返回 (payload, from_cache, elapsed_ms)。
    """
    path = Path(path)
    t0 = time.perf_counter()
    book, fp, _book_hit = get_workbook_cached(path)

    with _lock:
        mem = _luck_by_fp.get(fp)
    if mem is not None:
        out = dict(mem)
        out["cache"] = {"hit": True, "source": "memory", "fingerprint": fp}
        return out, True, (time.perf_counter() - t0) * 1000

    disk = cache_dir() / f"luck_{_fp_hash(fp)}.json"
    disk_payload = _read_json(disk)
    if disk_payload and disk_payload.get("_fingerprint") == fp:
        body = {k: v for k, v in disk_payload.items() if not k.startswith("_")}
        with _lock:
            _luck_by_fp[fp] = body
        out = dict(body)
        out["cache"] = {"hit": True, "source": "disk", "fingerprint": fp}
        return out, True, (time.perf_counter() - t0) * 1000

    payload = builder(book, fp)
    with _lock:
        _luck_by_fp.clear()
        _luck_by_fp[fp] = payload
        # 数据变了，概率缓存也应失效
        _prob_by_key.clear()
    _write_json(disk, {**payload, "_fingerprint": fp, "_saved_at": time.time()})
    out = dict(payload)
    out["cache"] = {"hit": False, "source": "compute", "fingerprint": fp}
    return out, False, (time.perf_counter() - t0) * 1000


PROB_CACHE_VER = "v4"


def get_or_build_probability(
    path: Path | str,
    pulls: int,
    builder: Callable[[dict[str, Any], str, int], dict[str, Any]],
    *,
    pool_key: str = "character",
) -> tuple[dict[str, Any], bool, float]:
    path = Path(path)
    t0 = time.perf_counter()
    book, fp, _ = get_workbook_cached(path)
    key = f"{fp}|{PROB_CACHE_VER}|{pool_key}|{pulls}"

    with _lock:
        mem = _prob_by_key.get(key)
    if mem is not None:
        out = dict(mem)
        out["cache"] = {"hit": True, "source": "memory", "fingerprint": fp}
        return out, True, (time.perf_counter() - t0) * 1000

    disk = cache_dir() / f"prob_{_fp_hash(key)}.json"
    disk_payload = _read_json(disk)
    if disk_payload and disk_payload.get("_key") == key:
        body = {k: v for k, v in disk_payload.items() if not k.startswith("_")}
        with _lock:
            _prob_by_key[key] = body
        out = dict(body)
        out["cache"] = {"hit": True, "source": "disk", "fingerprint": fp}
        return out, True, (time.perf_counter() - t0) * 1000

    payload = builder(book, fp, pulls)
    with _lock:
        # 只清同文件旧 pulls，避免无限涨
        stale = [k for k in _prob_by_key if not k.startswith(fp)]
        for k in stale:
            _prob_by_key.pop(k, None)
        _prob_by_key[key] = payload
    _write_json(disk, {**payload, "_key": key, "_fingerprint": fp, "_saved_at": time.time()})
    out = dict(payload)
    out["cache"] = {"hit": False, "source": "compute", "fingerprint": fp}
    return out, False, (time.perf_counter() - t0) * 1000
