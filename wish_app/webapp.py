"""本地网页服务：登录、同步、欧非分析。"""

from __future__ import annotations

import base64
import threading
import time
import uuid
import webbrowser
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .api import sync_wishes_to_xlsx
from .auth import clear_session, load_session, login_with_qr
from .cache_store import (
    get_or_build_luck,
    get_or_build_probability,
    invalidate_all,
    light_db_info,
)
from .luck import analyze_luck, luck_to_dict
from .xlsx_store import current_db_path, extract_five_star_stats, find_latest_xlsx

STATIC_DIR = Path(__file__).resolve().parent / "web_static"

app = FastAPI(title="原神抽卡欧非分析")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_qr_jobs: dict[str, dict[str, Any]] = {}
_sync_lock = threading.Lock()
_logs: list[str] = []


def _log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    _logs.append(line)
    if len(_logs) > 200:
        del _logs[:50]
    print(line)


def _status_payload() -> dict[str, Any]:
    session = load_session()
    logged_in = bool(session and session.is_usable())
    role = session.selected_role() if logged_in else None
    path = None
    if role:
        p = current_db_path(role.game_uid)
        if p.exists():
            path = p
    if path is None:
        path = find_latest_xlsx()
    db = light_db_info(path) if path and Path(path).exists() else None
    return {
        "logged_in": logged_in,
        "uid": role.game_uid if role else "",
        "nickname": role.nickname if role else "",
        "region": role.region if role else "",
        "db": db,
    }


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
def api_status():
    return _status_payload()


@app.get("/api/logs")
def api_logs():
    return {"lines": _logs[-80:]}


@app.post("/api/logout")
def api_logout():
    clear_session()
    _log("已清除登录")
    return _status_payload()


@app.post("/api/login/qr/start")
def api_qr_start():
    job_id = uuid.uuid4().hex
    state: dict[str, Any] = {
        "id": job_id,
        "status": "pending",
        "message": "正在生成二维码…",
        "qr_png_b64": None,
        "error": None,
        "abort": False,
    }
    _qr_jobs[job_id] = state

    def work():
        try:
            def on_qr(url: str):
                from .qr_login import make_qr_png
                import tempfile
                import os

                path = os.path.join(tempfile.gettempdir(), f"mhy_qr_{job_id}.png")
                make_qr_png(url, path)
                with open(path, "rb") as f:
                    state["qr_png_b64"] = base64.b64encode(f.read()).decode("ascii")
                state["status"] = "scan"
                state["message"] = "请用米游社 APP 扫一扫"

            def on_status(msg: str):
                state["message"] = msg
                _log(msg)

            session = login_with_qr(
                timeout=180,
                on_qr=on_qr,
                on_status=on_status,
                should_abort=lambda: state["abort"],
            )
            if state["abort"]:
                state["status"] = "aborted"
                return
            role = session.selected_role()
            state["status"] = "ok"
            state["message"] = f"登录成功 UID {role.game_uid}"
            _log(state["message"])
        except Exception as e:
            if state["abort"]:
                state["status"] = "aborted"
                return
            state["status"] = "error"
            state["error"] = str(e) or repr(e)
            state["message"] = state["error"]
            _log(f"扫码失败: {state['error']}")

    threading.Thread(target=work, daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/login/qr/{job_id}")
def api_qr_poll(job_id: str):
    state = _qr_jobs.get(job_id)
    if not state:
        raise HTTPException(404, "任务不存在")
    return {
        "status": state["status"],
        "message": state["message"],
        "qr_png_b64": state.get("qr_png_b64"),
        "error": state.get("error"),
        "account": _status_payload() if state["status"] == "ok" else None,
    }


@app.post("/api/login/qr/{job_id}/cancel")
def api_qr_cancel(job_id: str):
    state = _qr_jobs.get(job_id)
    if state:
        state["abort"] = True
        state["status"] = "aborted"
        state["message"] = "已取消"
    return {"ok": True}


@app.post("/api/sync")
def api_sync():
    session = load_session()
    if not session or not session.is_usable():
        raise HTTPException(401, "未登录，请先扫码")
    if not _sync_lock.acquire(blocking=False):
        raise HTTPException(409, "同步进行中")

    def work():
        try:
            saved, report, book = sync_wishes_to_xlsx(session, on_progress=_log)
            invalidate_all()
            _log(f"同步完成 {saved.name}（已刷新分析缓存）")
            _log(report.summary())
        except Exception as e:
            _log(f"同步失败: {e}")
        finally:
            _sync_lock.release()

    threading.Thread(target=work, daemon=True).start()
    return {"ok": True, "message": "同步已开始"}


def _resolve_db_path(status: dict[str, Any]):
    path = find_latest_xlsx()
    if status.get("uid"):
        p = current_db_path(status["uid"])
        if p.exists():
            path = p
    return path


def _build_luck_payload(book: dict[str, Any], path: Path, status: dict[str, Any]) -> dict[str, Any]:
    uid = str((book.get("meta") or {}).get("uid") or status.get("uid") or "")
    report = analyze_luck(book["raw"], source=path.name, uid=uid)
    payload = luck_to_dict(report)
    try:
        _, _, pity = extract_five_star_stats(book["sheets"])
        payload["char_pity"] = pity
    except Exception:
        payload["char_pity"] = None
    return payload


@app.get("/api/luck")
def api_luck():
    status = _status_payload()
    path = _resolve_db_path(status)
    if path is None or not path.exists():
        raise HTTPException(404, "没有本地抽卡库，请先同步")

    payload, hit, ms = get_or_build_luck(
        path, lambda book, _fp: _build_luck_payload(book, path, status)
    )
    payload["status"] = status
    payload["elapsed_ms"] = round(ms, 1)
    _log(f"欧非分析 {'缓存命中' if hit else '已重算'} {ms:.0f}ms ← {path.name}")
    return payload


@app.get("/api/probability")
def api_probability(pulls: int | None = None, pool: str = "character"):
    """按卡池规则的垫刀概率 + 该池历史分布洞察。"""
    from .analysis import GenshinWishProbability
    from .constants import LUCK_POOLS, POOL_PITY_RULES

    status = _status_payload()
    path = _resolve_db_path(status)
    if path is None or not path.exists():
        raise HTTPException(404, "没有本地抽卡库，请先同步")

    pool_key = (pool or "character").strip().lower()
    if pool_key not in POOL_PITY_RULES:
        raise HTTPException(400, f"未知卡池: {pool}")
    rule = POOL_PITY_RULES[pool_key]
    hard = int(rule["hard_pity"])

    luck_payload, _, _ = get_or_build_luck(
        path, lambda book, _fp: _build_luck_payload(book, path, status)
    )
    pool_rep = next(
        (x for x in luck_payload.get("pools") or [] if x.get("key") == pool_key),
        None,
    )
    auto_pulls = int(pool_rep["current_pity"]) if pool_rep else 0
    if pulls is None:
        pulls = auto_pulls
    pulls = max(0, min(int(pulls), hard - 1))

    pool_meta = [
        {
            "key": p["key"],
            "title": p["title"],
            "track_5050": p["track_5050"],
            "current_pity": next(
                (
                    x.get("current_pity", 0)
                    for x in (luck_payload.get("pools") or [])
                    if x.get("key") == p["key"]
                ),
                0,
            ),
            "long_term": next(
                (
                    {
                        "luck_label": (x.get("long_term") or {}).get("luck_label"),
                        "luck_color": (x.get("long_term") or {}).get("luck_color"),
                    }
                    for x in (luck_payload.get("pools") or [])
                    if x.get("key") == p["key"]
                ),
                {},
            ),
            "recent": next(
                (
                    {
                        "luck_label": (x.get("recent") or {}).get("luck_label"),
                        "luck_color": (x.get("recent") or {}).get("luck_color"),
                    }
                    for x in (luck_payload.get("pools") or [])
                    if x.get("key") == p["key"]
                ),
                {},
            ),
        }
        for p in LUCK_POOLS
    ]

    def build(book, _fp, pulls_n: int):
        from .prob_insights import (
            build_history_insights,
            fifty_fifty_flags_from_hits,
            theoretical_first_five_star_pmf,
        )

        calc = GenshinWishProbability.for_pool(pool_key)
        probs = calc.calculate_probability(pulls_n)
        cums = calc.calculate_cumulative_prob(probs)
        stats = calc.generate_statistics(pulls_n, probs, cums)
        clean_stats = {}
        for k, v in stats.items():
            if isinstance(v, float):
                clean_stats[k] = round(v, 6)
            else:
                clean_stats[k] = v

        xs = list(range(pulls_n, pulls_n + len(probs)))
        hits = list((pool_rep or {}).get("all_hits") or [])
        ff_flags = (
            fifty_fifty_flags_from_hits(hits)
            if hits and rule.get("track_5050")
            else None
        )
        if hits:
            hit_counts = [int(h.get("pity") or 0) for h in hits]
            hit_names = [str(h.get("name") or "") for h in hits]
            insights = build_history_insights(
                hit_counts,
                hit_names,
                fifty_fifty_flags=ff_flags,
                pool_key=pool_key,
            )
        else:
            insights = build_history_insights([], [], pool_key=pool_key)

        theory = theoretical_first_five_star_pmf(pool_key)
        from .four_star import extract_four_star_state

        four = extract_four_star_state(book["raw"], pool_key)
        return {
            "source": path.name,
            "pool": pool_key,
            "pool_title": rule.get("title", pool_key),
            "pool_rule": {
                "base_rate": rule["base_rate"],
                "soft_pity": rule["soft_pity"],
                "hard_pity": rule["hard_pity"],
                "rate_increase": rule["rate_increase"],
                "track_5050": rule["track_5050"],
                "featured_note": rule.get("featured_note", ""),
            },
            "pulls": pulls_n,
            "auto_pulls": auto_pulls,
            "stats": clean_stats,
            "curve": {
                "x": xs,
                "single": [round(p, 6) for p in probs],
                "cumulative": [round(c, 6) for c in cums],
                "soft_pity": calc.SOFT_PITY_START,
                "hard_pity": calc.HARD_PITY,
            },
            "theory": theory,
            "insights": insights,
            "four_star": four,
            "history": {
                "wish_counts": insights["sequence"]["pity"],
                "characters": insights["sequence"]["names"],
                "avg": insights["avg"],
                "median": insights["median"],
                "std": insights["std"],
                "n": insights["n"],
            },
        }

    payload, hit, ms = get_or_build_probability(
        path, pulls, build, pool_key=pool_key
    )
    payload["status"] = status
    payload["elapsed_ms"] = round(ms, 1)
    payload["pools"] = pool_meta
    _log(
        f"概率分析[{pool_key}] {'缓存命中' if hit else '已重算'} "
        f"{ms:.0f}ms pulls={pulls}"
    )
    return payload


def run(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    import uvicorn

    url = f"http://{host}:{port}/"
    _log(f"网页界面: {url}")
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
