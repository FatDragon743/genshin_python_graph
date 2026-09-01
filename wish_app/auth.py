"""米游社 Cookie / SToken / AuthKey 与浏览器手动登录。"""

from __future__ import annotations

import hashlib
import json
import random
import string
import threading
import time
import urllib.parse
import webbrowser
from dataclasses import asdict, dataclass, field, fields
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import parse_qs, urlparse

import requests

from .constants import (
    DATA_DIR,
    GAME_ROLES_URL,
    GEN_AUTHKEY_URL,
    LOGIN_BY_COOKIE_URL,
    MIHOYO_LOGIN_URL,
    MULTI_TOKEN_URL,
    SESSION_PATH,
)


@dataclass
class GameRole:
    game_uid: str
    nickname: str
    region: str
    game_biz: str = "hk4e_cn"
    level: int = 0


@dataclass
class Session:
    cookie: str = ""
    stoken_cookie: str = ""
    account_id: str = ""
    roles: list[dict] = field(default_factory=list)
    selected_uid: str = ""
    updated_at: float = 0.0

    def selected_role(self) -> Optional[GameRole]:
        uid = str(self.selected_uid or "")
        for r in self.roles:
            if str(r.get("game_uid")) == uid:
                return GameRole(
                    game_uid=str(r.get("game_uid")),
                    nickname=str(r.get("nickname", "")),
                    region=str(r.get("region", "")),
                    game_biz=str(r.get("game_biz", "hk4e_cn")),
                    level=int(r.get("level", 0) or 0),
                )
        if self.roles:
            r = self.roles[0]
            return GameRole(
                game_uid=str(r.get("game_uid")),
                nickname=str(r.get("nickname", "")),
                region=str(r.get("region", "")),
                game_biz=str(r.get("game_biz", "hk4e_cn")),
                level=int(r.get("level", 0) or 0),
            )
        return None

    def is_usable(self) -> bool:
        return bool(self.stoken_cookie or self.cookie) and bool(self.roles)


def ensure_data_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def save_session(session: Session, path: Path = SESSION_PATH) -> None:
    ensure_data_dir()
    path.write_text(json.dumps(asdict(session), ensure_ascii=False, indent=2), encoding="utf-8")


def load_session(path: Path = SESSION_PATH) -> Optional[Session]:
    """从 data/session.json 恢复登录；字段容错，避免版本升级后读失败。"""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        allowed = {f.name for f in fields(Session)}
        filtered = {k: v for k, v in data.items() if k in allowed}
        session = Session(**filtered)
        if not session.is_usable():
            return None
        return session
    except Exception:
        return None


def clear_session(path: Path = SESSION_PATH) -> None:
    if path.exists():
        path.unlink()


def normalize_cookie_input(raw: str) -> str:
    """支持 document.cookie 字符串，或 Cookie 提取器导出的 JSON 数组。"""
    text = (raw or "").strip()
    if not text:
        return ""
    if text.startswith("["):
        try:
            arr = json.loads(text)
            if isinstance(arr, list):
                pairs = []
                for item in arr:
                    if not isinstance(item, dict):
                        continue
                    name = item.get("name")
                    value = item.get("value")
                    if name and value is not None:
                        pairs.append(f"{name}={value}")
                if pairs:
                    return "; ".join(pairs)
        except json.JSONDecodeError:
            pass
    return text


def _parse_cookie_str(cookie: str) -> dict[str, str]:
    out = {}
    for part in normalize_cookie_input(cookie).replace("\n", "").split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _cookie_header(d: dict[str, str]) -> str:
    return "; ".join(f"{k}={v}" for k, v in d.items())


def random_text(num: int = 6) -> str:
    return "".join(random.sample(string.ascii_lowercase + string.digits, num))


def get_ds(salt: str = "ulInCDohgEs557j0VsPDYnQaaz6KJcv5") -> str:
    i = str(int(time.time()))
    r = random_text(6)
    c = hashlib.md5(f"salt={salt}&t={i}&r={r}".encode()).hexdigest()
    return f"{i},{r},{c}"


def build_stoken_cookie(cookie: str) -> tuple[str, str, list[GameRole]]:
    """
    Cookie -> 带 stoken 的 cookie、account_id、原神角色列表。
    若 Cookie 已含 stoken/stoken_v2，则直接使用。
    """
    cookie = normalize_cookie_input(cookie)
    jar = _parse_cookie_str(cookie)
    account_id = (
        jar.get("account_id")
        or jar.get("account_id_v2")
        or jar.get("ltuid")
        or jar.get("ltuid_v2")
        or jar.get("stuid")
        or ""
    )
    mid = jar.get("mid") or jar.get("account_mid_v2") or jar.get("ltmid_v2") or ""
    stoken = jar.get("stoken") or jar.get("stoken_v2") or ""

    # 已有 stoken（成熟项目要求：v2 必须带 mid）
    if stoken:
        if account_id and "stuid" not in jar:
            jar["stuid"] = account_id
        jar["stoken"] = stoken
        if mid:
            jar["mid"] = mid
        # genAuthKey 最小 cookie
        minimal = {"stuid": jar.get("stuid") or account_id, "stoken": stoken}
        if mid:
            minimal["mid"] = mid
        stoken_cookie = _cookie_header({**jar, **minimal})
        roles = fetch_game_roles(stoken_cookie)
        if not roles:
            # 有时仅 stoken+mid 即可拉角色
            roles = fetch_game_roles(_cookie_header(minimal))
        return stoken_cookie, str(account_id), roles

    # 通过 login_ticket 换 multi token（旧链路，多数情况下已不可用）
    login_ticket = jar.get("login_ticket")
    if not login_ticket:
        headers = {"cookie": cookie}
        r = requests.get(LOGIN_BY_COOKIE_URL + str(int(time.time() * 1000)), headers=headers, timeout=20)
        r.raise_for_status()
        info = r.json()
        if info.get("data", {}).get("account_info"):
            account_id = str(info["data"]["account_info"].get("account_id") or account_id)
            login_ticket = info["data"]["account_info"].get("weblogin_token") or login_ticket

    if not login_ticket:
        raise ValueError(
            "Cookie 中缺少 stoken。抽卡记录需要 stoken（+ mid）。\n"
            "请使用界面「米游社扫码登录」（移植自 mihoyo_qr_login），"
            "或粘贴含 stoken/stoken_v2 与 mid 的 Cookie。"
        )

    if not account_id:
        # 再试一次拿 account_id
        headers = {"cookie": cookie}
        r = requests.get(LOGIN_BY_COOKIE_URL + str(int(time.time() * 1000)), headers=headers, timeout=20)
        data = r.json().get("data") or {}
        account_info = data.get("account_info") or {}
        account_id = str(account_info.get("account_id") or "")

    if not account_id:
        raise ValueError("无法解析 account_id，请确认 Cookie 来自已登录的米哈游通行证。")

    url = (
        f"{MULTI_TOKEN_URL}"
        f"?login_ticket={urllib.parse.quote(login_ticket)}&token_types=3&uid={account_id}"
    )
    r = requests.get(url, headers={"cookie": cookie}, timeout=20)
    r.raise_for_status()
    body = r.json()
    token_list = (body.get("data") or {}).get("list") or []
    if not token_list:
        raise ValueError(f"换取 stoken 失败: {body.get('message') or body}")

    jar["stuid"] = str(account_id)
    for d in token_list:
        jar[d["name"]] = d["token"]
    # 合并原 cookie
    for k, v in _parse_cookie_str(cookie).items():
        jar.setdefault(k, v)
    stoken_cookie = _cookie_header(jar)
    roles = fetch_game_roles(stoken_cookie)
    return stoken_cookie, str(account_id), roles


def fetch_game_roles(cookie: str) -> list[GameRole]:
    headers = {"cookie": cookie}
    r = requests.get(f"{GAME_ROLES_URL}?game_biz=hk4e_cn", headers=headers, timeout=20)
    r.raise_for_status()
    body = r.json()
    items = (body.get("data") or {}).get("list") or []
    roles = []
    for v in items:
        roles.append(
            GameRole(
                game_uid=str(v.get("game_uid")),
                nickname=str(v.get("nickname", "")),
                region=str(v.get("region", "")),
                game_biz=str(v.get("game_biz", "hk4e_cn")),
                level=int(v.get("level", 0) or 0),
            )
        )
    return roles


def gen_authkey(session: Session, role: GameRole | None = None) -> str:
    role = role or session.selected_role()
    if role is None:
        raise ValueError("未选择原神账号")
    cookie = session.stoken_cookie or session.cookie
    jar = _parse_cookie_str(cookie)
    stoken = jar.get("stoken") or jar.get("stoken_v2")
    mid = jar.get("mid") or jar.get("account_mid_v2") or jar.get("ltmid_v2")
    if not stoken:
        raise ValueError("会话中无 stoken，请重新扫码登录")
    # 成熟项目惯例：stoken_v2 请求 Cookie 为 mid + stoken
    auth_cookie = f"mid={mid};stoken={stoken}" if mid else f"stoken={stoken}"
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://webstatic.mihoyo.com",
        "x-rpc-app_version": "2.28.1",
        "x-rpc-client_type": "5",
        "x-rpc-device_id": "CBEC8312-AA77-489E-AE8A-8D498DE24E90",
        "x-requested-with": "com.mihoyo.hyperion",
        "DS": get_ds(),
        "Cookie": auth_cookie,
    }
    data = {
        "auth_appid": "webview_gacha",
        "game_biz": role.game_biz,
        "game_uid": role.game_uid,
        "region": role.region,
    }
    r = requests.post(GEN_AUTHKEY_URL, headers=headers, json=data, timeout=20)
    r.raise_for_status()
    body = r.json()
    if body.get("retcode") not in (0, "0") or not (body.get("data") or {}).get("authkey"):
        raise ValueError(f"genAuthKey 失败: {body.get('message') or body}")
    return body["data"]["authkey"]


def _session_from_parts(
    cookie: str,
    stoken_cookie: str,
    account_id: str,
    roles: list[GameRole],
    selected_uid: str | None = None,
) -> Session:
    if not roles:
        raise ValueError("米游社未绑定原神账号（官服/渠道服需已绑定）。")
    session = Session(
        cookie=cookie,
        stoken_cookie=stoken_cookie,
        account_id=account_id,
        roles=[
            {
                "game_uid": r.game_uid,
                "nickname": r.nickname,
                "region": r.region,
                "game_biz": r.game_biz,
                "level": r.level,
            }
            for r in roles
        ],
        selected_uid=str(selected_uid or roles[0].game_uid),
        updated_at=time.time(),
    )
    save_session(session)
    return session


def login_with_cookie(cookie: str, selected_uid: str | None = None) -> Session:
    cookie = normalize_cookie_input(cookie)
    stoken_cookie, account_id, roles = build_stoken_cookie(cookie)
    return _session_from_parts(cookie, stoken_cookie, account_id, roles, selected_uid)


def login_with_qr(
    timeout: int = 180,
    on_qr: Callable[[str], None] | None = None,
    on_status: Callable[[str], None] | None = None,
    should_abort: Callable[[], bool] | None = None,
    selected_uid: str | None = None,
) -> Session:
    """米游社 APP 扫码登录（mihoyo_qr_login 流程）。"""
    from .qr_login import QRLogin

    qr = QRLogin()
    try:
        creds = qr.login(
            timeout=timeout,
            on_qr=on_qr,
            on_status=on_status,
            should_abort=should_abort,
        )
    finally:
        qr.close()

    stoken_cookie = creds.to_stoken_cookie()
    full_cookie = creds.to_full_cookie()
    roles = fetch_game_roles(stoken_cookie)
    if not roles:
        # 部分环境需带 cookie_token 才能列角色
        roles = fetch_game_roles(full_cookie)
    return _session_from_parts(full_cookie, stoken_cookie, creds.stuid, roles, selected_uid)


# ---------- 浏览器手动登录 + 本地回调 ----------

_LOGIN_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <title>米游社登录助手</title>
  <style>
    body { font-family: "Microsoft YaHei", sans-serif; max-width: 720px; margin: 40px auto; padding: 0 16px; line-height: 1.6; }
    code, textarea { font-family: Consolas, monospace; }
    textarea { width: 100%; height: 120px; }
    .box { background: #f6f8fa; border: 1px solid #ddd; padding: 16px; border-radius: 8px; margin: 12px 0; }
    button { padding: 8px 16px; margin-right: 8px; cursor: pointer; }
    a.bookmark { display: inline-block; background: #0969da; color: #fff; padding: 8px 12px; border-radius: 6px; text-decoration: none; }
    .ok { color: #1a7f37; font-weight: bold; }
    .err { color: #cf222e; }
  </style>
</head>
<body>
  <h1>原神抽卡记录 · 手动登录</h1>
  <p>程序<strong>不会</strong>自动填写账号密码。请你在浏览器里完成登录（含验证码/短信）。</p>

  <div class="box">
    <h3>步骤 1：打开通行证并登录</h3>
    <p><a href="__LOGIN_URL__" target="_blank" rel="noopener">打开米哈游通行证登录页</a></p>
  </div>

  <div class="box">
    <h3>步骤 2A（推荐）：登录后用书签一键提交 Cookie</h3>
    <p>将下面按钮拖到书签栏；登录成功后，在通行证页面点击该书签：</p>
    <p><a class="bookmark" href="__BOOKMARKLET__">提交Cookie到本地工具</a></p>
  </div>

  <div class="box">
    <h3>步骤 2B（备用）：粘贴 Cookie</h3>
    <p>在通行证页面按 F12 → Console，执行 <code>copy(document.cookie)</code>，然后粘贴到下方：</p>
    <textarea id="cookie" placeholder="粘贴 cookie 字符串"></textarea>
    <p>
      <button onclick="submitCookie()">提交 Cookie</button>
      <span id="msg"></span>
    </p>
  </div>

  <script>
    function submitCookie(c) {
      var cookie = (c || document.getElementById('cookie').value || '').trim();
      var msg = document.getElementById('msg');
      if (!cookie) { msg.className='err'; msg.textContent='Cookie 为空'; return; }
      fetch('/submit_cookie', {
        method: 'POST',
        headers: {'Content-Type': 'application/x-www-form-urlencoded'},
        body: 'cookie=' + encodeURIComponent(cookie)
      }).then(r => r.json()).then(d => {
        if (d.ok) { msg.className='ok'; msg.textContent='已提交，可关闭此页返回程序。'; }
        else { msg.className='err'; msg.textContent=d.error || '失败'; }
      }).catch(e => { msg.className='err'; msg.textContent=String(e); });
    }
    // 支持 ?cookie= 查询（书签回调）
    (function(){
      var q = new URLSearchParams(location.search);
      var c = q.get('cookie');
      if (c) {
        document.getElementById('cookie').value = c;
        submitCookie(c);
      }
    })();
  </script>
</body>
</html>
"""


class _LoginServerState:
    def __init__(self) -> None:
        self.cookie: Optional[str] = None
        self.error: Optional[str] = None
        self.done = threading.Event()
        self.port = 0


def start_browser_login(
    timeout: float = 300.0,
    on_ready: Callable[[str], None] | None = None,
) -> str:
    """
    打开本机引导页 + 系统浏览器；等待用户提交 Cookie。
    返回 cookie 字符串。超时抛 TimeoutError。
    """
    state = _LoginServerState()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # noqa: N802
            return

        def _send(self, code: int, body: bytes, content_type: str = "text/html; charset=utf-8") -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                # bookmarklet: 在 mihoyo 域读取 cookie 后跳回本地
                bookmark = (
                    "javascript:(function(){var c=document.cookie;"
                    f"location.href='http://127.0.0.1:{state.port}/?cookie='+encodeURIComponent(c);}})();"
                )
                html = (
                    _LOGIN_HTML.replace("__LOGIN_URL__", MIHOYO_LOGIN_URL)
                    .replace("__BOOKMARKLET__", bookmark)
                )
                self._send(200, html.encode("utf-8"))
                return
            if parsed.path == "/favicon.ico":
                self._send(204, b"")
                return
            qs = parse_qs(parsed.query)
            if "cookie" in qs and qs["cookie"]:
                # 书签跳回：展示页并自动提交
                cookie = qs["cookie"][0]
                bookmark = (
                    "javascript:(function(){var c=document.cookie;"
                    f"location.href='http://127.0.0.1:{state.port}/?cookie='+encodeURIComponent(c);}})();"
                )
                html = (
                    _LOGIN_HTML.replace("__LOGIN_URL__", MIHOYO_LOGIN_URL)
                    .replace("__BOOKMARKLET__", bookmark)
                )
                # 注入 cookie 到页面已有逻辑；同时服务端也接收
                state.cookie = cookie
                state.done.set()
                self._send(200, html.encode("utf-8"))
                return
            self._send(404, b"not found")

        def do_POST(self):  # noqa: N802
            if self.path != "/submit_cookie":
                self._send(404, b'{"ok":false}', "application/json")
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8", errors="replace")
            qs = parse_qs(raw)
            cookie = (qs.get("cookie") or [""])[0].strip()
            if not cookie:
                body = b'{"ok":false,"error":"empty cookie"}'
                self._send(400, body, "application/json")
                return
            state.cookie = cookie
            state.done.set()
            self._send(200, b'{"ok":true}', "application/json")

    server = HTTPServer(("127.0.0.1", 0), Handler)
    state.port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    url = f"http://127.0.0.1:{state.port}/"
    if on_ready:
        on_ready(url)
    webbrowser.open(url)

    finished = state.done.wait(timeout)
    server.shutdown()
    thread.join(timeout=2)
    if not finished or not state.cookie:
        raise TimeoutError("等待登录超时：请在浏览器中完成登录并提交 Cookie。")
    return state.cookie


def browser_login_and_save(
    selected_uid: str | None = None,
    timeout: float = 300.0,
    on_ready: Callable[[str], None] | None = None,
) -> Session:
    cookie = start_browser_login(timeout=timeout, on_ready=on_ready)
    return login_with_cookie(cookie, selected_uid=selected_uid)
