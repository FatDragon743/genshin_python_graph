"""
米游社扫码登录 —— passport API（UIGF / PizzaHelper）。

优先使用 app 接口（确认后 body 内返回 stoken）；
web 接口往往只有 cookie_token_v2 / ltoken_v2，不够 genAuthKey。
"""

from __future__ import annotations

import hashlib
import random
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable

import requests

CREATE_QR_WEB = "https://passport-api.miyoushe.com/account/ma-cn-passport/web/createQRLogin"
QUERY_QR_WEB = "https://passport-api.miyoushe.com/account/ma-cn-passport/web/queryQRLoginStatus"
CREATE_QR_APP = "https://passport-api.mihoyo.com/account/ma-cn-passport/app/createQRLogin"
QUERY_QR_APP = "https://passport-api.mihoyo.com/account/ma-cn-passport/app/queryQRLoginStatus"

GET_LTOKEN_URL = "https://passport-api.mihoyo.com/account/auth/api/getLTokenBySToken"
GET_COOKIE_TOKEN_URL = (
    "https://passport-api.mihoyo.com/account/auth/api/getCookieAccountInfoBySToken"
)

APP_ID_WEB = "bll8iq97cem8"
APP_ID_APP = "ddxf5dufpuyo"
DS_SALT_X4 = "xV8v4Qu54lUKrEYFZkJhB8cuOh9Asafs"
BBS_VERSION = "2.102.1"
BBS_UA = f"Mozilla/5.0 (Linux; Android 12) Mobile miHoYoBBS/{BBS_VERSION}"
HYP_UA = "HYPContainer/1.3.3.182"


@dataclass
class QRCredentials:
    stuid: str
    stoken: str
    mid: str
    ltoken: str = ""
    cookie_token: str = ""

    def to_stoken_cookie(self) -> str:
        parts = [f"stuid={self.stuid}", f"stoken={self.stoken}", f"mid={self.mid}"]
        if self.ltoken:
            parts += [f"ltoken={self.ltoken}", f"ltuid={self.stuid}"]
        if self.cookie_token:
            parts += [f"cookie_token={self.cookie_token}", f"account_id={self.stuid}"]
        return "; ".join(parts)

    def to_full_cookie(self) -> str:
        return (
            f"account_id={self.stuid}; account_id_v2={self.stuid}; "
            f"account_mid_v2={self.mid}; stuid={self.stuid}; "
            f"stoken={self.stoken}; mid={self.mid}; "
            f"ltmid_v2={self.mid}; ltuid={self.stuid}; ltuid_v2={self.stuid}"
            + (f"; ltoken={self.ltoken}" if self.ltoken else "")
            + (f"; cookie_token={self.cookie_token}" if self.cookie_token else "")
        )


def _device_id() -> str:
    return str(uuid.uuid4()).lower().replace("-", "")


def _device_fp() -> str:
    return "".join(random.choices("0123456789abcdef", k=13))


def _ds_x4(query: str = "", body: str = "") -> str:
    t = str(int(time.time()))
    r = str(random.randint(100000, 200000))
    h = hashlib.md5(f"salt={DS_SALT_X4}&t={t}&r={r}&b={body}&q={query}".encode()).hexdigest()
    return f"{t},{r},{h}"


def make_qr_png(url: str, path: str) -> str:
    import qrcode

    img = qrcode.make(url)
    img.save(path)
    return path


def _parse_set_cookie(resp: requests.Response) -> dict[str, str]:
    out: dict[str, str] = {}
    for c in resp.cookies:
        out[c.name] = c.value
    getlist = getattr(getattr(resp.raw, "headers", None), "getlist", None)
    if callable(getlist):
        for item in getlist("Set-Cookie") or []:
            part = item.split(";", 1)[0]
            if "=" in part:
                k, v = part.split("=", 1)
                out[k.strip()] = v.strip()
    # session jar
    return out


def _pick(d: dict[str, str], *keys: str) -> str:
    for k in keys:
        v = d.get(k)
        if v:
            return str(v)
    return ""


def _extract_creds(data: dict[str, Any], cookies: dict[str, str], session: requests.Session) -> tuple[str, str, str, str, str]:
    """返回 stuid, stoken, mid, ltoken, cookie_token。"""
    jar = {c.name: c.value for c in session.cookies}
    merged = {**jar, **cookies}

    user = data.get("user_info") if isinstance(data.get("user_info"), dict) else {}
    stuid = str(user.get("aid") or user.get("account_id") or "") or _pick(
        merged, "stuid", "account_id", "account_id_v2", "ltuid", "ltuid_v2"
    )
    mid = str(user.get("mid") or "") or _pick(merged, "mid", "account_mid_v2", "ltmid_v2")

    stoken = ""
    token_obj = data.get("token")
    if isinstance(token_obj, dict) and token_obj.get("token"):
        # PizzaHelper: token_type 1 = stoken
        stoken = str(token_obj.get("token"))
    for item in data.get("tokens") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").lower()
        tok = str(item.get("token") or "")
        ttype = item.get("token_type")
        if (name == "stoken" or ttype in (1, "1")) and tok:
            stoken = tok
    stoken = stoken or _pick(merged, "stoken", "stoken_v2")

    ltoken = _pick(merged, "ltoken", "ltoken_v2")
    cookie_token = _pick(merged, "cookie_token", "cookie_token_v2")
    return stuid, stoken, mid, ltoken, cookie_token


class QRLogin:
    def __init__(self) -> None:
        self.device_id = _device_id()
        self.device_fp = _device_fp()
        self.session = requests.Session()
        self._mode = "app"

    def _headers_web(self) -> dict[str, str]:
        return {
            "x-rpc-app_id": APP_ID_WEB,
            "x-rpc-client_type": "4",
            "x-rpc-device_id": self.device_id,
            "x-rpc-device_fp": self.device_fp,
            "user-agent": BBS_UA,
            "content-type": "application/json",
            "referer": "https://user.miyoushe.com/",
        }

    def _headers_app(self) -> dict[str, str]:
        return {
            "x-rpc-app_id": APP_ID_APP,
            "x-rpc-client_type": "3",
            "x-rpc-device_id": self.device_id,
            "x-rpc-device_fp": self.device_fp,
            "user-agent": HYP_UA,
            "content-type": "application/json",
        }

    def fetch_qrcode(self, prefer: str = "app") -> dict[str, str]:
        order = (
            (("app", CREATE_QR_APP, self._headers_app()), ("web", CREATE_QR_WEB, self._headers_web()))
            if prefer == "app"
            else (("web", CREATE_QR_WEB, self._headers_web()), ("app", CREATE_QR_APP, self._headers_app()))
        )
        errors = []
        for mode, url, headers in order:
            try:
                resp = self.session.post(url, json={}, headers=headers, timeout=30)
                data = resp.json()
            except Exception as e:
                errors.append(f"{mode}:{e}")
                continue
            if data.get("retcode") != 0:
                errors.append(f"{mode}:{data.get('message') or data}")
                continue
            payload = data.get("data") or {}
            qr_url = payload.get("url") or ""
            ticket = payload.get("ticket") or ""
            if not ticket and "tk=" in qr_url:
                ticket = qr_url.split("tk=")[1].split("&")[0].split("#")[0]
            if qr_url and ticket:
                self._mode = mode
                return {"url": qr_url, "ticket": ticket}
        raise RuntimeError("生成二维码失败: " + "; ".join(errors))

    def query_qrcode(self, ticket: str) -> tuple[dict[str, Any], requests.Response]:
        if self._mode == "web":
            url, headers = QUERY_QR_WEB, self._headers_web()
        else:
            url, headers = QUERY_QR_APP, self._headers_app()
        resp = self.session.post(url, json={"ticket": ticket}, headers=headers, timeout=30)
        try:
            body = resp.json()
        except Exception:
            body = {"retcode": -1, "message": "响应非 JSON", "data": None}
        return body, resp

    def wait_for_scan(
        self,
        ticket: str,
        timeout: int = 180,
        on_status: Callable[[str], None] | None = None,
        should_abort: Callable[[], bool] | None = None,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        last = ""
        start = time.time()
        while time.time() - start < timeout:
            if should_abort and should_abort():
                raise TimeoutError("用户取消扫码登录")
            body, resp = self.query_qrcode(ticket)
            ret = body.get("retcode")
            if ret in (-3501, -3505):
                raise RuntimeError(body.get("message") or f"扫码失败 retcode={ret}")
            data = body.get("data") or {}
            status = data.get("status") or data.get("stat") or ""
            if status != last:
                msg = {
                    "Created": "等待扫码…（米游社 APP → 我的 → 扫一扫）",
                    "Init": "等待扫码…（米游社 APP → 我的 → 扫一扫）",
                    "Scanned": "已扫码，请在 APP 上确认登录",
                    "Confirmed": "已确认，正在读取凭证…",
                }.get(status, f"状态: {status or body.get('message') or '未知'}")
                if on_status:
                    on_status(msg)
                last = status
            if status == "Confirmed":
                return data, _parse_set_cookie(resp)
            if ret not in (0, "0") and not data:
                raise RuntimeError(body.get("message") or f"查询失败: {body}")
            time.sleep(2)
        raise TimeoutError("扫码超时，请重试")

    def _exchange_headers(self, stoken: str, mid: str, query_str: str) -> dict[str, str]:
        return {
            "user-agent": BBS_UA,
            "x-rpc-app_version": BBS_VERSION,
            "x-rpc-client_type": "5",
            "x-requested-with": "com.mihoyo.hyperion",
            "referer": "https://webstatic.mihoyo.com",
            "x-rpc-device_id": self.device_id,
            "x-rpc-device_fp": self.device_fp,
            "ds": _ds_x4(query=query_str),
            "cookie": f"mid={mid};stoken={stoken}",
        }

    def get_ltoken(self, stoken: str, mid: str) -> str:
        params = {"stoken": stoken}
        headers = self._exchange_headers(stoken, mid, f"stoken={stoken}")
        data = self.session.get(GET_LTOKEN_URL, headers=headers, params=params, timeout=30).json()
        if data.get("retcode") == 0:
            return str((data.get("data") or {}).get("ltoken") or "")
        return ""

    def get_cookie_token(self, stoken: str, mid: str) -> str:
        params = {"stoken": stoken}
        headers = self._exchange_headers(stoken, mid, f"stoken={stoken}")
        data = self.session.get(GET_COOKIE_TOKEN_URL, headers=headers, params=params, timeout=30).json()
        if data.get("retcode") == 0:
            return str((data.get("data") or {}).get("cookie_token") or "")
        return ""

    def login(
        self,
        timeout: int = 180,
        on_qr: Callable[[str], None] | None = None,
        on_status: Callable[[str], None] | None = None,
        should_abort: Callable[[], bool] | None = None,
    ) -> QRCredentials:
        # app 优先：确认后可直接拿到 stoken（web 通常没有）
        info = self.fetch_qrcode(prefer="app")
        if on_qr:
            on_qr(info["url"])
        if on_status:
            on_status(f"二维码已生成（{self._mode}），请用米游社 APP 扫码")

        data, cookies = self.wait_for_scan(
            info["ticket"],
            timeout=timeout,
            on_status=on_status,
            should_abort=should_abort,
        )
        stuid, stoken, mid, ltoken, cookie_token = _extract_creds(data, cookies, self.session)

        if not stoken:
            # web 扫码常只有 cookie_token —— 自动再试 app（若刚才是 web）
            if self._mode == "web":
                if on_status:
                    on_status("web 凭证无 stoken，改用 app 扫码…")
                info = self.fetch_qrcode(prefer="app")
                if on_qr:
                    on_qr(info["url"])
                data, cookies = self.wait_for_scan(
                    info["ticket"],
                    timeout=timeout,
                    on_status=on_status,
                    should_abort=should_abort,
                )
                stuid, stoken, mid, ltoken, cookie_token = _extract_creds(data, cookies, self.session)

        if not stoken:
            raise RuntimeError(
                "扫码成功但未拿到 stoken（仅有 cookie/ltoken 无法拉抽卡）。"
                f" keys={sorted(set(cookies)|{c.name for c in self.session.cookies})}"
            )
        if not mid:
            raise RuntimeError("扫码成功但未拿到 mid")
        if not stuid:
            raise RuntimeError("扫码成功但未拿到账号 id")

        if on_status:
            on_status("正在补全 ltoken / cookie_token…")
        if not ltoken:
            ltoken = self.get_ltoken(stoken, mid)
        if not cookie_token:
            cookie_token = self.get_cookie_token(stoken, mid)

        return QRCredentials(
            stuid=str(stuid),
            stoken=stoken,
            mid=mid,
            ltoken=ltoken,
            cookie_token=cookie_token,
        )

    def close(self) -> None:
        self.session.close()
