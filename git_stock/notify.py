"""Telegram helpers via the HTTP API (no python-telegram-bot dependency)."""
from __future__ import annotations

import logging

log = logging.getLogger("git_stock.notify")
_API = "https://api.telegram.org/bot{token}/{method}"


def _post(token: str, method: str, **kw):
    import requests

    files = kw.pop("files", None)
    try:
        r = requests.post(_API.format(token=token, method=method),
                          data=kw if files else None, json=None if files else kw,
                          files=files, timeout=20)
        if r.status_code != 200:
            log.warning("telegram %s -> %s: %s", method, r.status_code, r.text[:200])
        return r.json() if r.headers.get("content-type", "").startswith("application/json") else None
    except Exception as e:  # noqa: BLE001
        log.warning("telegram %s failed: %s", method, e)
        return None


def mono_table(headers: list[str], rows: list[list], widths: list[int] | None = None) -> str:
    """A monospace table wrapped in <pre> for Telegram HTML."""
    cols = list(zip(*([headers] + rows))) if rows else [[h] for h in headers]
    widths = widths or [max(len(str(x)) for x in col) for col in cols]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    line = fmt.format(*[str(h) for h in headers])
    sep = "  ".join("-" * w for w in widths)
    body = "\n".join(fmt.format(*[str(c) for c in r]) for r in rows)
    return f"<pre>{line}\n{sep}\n{body}</pre>"


def send_message(token: str, chat_id: str | int, text: str,
                 buttons: list[list[dict]] | None = None, html: bool = True) -> dict | None:
    if not token or chat_id in (None, "", 0):
        return None
    kw = {"chat_id": chat_id, "text": text[:4090], "disable_web_page_preview": True}
    if html:
        kw["parse_mode"] = "HTML"
    if buttons:
        kw["reply_markup"] = {"inline_keyboard": buttons}
    return _post(token, "sendMessage", **kw)


def send_photo(token: str, chat_id: str | int, png: bytes, caption: str = "") -> dict | None:
    if not token or chat_id in (None, "", 0):
        return None
    return _post(token, "sendPhoto", chat_id=chat_id, caption=caption[:1000],
                 parse_mode="HTML", files={"photo": ("chart.png", png, "image/png")})


def answer_callback(token: str, callback_id: str, text: str = "") -> None:
    _post(token, "answerCallbackQuery", callback_query_id=callback_id, text=text)


def get_updates(token: str, offset: int = 0, timeout: int = 25) -> list[dict]:
    res = _post(token, "getUpdates", offset=offset, timeout=timeout)
    return (res or {}).get("result", []) if isinstance(res, dict) else []


def get_bot_username(token: str) -> str | None:
    """The bot's @username as Telegram reports it (getMe), or None."""
    res = _post(token, "getMe")
    return ((res or {}).get("result") or {}).get("username") if isinstance(res, dict) else None


def set_my_commands(token: str, commands: list[tuple[str, str]]) -> bool:
    """Registers the bot's "/" command menu in Telegram clients (the same feature
    freqtrade's bot uses) - `commands` is [(name, description), ...], names WITHOUT
    the leading slash, each description under Telegram's 256-char limit."""
    payload = [{"command": name, "description": desc[:256]} for name, desc in commands]
    res = _post(token, "setMyCommands", commands=payload)
    return bool(res and res.get("ok"))


# backwards-compatible alias
def send_telegram(token, chat_id, text, html=True):
    return send_message(token, chat_id, text, html=html)
