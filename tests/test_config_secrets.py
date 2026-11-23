"""Secrets stay out of the committed config: secrets.json / env vars overlay it at load time."""
import json

from git_stock.config import Config


def _cfg(tmp_path, telegram=None):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"strategy": "cleci", "telegram": telegram or {"enabled": True, "token": "", "chat_id": ""}}))
    return p


def test_secrets_file_overlays_config(tmp_path, monkeypatch):
    monkeypatch.delenv("GIT_STOCK_TELEGRAM_TOKEN", raising=False)
    monkeypatch.delenv("GIT_STOCK_TELEGRAM_CHAT_ID", raising=False)
    monkeypatch.delenv("GIT_STOCK_SECRETS", raising=False)
    p = _cfg(tmp_path)
    (tmp_path / "secrets.json").write_text(json.dumps({"telegram": {"token": "1:abc", "chat_id": "42"}}))
    tg = Config.load(p).telegram
    assert tg == {"enabled": True, "token": "1:abc", "chat_id": "42"}


def test_env_wins_over_file(tmp_path, monkeypatch):
    p = _cfg(tmp_path)
    (tmp_path / "secrets.json").write_text(json.dumps({"telegram": {"token": "1:file"}}))
    monkeypatch.setenv("GIT_STOCK_TELEGRAM_TOKEN", "2:env")
    assert Config.load(p).telegram["token"] == "2:env"


def test_no_secrets_is_fine(tmp_path, monkeypatch):
    monkeypatch.delenv("GIT_STOCK_TELEGRAM_TOKEN", raising=False)
    monkeypatch.delenv("GIT_STOCK_SECRETS", raising=False)
    assert Config.load(_cfg(tmp_path)).telegram["token"] == ""
