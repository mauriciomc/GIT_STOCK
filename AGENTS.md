# AGENTS.md — GIT_STOCK

Instructions for AI agents working in this repo. Also see `../../AGENTS.md`.

## What this is

A small toolkit that pulls daily stock data from Yahoo Finance, runs rule strategies over
it, and either **backtests / optimises** them or emits **live paper signals** (SQLite +
Telegram). No broker connection. Runs on a Raspberry Pi (`pi@guri`) as the daily job; the
dev machine is used for backtests / hyperopt.

## Layout

```
git_stock/                package (v2 — use this)
  config.py               Config dataclass, loads config/*.json
  data.py                 DataFeed — yfinance download + CSV cache (fixed for modern yfinance)
  indicators.py           crossed_above/below, ema, rsi, bbands, linreg_angle, ... (no TA-Lib/qtpylib)
  strategy.py             Strategy ABC: populate_indicators / entry_signal / exit_signal + a param `space`
  strategies/             cleci.py (the flagship, bug-fixed), ema_cross.py, bbrsi.py + registry
  backtest.py             vectorised-prep event-loop backtester -> BacktestResult(trades, equity, stats)
  metrics.py              return / CAGR / Sharpe / Sortino / Calmar / drawdown / win-rate / PF ...
  optimize.py             random / grid / optuna hyperopt over a strategy's space; objectives
  signals.py              SignalRunner — daily paper ledger in SQLite
  seed.py                 one-off import of a starting book from a dashboard portfolio.yaml
  notify.py               Telegram HTTP API — messages, <pre> tables, photos, inline buttons
  charts.py               matplotlib PNG: price + strategy bands + entry/exit markers
  telegrambot.py          long-poll bot: /status /report /chart SYMBOL + "📈 SYM" chart buttons
  cli.py                  `python -m git_stock <cmd>`
tests/                    pytest, synthetic data, no network
src/                      LEGACY (pre-2024 trader.py / exchange.py / strategies/*). Reference only — do not extend.
config/                   config.json (live), config_mdc_1h.json, config_test.json
```

## Rules

- **Do not extend `src/`** — it's the old broken version (dead yfinance 0.1.64, class-state
  leak). All new work goes in `git_stock/`.
- **Keep it Python 3.9-compatible** (the Pi). Use `from __future__ import annotations`; no
  runtime `X | Y` unions, no `match`.
- No TA-Lib, no qtpylib, no `freqtrade` dependency — `indicators.py` covers what's needed.
- A live Telegram token sits in `config/config.json`. Don't commit it; flag for rotation.
- The bot is **not linked to the dashboard**. `seed` may read a `portfolio.yaml` file; that's all.

## Commands

```bash
./.venv/bin/python -m git_stock strategies
./.venv/bin/python -m git_stock --cache stock_data data        # refresh price cache
./.venv/bin/python -m git_stock --cache stock_data backtest -s cleci -t 20200101-20240801
./.venv/bin/python -m git_stock --cache stock_data optimise -s cleci -n 300 --objective recipe
./.venv/bin/python -m git_stock seed --from ../../dashboard/portfolio.yaml --dry-run
./.venv/bin/python -m git_stock signals --quiet                # daily paper run (no Telegram)
./.venv/bin/python -m git_stock telegram-bot                   # command/chart bot loop
./.venv/bin/python -m pytest -q
```

`--cache stock_data` uses the recovered Pi archive (legacy `Datetime,Open,…,Adj Close` CSVs
— `DataFeed` reads both that and the new `date,open,…` format). Default cache dir is
`data_cache/`.

## Config schema (`config/config.json`)

`strategy`, `timeframe`, `daterange` (bars), `timewindow` (→ Cleci `window`),
`initial_balance`, `stake_amount`, `max_open_trades`, `open_fee`/`close_fee`,
`stop_loss`/`take_profit` (percent), `database`, `sectors` (`{sector: [tickers]}`),
`telegram` (`enabled`, `token`, `chat_id`, optional `send_charts`).

## Gotchas

- Modern yfinance returns a column MultiIndex and defaults `auto_adjust=True`. `data.py`
  handles both (uses `auto_adjust=False` to keep `Adj Close`, flattens the MultiIndex).
- LSE tickers quote in pence — `seed._quote` divides GBp by 100; `data.py` keeps raw.
- The backtester enters/exits at the **signal candle's close**; stop-loss fills at the stop
  if the low pierces it. Documented in `backtest.py`.
- Cleci's old bug: `highest_low`/`lowest_low` were class attributes never reset between
  tickers, so only the first symbol could signal. Fixed — `populate_indicators` is now
  per-call and vectorised; `tests/` guards this.
