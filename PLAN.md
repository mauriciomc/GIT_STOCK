# GIT_STOCK — plan

_Last updated: 2026-09-07_

## Done (v2 rebuild, this session)

- **Fixed the dead data feed.** `src/exchange.py` (yfinance 0.1.64) stopped working when
  Yahoo retired that endpoint on ~2024-08-01 — every cached CSV froze, so no candle ever
  changed and no strategy signalled for ~13 months. Rebuilt in `git_stock/data.py` on modern
  yfinance (MultiIndex flatten, `auto_adjust=False`, `pd.concat` merge, retries). Verified:
  frozen 2024-08-01 CSVs extend to today.
- **Fixed the Cleci strategy.** The old class kept `highest_low`/`lowest_low` as class
  attributes that `trader.py` never reset between tickers, so only the first symbol in a run
  could fire. `git_stock/strategies/cleci.py` computes the bands per-call, vectorised, with
  `window` (the old unused `config["timewindow"]`) now a real parameter.
- **Clean package** — `config / data / indicators / strategy / strategies / backtest /
  metrics / optimize / signals / seed / notify / charts / telegrambot / cli`. No TA-Lib /
  qtpylib / freqtrade deps. 10 pytest tests, all green. Python 3.9-compatible (for the Pi).
- **Working backtester** — vectorised prep + event loop; fees, stop-loss, take-profit,
  `max_open_trades`, fully-invested equity curve, full stats. ~2 s for 200 symbols × 4.5 y.
  Sanity: Cleci (defaults) 2020-01→2024-08 = **+42 % / CAGR 7.8 % / Sharpe 0.31 / −31 % DD**.
- **Hyperopt** — freqtrade-style: parameter `space` on each strategy, random / grid / optuna
  search, objectives (sharpe/sortino/calmar/return/expectancy/`recipe`), per-window
  `--timerange`. 30-trial demo found `window 360 / entry 1.09 / exit 0.85` → **+78 % / Sharpe
  0.69 / −16 % DD**.
- **Telegram** — `<pre>` monospace tables, per-signal chart PNGs (price + strategy bands +
  entry/exit markers), inline "📈 SYM" buttons, and a `telegram-bot` polling loop for
  `/status` `/report` `/chart SYMBOL`.
- **Seed** — one-off import of a starting paper book from `dashboard/portfolio.yaml`
  (stock/etf/fund only, priceable ones by default). Not linked to the dashboard.

## Next

### 1. Validate Cleci — is there a ~47 % rule? (do first)
- [ ] Backtest `use_rolling` True vs False across `window` on 2020–2024, and specifically on
      **2024** alone, on the recovered archive. Find the config that gives ~47 % in 2024.
- [ ] Also backtest `emacross` and `bbrsi` — the pre-class `strategy_ema_cross` is a 47 %
      candidate (`git show 10afcd5^:src/trader.py`).
- [ ] Run the proper hyperopt: `optimise -n 500`, per-year / per-regime windows, then
      re-backtest the top 20 standalone (per the crypto-side recipe) before trusting a score.
- [ ] Lock a champion strategy + params into `config/config.json`.

### 2. Deploy to the Raspberry Pi (`pi@guri`) — user approved
- [ ] `rsync` `git_stock/`, `pyproject.toml`, `config/` to `~/workspace/GIT_STOCK/`.
- [ ] Fresh venv (`python3 -m venv`, piwheels for arm); `pip install -e .` (+ `[charts]`).
- [ ] `git_stock data` → `git_stock backtest` sanity → `git_stock seed --from <copy>` →
      `git_stock signals --quiet` dry run.
- [ ] `systemd` timer for the daily `signals` run + a service for `telegram-bot`.
- [ ] Retire the old `src/telegram_bot.py` job. **Rotate the Telegram token first.**

### 3. Execution layer (real money — later)
- Cleci emits signals only. To trade the £1 000 IBKR account: `ib_async`, with hard limits
  (max position, max open, daily-loss stop, kill switch). Paper on IBKR first, then a small
  real allocation. Or run signals-only and act manually.

### 4. Nice-to-haves
- [ ] More strategies (ichimoku/heiken from `src` history; `redesing` branch ideas).
- [ ] Walk-forward / out-of-sample split in `optimize.py` to fight overfitting.
- [ ] Per-sector position caps; volatility-scaled position sizing.
- [ ] Share `git_stock/data.py` as the workspace-wide `marketdata` package (also for the dashboard).
- [ ] A tiny results dashboard (equity curve, trades) — or feed `signals.db` into the main dashboard as a read-only panel.
