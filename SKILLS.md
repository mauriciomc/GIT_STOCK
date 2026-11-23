# SKILLS.md — GIT_STOCK

Playbooks for common tasks. Assume `cd ~/Workspace/Finance/stocks/GIT_STOCK` and a venv at
`./.venv`.

---

## Add a new strategy

1. Create `git_stock/strategies/<name>.py` with a class subclassing `Strategy`:
   ```python
   class MyStrat(Strategy):
       timeframe = "1d"
       space = {"lookback": (10, 200, 50, 5), "mode": ["a", "b"]}
       def populate_indicators(self, df):
           df["x"] = ind.ema(df["close"], int(self.p["lookback"]))
           return df
       def entry_signal(self, df): return ind.crossed_above(df["close"], df["x"])
       def exit_signal(self, df):  return ind.crossed_below(df["close"], df["x"])
   ```
2. It auto-registers by class name lowercased (`mystrat`). Check: `python -m git_stock strategies`.
3. Add a test in `tests/` (synthetic OHLCV, assert signal counts + idempotence).
4. `space` entries: `(low, high, default[, step])` numeric, or `[choice, ...]` categorical.

## Run a backtest

```bash
./.venv/bin/python -m git_stock --cache stock_data backtest \
    -s cleci -t 20200101-20240801 -o bt/cleci_2020_2024
```
Prints full stats + top/bottom symbols; `-o` writes `trades.csv` + `equity.csv`.
Risk params come from `config/config.json` (`stop_loss`, `take_profit`, `stake_amount`,
`max_open_trades`, fees); a strategy can override `stoploss_pct` / `take_profit_pct`.

## Run a hyperopt properly (the recipe)

```bash
# 1. broad search per window
for yr in 2021 2022 2023 2024; do
  ./.venv/bin/python -m git_stock --cache stock_data optimise -s cleci \
     -n 500 --method random --objective recipe \
     -t ${yr}0101-${yr}1231 --min-trades 30 -o bt/cleci_${yr}.csv
done
# 2. take the top ~20 rows of each csv, re-run each as a standalone backtest
#    (the optimiser's stats are recomputed per trial but always sanity-check the winner)
# 3. pick params that hold up across windows, not just the single best score
```
Objectives: `sharpe`, `sortino`, `calmar`, `return`, `expectancy`, `recipe`
(= return + 0.5·Sharpe − 2·|maxDD|). `--method optuna` uses TPE if `optuna` is installed.
`--seed` for reproducibility. Grid: `--method grid` (auto-capped to `-n`).

## Reconstruct the "47 % in 2024" rule

```bash
# expanding vs rolling, sweep the window, 2024 only
./.venv/bin/python -m git_stock --cache stock_data optimise -s cleci -n 200 \
    --objective return -t 20240101-20241231 --min-trades 20 -o bt/cleci_2024.csv
# also test the pre-class trend follower
./.venv/bin/python -m git_stock --cache stock_data backtest -s emacross -t 20240101-20241231
git show 10afcd5^:src/trader.py    # the original inline strategy_ema_cross / strategy_ichi
```
Whichever config reproduces ~+47 % over 2024 across the `sectors` universe is the target;
then confirm it out-of-sample on 2024-08 → now once the feed is refreshed.

## Refresh price data

```bash
./.venv/bin/python -m git_stock --cache data_cache data     # all config tickers, ~mins
```
`DataFeed` merges new candles into the existing CSV (dedupe on date). Legacy archive CSVs in
`stock_data/` are read as-is; new pulls write `date,open,high,low,close,adj_close,volume`.

## Daily paper signals

```bash
./.venv/bin/python -m git_stock signals            # refresh data, evaluate, write ledger, Telegram
./.venv/bin/python -m git_stock signals --no-update --quiet   # ledger only, no network/Telegram
./.venv/bin/python -m git_stock report             # show open + closed paper trades
```
Ledger: `databases/<config.database>` (`trades` table, `status` open|closed).

## Seed a starting book from the dashboard (one-off)

```bash
./.venv/bin/python -m git_stock seed --from ../../dashboard/portfolio.yaml --dry-run
./.venv/bin/python -m git_stock seed --from ../../dashboard/portfolio.yaml          # priceable stock/etf/fund
./.venv/bin/python -m git_stock seed --from ../../dashboard/portfolio.yaml --all    # also un-priceable (pension funds)
```
Reads the YAML once, inserts `open` paper trades tagged `strategy='seed'`. Refuses if the
ledger already has open trades (`--force` to reseed). Honours a `yahoo:` field per position.
**No link to the dashboard is created.**

## Deploy to the Raspberry Pi

```bash
rsync -av --exclude .venv --exclude data_cache --exclude '*.db' \
    git_stock pyproject.toml config pi@guri:~/workspace/GIT_STOCK/
ssh pi@guri
  cd ~/workspace/GIT_STOCK
  python3 -m venv .venv && ./.venv/bin/pip install -e ".[charts]"   # piwheels provides arm wheels
  ./.venv/bin/python -m git_stock data
  ./.venv/bin/python -m git_stock backtest -s cleci -t 20230101-20240801   # sanity
  ./.venv/bin/python -m git_stock signals --quiet
```
Then a `systemd` timer for `signals` (daily, after US close) + a `systemd` service for
`telegram-bot`. Rotate the Telegram token before going live. Retire the old
`src/telegram_bot.py` cron/nohup job.

## Telegram bot

```bash
./.venv/bin/python -m git_stock telegram-bot       # long-poll loop
```
Handles `/status`, `/report`, `/chart SYMBOL`, and the "📈 SYM" inline buttons that appear
on signal messages. Set `telegram.send_charts: true` in config to auto-attach charts to the
daily `signals` summary.
