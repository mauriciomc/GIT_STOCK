# Strategy bake-off — results log

Walk-forward: hyperopt (random search, recipe objective) on **IS 2018-01-01 … 2022-06-30**,
then validate the top-8 configs on **OOS 2022-07-01 … 2026-09-04** (never tuned on).
Universe: 207 daily-bar tickers. Stake £40, max 40 open, £1,000 start.

`recipe = return/100 + 0.5·Sharpe − 2·|maxDD|/100`

---

## Run 1 — ORIGINAL (frictionless, loose param bounds)  ·  2026-09-07

Cost model: 0.1 %/side commission only. No slippage, no spread, same-bar-close fills,
stops always fill exactly at the stop level. Trailing-stop search allowed down to 2 %.

CSV: `bakeoff_ORIG_frictionless_loosebounds.csv` (ichimoku row) + numbers below from
task logs `bgk0r7m2n` / `bn7med580` / `bvlkjpue4`.

| Strategy   | IS ret % | IS Sharpe | OOS ret % | OOS Sharpe | OOS maxDD % | OOS trades | OOS hold | best params |
|------------|---------:|----------:|----------:|-----------:|------------:|-----------:|---------:|-------------|
| ichimoku   | 563 | 11.6 | **467** | 13.2 | −0.7 | 21,024 | 1.1 d | conv14 base28 spanB64 HA exit=cloud_loss sl28 trail1 |
| clecitrail | 249 | 4.7  | **164** | 6.6  | −1.1 | 4,330  | 1.5 d | window120 entry1.16 exit0.77 trail2 sl29 |
| cleci      | —   | —    | **72**  | 3.8  | −0.8 | —      | —     | (recipe 2.48 IS) |
| bbrsi      | —   | —    | **60**  | 0.9  | −12.2| —      | —     | (recipe 0.68 IS) |
| emacross   | —   | —    | **−8**  | −0.4 | −17.5| —      | —     | (recipe 0.20 IS) |

**Verdict:** ichimoku & clecitrail headline numbers are artifacts — 1-bar holds, tens of
thousands of trades, sub-1 % drawdown. A 1–2 % trailing stop on daily bars just scalps one
day's noise; the frictionless fill model can't cost that. Not tradeable as shown.

Known unrealistic assumptions identified:
1. **No commission beyond 0.1 %/side** — real all-in is ~0.3 %/side for global equities.
2. **No bid/ask spread** — 0.1–1 %+ on the illiquid names (Brazil, LSE small-caps, ADRs).
3. **No slippage** — every fill assumed at the exact reference price.
4. **Same-bar-close fills** — enters/exits at the close of the bar the signal is computed on
   (you can only act *after* that close → next bar).
5. **Stops fill exactly at the stop** — a gap-down through the level should fill at the open.
6. **No anti-scalp guard** — the optimiser is rewarded for 1-bar round trips.

---

## Run 2 — frictionless, TIGHTENED bounds  ·  2026-09-07

Same 0.1 %/side-only cost model as Run 1, but `cleci_trail.py`'s trailing-stop floor
raised to 6 % (was 2 %) after Run 1 showed sub-6 % trails just scalp one day's noise.
150 trials/strategy.

CSV: `bakeoff_frictionless.csv`  ·  log: `run_frictionless.log`

| Strategy | IS ret % | IS Sharpe | OOS ret % | OOS Sharpe | OOS maxDD % | OOS trades | OOS hold | best params |
|----------|---------:|----------:|----------:|-----------:|------------:|-----------:|---------:|-------------|
| clecitrail | 67.2 | 0.73 | **69.9** | 1.03 | −14.2 | 1,811 | 15.7 d | window60 entry1.08 exit0.87 trail8 sl29 |
| bbrsi      | 69.0 | 0.69 | **60.1** | 0.93 | −12.2 | 207   | 62.5 d | bb14 rsiWin27 buy38 sell75 |
| cleci      | 77.6 | 0.65 | **50.3** | 0.93 | −8.9  | 178   | 58.8 d | window270 entry1.12 exit0.85 exit=band sl22 |
| ichimoku   | 47.1 | 0.48 | **22.0** | 0.23 | −17.9 | 1,513 | 15.2 d | conv11 base34 spanB84 HA exit=kijun_break sl8 trail18 |
| emacross   | 28.3 | 0.31 | **−8.4** | −0.39| −17.5 | 2,661 | 7.1 d  | fast16 mid30 slow160 |

Raising the trail floor alone (still no realistic costs) already cuts clecitrail's OOS
return from 164 % to 70 % and its trades from 4,330 to 1,811 — most of Run 1's headline
number was the sub-6 % trail scalping noise, not the strategy's edge.

---

## Run 3 — REALISTIC cost model, tightened bounds  ·  2026-09-07

Cost model: **0.3 %/side commission**, 0.05 % slippage per fill, 0.10 % bid/ask spread
(half per side), **next-bar-open fills** for signals, **overnight-gap stops** (a bar that
opens through the stop fills at the open), and an **anti-scalp guard**: any config whose
average hold is < 2 bars is rejected in both IS selection and OOS validation. 150 trials/strategy.

CSV: `bakeoff_realistic.csv`  ·  log: `run_realistic.log`

| Strategy | IS ret % | IS Sharpe | OOS ret % | OOS Sharpe | OOS maxDD % | OOS trades | OOS hold | best params |
|----------|---------:|----------:|----------:|-----------:|------------:|-----------:|---------:|-------------|
| **cleci**  | 19.8 | 0.11 | **58.4** | 0.74 | −13.9 | 239 | 100.3 d | window210 entry1.09 exit0.82 exit=trail tp20 sl23 trail17.5 |
| clecitrail | 22.1 | 0.14 | **55.9** | 0.74 | −17.3 | 243 | 91.8 d  | window210 entry1.22 exit0.79 trail23.5 sl17 |
| bbrsi      | 60.8 | 0.57 | **40.9** | 0.59 | −15.7 | 293 | 53.9 d  | bb30 rsiWin14 buy32 sell64 |
| ichimoku   | 40.6 | 0.33 | **13.9** | 0.09 | −21.6 | 267 | 81.1 d  | conv16 base44 spanB104 HA exit=cloud_loss sl29 |
| emacross   | −44.0| −1.02| **−54.7**| −1.49| −55.1 | 1,701| 7.8 d  | fast20 mid25 slow220 |

Once real costs + the anti-scalp guard are on, both `cleci` and `clecitrail` converge to
similar 90–100-bar swing profiles with near-identical OOS return/Sharpe — but plain
`cleci` edges it on both return and drawdown, with a wide `trail` (17.5 %), not a tight
one. `clecitrail`'s extra turnover (243 vs 239 trades, both ~100 d avg hold — this
hyperopt run picked a *wide* trail, not a scalp config) buys nothing once costs are real.

**Verdict so far: `cleci` ≥ `clecitrail` under realistic costs, and neither strategy's OOS
number from Run 1/2 (frictionless) survives real trading costs.**

---

## Fixed-config check — is `config.json`'s clecitrail actually viable?  ·  2026-09-07

`config/config.json` doesn't hold a *hyperopted* clecitrail — its `strategy_params`
(`window120 entry1.16 exit0.77 trail2 sl29`) are frozen from **Run 1** (frictionless,
loose bounds, before the 6 % trail floor existed). Neither Run 2 nor Run 3 re-tests that
exact config — they search independently and land on different parameters each time.
So a new strategy, `git_stock/strategies/cleci_trail_live.py` (`CleciTrailLive`), pins
those exact fixed params (not hyperopted, `space = {}`) and both cost models were run
against it directly via `scripts/compare_live_config.py`, merged into both CSVs as
`clecitrail_live`.

| Cost model | OOS ret % | OOS Sharpe | OOS maxDD % | OOS trades | OOS hold |
|---|---:|---:|---:|---:|---:|
| frictionless | **163.75** | 6.62 | −1.05 | 4,330 | 1.5 d |
| **realistic** | **−96.10** | −3.53 | **−96.10** | 2,211 | 1.8 d |

The frictionless number reproduces config.json's own `_strategy_note` almost exactly
(164 % / Sharpe 6.6 / −1 % DD / 4,330 trades) — confirms `CleciTrailLive` is faithfully
pinned to the live config. **Under realistic costs the same exact config loses 96 % of
capital** — a 2 % trailing stop on daily bars gets stopped out by ordinary noise on
nearly every trade, and at ~4,300 round trips/OOS-window the 0.3 %/side commission +
spread + slippage alone is a brutal, compounding bleed. This is the clearest evidence yet
that Run 1's headline number was pure cost-model artifact, not edge.

---

## Sharpe/Sortino-targeted hyperopt — `clecitrailsharpe`  ·  2026-09-07

Added a dedicated objective `sharpe_sortino` (average of Sharpe and Sortino —
`git_stock/optimize.py`) and a separate registered strategy `CleciTrailSharpe`
(`git_stock/strategies/cleci_trail_sharpe.py`, same CleciTrail mechanics/space) so it
gets its own tuned config instead of overwriting `clecitrail`'s return-blended one.
300-trial walk-forward hyperopt, same realistic cost model + anti-scalp guard as Run 3,
OOS config selection also on `sharpe_sortino` (script: `scripts/hyperopt_sharpe_sortino.py`).

| Strategy | OOS ret % | OOS Sharpe | OOS Sortino | OOS maxDD % | OOS PF | OOS win % | OOS trades | OOS hold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **clecitrailsharpe** | **60.4** | **0.82** | **1.24** | −14.0 | **2.27** | **56.2** | 283 | 76.5 d |
| cleci | 58.4 | 0.74 | — | −13.9 | 2.06 | 45.6 | 239 | 100.3 d |
| clecitrail | 55.9 | 0.74 | — | −17.3 | 2.19 | 52.3 | 243 | 91.8 d |

Params: `window=160 entry_mult=1.12 exit_mult=0.77 trail=21.5 sl=20.0`.

Optimising for Sharpe+Sortino instead of the return-weighted "recipe" objective found a
config that's **better on every axis** — higher OOS return *and* Sharpe *and* Sortino
*and* profit factor *and* win rate, similar drawdown to `cleci` — by trading a bit more
often (283 vs 239 trades) at a slightly shorter average hold (76 d vs 100 d) with a wider
trail (21.5% vs `cleci`'s 17.5%). (Sortino is only populated for this run — added it to
the bake-off's row helper here; the earlier rows would need a quick rerun to backfill it,
not done yet.)

**New leading candidate: `clecitrailsharpe`, not `cleci`.** Same caveats as before apply
(still need to reconcile against the ~47%-in-2024 target) before locking a champion into
`config/config.json`.

### Dirty (realistic) vs clean (frictionless) — same algorithm, same search, two cost models

Re-ran the identical 300-trial `sharpe_sortino` walk-forward hyperopt with the cost model
stripped to frictionless (0.1%/side only, no slippage/spread/next-open-fill/gap-stop
penalties, no anti-scalp guard) — `scripts/hyperopt_sharpe_sortino.py --frictionless`.

| Setup | OOS ret % | OOS Sharpe | OOS Sortino | OOS maxDD % | OOS trades | OOS hold | params |
|---|---:|---:|---:|---:|---:|---:|---|
| **dirty** (realistic) | 60.4 | 0.82 | 1.24 | −14.0 | 283 | 76.5 d | window160 entry1.12 exit0.77 trail21.5 sl20 |
| **clean** (frictionless) | 69.9 | 1.03 | 1.65 | −14.2 | 1,811 | 15.7 d | window60 entry1.08 exit0.87 trail8 sl29 |

Two things stand out:
1. **Costs cost about 9.5 percentage points of OOS return** and ~0.2-0.4 off both Sharpe
   and Sortino for this algorithm family — a real but not catastrophic drag (nothing like
   `clecitrail_live`'s −96%, because these params were searched *with* the anti-scalp
   guard, so they don't hinge on a scalping edge that costs erase).
2. **The optimiser picks a structurally different strategy once costs are removed** — clean
   converges to a faster, ~15-day-hold, 1,811-trade config; dirty converges to a slower,
   ~76-day-hold, 283-trade config. This isn't the same algorithm measured two ways so much
   as two different equilibria for the same search space — costs don't just shave the
   clean number, they change *what the optimal strategy even looks like*. The realistic
   number (60.4%/Sharpe 0.82) is the one that means anything for actual trading.

### Reality check — passive buy & hold of the same universe, same window

Equal-weight buy & hold of all 207 tickers, no strategy, no costs, 2022-07-01→2026-09-04
(4.18y): **mean total return 86.7% (CAGR ~16%), median 47.6%**, 75.8% of tickers
profitable. The mean is dominated by a handful of AI-boom outliers (NVDA alone: +1,486%);
the median is the more representative "typical stock" outcome.

So `clecitrailsharpe`'s dirty-cost 60.4% **beats a typical individual stock** (median
47.6%) but **loses to an equal-weight-everything passive benchmark on average** (86.7%,
skewed hard by a few mega-winners) — with none of the strategy's complexity, signal risk,
or trading costs. That's the honest context for "60% in 4 years feels bad": on a
risk-adjusted / drawdown-controlled basis the strategy is doing real work (Sharpe 0.82 vs.
an undiversified, unmanaged buy-and-hold that would have also eaten NVDA's -60%+
drawdowns along the way with no exit), but on raw return it is not obviously beating
"just buy everything and don't sell."

**Caveat raised (correctly):** the 207-ticker equal-weight benchmark isn't capital-fair —
it assumes you can hold all 207 names at once, when the strategies themselves are capped
at `max_open_trades=40`. See the next section for capital-matched benchmarks.

### Capital-matched benchmarks — respecting the 40-position limit

`scripts/benchmark_rotation.py`. Both hold **≤40 positions**, same as the strategies:

**A — Random-40 buy & hold, no rebalance** (300 draws, same OOS window, one-side entry
cost only): **mean +87.0% / median +83.8%** (p10 +60.9%, p90 +117.0%); avg per-draw
Sharpe **0.88**, Sortino 1.31, maxDD **−18.1%** (worst draw −23.6%). Diversifying across
just 40 random names already captures nearly all of the full-207 population's mean return
(law of large numbers washes out single-stock skew) — ticker-picking skill isn't doing
much work here, this bull market was just broad.

**B — Naive monthly top-40 momentum rotation** (no real signal — just chase the 40
highest trailing-6-month-return names every month, realistic 0.4%/side cost, 51
rebalances): **+25.6% / CAGR 5.5% / Sharpe 0.21 / Sortino 0.28 / maxDD −22.7%.** This is
the literal "keep rebalancing to find better positions" baseline with zero edge — and
it's the **worst performer of everything tested**, worse than doing nothing (random-40
hold) and worse than the signal-driven strategies. Monthly turnover into whatever just
went up gets whipsawed and bleeds to transaction costs; blind rotation is not free money.

| Approach | Return | Sharpe | Sortino | maxDD |
|---|---:|---:|---:|---:|
| Random-40 buy & hold (no skill, no rotation) | +87.0% (mean) | 0.88 | 1.31 | −18.1% |
| **clecitrailsharpe** (dirty, signal-driven) | +60.4% | 0.82 | 1.24 | −14.0% |
| cleci (dirty, signal-driven) | +58.4% | 0.74 | — | −13.9% |
| Naive monthly momentum rotation (no skill) | +25.6% | 0.21 | 0.28 | −22.7% |

**Verdict:** the signal-driven strategies clearly beat naive rotation (2.4× the return,
~4× the Sharpe) — so "constantly rebalancing to find better positions" only pays off if
the rebalancing logic has real edge; blind momentum-chasing is actively harmful once
costs are real. But **static random diversification still edges out the strategies** on
both return and Sharpe in this specific (strongly bullish) window, with a deeper average
drawdown as the price of not managing risk. The honest reading: `cleci`/`clecitrailsharpe`
are buying real downside control (best maxDD of the four) at some cost in upside versus
just holding a diversified basket through a bull run — whether that trade-off is worth it
depends on how this window's regime compares to what you expect going forward, and it's
exactly the kind of edge that's hardest to see in one 4-year backtest window.

---

## Portfolio rotation — close weak positions to fund stronger new signals  ·  2026-09-07

User's objection to naive momentum rotation (above): they don't want *blind* rotation —
they want the engine to close positions it doesn't see much further profit in, in favour
of new candidates with more room to run, using the strategy's own logic. Built this as an
opt-in backtest capability rather than a separate naive baseline:

- `Strategy.score(df)` (`git_stock/strategy.py`) — optional per-bar, cross-sectional
  "how much upside is left" estimate. Default: all-NaN (rotation disabled). Cleci family
  (`cleci.py`) overrides it as `band_high/close - 1` — room left to the recent
  high-water mark; a name just off its trailing low scores high (lots of mean-reversion
  room), a name that's already run up near its band scores near zero.
- `BacktestConfig.rotate` (+ `rotate_margin_pct`, `min_hold_bars`) — when at
  `max_open_trades` and a new entry fires, force-close the **weakest-scoring eligible
  open position** (past its cooldown) if the candidate's score clears it by
  `rotate_margin_pct`; exit reason `"rotated_out"`. `git_stock/backtest.py`.
- Also fixed a latent bug while wiring this up: survivor positions (still open at the
  end of the backtest window) were always recorded with `bars_held=0` regardless of true
  duration (an index-scope bug — `i - p["bar"]` used a bar index from the wrong symbol's
  frame). Now uses a universal day counter. This **only affects the `avg_bars_held`
  stat** (already-reported returns/Sharpe/maxDD/trade counts are unaffected — they come
  from `pnl`, which was always computed correctly); e.g. `clecitrailsharpe`'s OOS avg
  hold corrected from 76.5b to 104.4b. Every table above pre-dates this fix.

**Ablation** (`scripts/rotation_ablation.py`), `clecitrailsharpe`'s own OOS params, dirty
costs:

| Setup | Return | Sharpe | Sortino | maxDD | trades | avg hold | rotations |
|---|---:|---:|---:|---:|---:|---:|---:|
| cap=40 (natural — never binds), rotation OFF/ON | 60.4% | 0.81 | 1.23 | −14.0% | 283 | 104.4d | 0 |
| cap=15 (forced binding), rotation OFF | 29.9% | 0.59 | 0.84 | −9.6% | 113 | 139.7d | 0 |
| cap=15, rotation ON (1.0pp margin, 5d cooldown) | 31.3% | **0.38** | 0.62 | **−21.8%** | 1,093 | 14.3d | 719 |
| cap=15, rotation ON (0.5pp margin, 3d cooldown) | 36.6% | 0.45 | 0.74 | −21.9% | 1,132 | 13.8d | 780 |

First finding: at the strategy's own natural capacity (`max_open_trades=40`), rotation
never engages — peak concurrent positions over the whole OOS run is **33**, average
**~19**. Capital sits idle roughly half the time; the 40-slot cap was never actually the
constraint. Second finding: once capacity genuinely binds, hand-picked rotation
margins/cooldowns raise raw return a little but **hurt risk-adjusted performance** —
churn jumps from ~140-day to ~14-day average holds (700+ forced swaps), each paying the
0.4%/side realistic cost, and drawdown nearly doubles. Mechanically identical lesson to
the naive-momentum benchmark: an unproven "smart-looking" trigger with an untuned
threshold can still lose to doing nothing. `rotate_margin_pct` / `min_hold_bars` need to
be searched properly, not guessed — see next section.

---

## Joint strategy + portfolio hyperopt — stake, position count, rotation  ·  in progress

Everything above searched only the strategy's own 5 knobs (`window`, `entry_mult`,
`exit_mult`, `trail`, `sl`) with `stake_amount`/`max_open_trades` fixed at config.json's
£40 / 40. `git_stock/optimize.py`'s new `optimise_portfolio()` + `PORTFOLIO_SPACE` now
also search `stake_amount` (£10–200), `max_open_trades` (5–60), `rotate` (on/off),
`rotate_margin_pct` (0.5–10pp), `min_hold_bars` (0–30d) jointly with the strategy space —
`scripts/hyperopt_portfolio.py`, 500-trial walk-forward, same realistic cost model,
objective `sharpe_sortino`, results merge into `bakeoff_realistic.csv` as
`clecitrailportfolio`. Result pending — will log here once it finishes.

---

## Realised vs. unrealised P&L — a hidden problem in every result above  ·  2026-09-07

User's question: "make sure to close all positions at the end, otherwise we won't know
the real advantage" + "is there a better way to handle realised vs. unrealised % and
still let hyperopt find good Sharpe/Sortino/profit?"

**The engine already force-closes every open position at the last available price**
(`backtest.py`'s "close survivors" step existed before this session) — `total_return_pct`
/ Sharpe / Sortino / maxDD are computed off the equity curve, which marks every open
position to market every single day, so nothing was ever hidden from those numbers.

**But** checking `clecitrailsharpe`'s own winning OOS config directly exposed a real
problem: **40.6% of its total P&L is `unrealised`** — 33 of its positions were simply
still open when the data ran out (2026-09-04) and got closed only because the test
window ended, not because the strategy's own logic chose to exit (`exit_reason="end"`).
Its apparent Sharpe 0.82 is meaningfully inflated by paper gains that were never locked
in and could have gone either way with one more week of data. Since every strategy in
this bake-off shares the same OOS end date, this risk isn't unique to one config — it's
a structural property of testing right up to "today."

**Fix, implemented:**
- `metrics.stats()` now reports `realised_pct` / `unrealised_pct` (split by
  `exit_reason == "end"` vs. everything else — genuine stoploss/trailing-stop/
  take-profit/signal exits) and `unrealised_frac` (0–1). `realised_pct + unrealised_pct
  == total_return_pct` exactly, always — nothing is discarded, just made visible.
- All three walk-forward scripts (`bakeoff.py`, `hyperopt_sharpe_sortino.py`,
  `hyperopt_portfolio.py`) now reject an OOS "winner" whose `unrealised_frac > 0.35`
  during top-N validation — same mechanism as the existing anti-scalp (`min_hold`) /
  `min_trades` guards, so hyperopt still searches freely for Sharpe/Sortino/return, it
  just can't win by getting lucky on the snapshot date. Each also now saves its IS
  hyperopt table to `bt/is_table_*.csv` so a guard-only re-check never needs a full
  re-search again.
- Confirms the joint portfolio-search winner from the previous (unguarded) run was
  actually the more trustworthy one despite its lower headline Sharpe: stake=£125/
  max_open=31/rotate=on scored Sharpe 0.66 but only **23.8% unrealised** (10 still open)
  — vs. `clecitrailsharpe`'s Sharpe 0.82 built on 40.6% unrealised (33 still open).
  Comparing raw Sharpe across configs without this context is misleading.

**Re-running both `clecitrailsharpe` and `clecitrailportfolio` now with the guard
actually enforced during OOS selection** (rather than checked after the fact) — result
pending, will replace the entries logged above once done.

**Update (then reverted — see below):** the guarded 300-trial `clecitrailsharpe` rerun
came back "no OOS-valid config" (none of its top-8 IS candidates cleared 35% unrealised
OOS). Reused the saved `bt/is_table_clecitrailsharpe.csv` and re-validated its top 100
IS-ranked configs with the guard via `scripts/rescan_is_table.py` — 50 of 100 passed, and
the top passer (`window=350, entry_mult=1.17, exit_mult=0.7, trail=15.0, sl=29.0`) scored
OOS return 69.95% / Sharpe 0.92 / Sortino 1.41 — better than the original ungated
`window=160` config (60.4% / 0.82 / 1.24) on every axis, at a lower 27.8% unrealised
(vs. 40.6%). At the time this looked like the guard had done its job.

**Then the 2000-2006 cross-check (next section) proved it hadn't.** Tested both
`window=160` (ungated) and `window=350` (guarded "winner") cold on 2000-01-01..2006-12-31
— data neither had ever seen, no re-hyperopting:

| | 2018-2026 OOS return | 2018-2026 Sharpe | **2000-2006 return** | **2000-2006 Sharpe** |
|---|---:|---:|---:|---:|
| `window=160` (ungated, "worse", 40.6% unrealised) | 60.4% | 0.82 | **85.5%** | **0.56** |
| `window=350` (guarded "winner", 27.8% unrealised) | 69.95% | 0.92 | **26.0%** | **0.10** |

The config the guard *rejected* generalises far better to an unseen regime than the one
it picked. **The guard didn't select for robustness — it selected for "how much happened
to be closed on this one specific snapshot date," which is a different and much weaker
property**, and here it actively preferred the more curve-fit config. User's objection was
correct: unrealised P&L isn't a defect to gate on. It's already handled correctly —
`metrics.stats()`'s equity curve marks every open position to market every day, so
Sharpe/Sortino/maxDD already reflect the true daily risk of holding them; a strategy with
a long average hold (`cleci` family: 60-140 days) will *naturally* have a lot open at any
single arbitrary cutoff, and that's a property of patience, not an error. Penalising it
just biases the search toward faster turnover — which this session has separately shown
tends to bleed more on realistic trading costs.

**Reverted:** `MAX_UNREALISED_FRAC` rejection removed from `bakeoff.py`,
`hyperopt_sharpe_sortino.py`, `hyperopt_portfolio.py`. `realised_pct` / `unrealised_pct` /
`unrealised_frac` stay in `metrics.stats()` and every results table — genuinely useful
context for reading a result — they're just no longer used to disqualify a candidate.
**`clecitrailsharpe`'s champion reverts to `window=160`** (the original, ungated pick);
`window=350` is set aside, having only "won" a flawed filter and then failed
2000-2006 outright.

---

## Live config — ACTION NEEDED

`config/config.json` currently has `"strategy": "clecitrail"` with the **catastrophic**
Run-1 params above (verified −96 % OOS under realistic costs — see previous section).
**This must not be deployed / run live as-is.** Nothing has been pushed to the Pi yet
(GIT_STOCK's deploy step is still unchecked), so no live capital is at risk, but the
config file itself is wrong and should be updated before that step is taken.

Recommendation from Runs 2–3: promote **`cleci`** (Run 3 realistic winner: `window210
entry1.09 exit0.82 exit_mode=trail tp20 sl23 trail17.5` → OOS +58 % / Sharpe 0.74 /
−13.9 % maxDD / 100 d avg hold), not `clecitrail` — the trail variant's extra turnover
doesn't pay for itself once costs are real, and its Run-1 config as currently saved is
actively dangerous. Still pending: reconcile against the ~47 % 2024 target (GIT_STOCK
PLAN.md §"Validate Cleci") before locking a champion into config.json.
