# fantastic-spoon
Grok bot msft

## XRP Research Loop Supervisor

Use the supervisor to run repeated backtest -> Grok -> Codex iterations with:
- best-so-far tracking,
- accept/reject logic,
- run logs,
- stop conditions.

### Entry Point

Run the supervisor loop:

```bash
python -m crypto_xrp.research_loop.run_supervisor_loop \
	--csv crypto_xrp/data/xrp_4h_clean.csv \
	--supervisor-config-json crypto_xrp/research_loop/supervisor_config_conservative.json \
	--output-json crypto_xrp/research_loop/supervisor_last_run.json
```

### Mode Configs

Conservative mode (safer promotion criteria):
- Config: `crypto_xrp/research_loop/supervisor_config_conservative.json`
- Use when you want tighter drawdown control and fewer but higher-confidence changes.

Exploration mode (broader search):
- Config: `crypto_xrp/research_loop/supervisor_config_exploration.json`
- Use when you want to explore more iterations and accept smaller incremental gains.

Example (exploration mode):

```bash
python -m crypto_xrp.research_loop.run_supervisor_loop \
	--csv crypto_xrp/data/xrp_4h_clean.csv \
	--supervisor-config-json crypto_xrp/research_loop/supervisor_config_exploration.json \
	--output-json crypto_xrp/research_loop/supervisor_last_run_exploration.json
```

### Recommended Starting Thresholds

These are practical initial values for your current XRP loop before introducing Optuna/Hyperopt:

- Conservative:
	- `min_improvement`: `0.02`
	- `max_drawdown_limit`: `0.25`
	- `min_trades`: `12`
	- `dd_penalty_weight`: `1.5`

- Exploration:
	- `min_improvement`: `0.005`
	- `max_drawdown_limit`: `0.35`
	- `min_trades`: `8`
	- `dd_penalty_weight`: `1.0`

These values are meant as first-pass operating points. After 10-20 runs, tighten or relax thresholds based on how often candidate configs are accepted and whether accepted configs improve out-of-sample behavior.

### Artifacts

The supervisor writes:
- Per-iteration run logs (`*.jsonl`) with metrics and candidate decisions.
- Best configuration snapshot (`best_config_*.json`) when a new best is found.
- Optional summary output (`--output-json`) for the final run report.

## Walk-Forward Scoring Baseline

Use this before Optuna/Hyperopt integration so the optimizer targets a validated walk-forward objective.

Run with defaults:

```bash
python -m crypto_xrp.research_loop.run_walk_forward_score \
	--csv crypto_xrp/data/xrp_4h_clean.csv \
	--output-json crypto_xrp/research_loop/walk_forward_score_baseline.json
```

The scorer reports:
- aggregate walk-forward score,
- total/valid/rejected windows,
- mean Sharpe,
- Sharpe stability (std),
- worst window max drawdown,
- average trades.

Optional overrides:
- `--wf-spec-json`: override train/test/step window settings.
- `--wf-score-json`: override penalties and optional hard gates.

## Optuna Proposal Mode

The supervisor can use Optuna instead of Codex as the proposal engine.

Config file:
- `crypto_xrp/research_loop/supervisor_config_optuna.json`

Run Optuna mode:

```bash
python -m crypto_xrp.research_loop.run_supervisor_loop \
	--csv crypto_xrp/data/xrp_4h_clean.csv \
	--supervisor-config-json crypto_xrp/research_loop/supervisor_config_optuna.json \
	--output-json crypto_xrp/research_loop/supervisor_last_run_optuna.json
```

CLI override (without editing config):

```bash
python -m crypto_xrp.research_loop.run_supervisor_loop \
	--proposal-source optuna \
	--supervisor-config-json crypto_xrp/research_loop/supervisor_config_conservative.json
```

Notes:
- Optuna trials optimize the walk-forward score, not a single backtest score.
- Supervisor accept/reject and best-tracking logic stays the same.

## Kalshi Shadow Mode Calibration

Recommended starter environment values for running the Kalshi bot in shadow mode:

```bash
export SHADOW_MODE=true
export MIN_EDGE=0.08
export TRADING_FEES=0.01
export SLIPPAGE=0.002
export MODEL_WEIGHT_BASE=0.40
export MODEL_WEIGHT_MAX=0.60
export MIN_MODEL_CONFIDENCE=0.55
export MAX_POSITION_PCT=0.02
export FRACTIONAL_KELLY=0.50
export SHADOW_LOG_JSONL=kalshi/shadow_calibration_log.jsonl
export SHADOW_LOG_CSV=kalshi/shadow_calibration_log.csv
```

Run the bot (shadow mode will log calibration rows and skip live execution):

```bash
python kalshi/script.py
```

Backfill realized outcomes after markets settle:

```bash
python kalshi/script.py --backfill-outcomes --dry-run
python kalshi/script.py --backfill-outcomes
```

Tuning guidance:
- Keep `SHADOW_MODE=true` until calibration logs show stable positive edge after costs.
- If too few candidates are logged, lower `MIN_MODEL_CONFIDENCE` to `0.50`.
- If too many weak candidates are logged, raise `MIN_EDGE` to `0.10`.
