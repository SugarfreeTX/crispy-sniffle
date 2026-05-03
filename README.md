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
