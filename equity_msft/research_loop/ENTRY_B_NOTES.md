# Entry B — grind / trend continuation (backtest)

Date: 2026-10-13

## What changed
- Kept Entry A: bullish + `rsi <= buy_pullback_rsi`.
- Added Entry B: flat + bullish + `buy_pullback_rsi < rsi <= buy_grind_rsi_max`
  (and below `sell_overbought_rsi`).
- Soft gates (backtest `should_auto_hold` only):
  - Neutral-band auto-HOLD no longer applies when trend is Bullish.
  - Flat + bullish uses `buy_grind_rsi_max` as the extended-RSI ceiling for new entries.
- New research knob: `buy_grind_rsi_max` (default 68), searchable by Grok/Optuna.

## Live / shadow loop (promoted 2026-10-13)
`complete_daily_loop.py` now mirrors Entry B for **shadow** collection:
- `should_auto_hold` matches backtest soft gates (bullish exempt from neutral band;
  flat+bullish uses `buy_grind_rsi_max`).
- `ENTRY_B_RESEARCH_THRESHOLDS` = knobs from `best_config_grok_then_optuna_entry_b.json`.
- `DEFAULT_SOFT_THRESHOLDS` / `RELAXED_SOFT_THRESHOLDS` both point at that set
  (launchd uses `--relax-gates`).
- `determine_deterministic_action()` logs Entry A/B as `strategy_action` /
  `strategy_reason` in the shadow jsonl (alongside legacy `deterministic_action`).
- Packet `constraints` and the Grok prompt include Entry A/B knobs.

Hard DD / loss-streak / ATR execution blocks are unchanged. No live orders until
you remove `--dry-run` from launchd after reviewing shadow logs / `shadow_replay.py`.

## Launch research (already done for entry_b)
```bash
python -m equity_msft.research_loop.run_supervisor_loop \
  --supervisor-config-json equity_msft/research_loop/supervisor_config_grok_then_optuna_entry_b.json \
  --output-json equity_msft/research_loop/supervisor_last_run_grok_then_optuna_entry_b.json
```
