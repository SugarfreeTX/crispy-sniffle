"""Execution-realistic variant of backtest.py.

backtest.py computes each day's signal from that day's fully-closed bar (final
close, volume, RSI/ATR) and then fills the resulting order with
`trade_on_close=True` — i.e. at that SAME bar's close, with zero latency.
That is not achievable in live trading (the live loop in complete_daily_loop.py
runs after the market has already closed and any resulting order fills at the
NEXT session's open, not at the just-printed close).

This script reuses the identical strategy, data loading, and CLI from
backtest.py but fills orders at the next bar's Open (`trade_on_close=False`,
the backtesting.py default), which matches how the live loop actually trades.
Run this alongside backtest.py to see how much the same-close assumption was
inflating (or otherwise distorting) results.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict

import pandas as pd
from backtesting import Backtest

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from equity_msft.backtest import (  # noqa: F401 (re-exported for CLI parity)
    build_strategy_params,
    load_or_refresh_data,
    parse_args,
    parse_csv_floats,
    safe_float,
    sort_sweep_results,
    validate_args,
    write_outputs,
)
from equity_msft.backtest_strategy import MSFTDailyBacktestStrategy

DEFAULT_OUTPUT_DIR = BASE_DIR / "backtest_outputs_next_open"


def run_backtest(
    data: pd.DataFrame,
    args: Any,
    strategy_overrides: Dict[str, Any] | None = None,
) -> pd.Series:
    """Same as backtest.run_backtest but fills at the next bar's Open."""
    total_execution_cost_rate = (args.commission_bps + args.slippage_bps) / 10000.0

    bt = Backtest(
        data,
        MSFTDailyBacktestStrategy,
        cash=args.initial_cash,
        commission=total_execution_cost_rate,
        trade_on_close=False,
        hedging=False,
        exclusive_orders=True,
        finalize_trades=True,
    )

    strategy_params = build_strategy_params(args)
    if strategy_overrides:
        strategy_params.update(strategy_overrides)

    return bt.run(**strategy_params)


def plot_backtest(data: pd.DataFrame, args: Any) -> None:
    total_execution_cost_rate = (args.commission_bps + args.slippage_bps) / 10000.0
    bt = Backtest(
        data,
        MSFTDailyBacktestStrategy,
        cash=args.initial_cash,
        commission=total_execution_cost_rate,
        trade_on_close=False,
        hedging=False,
        exclusive_orders=True,
        finalize_trades=True,
    )
    bt.run(**build_strategy_params(args))
    bt.plot()


def main() -> None:
    args = parse_args()
    validate_args(args)

    csv_path = Path(args.csv).expanduser().resolve()
    output_dir = DEFAULT_OUTPUT_DIR

    data = load_or_refresh_data(
        symbol=args.symbol,
        csv_path=csv_path,
        start=args.start,
        end=args.end,
        refresh_data=args.refresh_data,
    )

    if args.sweep:
        raise SystemExit(
            "--sweep is not supported here; run backtest.py --sweep for parameter "
            "sweeps, then re-validate the winning config with this script."
        )

    if args.plot:
        plot_backtest(data, args)
        return

    stats = run_backtest(data, args)

    config = {
        "csv": str(csv_path),
        "start": args.start,
        "end": args.end,
        "initial_cash": args.initial_cash,
        "commission_bps": args.commission_bps,
        "slippage_bps": args.slippage_bps,
        "fill_timing": "next_bar_open",
        "strategy_params": build_strategy_params(args),
    }

    write_outputs(output_dir, args.symbol, stats, len(data), config)

    print("Backtest complete (next-open fills)")
    print(f"Data rows: {len(data)}")
    print(f"Return [%]: {stats.get('Return [%]')}")
    print(f"Max Drawdown [%]: {stats.get('Max. Drawdown [%]')}")
    print(f"Win Rate [%]: {stats.get('Win Rate [%]')}")
    print(f"Trades: {stats.get('# Trades')}")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
