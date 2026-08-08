# Kalshi Bot Quickstart

This README explains how to run the trading script and query its SQLite logs.

## Script Location

- Script: `/Users/khemra/dev/crispy-sniffle/kalshi/script.py`
- Default SQLite DB: `/Users/khemra/dev/crispy-sniffle/kalshi/shadow_calibration.db`

## Python Environment

Use the same interpreter used in this workspace:

```bash
/opt/homebrew/Caskroom/miniforge/base/envs/kalshi_env/bin/python
```

## How To Execute

From anywhere:

```bash
/opt/homebrew/Caskroom/miniforge/base/envs/kalshi_env/bin/python /Users/khemra/dev/crispy-sniffle/kalshi/script.py
```

From repo root (`/Users/khemra/dev/crispy-sniffle`):

```bash
/opt/homebrew/Caskroom/miniforge/base/envs/kalshi_env/bin/python kalshi/script.py
```

## Usable Flags

### 1) Shadow calibration summary (lightweight)

```bash
/opt/homebrew/Caskroom/miniforge/base/envs/kalshi_env/bin/python kalshi/script.py --shadow-summary
```

Returns JSON summary including:
- total/pending/resolved/missing calibration rows
- average edge/confidence
- Brier score (blended only)
- order counters (`orders_total`, `orders_submitted`, `orders_failed`)
- calibration bins

`rows_missing` counts markets Kalshi no longer returns (usually delisted). Those cannot be backfilled later — run `--backfill-outcomes` regularly to avoid this gap.

### 1b) Detailed shadow performance report (recommended)

```bash
/opt/homebrew/Caskroom/miniforge/base/envs/kalshi_env/bin/python kalshi/script.py --shadow-report
```

Returns a richer JSON report for deciding whether the bot is ready to leave shadow mode:
- model vs market vs blended Brier/MAE and “who is closer” counts
- trade-signal-only hit rate and fee-aware unit PnL
- edge / confidence buckets on resolved trade signals
- pending age buckets and unresolvable (`resolution_status='missing'`) counts
- top ticker prefixes (universe composition)
- traffic-light `verdict` with plain-language notes (`ready_for_live`)

Use this after each backfill when interpreting performance.

### 2) Recent executed orders

Default 20:

```bash
/opt/homebrew/Caskroom/miniforge/base/envs/kalshi_env/bin/python kalshi/script.py --orders-recent
```

Custom N (example 25):

```bash
/opt/homebrew/Caskroom/miniforge/base/envs/kalshi_env/bin/python kalshi/script.py --orders-recent 25
```

Returns JSON with recent rows from `executed_orders` joined to calibration context (`edge`, `confidence`, `blended_prob`, etc.).

### 3) Backfill settled outcomes

```bash
/opt/homebrew/Caskroom/miniforge/base/envs/kalshi_env/bin/python kalshi/script.py --backfill-outcomes
```

Dry run:

```bash
/opt/homebrew/Caskroom/miniforge/base/envs/kalshi_env/bin/python kalshi/script.py --backfill-outcomes --dry-run
```

**Why timing matters:** backfill looks up each pending ticker on Kalshi. Settled markets still available return `result=yes|no` and get `realized_outcome` filled. Markets that have been delisted return HTTP 404; those rows are marked `resolution_status='missing'` and can never recover an outcome. Waiting too long therefore permanently shrinks your resolved sample.

Suggested habit: run `--backfill-outcomes` daily (or after each bot session), then `--shadow-report` to re-read metrics.

Backfill JSON includes `tickers_missing`, `rows_marked_missing`, and `sample_missing_tickers` so the gap is visible.

### 4) Failed orders report

Default 20:

```bash
/opt/homebrew/Caskroom/miniforge/base/envs/kalshi_env/bin/python kalshi/script.py --orders-failures
```

Custom N (example 50):

```bash
/opt/homebrew/Caskroom/miniforge/base/envs/kalshi_env/bin/python kalshi/script.py --orders-failures 50
```

Returns JSON including:
- recent failed orders joined with calibration context
- grouped error summaries (error text + failure counts)
- failure rate across attempted orders

## SQLite Useful Commands

Open the DB:

```bash
sqlite3 /Users/khemra/dev/crispy-sniffle/kalshi/shadow_calibration.db
```

Inside `sqlite3`:

```sql
.headers on
.mode column
.tables
.schema shadow_calibration
.schema executed_orders
```

### Row counts

```sql
SELECT COUNT(*) AS shadow_rows FROM shadow_calibration;
SELECT COUNT(*) AS executed_orders_rows FROM executed_orders;
```

### Latest calibration rows

```sql
SELECT id, timestamp, ticker, direction, edge, confidence, realized_outcome, shadow_mode, executed
FROM shadow_calibration
ORDER BY id DESC
LIMIT 20;
```

### Latest orders with model context

```sql
SELECT
  eo.id,
  eo.timestamp,
  eo.ticker,
  eo.side,
  eo.count,
  eo.yes_price,
  eo.status,
  eo.error,
  sc.edge,
  sc.confidence,
  sc.market_prob,
  sc.model_prob,
  sc.blended_prob
FROM executed_orders eo
LEFT JOIN shadow_calibration sc ON sc.id = eo.calibration_id
ORDER BY eo.id DESC
LIMIT 20;
```

### Order status breakdown

```sql
SELECT status, COUNT(*) AS n
FROM executed_orders
GROUP BY status
ORDER BY n DESC;
```

### Grouped failure errors

```sql
SELECT
  COALESCE(NULLIF(TRIM(error), ''), '<empty>') AS error_text,
  COUNT(*) AS failures
FROM executed_orders
WHERE status = 'failed'
GROUP BY COALESCE(NULLIF(TRIM(error), ''), '<empty>')
ORDER BY failures DESC, error_text ASC;
```

### Recent failed orders with model context

```sql
SELECT
  eo.id,
  eo.timestamp,
  eo.ticker,
  eo.side,
  eo.count,
  eo.yes_price,
  eo.status,
  eo.error,
  sc.edge,
  sc.confidence,
  sc.market_prob,
  sc.model_prob,
  sc.blended_prob
FROM executed_orders eo
LEFT JOIN shadow_calibration sc ON sc.id = eo.calibration_id
WHERE eo.status = 'failed'
ORDER BY eo.id DESC
LIMIT 20;
```

### Basic calibration quality (resolved only)

```sql
SELECT
  COUNT(*) AS n,
  AVG((blended_prob - realized_outcome) * (blended_prob - realized_outcome)) AS brier_score,
  AVG(edge) AS avg_edge,
  AVG(confidence) AS avg_confidence
FROM shadow_calibration
WHERE realized_outcome IS NOT NULL;
```

### Calibration bins

```sql
SELECT
  CAST(blended_prob * 10 AS INTEGER) AS prob_bin,
  COUNT(*) AS samples,
  AVG(blended_prob) AS avg_pred,
  AVG(realized_outcome) AS realized_rate,
  AVG(blended_prob) - AVG(realized_outcome) AS calibration_gap
FROM shadow_calibration
WHERE realized_outcome IS NOT NULL
GROUP BY prob_bin
ORDER BY prob_bin;
```

## Optional DB Path Override

Set this before running if you want a different database file:

```bash
export SHADOW_DB_PATH="/path/to/custom/shadow_calibration.db"
```

The script will create tables/indexes automatically on first use.
