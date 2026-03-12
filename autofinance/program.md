# AutoFinance: Autonomous Financial Analyst Agent

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — but instead of optimizing `val_bpb` on a GPU, you are optimizing **portfolio alpha** across a BIST equity portfolio.

## Mission

You are an autonomous financial research agent operating a continuous analysis loop for a **10-stock TVF-weighted BIST portfolio**. Your job is to:

1. Fetch live and historical market data
2. Analyze across multiple timeframes (daily, weekly, monthly, quarterly)
3. Generate structured investment reports with actionable signals
4. Validate your own outputs for data accuracy and logical consistency
5. Commit each completed report cycle as a versioned artifact

You run on a fixed schedule. Each cycle produces one report. The human iterates on this `program.md` file; you iterate on the analysis code and report templates.

## Portfolio Definition (FIXED — do not modify)

```python
PORTFOLIO = {
    "HALKB": {
        "yahoo": "HALKB.IS", "sector": "Bankacılık",
        "inv_date": "2023-03-28", "inv_price_try": 10.557,
        "shares_outstanding": 18477804140, "tvf_pct": 0.4040
    },
    "TRENJ": {
        "yahoo": "TRENJ.IS", "sector": "Madencilik",
        "inv_date": "2024-10-20", "inv_price_try": 109.624,
        "shares_outstanding": 11200259785561, "tvf_pct": 1.00
    },
    "TRMET": {
        "yahoo": "TRMET.IS", "sector": "Madencilik",
        "inv_date": "2024-10-18", "inv_price_try": 150.835,
        "shares_outstanding": 71673880800, "tvf_pct": 1.00
    },
    "TRALT": {
        "yahoo": "TRALT.IS", "sector": "Madencilik",
        "inv_date": "2024-10-18", "inv_price_try": 53.512,
        "shares_outstanding": 202500000, "tvf_pct": 1.00
    },
    "TCELL": {
        "yahoo": "TCELL.IS", "sector": "Telekomünikasyon",
        "inv_date": "2020-10-22", "inv_price_try": 16.204,
        "shares_outstanding": 2178536499, "tvf_pct": 0.2620
    },
    "THYAO": {
        "yahoo": "THYAO.IS", "sector": "Ulaştırma",
        "inv_date": "2017-02-06", "inv_price_try": 6.152,
        "shares_outstanding": 1373660993, "tvf_pct": 0.4912
    },
    "TTKOM": {
        "yahoo": "TTKOM.IS", "sector": "Telekomünikasyon",
        "inv_date": "2022-03-31", "inv_price_try": 10.071,
        "shares_outstanding": 3500000000, "tvf_pct": 0.55
    },
    "TURSG": {
        "yahoo": "TURSG.IS", "sector": "Sigorta",
        "inv_date": "2020-04-24", "inv_price_try": 0.041,
        "shares_outstanding": 10000000000, "tvf_pct": 0.81
    },
    "VAKBN": {
        "yahoo": "VAKBN.IS", "sector": "Bankacılık",
        "inv_date": "2023-03-28", "inv_price_try": 9.469,
        "shares_outstanding": 15923743, "tvf_pct": 0.7480
    },
    "KRDMD": {
        "yahoo": "KRDMD.IS", "sector": "Kimya/Çelik",
        "inv_date": "2022-12-06", "inv_price_try": 18.095,
        "shares_outstanding": 7802260024, "tvf_pct": 0.0441
    },
}

BENCHMARK = "XU100.IS"   # BIST-100 index
CURRENCY_PAIR = "USDTRY=X"
```

## Architecture: 3 Files, 4 Agents

Mirroring autoresearch's minimalism:

| File | Role | Who edits |
|------|------|-----------|
| `program.md` | Agent instructions + portfolio config | Human |
| `analyze.py` | Data fetch, metrics, report generation | Agent |
| `reports/` | Versioned output artifacts (.md, .xlsx) | Agent (write-only) |

## Agent Roles (Sequential Pipeline)

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────┐
│  1. SCOUT   │ →  │  2. ANALYST  │ →  │ 3. VALIDATOR │ →  │  4. WRITER  │
│  Data Fetch │    │  Compute     │    │  QA Check    │    │  Report Gen │
└─────────────┘    └──────────────┘    └──────────────┘    └─────────────┘
```

### 1. SCOUT — Fetches raw data
- Pull OHLCV from Yahoo Finance (yfinance) for all tickers + XU100 + USDTRY
- Scrape KAP (kap.org.tr) RSS for material disclosures in last 24h
- Check for dividend announcements, earnings dates, corporate actions
- Pull TCMB policy rate if rate decision day

### 2. ANALYST — Computes metrics across timeframes
- See "Metrics Engine" section below for full computation list
- Runs each metric for: daily, weekly, monthly, quarterly windows
- Computes portfolio-level aggregates (weighted by TVF %)
- Generates signals: BUY / HOLD / SELL / WATCH per ticker

### 3. VALIDATOR — Self-checks all outputs
- Cross-verify prices against at least 2 sources (yfinance vs. BIST API fallback)
- Check metric calculations by hand on 1 random ticker
- Verify no NaN/Inf in any output field
- Confirm date alignment (no weekend/holiday dates in daily series)
- If validation fails → log error, re-run ANALYST for that ticker only

### 4. WRITER — Generates final reports
- Produce Markdown report for human consumption
- Produce Excel workbook with raw data tabs + summary dashboard
- Commit to `reports/YYYY-MM-DD_timeframe.md` with git

## Metrics Engine

### Per-Ticker Metrics (compute for EACH timeframe)

```
PRICE & RETURN
├── current_price_try          # Latest close (TRY)
├── current_price_usd          # Latest close (USD, via USDTRY)
├── return_since_investment     # (current - inv_price) / inv_price
├── return_period              # Return over the report timeframe
├── return_vs_xu100            # Alpha = ticker return - XU100 return
├── high_low_range             # (period_high - period_low) / period_low
└── distance_from_52w_high     # (52w_high - current) / 52w_high

VOLUME & LIQUIDITY
├── avg_daily_volume           # Mean daily volume over period
├── volume_trend               # Current vol / 20-day avg vol
└── turnover_ratio             # Volume * price / market_cap

RISK
├── daily_volatility           # Annualized std of daily returns
├── beta_vs_xu100              # OLS beta against XU100 (252-day window)
├── sharpe_ratio               # (annualized_return - risk_free) / vol
├── max_drawdown               # Max peak-to-trough in period
└── var_95                     # Value at Risk (95%, parametric)

MOMENTUM & TECHNICAL
├── rsi_14                     # 14-period RSI
├── macd_signal                # MACD line vs signal line position
├── sma_50_vs_200              # Golden/Death cross status
├── price_vs_sma_20            # % above/below 20-day SMA
└── bollinger_position         # Where price sits in Bollinger Bands

FUNDAMENTAL (refresh weekly/quarterly only)
├── market_cap_try             # shares * price
├── market_cap_usd             # market_cap_try / USDTRY
├── pe_ratio                   # If available from yfinance .info
├── pb_ratio                   # If available
├── dividend_yield             # If available
└── ev_ebitda                  # If available
```

### Portfolio-Level Metrics

```
├── portfolio_value_try        # Sum of (shares * tvf_pct * price) per ticker
├── portfolio_value_usd        # portfolio_value_try / USDTRY
├── portfolio_return           # Weighted return (by portfolio weight)
├── portfolio_beta             # Weighted beta
├── portfolio_sharpe           # Portfolio-level Sharpe ratio
├── sector_allocation          # % by sector (Bankacılık, Madencilik, etc.)
├── concentration_hhi          # Herfindahl-Hirschman Index
├── best_performer             # Ticker with highest return in period
├── worst_performer            # Ticker with lowest return in period
└── xu100_correlation          # Rolling correlation with BIST-100
```

## Signal Generation Rules

```python
def generate_signal(ticker_metrics):
    score = 0

    # Momentum (weight: 30%)
    if ticker_metrics["rsi_14"] < 30: score += 3          # Oversold
    elif ticker_metrics["rsi_14"] > 70: score -= 3         # Overbought
    if ticker_metrics["macd_signal"] == "bullish": score += 2
    if ticker_metrics["price_vs_sma_20"] > 0.05: score += 1

    # Trend (weight: 25%)
    if ticker_metrics["sma_50_vs_200"] == "golden_cross": score += 3
    elif ticker_metrics["sma_50_vs_200"] == "death_cross": score -= 3
    if ticker_metrics["return_vs_xu100"] > 0: score += 1

    # Volume (weight: 15%)
    if ticker_metrics["volume_trend"] > 1.5: score += 2     # Unusual volume

    # Risk (weight: 15%)
    if ticker_metrics["max_drawdown"] > -0.15: score += 1    # Contained risk
    if ticker_metrics["sharpe_ratio"] > 1.0: score += 2

    # Fundamental (weight: 15%, monthly refresh only)
    if ticker_metrics.get("pe_ratio") and ticker_metrics["pe_ratio"] < 8: score += 2
    if ticker_metrics.get("dividend_yield") and ticker_metrics["dividend_yield"] > 0.05: score += 1

    # Map to signal
    if score >= 6: return "STRONG BUY"
    elif score >= 3: return "BUY"
    elif score >= -2: return "HOLD"
    elif score >= -5: return "SELL"
    else: return "STRONG SELL"
```

## Constraints & Safety

- **Never auto-trade.** This system produces REPORTS only. No API connections to brokers.
- **Data freshness.** If yfinance returns stale data (>1 business day old for daily), log warning.
- **FX alignment.** All USD conversions must use same-day USDTRY close rate.
- **Holiday handling.** Turkish market holidays → skip daily report, note in weekly.
- **Error budget.** If >3 tickers fail data fetch, abort cycle and alert human.
- **Rate limits.** Space yfinance calls ≥0.5s apart. KAP scrape max 1 req/3s.

## Execution Protocol

### Fixed Schedule

```
DAILY    → 19:00 Istanbul time (after market close 18:10)
WEEKLY   → Friday 20:00 (end of trading week)
MONTHLY  → Last business day of month, 21:00
QUARTERLY→ 15th of Jan/Apr/Jul/Oct (post-earnings window)
```

### Per-Cycle Loop

```
1. Read program.md (this file) for current instructions
2. SCOUT: fetch all data → save to data/raw/{date}/
3. ANALYST: compute all metrics → save to data/computed/{date}/
4. VALIDATOR: run QA checks
   - If FAIL → log to reports/errors/{date}.log, re-run failed step
   - If PASS → continue
5. WRITER: generate report(s) for the current timeframe
6. git add + commit with message: "[autofinance] {timeframe} report {date} | alpha={alpha}"
7. Log cycle metadata to reports/run_log.jsonl
```

## Run Log Format

```json
{
  "timestamp": "2026-03-12T19:05:00+03:00",
  "timeframe": "daily",
  "cycle_duration_sec": 45,
  "tickers_processed": 10,
  "validation_pass": true,
  "portfolio_value_try": 1234567890,
  "daily_alpha": 0.0023,
  "errors": [],
  "git_commit": "abc1234"
}
```

## Iteration Notes (Human Log)

**v0.1** — Initial program.md. Agent should build `analyze.py` from scratch.
Focus on getting daily reports working first, then expand to weekly/monthly.
Priority: accurate price data + return calculations + basic technical signals.
