#!/usr/bin/env python3
"""
AutoFinance: Autonomous Financial Analyst Agent
Inspired by karpathy/autoresearch — optimizing portfolio alpha on BIST equities.

Usage:
    python analyze.py                    # Show help
    python analyze.py --timeframe daily  # Run daily analysis cycle
    python analyze.py --timeframe weekly
    python analyze.py --timeframe monthly
    python analyze.py --timeframe quarterly
    python analyze.py --dry-run --timeframe daily  # Preview without writing
"""

import argparse
import json
import logging
import os
import sys
import time
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import requests
import xml.etree.ElementTree as ET

# ---------------------------------------------------------------------------
# PORTFOLIO DEFINITION (FIXED — mirrors program.md)
# ---------------------------------------------------------------------------
PORTFOLIO = {
    "HALKB": {
        "yahoo": "HALKB.IS", "sector": "Bankacılık",
        "inv_date": "2023-03-28", "inv_price_try": 10.557,
        "shares_outstanding": 18477804140, "tvf_pct": 0.4040,
    },
    "TRENJ": {
        "yahoo": "TRENJ.IS", "sector": "Madencilik",
        "inv_date": "2024-10-20", "inv_price_try": 109.624,
        "shares_outstanding": 11200259785561, "tvf_pct": 1.00,
    },
    "TRMET": {
        "yahoo": "TRMET.IS", "sector": "Madencilik",
        "inv_date": "2024-10-18", "inv_price_try": 150.835,
        "shares_outstanding": 71673880800, "tvf_pct": 1.00,
    },
    "TRALT": {
        "yahoo": "TRALT.IS", "sector": "Madencilik",
        "inv_date": "2024-10-18", "inv_price_try": 53.512,
        "shares_outstanding": 202500000, "tvf_pct": 1.00,
    },
    "TCELL": {
        "yahoo": "TCELL.IS", "sector": "Telekomünikasyon",
        "inv_date": "2020-10-22", "inv_price_try": 16.204,
        "shares_outstanding": 2178536499, "tvf_pct": 0.2620,
    },
    "THYAO": {
        "yahoo": "THYAO.IS", "sector": "Ulaştırma",
        "inv_date": "2017-02-06", "inv_price_try": 6.152,
        "shares_outstanding": 1373660993, "tvf_pct": 0.4912,
    },
    "TTKOM": {
        "yahoo": "TTKOM.IS", "sector": "Telekomünikasyon",
        "inv_date": "2022-03-31", "inv_price_try": 10.071,
        "shares_outstanding": 3500000000, "tvf_pct": 0.55,
    },
    "TURSG": {
        "yahoo": "TURSG.IS", "sector": "Sigorta",
        "inv_date": "2020-04-24", "inv_price_try": 0.041,
        "shares_outstanding": 10000000000, "tvf_pct": 0.81,
    },
    "VAKBN": {
        "yahoo": "VAKBN.IS", "sector": "Bankacılık",
        "inv_date": "2023-03-28", "inv_price_try": 9.469,
        "shares_outstanding": 15923743, "tvf_pct": 0.7480,
    },
    "KRDMD": {
        "yahoo": "KRDMD.IS", "sector": "Kimya/Çelik",
        "inv_date": "2022-12-06", "inv_price_try": 18.095,
        "shares_outstanding": 7802260024, "tvf_pct": 0.0441,
    },
}

BENCHMARK = "XU100.IS"
CURRENCY_PAIR = "USDTRY=X"

# Risk-free rate proxy (TCMB policy rate — update periodically)
RISK_FREE_RATE = 0.425  # 42.5% annual as of early 2026

# Paths
BASE_DIR = Path(__file__).parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_COMPUTED = BASE_DIR / "data" / "computed"
REPORTS_DIR = BASE_DIR / "reports"
TEMPLATES_DIR = BASE_DIR / "templates"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("autofinance")

# Timeframe → lookback period mapping
TIMEFRAME_DAYS = {
    "daily": 1,
    "weekly": 7,
    "monthly": 30,
    "quarterly": 90,
}

# ---------------------------------------------------------------------------
# 1. SCOUT — Data Fetching
# ---------------------------------------------------------------------------

class Scout:
    """Fetches raw market data from Yahoo Finance and KAP."""

    def __init__(self, date_str: str):
        self.date_str = date_str
        self.raw_dir = DATA_RAW / date_str
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def fetch_ohlcv(self, ticker_yahoo: str, period: str = "1y") -> pd.DataFrame:
        """Fetch OHLCV data for a single ticker."""
        time.sleep(0.5)  # Rate limit: >=0.5s between calls
        try:
            tk = yf.Ticker(ticker_yahoo)
            df = tk.history(period=period, auto_adjust=True)
            if df.empty:
                log.warning(f"No data returned for {ticker_yahoo}")
            return df
        except Exception as e:
            log.error(f"Failed to fetch {ticker_yahoo}: {e}")
            return pd.DataFrame()

    def fetch_all(self) -> dict:
        """Fetch data for all portfolio tickers, benchmark, and FX."""
        data = {}
        all_tickers = {name: info["yahoo"] for name, info in PORTFOLIO.items()}
        all_tickers["XU100"] = BENCHMARK
        all_tickers["USDTRY"] = CURRENCY_PAIR

        failed = 0
        for name, yahoo in all_tickers.items():
            log.info(f"SCOUT: Fetching {name} ({yahoo})...")
            df = self.fetch_ohlcv(yahoo)
            if df.empty:
                failed += 1
                if failed > 3:
                    log.error("ERROR BUDGET EXCEEDED: >3 tickers failed. Aborting.")
                    sys.exit(1)
            else:
                csv_path = self.raw_dir / f"{name}.csv"
                df.to_csv(csv_path)
                data[name] = df

        # Check data freshness
        for name, df in data.items():
            if name in ("USDTRY",):
                continue
            if not df.empty:
                last_date = df.index[-1].date()
                today = datetime.now().date()
                biz_days = np.busday_count(last_date, today)
                if biz_days > 1:
                    log.warning(
                        f"STALE DATA: {name} last date {last_date} "
                        f"({biz_days} business days old)"
                    )

        return data

    def fetch_kap_disclosures(self) -> list[dict]:
        """Fetch recent KAP disclosures via RSS (using requests + xml.etree)."""
        disclosures = []
        portfolio_tickers = [info["yahoo"].replace(".IS", "") for info in PORTFOLIO.values()]
        try:
            resp = requests.get(
                "https://www.kap.org.tr/tr/rss/bildiri",
                timeout=10,
                headers={"User-Agent": "AutoFinance/0.1"},
            )
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
            # RSS items are under channel/item
            for item in root.findall(".//item")[:50]:
                title = item.findtext("title", "")
                link = item.findtext("link", "")
                pub_date = item.findtext("pubDate", "")
                for ticker in portfolio_tickers:
                    if ticker in title.upper():
                        disclosures.append({
                            "title": title,
                            "link": link,
                            "published": pub_date,
                            "ticker": ticker,
                        })
        except Exception as e:
            log.warning(f"KAP RSS fetch failed: {e}")
        return disclosures

    def fetch_fundamentals(self, ticker_yahoo: str) -> dict:
        """Fetch fundamental data from yfinance .info."""
        time.sleep(0.5)
        try:
            tk = yf.Ticker(ticker_yahoo)
            info = tk.info
            return {
                "pe_ratio": info.get("trailingPE"),
                "pb_ratio": info.get("priceToBook"),
                "dividend_yield": info.get("dividendYield"),
                "ev_ebitda": info.get("enterpriseToEbitda"),
            }
        except Exception as e:
            log.warning(f"Fundamentals fetch failed for {ticker_yahoo}: {e}")
            return {}


# ---------------------------------------------------------------------------
# 2. ANALYST — Metrics Computation
# ---------------------------------------------------------------------------

class Analyst:
    """Computes all per-ticker and portfolio-level metrics."""

    def __init__(self, data: dict, timeframe: str, date_str: str):
        self.data = data
        self.timeframe = timeframe
        self.date_str = date_str
        self.lookback = TIMEFRAME_DAYS[timeframe]
        self.computed_dir = DATA_COMPUTED / date_str
        self.computed_dir.mkdir(parents=True, exist_ok=True)
        self.usdtry = self._get_usdtry()

    def _get_usdtry(self) -> float:
        """Get latest USDTRY rate."""
        if "USDTRY" in self.data and not self.data["USDTRY"].empty:
            return float(self.data["USDTRY"]["Close"].iloc[-1])
        log.warning("USDTRY data missing, using fallback 36.0")
        return 36.0

    def _get_xu100_return(self, days: int) -> float:
        """Get XU100 return over given period."""
        if "XU100" not in self.data or self.data["XU100"].empty:
            return 0.0
        df = self.data["XU100"]
        if len(df) < days + 1:
            days = len(df) - 1
        if days <= 0:
            return 0.0
        return float((df["Close"].iloc[-1] / df["Close"].iloc[-days - 1]) - 1)

    def compute_ticker_metrics(self, name: str) -> dict:
        """Compute all metrics for a single ticker."""
        info = PORTFOLIO[name]
        df = self.data.get(name)
        if df is None or df.empty:
            return {"ticker": name, "error": "no_data"}

        close = df["Close"]
        volume = df["Volume"]
        current_price = float(close.iloc[-1])

        # --- PRICE & RETURN ---
        inv_price = info["inv_price_try"]
        return_since_inv = (current_price - inv_price) / inv_price

        lookback = min(self.lookback, len(close) - 1)
        if lookback > 0:
            period_start = float(close.iloc[-lookback - 1])
            return_period = (current_price - period_start) / period_start
        else:
            return_period = 0.0

        xu100_return = self._get_xu100_return(lookback)
        alpha = return_period - xu100_return

        # High-low range
        if lookback > 0:
            period_slice = close.iloc[-lookback - 1:]
            period_high = float(period_slice.max())
            period_low = float(period_slice.min())
            high_low_range = (period_high - period_low) / period_low if period_low > 0 else 0
        else:
            high_low_range = 0.0

        # 52-week high distance
        yearly_data = close.iloc[-252:] if len(close) >= 252 else close
        high_52w = float(yearly_data.max())
        dist_52w_high = (high_52w - current_price) / high_52w if high_52w > 0 else 0

        # --- VOLUME & LIQUIDITY ---
        avg_vol = float(volume.iloc[-20:].mean()) if len(volume) >= 20 else float(volume.mean())
        current_vol = float(volume.iloc[-1])
        volume_trend = current_vol / avg_vol if avg_vol > 0 else 1.0

        shares = info["shares_outstanding"]
        market_cap_try = shares * current_price
        turnover_ratio = (current_vol * current_price) / market_cap_try if market_cap_try > 0 else 0

        # --- RISK ---
        daily_returns = close.pct_change().dropna()

        if len(daily_returns) >= 20:
            daily_vol = float(daily_returns.std())
            ann_vol = daily_vol * np.sqrt(252)
        else:
            ann_vol = 0.0

        # Beta vs XU100
        beta = 1.0
        if "XU100" in self.data and not self.data["XU100"].empty:
            xu100_close = self.data["XU100"]["Close"]
            # Align on common dates
            common_idx = close.index.intersection(xu100_close.index)
            if len(common_idx) >= 60:
                tk_ret = close.reindex(common_idx).pct_change().dropna()
                xu_ret = xu100_close.reindex(common_idx).pct_change().dropna()
                common = tk_ret.index.intersection(xu_ret.index)
                tk_ret = tk_ret.reindex(common)
                xu_ret = xu_ret.reindex(common)
                if len(common) >= 30:
                    cov = np.cov(tk_ret.values, xu_ret.values)
                    if cov[1, 1] > 0:
                        beta = float(cov[0, 1] / cov[1, 1])

        # Annualized return for Sharpe
        if len(close) >= 252:
            ann_return = float((close.iloc[-1] / close.iloc[-252]) - 1)
        else:
            n = len(close) - 1
            if n > 0:
                total_ret = float((close.iloc[-1] / close.iloc[0]) - 1)
                ann_return = (1 + total_ret) ** (252 / n) - 1
            else:
                ann_return = 0.0

        sharpe = (ann_return - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0.0

        # Max drawdown
        cummax = close.cummax()
        drawdown = (close - cummax) / cummax
        max_dd = float(drawdown.min())

        # VaR 95%
        if len(daily_returns) >= 30:
            var_95 = float(np.percentile(daily_returns.values, 5))
        else:
            var_95 = 0.0

        # --- MOMENTUM & TECHNICAL (manual implementations) ---
        # RSI 14
        if len(close) >= 15:
            delta = close.diff()
            gain = delta.where(delta > 0, 0.0)
            loss = (-delta).where(delta < 0, 0.0)
            avg_gain = gain.rolling(window=14, min_periods=14).mean().iloc[-1]
            avg_loss = loss.rolling(window=14, min_periods=14).mean().iloc[-1]
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi_val = float(100 - (100 / (1 + rs)))
            else:
                rsi_val = 100.0
        else:
            rsi_val = 50.0

        # MACD (12, 26, 9)
        macd_signal_str = "neutral"
        if len(close) >= 35:
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd_line = (ema12 - ema26).iloc[-1]
            signal_line = (ema12 - ema26).ewm(span=9, adjust=False).mean().iloc[-1]
            if not (np.isnan(macd_line) or np.isnan(signal_line)):
                macd_signal_str = "bullish" if macd_line > signal_line else "bearish"

        # SMA 50 vs 200
        sma_cross = "neutral"
        if len(close) >= 200:
            sma50 = float(close.rolling(50).mean().iloc[-1])
            sma200 = float(close.rolling(200).mean().iloc[-1])
            if not (np.isnan(sma50) or np.isnan(sma200)):
                sma_cross = "golden_cross" if sma50 > sma200 else "death_cross"

        # Price vs SMA 20
        if len(close) >= 20:
            sma20 = float(close.rolling(20).mean().iloc[-1])
            price_vs_sma20 = (current_price - sma20) / sma20 if sma20 > 0 else 0
        else:
            price_vs_sma20 = 0.0

        # Bollinger position (20-day, 2 std dev)
        boll_pos = 0.5
        if len(close) >= 20:
            sma20_bb = close.rolling(20).mean().iloc[-1]
            std20 = close.rolling(20).std().iloc[-1]
            bb_high = float(sma20_bb + 2 * std20)
            bb_low = float(sma20_bb - 2 * std20)
            if bb_high != bb_low:
                boll_pos = (current_price - bb_low) / (bb_high - bb_low)

        metrics = {
            "ticker": name,
            "sector": info["sector"],
            "current_price_try": round(current_price, 4),
            "current_price_usd": round(current_price / self.usdtry, 4),
            "return_since_investment": round(return_since_inv, 4),
            "return_period": round(return_period, 4),
            "return_vs_xu100": round(alpha, 4),
            "high_low_range": round(high_low_range, 4),
            "distance_from_52w_high": round(dist_52w_high, 4),
            "avg_daily_volume": round(avg_vol, 0),
            "volume_trend": round(volume_trend, 2),
            "turnover_ratio": round(turnover_ratio, 6),
            "daily_volatility": round(ann_vol, 4),
            "beta_vs_xu100": round(beta, 4),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown": round(max_dd, 4),
            "var_95": round(var_95, 4),
            "rsi_14": round(rsi_val, 2),
            "macd_signal": macd_signal_str,
            "sma_50_vs_200": sma_cross,
            "price_vs_sma_20": round(price_vs_sma20, 4),
            "bollinger_position": round(boll_pos, 4),
            "market_cap_try": round(market_cap_try, 0),
            "market_cap_usd": round(market_cap_try / self.usdtry, 0),
        }

        return metrics

    def compute_all(self) -> dict:
        """Compute metrics for all tickers + portfolio aggregates."""
        ticker_metrics = {}
        for name in PORTFOLIO:
            log.info(f"ANALYST: Computing metrics for {name}...")
            metrics = self.compute_ticker_metrics(name)
            ticker_metrics[name] = metrics

        # Generate signals
        for name, m in ticker_metrics.items():
            if "error" not in m:
                m["signal"] = generate_signal(m)

        # Portfolio-level metrics
        portfolio = self._compute_portfolio_metrics(ticker_metrics)

        result = {
            "date": self.date_str,
            "timeframe": self.timeframe,
            "usdtry": self.usdtry,
            "xu100_return": round(self._get_xu100_return(self.lookback), 4),
            "tickers": ticker_metrics,
            "portfolio": portfolio,
        }

        # Save computed data
        out_path = self.computed_dir / f"{self.timeframe}_metrics.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        log.info(f"ANALYST: Metrics saved to {out_path}")

        return result

    def _compute_portfolio_metrics(self, ticker_metrics: dict) -> dict:
        """Compute portfolio-level aggregate metrics."""
        total_value_try = 0.0
        weighted_return = 0.0
        weighted_beta = 0.0
        sector_values = {}
        weights = {}

        # First pass: compute total value for weights
        for name, info in PORTFOLIO.items():
            m = ticker_metrics.get(name, {})
            if "error" in m:
                continue
            tvf_value = info["shares_outstanding"] * info["tvf_pct"] * m["current_price_try"]
            total_value_try += tvf_value
            sector = info["sector"]
            sector_values[sector] = sector_values.get(sector, 0) + tvf_value
            weights[name] = tvf_value

        # Second pass: weighted metrics
        if total_value_try > 0:
            for name, val in weights.items():
                w = val / total_value_try
                weights[name] = w
                m = ticker_metrics[name]
                weighted_return += w * m.get("return_period", 0)
                weighted_beta += w * m.get("beta_vs_xu100", 1)

        # Sector allocation
        sector_alloc = {}
        for sector, val in sector_values.items():
            sector_alloc[sector] = round(val / total_value_try, 4) if total_value_try > 0 else 0

        # HHI concentration
        hhi = sum(w ** 2 for w in weights.values()) if weights else 0

        # Best / worst performers
        valid = {n: m for n, m in ticker_metrics.items() if "error" not in m}
        best = max(valid, key=lambda n: valid[n]["return_period"]) if valid else "N/A"
        worst = min(valid, key=lambda n: valid[n]["return_period"]) if valid else "N/A"

        # Portfolio Sharpe (simplified)
        port_returns = []
        for name, w in weights.items():
            m = ticker_metrics[name]
            port_returns.append(w * m.get("return_period", 0))
        port_return_total = sum(port_returns)
        # Use weighted vol approximation
        weighted_vol = 0.0
        for name, w in weights.items():
            m = ticker_metrics[name]
            weighted_vol += w * m.get("daily_volatility", 0)
        port_sharpe = ((port_return_total * (252 / max(self.lookback, 1))) - RISK_FREE_RATE) / weighted_vol if weighted_vol > 0 else 0

        # XU100 correlation (portfolio weighted returns vs XU100)
        xu100_corr = 0.0
        if "XU100" in self.data and not self.data["XU100"].empty:
            xu100_close = self.data["XU100"]["Close"]
            xu100_ret = xu100_close.pct_change().dropna()
            # Build portfolio return series
            port_ret_series = pd.Series(0.0, index=xu100_ret.index)
            for name, w in weights.items():
                if name in self.data and not self.data[name].empty:
                    tk_ret = self.data[name]["Close"].pct_change().dropna()
                    common = port_ret_series.index.intersection(tk_ret.index)
                    port_ret_series.loc[common] += w * tk_ret.reindex(common).fillna(0)
            common = port_ret_series.index.intersection(xu100_ret.index)
            if len(common) >= 30:
                xu100_corr = float(port_ret_series.loc[common].corr(xu100_ret.loc[common]))

        return {
            "portfolio_value_try": round(total_value_try, 0),
            "portfolio_value_usd": round(total_value_try / self.usdtry, 0),
            "portfolio_return": round(weighted_return, 4),
            "portfolio_beta": round(weighted_beta, 4),
            "portfolio_sharpe": round(port_sharpe, 4),
            "sector_allocation": sector_alloc,
            "concentration_hhi": round(hhi, 4),
            "best_performer": best,
            "worst_performer": worst,
            "xu100_correlation": round(xu100_corr, 4),
        }


# ---------------------------------------------------------------------------
# Signal Generation
# ---------------------------------------------------------------------------

def generate_signal(m: dict) -> str:
    """Generate BUY/HOLD/SELL signal from ticker metrics."""
    score = 0

    # Momentum (30%)
    rsi = m.get("rsi_14", 50)
    if rsi < 30:
        score += 3
    elif rsi > 70:
        score -= 3
    if m.get("macd_signal") == "bullish":
        score += 2
    if m.get("price_vs_sma_20", 0) > 0.05:
        score += 1

    # Trend (25%)
    if m.get("sma_50_vs_200") == "golden_cross":
        score += 3
    elif m.get("sma_50_vs_200") == "death_cross":
        score -= 3
    if m.get("return_vs_xu100", 0) > 0:
        score += 1

    # Volume (15%)
    if m.get("volume_trend", 1) > 1.5:
        score += 2

    # Risk (15%)
    if m.get("max_drawdown", -1) > -0.15:
        score += 1
    if m.get("sharpe_ratio", 0) > 1.0:
        score += 2

    # Fundamental (15%)
    pe = m.get("pe_ratio")
    if pe and pe < 8:
        score += 2
    div_y = m.get("dividend_yield")
    if div_y and div_y > 0.05:
        score += 1

    if score >= 6:
        return "STRONG BUY"
    elif score >= 3:
        return "BUY"
    elif score >= -2:
        return "HOLD"
    elif score >= -5:
        return "SELL"
    else:
        return "STRONG SELL"


# ---------------------------------------------------------------------------
# 3. VALIDATOR — QA Checks
# ---------------------------------------------------------------------------

class Validator:
    """Validates computed metrics for correctness."""

    def __init__(self, result: dict, data: dict):
        self.result = result
        self.data = data
        self.errors = []

    def validate(self) -> bool:
        """Run all validation checks. Returns True if passed."""
        self._check_nan_inf()
        self._check_date_alignment()
        self._check_price_sanity()
        self._spot_check_return()

        if self.errors:
            log.warning(f"VALIDATOR: {len(self.errors)} issue(s) found:")
            for err in self.errors:
                log.warning(f"  - {err}")
            return False

        log.info("VALIDATOR: All checks passed.")
        return True

    def _check_nan_inf(self):
        """Verify no NaN/Inf in any output field."""
        for name, m in self.result["tickers"].items():
            if "error" in m:
                self.errors.append(f"{name}: has error flag ({m['error']})")
                continue
            for key, val in m.items():
                if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                    self.errors.append(f"{name}.{key} = {val} (NaN/Inf)")

    def _check_date_alignment(self):
        """Confirm no weekend dates in latest data points."""
        for name, df in self.data.items():
            if name in ("USDTRY",) or df.empty:
                continue
            last_date = df.index[-1]
            if last_date.weekday() >= 5:  # Saturday=5, Sunday=6
                self.errors.append(
                    f"{name}: last data point on weekend ({last_date.strftime('%A %Y-%m-%d')})"
                )

    def _check_price_sanity(self):
        """Basic sanity: prices should be positive."""
        for name, m in self.result["tickers"].items():
            if "error" in m:
                continue
            price = m.get("current_price_try", 0)
            if price <= 0:
                self.errors.append(f"{name}: price <= 0 ({price})")

    def _spot_check_return(self):
        """Verify return calculation on 1 random ticker."""
        valid_tickers = [n for n, m in self.result["tickers"].items() if "error" not in m]
        if not valid_tickers:
            return

        check_name = random.choice(valid_tickers)
        m = self.result["tickers"][check_name]
        info = PORTFOLIO[check_name]

        expected_return = (m["current_price_try"] - info["inv_price_try"]) / info["inv_price_try"]
        actual_return = m["return_since_investment"]
        diff = abs(expected_return - actual_return)

        if diff > 0.001:
            self.errors.append(
                f"{check_name}: return_since_investment mismatch "
                f"(expected {expected_return:.4f}, got {actual_return})"
            )
        else:
            log.info(f"VALIDATOR: Spot check passed for {check_name}")


# ---------------------------------------------------------------------------
# 4. WRITER — Report Generation
# ---------------------------------------------------------------------------

class Writer:
    """Generates Markdown and Excel reports."""

    def __init__(self, result: dict, kap_disclosures: list, timeframe: str, date_str: str):
        self.result = result
        self.kap = kap_disclosures
        self.timeframe = timeframe
        self.date_str = date_str

    def generate_markdown(self) -> str:
        """Generate the Markdown report."""
        p = self.result["portfolio"]
        tickers = self.result["tickers"]
        usdtry = self.result["usdtry"]

        # Header
        timeframe_tr = {
            "daily": "Günlük", "weekly": "Haftalık",
            "monthly": "Aylık", "quarterly": "Çeyreklik",
        }
        title = f"TVF Portföy {timeframe_tr[self.timeframe]} Rapor — {self.date_str}"

        lines = [f"# {title}", ""]

        # Summary
        period_label = {
            "daily": "Günlük", "weekly": "Haftalık",
            "monthly": "Aylık", "quarterly": "Çeyreklik",
        }
        lines.append("## Özet")
        lines.append(f"- **Portföy Değeri:** ₺{p['portfolio_value_try']:,.0f} (${p['portfolio_value_usd']:,.0f})")
        lines.append(f"- **{period_label[self.timeframe]} Getiri:** {p['portfolio_return']:+.2%} | XU100: {self.result['xu100_return']:+.2%}")
        lines.append(f"- **Alfa:** {p['portfolio_return'] - self.result['xu100_return']:+.2%}")
        lines.append(f"- **USD/TRY:** {usdtry:.4f}")
        lines.append("")

        # Per-ticker table
        lines.append("## Hisse Bazında Performans")
        lines.append("")
        if self.timeframe == "daily":
            lines.append("| Ticker | Fiyat (₺) | Günlük Δ | Hacim Trend | RSI | Sinyal |")
            lines.append("|--------|-----------|----------|-------------|-----|--------|")
        elif self.timeframe == "weekly":
            lines.append("| Ticker | Fiyat (₺) | Haftalık Δ | Hacim Trend | RSI | Sinyal |")
            lines.append("|--------|-----------|------------|-------------|-----|--------|")
        elif self.timeframe == "monthly":
            lines.append("| Ticker | Fiyat (₺) | Aylık Δ | Yatırımdan Beri | RSI | Sinyal |")
            lines.append("|--------|-----------|---------|-----------------|-----|--------|")
        else:
            lines.append("| Ticker | Fiyat (₺) | Çeyreklik Δ | Yatırımdan Beri | Beta | Sinyal |")
            lines.append("|--------|-----------|-------------|-----------------|------|--------|")

        for name, m in sorted(tickers.items()):
            if "error" in m:
                lines.append(f"| {name} | ERROR | — | — | — | — |")
                continue
            if self.timeframe in ("daily", "weekly"):
                lines.append(
                    f"| {name} | {m['current_price_try']:.2f} "
                    f"| {m['return_period']:+.2%} "
                    f"| {m['volume_trend']:.2f}x "
                    f"| {m['rsi_14']:.0f} "
                    f"| {m.get('signal', 'N/A')} |"
                )
            elif self.timeframe == "monthly":
                lines.append(
                    f"| {name} | {m['current_price_try']:.2f} "
                    f"| {m['return_period']:+.2%} "
                    f"| {m['return_since_investment']:+.2%} "
                    f"| {m['rsi_14']:.0f} "
                    f"| {m.get('signal', 'N/A')} |"
                )
            else:
                lines.append(
                    f"| {name} | {m['current_price_try']:.2f} "
                    f"| {m['return_period']:+.2%} "
                    f"| {m['return_since_investment']:+.2%} "
                    f"| {m['beta_vs_xu100']:.2f} "
                    f"| {m.get('signal', 'N/A')} |"
                )
        lines.append("")

        # Highlights
        valid = {n: m for n, m in tickers.items() if "error" not in m}
        if valid:
            best = p["best_performer"]
            worst = p["worst_performer"]
            overbought = [n for n, m in valid.items() if m["rsi_14"] > 70]
            oversold = [n for n, m in valid.items() if m["rsi_14"] < 30]
            high_vol = [n for n, m in valid.items() if m["volume_trend"] > 1.5]

            lines.append("## Dikkat Çekenler")
            best_ret = valid[best]["return_period"] if best in valid else 0
            worst_ret = valid[worst]["return_period"] if worst in valid else 0
            lines.append(f"- **En İyi:** {best} ({best_ret:+.2%})")
            lines.append(f"- **En Kötü:** {worst} ({worst_ret:+.2%})")
            lines.append(f"- **Yüksek Hacim:** {', '.join(high_vol) if high_vol else 'Yok'}")
            lines.append(f"- **RSI Aşırı Alım (>70):** {', '.join(overbought) if overbought else 'Yok'}")
            lines.append(f"- **RSI Aşırı Satım (<30):** {', '.join(oversold) if oversold else 'Yok'}")
            lines.append("")

        # KAP disclosures
        lines.append("## KAP Bildirimleri (Son 24 Saat)")
        if self.kap:
            for d in self.kap:
                lines.append(f"- **{d['ticker']}**: {d['title']}")
        else:
            lines.append("- Portföy hisseleri ile ilgili yeni bildirim bulunmamaktadır.")
        lines.append("")

        # Technical signals summary
        lines.append("## Teknik Sinyaller Özeti")
        lines.append("")
        lines.append("| Ticker | MACD | SMA 50/200 | Bollinger | SMA20 Δ |")
        lines.append("|--------|------|------------|-----------|---------|")
        for name, m in sorted(tickers.items()):
            if "error" in m:
                continue
            lines.append(
                f"| {name} | {m['macd_signal']} "
                f"| {m['sma_50_vs_200']} "
                f"| {m['bollinger_position']:.2f} "
                f"| {m['price_vs_sma_20']:+.2%} |"
            )
        lines.append("")

        # Sector allocation (for weekly+)
        if self.timeframe in ("weekly", "monthly", "quarterly"):
            lines.append("## Sektör Dağılımı")
            for sector, pct in sorted(p["sector_allocation"].items()):
                bar = "█" * int(pct * 40)
                lines.append(f"- **{sector}:** {pct:.1%} {bar}")
            lines.append(f"- **HHI Yoğunlaşma:** {p['concentration_hhi']:.4f}")
            lines.append("")

        # Portfolio risk (for monthly+)
        if self.timeframe in ("monthly", "quarterly"):
            lines.append("## Portföy Risk Metrikleri")
            lines.append(f"- **Portföy Beta:** {p['portfolio_beta']:.2f}")
            lines.append(f"- **Portföy Sharpe:** {p['portfolio_sharpe']:.2f}")
            lines.append(f"- **XU100 Korelasyon:** {p['xu100_correlation']:.2f}")
            lines.append("")

        # Footer
        lines.append("---")
        lines.append(f"*AutoFinance v0.1 | Oluşturulma: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

        return "\n".join(lines)

    def generate_excel(self, output_path: Path):
        """Generate Excel workbook with data tabs."""
        tickers = self.result["tickers"]
        portfolio = self.result["portfolio"]

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            # Summary sheet
            summary_data = {
                "Metric": [
                    "Portföy Değeri (₺)", "Portföy Değeri ($)",
                    "Portföy Getiri", "Portföy Beta", "Portföy Sharpe",
                    "HHI", "XU100 Korelasyon", "En İyi", "En Kötü",
                    "USD/TRY",
                ],
                "Value": [
                    portfolio["portfolio_value_try"],
                    portfolio["portfolio_value_usd"],
                    portfolio["portfolio_return"],
                    portfolio["portfolio_beta"],
                    portfolio["portfolio_sharpe"],
                    portfolio["concentration_hhi"],
                    portfolio["xu100_correlation"],
                    portfolio["best_performer"],
                    portfolio["worst_performer"],
                    self.result["usdtry"],
                ],
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name="Özet", index=False)

            # Per-ticker sheet
            rows = []
            for name, m in sorted(tickers.items()):
                if "error" in m:
                    continue
                rows.append({
                    "Ticker": name,
                    "Sektör": m["sector"],
                    "Fiyat (₺)": m["current_price_try"],
                    "Fiyat ($)": m["current_price_usd"],
                    "Dönem Getiri": m["return_period"],
                    "Yatırımdan Beri": m["return_since_investment"],
                    "Alfa": m["return_vs_xu100"],
                    "RSI": m["rsi_14"],
                    "Beta": m["beta_vs_xu100"],
                    "Sharpe": m["sharpe_ratio"],
                    "Max DD": m["max_drawdown"],
                    "Volatilite": m["daily_volatility"],
                    "Hacim Trend": m["volume_trend"],
                    "Sinyal": m.get("signal", "N/A"),
                })
            if rows:
                pd.DataFrame(rows).to_excel(writer, sheet_name="Hisseler", index=False)

            # Sector allocation
            if portfolio["sector_allocation"]:
                sector_df = pd.DataFrame([
                    {"Sektör": k, "Ağırlık": v}
                    for k, v in portfolio["sector_allocation"].items()
                ])
                sector_df.to_excel(writer, sheet_name="Sektör Dağılımı", index=False)

        log.info(f"WRITER: Excel saved to {output_path}")

    def write_reports(self) -> Path:
        """Write all report files and return the markdown path."""
        report_dir = REPORTS_DIR / self.timeframe
        report_dir.mkdir(parents=True, exist_ok=True)

        # Markdown
        md_content = self.generate_markdown()
        md_path = report_dir / f"{self.date_str}_{self.timeframe}.md"
        md_path.write_text(md_content, encoding="utf-8")
        log.info(f"WRITER: Markdown saved to {md_path}")

        # Excel
        xlsx_path = report_dir / f"{self.date_str}_{self.timeframe}.xlsx"
        self.generate_excel(xlsx_path)

        return md_path


# ---------------------------------------------------------------------------
# Run Log
# ---------------------------------------------------------------------------

def append_run_log(entry: dict):
    """Append a cycle entry to run_log.jsonl."""
    log_path = REPORTS_DIR / "run_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AutoFinance: Autonomous Financial Analyst Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python analyze.py --timeframe daily
    python analyze.py --timeframe weekly
    python analyze.py --timeframe monthly --dry-run
    python analyze.py --timeframe quarterly
        """,
    )
    parser.add_argument(
        "--timeframe",
        choices=["daily", "weekly", "monthly", "quarterly"],
        required=True,
        help="Report timeframe to generate",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview mode — fetch data but don't write reports",
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    date_str = datetime.now().strftime("%Y-%m-%d")
    cycle_start = time.time()

    log.info(f"{'='*60}")
    log.info(f"AutoFinance — {args.timeframe.upper()} cycle — {date_str}")
    log.info(f"{'='*60}")

    # 1. SCOUT
    log.info("--- PHASE 1: SCOUT ---")
    scout = Scout(date_str)
    data = scout.fetch_all()
    kap = scout.fetch_kap_disclosures()

    # Fetch fundamentals for monthly/quarterly
    fundamentals = {}
    if args.timeframe in ("monthly", "quarterly"):
        log.info("SCOUT: Fetching fundamentals (monthly/quarterly refresh)...")
        for name, info in PORTFOLIO.items():
            fund = scout.fetch_fundamentals(info["yahoo"])
            if fund:
                fundamentals[name] = fund

    # 2. ANALYST
    log.info("--- PHASE 2: ANALYST ---")
    analyst = Analyst(data, args.timeframe, date_str)
    result = analyst.compute_all()

    # Merge fundamentals if available
    for name, fund in fundamentals.items():
        if name in result["tickers"] and "error" not in result["tickers"][name]:
            result["tickers"][name].update(fund)
            # Re-generate signal with fundamentals
            result["tickers"][name]["signal"] = generate_signal(result["tickers"][name])

    # 3. VALIDATOR
    log.info("--- PHASE 3: VALIDATOR ---")
    validator = Validator(result, data)
    passed = validator.validate()

    if not passed:
        error_dir = REPORTS_DIR / "errors"
        error_dir.mkdir(parents=True, exist_ok=True)
        error_log = error_dir / f"{date_str}.log"
        error_log.write_text(
            "\n".join(validator.errors), encoding="utf-8"
        )
        log.warning(f"Validation errors logged to {error_log}")
        # Continue anyway — errors are logged, report will note them

    if args.dry_run:
        log.info("DRY RUN: Skipping report generation.")
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return

    # 4. WRITER
    log.info("--- PHASE 4: WRITER ---")
    writer = Writer(result, kap, args.timeframe, date_str)
    md_path = writer.write_reports()

    # Run log
    cycle_duration = time.time() - cycle_start
    alpha = result["portfolio"]["portfolio_return"] - result["xu100_return"]
    run_entry = {
        "timestamp": datetime.now().isoformat(),
        "timeframe": args.timeframe,
        "cycle_duration_sec": round(cycle_duration, 1),
        "tickers_processed": len([m for m in result["tickers"].values() if "error" not in m]),
        "validation_pass": passed,
        "portfolio_value_try": result["portfolio"]["portfolio_value_try"],
        "daily_alpha": round(alpha, 4),
        "errors": validator.errors,
    }
    append_run_log(run_entry)

    log.info(f"{'='*60}")
    log.info(f"CYCLE COMPLETE in {cycle_duration:.1f}s")
    log.info(f"  Portfolio: ₺{result['portfolio']['portfolio_value_try']:,.0f}")
    log.info(f"  Alpha: {alpha:+.4f}")
    log.info(f"  Report: {md_path}")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
