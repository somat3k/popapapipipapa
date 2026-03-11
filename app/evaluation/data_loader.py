"""Historical OHLCV data loader for the evaluation module.

Supports three data sources:
  1. CoinGecko public API  — real market data for ETH, BTC, MATIC/POL, USDC
  2. CSV / Parquet files   — load pre-downloaded datasets
  3. Synthetic generator   — reproducible pseudo-random data for offline use

All sources return a list of :class:`~app.trading.algorithms.Bar` objects
which are directly consumable by the trading algorithms.

Timeframe handling
------------------
``fetch_from_api()`` always fetches a base daily (``1d``) dataset from
CoinGecko and then resamples it to the requested timeframe:

  ``"1d"``  → no resampling (raw daily bars)
  ``"4h"``  → resample 4 daily bars into each output bar  (proxy: 4d/bar)
  ``"1h"``  → resample 24 daily bars into each output bar (proxy: 24d/bar)

This gives genuinely different bar counts across timeframes so that
multi-timeframe analysis receives structurally distinct datasets.

When real historical intraday data is required, supply a pre-downloaded
CSV with the appropriate granularity via :meth:`load_from_csv`.

Supported instruments (CoinGecko coin IDs)
------------------------------------------
  "ETH"   → "ethereum"
  "BTC"   → "bitcoin"
  "MATIC" → "matic-network"
  "POL"   → "matic-network"
  "USDC"  → "usd-coin"
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.trading.algorithms import Bar

logger = logging.getLogger(__name__)

# CoinGecko coin-ID map (lowercase symbol → coingecko id)
COINGECKO_IDS: Dict[str, str] = {
    "eth": "ethereum",
    "btc": "bitcoin",
    "matic": "matic-network",
    "pol": "matic-network",
    "usdc": "usd-coin",
    "weth": "ethereum",
    "wbtc": "bitcoin",
    "wmatic": "matic-network",
}

# CoinGecko OHLC endpoint — no API key required for public data
_CG_OHLC_URL = (
    "https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
    "?vs_currency=usd&days={days}"
)
# Market chart endpoint for higher granularity
_CG_MARKET_URL = (
    "https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    "?vs_currency=usd&days={days}&interval={interval}"
)

_SUPPORTED_TIMEFRAMES: Tuple[str, ...] = ("1h", "4h", "1d")

# Delay between successive CoinGecko API requests to stay within rate limits
_API_RATE_LIMIT_DELAY = 0.10  # seconds

# Resample factors: how many base daily bars to aggregate per output bar.
# 1d → 1 (no resampling), 4h → 4 bars/day proxy, 1h → 24 bars/day proxy.
_TF_RESAMPLE_FACTOR: Dict[str, int] = {
    "1d": 1,
    "4h": 4,
    "1h": 24,
}


class OHLCVLoader:
    """Load OHLCV bars for a given instrument from multiple sources.

    Parameters
    ----------
    timeout:
        HTTP request timeout in seconds.  Defaults to 15.
    """

    def __init__(self, timeout: int = 15) -> None:
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fetch_from_api(
        self,
        symbol: str,
        days: int = 90,
        timeframe: str = "1d",
    ) -> List[Bar]:
        """Download OHLCV bars from the CoinGecko public API.

        Fetches daily-granularity data from CoinGecko and resamples it to the
        requested *timeframe*:

          - ``"1d"`` — one bar per day (no resampling)
          - ``"4h"`` — bars aggregated from groups of 4 daily bars (proxy)
          - ``"1h"`` — bars aggregated from groups of 24 daily bars (proxy)

        This ensures that ``fetch_multi_timeframe()`` returns structurally
        different datasets (different bar counts) for each timeframe, which is
        the prerequisite for meaningful multi-timeframe fusion.

        For high-fidelity intraday data use :meth:`load_from_csv` with a
        pre-downloaded hourly / 4h CSV file.

        Falls back to synthetic data when the network is unavailable.

        Parameters
        ----------
        symbol:
            Asset symbol such as ``"ETH"``, ``"BTC"``, or ``"MATIC"``.
        days:
            Number of historical *daily* bars to fetch before resampling.
        timeframe:
            Target candle timeframe: ``"1d"``, ``"4h"``, or ``"1h"``.

        Returns
        -------
        List[Bar]
            OHLCV bars in ascending timestamp order.
        """
        coin_id = COINGECKO_IDS.get(symbol.lower())
        if coin_id is None:
            logger.warning("[DataLoader] Unknown symbol '%s'. Using synthetic data.", symbol)
            return self._synthetic_for_timeframe(days, timeframe, symbol)

        api_success = False
        try:
            daily_bars = self._fetch_coingecko_ohlc(coin_id, days)
            if daily_bars:
                logger.info(
                    "[DataLoader] Fetched %d daily bars for %s from CoinGecko.",
                    len(daily_bars), symbol,
                )
                api_success = True
        except Exception as exc:
            logger.warning("[DataLoader] CoinGecko API unavailable (%s). Using synthetic data.", exc)
            daily_bars = []

        if not api_success:
            # Offline fallback: generate synthetic daily bars then resample
            daily_bars = self.generate_synthetic(n=days, symbol=symbol)

        return self._resample_to_timeframe(daily_bars, timeframe)

    def _resample_to_timeframe(self, daily_bars: List[Bar], timeframe: str) -> List[Bar]:
        """Resample *daily_bars* to the requested *timeframe*.

        Parameters
        ----------
        daily_bars:
            Source bars (one per day, or finer).
        timeframe:
            Target timeframe: ``"1d"``, ``"4h"``, or ``"1h"``.  Unknown
            timeframes are treated as ``"1d"`` (no resampling).
        """
        factor = _TF_RESAMPLE_FACTOR.get(timeframe, 1)
        if factor <= 1:
            return list(daily_bars)
        return resample_bars(daily_bars, factor)

    def _synthetic_for_timeframe(
        self, days: int, timeframe: str, symbol: str
    ) -> List[Bar]:
        """Generate synthetic bars and resample to *timeframe*."""
        # Generate enough daily bars to produce a reasonable output length
        factor = _TF_RESAMPLE_FACTOR.get(timeframe, 1)
        n_daily = max(days, factor * 10)  # at least 10 output bars
        daily = self.generate_synthetic(n=n_daily, symbol=symbol)
        return self._resample_to_timeframe(daily, timeframe)

    def load_from_csv(self, path: Path, symbol: str = "UNKNOWN") -> List[Bar]:
        """Load OHLCV bars from a CSV file.

        Expected columns (in order or by header):
            timestamp, open, high, low, close, volume

        Parameters
        ----------
        path:
            Path to the CSV file.
        symbol:
            Symbol label attached to each bar (not used in calculations).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")

        bars: List[Bar] = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            lower_headers = [h.lower() for h in headers]
            # Detect column positions
            col_map = {
                "timestamp": _find_col(lower_headers, ["timestamp", "ts", "time", "date", "unix"]),
                "open": _find_col(lower_headers, ["open", "o"]),
                "high": _find_col(lower_headers, ["high", "h"]),
                "low": _find_col(lower_headers, ["low", "l"]),
                "close": _find_col(lower_headers, ["close", "c", "price"]),
                "volume": _find_col(lower_headers, ["volume", "vol", "v"]),
            }
            # Validate required columns before consuming the reader
            required_fields = ["timestamp", "open", "high", "low", "close"]
            missing = [field for field in required_fields if col_map.get(field, -1) < 0]
            if missing:
                raise ValueError(
                    f"Missing required columns in CSV {path}: {', '.join(missing)}"
                )
            for row in reader:
                try:
                    ts = float(row[headers[col_map["timestamp"]]])
                    o = float(row[headers[col_map["open"]]])
                    h = float(row[headers[col_map["high"]]])
                    lo = float(row[headers[col_map["low"]]])
                    c = float(row[headers[col_map["close"]]])
                    vol = float(row[headers[col_map["volume"]]]) if col_map["volume"] >= 0 else 0.0
                    bars.append(Bar(ts, o, h, lo, c, vol))
                except (ValueError, KeyError, IndexError):
                    logger.debug("[DataLoader] Skipping malformed row: %s", row)

        logger.info("[DataLoader] Loaded %d bars from %s.", len(bars), path)
        return bars

    def generate_synthetic(
        self,
        n: int = 365,
        seed: int = 42,
        start_price: float = 1800.0,
        volatility: float = 0.025,
        drift: float = 0.0002,
        symbol: str = "ETH",
    ) -> List[Bar]:
        """Generate synthetic OHLCV bars using geometric Brownian motion.

        Parameters
        ----------
        n:
            Number of bars to generate.
        seed:
            Random seed for reproducibility.
        start_price:
            Starting price.
        volatility:
            Per-bar log-return volatility.
        drift:
            Per-bar log-return drift (positive = bullish trend).
        symbol:
            Symbol label (used only for logging).
        """
        rng = np.random.default_rng(seed)
        log_returns = rng.normal(drift, volatility, n)
        prices = start_price * np.exp(np.cumsum(log_returns))

        bars: List[Bar] = []
        for i, price in enumerate(prices):
            noise = rng.uniform(0.001, 0.015)
            h = price * (1 + noise)
            lo = price * (1 - noise)
            o = price * (1 + rng.uniform(-noise / 2, noise / 2))
            vol = float(rng.uniform(1_000_000, 10_000_000))
            ts = float(i * 86_400)  # daily timestamps
            bars.append(Bar(ts, float(o), float(h), float(lo), float(price), vol))

        logger.debug(
            "[DataLoader] Generated %d synthetic bars for %s (seed=%d).", n, symbol, seed
        )
        return bars

    # ------------------------------------------------------------------
    # Multi-instrument / multi-timeframe fetch
    # ------------------------------------------------------------------

    def fetch_multi(
        self,
        symbols: List[str],
        days: int = 90,
        timeframe: str = "1d",
    ) -> Dict[str, List[Bar]]:
        """Fetch bars for multiple instruments.

        Parameters
        ----------
        symbols:
            List of asset symbols to fetch.
        days:
            Historical window.
        timeframe:
            Candle timeframe.

        Returns
        -------
        Dict[str, List[Bar]]
            Mapping from symbol → list of bars.
        """
        result: Dict[str, List[Bar]] = {}
        for sym in symbols:
            coin_id = COINGECKO_IDS.get(sym.lower())
            result[sym] = self.fetch_from_api(sym, days=days, timeframe=timeframe)
            # Respect rate limits only when a real API endpoint would be used
            if coin_id is not None:
                time.sleep(_API_RATE_LIMIT_DELAY)
        return result

    def fetch_multi_timeframe(
        self,
        symbol: str,
        days: int = 90,
        timeframes: Optional[List[str]] = None,
    ) -> Dict[str, List[Bar]]:
        """Fetch bars for a single instrument across multiple timeframes.

        Parameters
        ----------
        symbol:
            Asset symbol.
        days:
            Historical window.
        timeframes:
            List of timeframes to fetch.  Defaults to ``["1h", "4h", "1d"]``.

        Returns
        -------
        Dict[str, List[Bar]]
            Mapping from timeframe → list of bars.
        """
        if timeframes is None:
            timeframes = ["1h", "4h", "1d"]
        result: Dict[str, List[Bar]] = {}
        for tf in timeframes:
            result[tf] = self.fetch_from_api(symbol, days=days, timeframe=tf)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_coingecko_ohlc(self, coin_id: str, days: int) -> List[Bar]:
        """Fetch OHLC data from CoinGecko.  Raises on network error."""
        import requests  # lazy import — may not be needed in offline mode

        url = _CG_OHLC_URL.format(coin_id=coin_id, days=days)
        resp = requests.get(url, timeout=self._timeout)
        resp.raise_for_status()
        data = resp.json()

        # CoinGecko OHLC response: [[timestamp_ms, open, high, low, close], ...]
        bars: List[Bar] = []
        for entry in data:
            if len(entry) < 5:
                continue
            ts_ms, o, h, lo, c = entry[:5]
            bars.append(Bar(
                timestamp=float(ts_ms) / 1000.0,
                open=float(o),
                high=float(h),
                low=float(lo),
                close=float(c),
                volume=0.0,  # OHLC endpoint doesn't include volume
            ))
        return bars


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _find_col(headers: List[str], candidates: List[str]) -> int:
    """Return the index of the first matching header, or -1."""
    for candidate in candidates:
        if candidate in headers:
            return headers.index(candidate)
    return -1


def resample_bars(bars: List[Bar], factor: int) -> List[Bar]:
    """Aggregate ``bars`` into higher-timeframe bars by grouping *factor* bars.

    For example, resample hourly bars into 4-hour bars by passing
    ``factor=4``.

    Parameters
    ----------
    bars:
        Input bars in ascending timestamp order.
    factor:
        Number of input bars to aggregate into one output bar.

    Returns
    -------
    List[Bar]
        Resampled bars.
    """
    if factor <= 1:
        return list(bars)
    result: List[Bar] = []
    for i in range(0, len(bars) - factor + 1, factor):
        chunk = bars[i: i + factor]
        ts = chunk[0].timestamp
        o = chunk[0].open
        h = max(b.high for b in chunk)
        lo = min(b.low for b in chunk)
        c = chunk[-1].close
        vol = sum(b.volume for b in chunk)
        result.append(Bar(ts, o, h, lo, c, vol))
    return result
