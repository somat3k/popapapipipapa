"""Text-based data visualisations for Morpho Blue market data.

Provides compact ASCII/Unicode charts and tables suitable for display in
terminal output, GUI text widgets (tkinter ScrolledText), or log files.
No external graphics libraries are required.

Available visualisations
------------------------
- ``apy_bar_chart(markets)``            — horizontal bar chart of APYs
- ``utilization_gauge(utilization)``    — single-market utilization meter
- ``position_table(positions)``         — formatted position overview table
- ``opportunity_ranking(scored)``       — ranked opportunity table with labels
- ``health_factor_meter(hf)``           — health factor indicator bar
- ``borrow_capacity_table(caps)``       — borrow capacity summary table
- ``market_summary(market)``            — compact single-market summary
- ``rewards_table(rewards)``            — reward accrual summary table
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

# Maximum width of the rendered output in characters (for bar charts etc.)
DEFAULT_WIDTH: int = 72

# Unicode block characters for smooth bars
_BLOCKS = " ▏▎▍▌▋▊▉█"

# ANSI escape codes (only used in colour-enabled mode)
_RESET = "\033[0m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_BOLD = "\033[1m"
_CYAN = "\033[36m"


def _bar(value: float, max_value: float, width: int = 30, colour: bool = False) -> str:
    """Render a horizontal Unicode bar for *value* / *max_value*."""
    if max_value <= 0:
        return " " * width
    ratio = max(0.0, min(1.0, value / max_value))
    filled = ratio * width
    full_blocks = int(filled)
    remainder = filled - full_blocks
    idx = int(remainder * (len(_BLOCKS) - 1))
    bar = "█" * full_blocks + (_BLOCKS[idx] if full_blocks < width else "")
    bar = bar.ljust(width)
    if colour:
        if ratio >= 0.8:
            bar = _GREEN + bar + _RESET
        elif ratio >= 0.4:
            bar = _YELLOW + bar + _RESET
        else:
            bar = _RED + bar + _RESET
    return bar


def _hr(width: int = DEFAULT_WIDTH, char: str = "─") -> str:
    return char * width


def _center(text: str, width: int = DEFAULT_WIDTH) -> str:
    return text.center(width)


# ---------------------------------------------------------------------------
# Public visualisation functions
# ---------------------------------------------------------------------------

def apy_bar_chart(
    markets: Sequence[Any],
    width: int = DEFAULT_WIDTH,
    colour: bool = False,
) -> str:
    """Horizontal bar chart comparing supply and borrow APYs across markets.

    Parameters
    ----------
    markets:
        Sequence of objects with attributes ``loan_symbol``,
        ``collateral_symbol``, ``supply_apy_pct``, ``borrow_apy_pct``.
        Alternatively, dicts with the same keys are accepted.
    width:
        Total character width of the output.
    colour:
        If True, add ANSI colour codes (green = high, red = low).

    Returns a multi-line string.
    """
    if not markets:
        return "No market data available."

    def _get(obj: Any, key: str, default: Any = 0) -> Any:
        return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)

    bar_width = max(10, width - 40)
    all_apys = [_get(m, "supply_apy_pct", 0) for m in markets] + \
               [_get(m, "borrow_apy_pct", 0) for m in markets]
    max_apy = max(all_apys) if all_apys else 1.0

    lines: List[str] = []
    lines.append(_hr(width))
    lines.append(_center("APY Comparison (% per year)", width))
    lines.append(_hr(width))
    lines.append(f"  {'Market':<22}  {'Supply':>6}  {'Borrow':>6}  {'Bar (supply)'}")
    lines.append(_hr(width, "─"))

    for m in markets:
        loan = _get(m, "loan_symbol", "?")
        coll = _get(m, "collateral_symbol", "?")
        s_apy = _get(m, "supply_apy_pct", 0)
        b_apy = _get(m, "borrow_apy_pct", 0)
        label = f"{coll}/{loan}"[:22].ljust(22)
        bar = _bar(s_apy, max_apy, bar_width, colour=colour)
        lines.append(f"  {label}  {s_apy:>5.2f}%  {b_apy:>5.2f}%  {bar}")

    lines.append(_hr(width))
    return "\n".join(lines)


def utilization_gauge(
    utilization: float,
    market_name: str = "",
    width: int = 40,
    colour: bool = False,
) -> str:
    """Single-line utilization meter.

    Parameters
    ----------
    utilization:  0–1 float (e.g. 0.72 for 72%).
    market_name:  Optional market label.
    width:        Width of the inner bar.
    colour:       ANSI colour output.
    """
    pct = utilization * 100
    bar = _bar(utilization, 1.0, width, colour=colour)
    name_part = f" [{market_name}]" if market_name else ""
    threshold_marker = ""
    if pct >= 90:
        threshold_marker = " ⚠ HIGH"
    elif pct >= 75:
        threshold_marker = " ◆ MED"
    return f"Util{name_part}: |{bar}| {pct:5.1f}%{threshold_marker}"


def position_table(
    positions: Sequence[Any],
    width: int = DEFAULT_WIDTH,
) -> str:
    """Formatted position summary table.

    Parameters
    ----------
    positions:
        Sequence of objects or dicts with keys:
        ``market_key``, ``loan_symbol``, ``collateral_symbol``,
        ``supply_assets``, ``borrow_assets``, ``collateral``,
        ``health_factor``, ``supply_apy``, ``borrow_apy``.
    """
    if not positions:
        return "No positions found."

    def _get(obj: Any, key: str, default: Any = 0) -> Any:
        return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)

    lines: List[str] = []
    lines.append(_hr(width))
    lines.append(_center("Position Overview", width))
    lines.append(_hr(width))
    header = f"  {'Market':<18} {'Supply $':>10} {'Borrow $':>10} {'Collat':>10} {'HF':>6} {'sAPY':>6} {'bAPY':>6}"
    lines.append(header)
    lines.append(_hr(width, "─"))

    for p in positions:
        loan = _get(p, "loan_symbol", "?")
        coll = _get(p, "collateral_symbol", "?")
        label = f"{coll}/{loan}"[:18].ljust(18)
        supply = _get(p, "supply_assets", 0)
        borrow = _get(p, "borrow_assets", 0)
        collateral = _get(p, "collateral", 0)
        hf = _get(p, "health_factor", None)
        s_apy = _get(p, "supply_apy", 0)
        b_apy = _get(p, "borrow_apy", 0)
        hf_str = f"{hf:.2f}" if hf is not None and not math.isinf(hf) else ("∞" if hf == float("inf") else "N/A")
        lines.append(
            f"  {label} {supply:>10.2f} {borrow:>10.2f} {collateral:>10.2f} "
            f"{hf_str:>6} {s_apy * 100:>5.2f}% {b_apy * 100:>5.2f}%"
        )

    lines.append(_hr(width))
    return "\n".join(lines)


def opportunity_ranking(
    scored: Sequence[Any],
    width: int = DEFAULT_WIDTH,
    top_n: Optional[int] = None,
) -> str:
    """Ranked opportunity table.

    Parameters
    ----------
    scored:
        Sequence of :class:`~morpho.opportunity.OpportunityScore` objects or
        equivalent dicts with keys ``loan_symbol``, ``collateral_symbol``,
        ``net_supply_apr_pct``, ``net_borrow_apr_pct``, ``utilization_pct``,
        ``liquidity_usd``, ``score``, ``label``.
    top_n:
        If set, show only the top N entries.
    """
    if not scored:
        return "No opportunities found."

    def _get(obj: Any, key: str, default: Any = 0) -> Any:
        return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)

    entries = list(scored)[:top_n] if top_n else list(scored)

    lines: List[str] = []
    lines.append(_hr(width))
    lines.append(_center("Opportunity Ranking", width))
    lines.append(_hr(width))
    header = (
        f"  {'#':>2} {'Market':<18} {'NetS%':>7} {'NetB%':>7} "
        f"{'Util%':>6} {'Liq$':>10} {'Score':>6} {'Label':<8}"
    )
    lines.append(header)
    lines.append(_hr(width, "─"))

    for rank, opp in enumerate(entries, start=1):
        loan = _get(opp, "loan_symbol", "?")
        coll = _get(opp, "collateral_symbol", "?")
        label_str = _get(opp, "label", "")
        market = f"{coll}/{loan}"[:18].ljust(18)
        net_s = _get(opp, "net_supply_apr_pct", 0)
        net_b = _get(opp, "net_borrow_apr_pct", 0)
        util = _get(opp, "utilization_pct", 0)
        liq = _get(opp, "liquidity_usd", 0)
        score = _get(opp, "score", 0)
        lines.append(
            f"  {rank:>2} {market} {net_s:>6.2f}% {net_b:>6.2f}% "
            f"{util:>5.1f}% {liq:>10,.0f} {score:>6.1f} {label_str:<8}"
        )

    lines.append(_hr(width))
    return "\n".join(lines)


def health_factor_meter(
    hf: float,
    width: int = 40,
    colour: bool = False,
) -> str:
    """Visual health factor indicator bar.

    Parameters
    ----------
    hf:     Health factor value (1.0 = liquidation threshold).
    width:  Width of the inner bar.
    colour: ANSI colour output.
    """
    if math.isinf(hf):
        bar = "█" * width
        label = f" HF: ∞ (no borrow)"
        if colour:
            bar = _GREEN + bar + _RESET
        return f"|{bar}|{label}"

    # Display up to HF = 3 for visual scale
    display_max = 3.0
    ratio = min(hf / display_max, 1.0)
    bar = _bar(ratio, 1.0, width)
    if colour:
        if hf >= 2.0:
            bar = _GREEN + bar + _RESET
        elif hf >= 1.3:
            bar = _YELLOW + bar + _RESET
        else:
            bar = _RED + bar + _RESET

    if hf < 1.0:
        zone = " ⛔ LIQUIDATABLE"
    elif hf < 1.15:
        zone = " ⚠ AT RISK"
    elif hf < 1.5:
        zone = " ◆ WATCH"
    else:
        zone = " ✓ SAFE"

    return f"|{bar}| HF: {hf:.3f}{zone}"


def borrow_capacity_table(
    capacities: Sequence[Any],
    width: int = DEFAULT_WIDTH,
) -> str:
    """Borrow capacity summary table.

    Parameters
    ----------
    capacities:
        Sequence of :class:`~morpho.opportunity.BorrowCapacity` objects or
        dicts with keys ``loan_symbol``, ``collateral_symbol``, ``lltv``,
        ``current_collateral_usd``, ``additional_supply_usd``,
        ``max_borrow_usd``, ``current_borrow_usd``,
        ``safe_additional_borrow_usd``.
    """
    if not capacities:
        return "No borrow capacity data."

    def _get(obj: Any, key: str, default: Any = 0) -> Any:
        return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)

    lines: List[str] = []
    lines.append(_hr(width))
    lines.append(_center("Borrow Capacity (after supply)", width))
    lines.append(_hr(width))
    header = (
        f"  {'Market':<18} {'LLTV':>6} {'Collat$':>10} "
        f"{'MaxBor$':>10} {'CurBor$':>10} {'SafeAdd$':>10}"
    )
    lines.append(header)
    lines.append(_hr(width, "─"))

    for cap in capacities:
        loan = _get(cap, "loan_symbol", "?")
        coll = _get(cap, "collateral_symbol", "?")
        market = f"{coll}/{loan}"[:18].ljust(18)
        lltv = _get(cap, "lltv", 0)
        collateral = _get(cap, "current_collateral_usd", 0)
        max_bor = _get(cap, "max_borrow_usd", 0)
        cur_bor = _get(cap, "current_borrow_usd", 0)
        safe_add = _get(cap, "safe_additional_borrow_usd", 0)
        lines.append(
            f"  {market} {lltv * 100:>5.1f}% {collateral:>10,.2f} "
            f"{max_bor:>10,.2f} {cur_bor:>10,.2f} {safe_add:>10,.2f}"
        )

    lines.append(_hr(width))
    return "\n".join(lines)


def market_summary(market: Any, width: int = DEFAULT_WIDTH) -> str:
    """Compact single-market summary block.

    Parameters
    ----------
    market:
        Object or dict with keys: ``loan_symbol``, ``collateral_symbol``,
        ``lltv``, ``supply_apy_pct``, ``borrow_apy_pct``,
        ``utilization_pct``, ``total_supply_usd``, ``total_borrow_usd``,
        ``liquidity_usd``.
    """
    def _get(obj: Any, key: str, default: Any = 0) -> Any:
        return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)

    loan = _get(market, "loan_symbol", "?")
    coll = _get(market, "collateral_symbol", "?")
    lltv = _get(market, "lltv", 0)
    s_apy = _get(market, "supply_apy_pct", 0)
    b_apy = _get(market, "borrow_apy_pct", 0)
    util = _get(market, "utilization_pct", 0)
    total_s = _get(market, "total_supply_usd", 0)
    total_b = _get(market, "total_borrow_usd", 0)
    liq = _get(market, "liquidity_usd", 0)

    lines: List[str] = []
    lines.append(_hr(width))
    lines.append(_center(f"Market: {coll}/{loan}  (LLTV {lltv * 100:.1f}%)", width))
    lines.append(_hr(width))
    lines.append(f"  Supply APY   : {s_apy:>7.4f}%")
    lines.append(f"  Borrow APY   : {b_apy:>7.4f}%")
    lines.append(f"  Utilization  : {util_gauge_inline(util / 100)}")
    lines.append(f"  Total Supply : ${total_s:>12,.2f}")
    lines.append(f"  Total Borrow : ${total_b:>12,.2f}")
    lines.append(f"  Liquidity    : ${liq:>12,.2f}")
    lines.append(_hr(width))
    return "\n".join(lines)


def util_gauge_inline(utilization: float, width: int = 20) -> str:
    """Inline utilization bar (no market label)."""
    bar = _bar(utilization, 1.0, width)
    return f"|{bar}| {utilization * 100:.1f}%"


def rewards_table(
    rewards: Sequence[Any],
    width: int = DEFAULT_WIDTH,
) -> str:
    """Reward accrual summary table.

    Parameters
    ----------
    rewards:
        Sequence of :class:`~morpho.api.MarketRewards` objects or dicts
        with keys ``market_key``, ``loan_symbol``, ``supply_rewards``,
        ``borrow_rewards``.  Each rewards list contains entries with
        ``asset_symbol``, ``claimable_now``, ``claimable_later``.
    """
    if not rewards:
        return "No reward programs found for this address."

    def _get(obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    lines: List[str] = []
    lines.append(_hr(width))
    lines.append(_center("Reward Programs", width))
    lines.append(_hr(width))

    for mr in rewards:
        loan = _get(mr, "loan_symbol", "?")
        mkey = _get(mr, "market_key", "?")[:12]
        lines.append(f"  Market: {loan}  (key: {mkey}…)")

        supply_rewards = _get(mr, "supply_rewards", []) or []
        borrow_rewards = _get(mr, "borrow_rewards", []) or []

        if supply_rewards:
            lines.append(f"    Supply rewards:")
            for r in supply_rewards:
                sym = _get(r, "asset_symbol", "?")
                now = _get(r, "claimable_now", 0)
                later = _get(r, "claimable_later", 0)
                lines.append(f"      {sym}: claimable now={now:.6f}  later={later:.6f}")
        if borrow_rewards:
            lines.append(f"    Borrow rewards:")
            for r in borrow_rewards:
                sym = _get(r, "asset_symbol", "?")
                now = _get(r, "claimable_now", 0)
                later = _get(r, "claimable_later", 0)
                lines.append(f"      {sym}: claimable now={now:.6f}  later={later:.6f}")
        if not supply_rewards and not borrow_rewards:
            lines.append("    (no active rewards)")
        lines.append("")

    lines.append(_hr(width))
    return "\n".join(lines)
