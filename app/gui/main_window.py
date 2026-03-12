"""Multiplex Financials — Main Window (tkinter).

Provides a sophisticated multi-panel windowed frame with:
- Dark theme
- Tabbed notebook: Dashboard, Trading, DeFi, ML, Chat, Settings
- Live metric tiles
- Agentic chat panel
- DeFi panel (Morpho)
- ML training panel
- Trading panel with signal display
"""

from __future__ import annotations

import threading
import time
import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox, scrolledtext, ttk
from typing import Any, Callable, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Theme constants
# ---------------------------------------------------------------------------

DARK_BG = "#0d1117"
DARK_BG2 = "#161b22"
DARK_BG3 = "#21262d"
DARK_BORDER = "#30363d"
ACCENT = "#58a6ff"
ACCENT2 = "#79c0ff"
SUCCESS = "#3fb950"
WARNING = "#d29922"
ERROR = "#f85149"
TEXT_PRIMARY = "#e6edf3"
TEXT_SECONDARY = "#8b949e"
TEXT_MUTED = "#484f58"

FONT_MONO = ("Courier New", 10)
FONT_BODY = ("Segoe UI", 10)
FONT_HEADER = ("Segoe UI", 13, "bold")
FONT_TITLE = ("Segoe UI", 16, "bold")
FONT_SMALL = ("Segoe UI", 9)


# ---------------------------------------------------------------------------
# Reusable widgets
# ---------------------------------------------------------------------------

class DarkFrame(tk.Frame):
    def __init__(self, parent: Any, **kwargs: Any) -> None:
        super().__init__(parent, bg=DARK_BG2, **kwargs)


class MetricTile(tk.Frame):
    """Tile displaying a metric label, value, and optional delta."""

    def __init__(
        self,
        parent: Any,
        label: str,
        value: str = "—",
        delta: str = "",
        delta_positive: bool = True,
    ) -> None:
        super().__init__(parent, bg=DARK_BG3, bd=0, relief="flat",
                         padx=10, pady=8, cursor="hand2")
        self._label_text = label
        tk.Label(self, text=label, bg=DARK_BG3, fg=TEXT_SECONDARY,
                 font=FONT_SMALL).pack(anchor="w")
        self._value_var = tk.StringVar(value=value)
        self._value_lbl = tk.Label(
            self, textvariable=self._value_var, bg=DARK_BG3,
            fg=TEXT_PRIMARY, font=FONT_HEADER
        )
        self._value_lbl.pack(anchor="w")
        self._delta_var = tk.StringVar(value=delta)
        color = SUCCESS if delta_positive else ERROR
        self._delta_lbl = tk.Label(
            self, textvariable=self._delta_var, bg=DARK_BG3,
            fg=color, font=FONT_SMALL
        )
        self._delta_lbl.pack(anchor="w")

    def update(self, value: str, delta: str = "", positive: bool = True) -> None:
        self._value_var.set(value)
        self._delta_var.set(delta)
        self._delta_lbl.config(fg=SUCCESS if positive else ERROR)


class SectionHeader(tk.Label):
    def __init__(self, parent: Any, text: str, **kwargs: Any) -> None:
        super().__init__(
            parent, text=text, bg=DARK_BG2, fg=ACCENT,
            font=FONT_HEADER, anchor="w", **kwargs
        )


class Divider(tk.Frame):
    def __init__(self, parent: Any, **kwargs: Any) -> None:
        super().__init__(parent, bg=DARK_BORDER, height=1, **kwargs)


class RoundedButton(tk.Button):
    def __init__(
        self,
        parent: Any,
        text: str,
        command: Optional[Callable[[], None]] = None,
        color: str = ACCENT,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            parent,
            text=text,
            command=command,
            bg=color,
            fg=DARK_BG,
            activebackground=ACCENT2,
            activeforeground=DARK_BG,
            font=("Segoe UI", 10, "bold"),
            bd=0,
            padx=14,
            pady=6,
            cursor="hand2",
            relief="flat",
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Dashboard Panel
# ---------------------------------------------------------------------------

class DashboardPanel(DarkFrame):
    """Live metrics dashboard."""

    def __init__(self, parent: Any, context: Dict[str, Any]) -> None:
        super().__init__(parent)
        self._ctx = context
        self._tiles: Dict[str, MetricTile] = {}
        self._build()
        self._start_refresh()

    def _build(self) -> None:
        SectionHeader(self, "  📊  Dashboard").pack(fill="x", padx=10, pady=(10, 4))
        Divider(self).pack(fill="x", padx=10, pady=(0, 8))

        # Metric grid
        grid = tk.Frame(self, bg=DARK_BG2)
        grid.pack(fill="x", padx=10, pady=4)

        metrics = [
            ("portfolio_nav", "Portfolio NAV", "$100,000.00", ""),
            ("pnl_today", "PnL Today", "+$0.00", "▲ 0.00%"),
            ("health_factor", "DeFi Health Factor", "∞", ""),
            ("active_agents", "Active Agents", "0", ""),
            ("ml_accuracy", "ML Model Accuracy", "—", ""),
            ("gas_price", "Gas (Gwei)", "—", ""),
            ("matic_price", "MATIC Price", "$0.85", ""),
            ("eth_price", "ETH Price", "$3,200.00", ""),
        ]
        for col, (key, label, val, delta) in enumerate(metrics):
            tile = MetricTile(grid, label, val, delta)
            tile.grid(row=col // 4, column=col % 4, padx=5, pady=5, sticky="ew")
            grid.columnconfigure(col % 4, weight=1)
            self._tiles[key] = tile

        # Market overview
        SectionHeader(self, "  📈  Market Overview").pack(fill="x", padx=10, pady=(12, 4))
        Divider(self).pack(fill="x", padx=10, pady=(0, 4))
        self._market_text = scrolledtext.ScrolledText(
            self, height=5, bg=DARK_BG3, fg=TEXT_PRIMARY,
            font=FONT_MONO, state="disabled", insertbackground=TEXT_PRIMARY
        )
        self._market_text.pack(fill="both", padx=10, pady=4)

        # Alerts
        SectionHeader(self, "  🔔  Recent Alerts").pack(fill="x", padx=10, pady=(8, 4))
        Divider(self).pack(fill="x", padx=10, pady=(0, 4))
        self._alert_text = scrolledtext.ScrolledText(
            self, height=4, bg=DARK_BG3, fg=WARNING,
            font=FONT_MONO, state="disabled", insertbackground=TEXT_PRIMARY
        )
        self._alert_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def _start_refresh(self) -> None:
        self._refresh()

    def _refresh(self) -> None:
        try:
            self._update_tiles()
            self._update_market()
        except Exception:
            pass
        self.after(5000, self._refresh)

    def _update_tiles(self) -> None:
        nav = self._ctx.get("portfolio_nav", 100_000.0)
        pnl = self._ctx.get("pnl_today", 0.0)
        hf = self._ctx.get("health_factor", float("inf"))
        agents = self._ctx.get("active_agents", 0)
        acc = self._ctx.get("ml_accuracy", None)
        gas = self._ctx.get("gas_price", None)
        matic = self._ctx.get("matic_price", 0.85)
        eth = self._ctx.get("eth_price", 3200.0)

        self._tiles["portfolio_nav"].update(f"${nav:,.2f}")
        self._tiles["pnl_today"].update(
            f"{'+'if pnl >= 0 else ''}{pnl:,.2f}",
            f"{'▲' if pnl >= 0 else '▼'} {abs(pnl/max(nav,1)*100):.2f}%",
            positive=pnl >= 0,
        )
        hf_str = f"{hf:.2f}" if hf < 1e10 else "∞"
        hf_color = hf > 1.5
        self._tiles["health_factor"].update(hf_str, positive=hf_color)
        self._tiles["active_agents"].update(str(agents))
        self._tiles["ml_accuracy"].update(
            f"{acc:.2%}" if acc is not None else "—"
        )
        self._tiles["gas_price"].update(
            f"{gas:.1f}" if gas is not None else "—"
        )
        self._tiles["matic_price"].update(f"${matic:.4f}")
        self._tiles["eth_price"].update(f"${eth:,.2f}")

    def _update_market(self) -> None:
        matic = self._ctx.get("matic_price", 0.85)
        eth = self._ctx.get("eth_price", 3200.0)
        btc = self._ctx.get("btc_price", 65000.0)
        ts = time.strftime("%H:%M:%S")
        text = (
            f"[{ts}] MATIC/USDC  {matic:.4f}  |  ETH/USDC  {eth:,.2f}  |  BTC/USDC  {btc:,.2f}\n"
        )
        self._market_text.config(state="normal")
        self._market_text.insert("end", text)
        self._market_text.see("end")
        self._market_text.config(state="disabled")

    def add_alert(self, message: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self._alert_text.config(state="normal")
        self._alert_text.insert("end", f"[{ts}] {message}\n")
        self._alert_text.see("end")
        self._alert_text.config(state="disabled")


# ---------------------------------------------------------------------------
# Trading Panel
# ---------------------------------------------------------------------------

class TradingPanel(DarkFrame):
    """Trading interface: algorithm selector, order form, signals."""

    def __init__(self, parent: Any, context: Dict[str, Any]) -> None:
        super().__init__(parent)
        self._ctx = context
        self._build()

    def _build(self) -> None:
        SectionHeader(self, "  💹  Trading").pack(fill="x", padx=10, pady=(10, 4))
        Divider(self).pack(fill="x", padx=10, pady=(0, 8))

        pane = tk.PanedWindow(self, orient="horizontal", bg=DARK_BG2, sashwidth=4,
                              sashrelief="flat", handlesize=0)
        pane.pack(fill="both", expand=True, padx=10, pady=4)

        # Left: order form
        left = DarkFrame(pane)
        pane.add(left, minsize=260)
        self._build_order_form(left)

        # Right: signals log
        right = DarkFrame(pane)
        pane.add(right, minsize=300)
        self._build_signal_log(right)

    def _build_order_form(self, parent: Any) -> None:
        SectionHeader(parent, "  Order Form").pack(fill="x", padx=6, pady=(6, 2))

        fields: List[tuple[str, str, List[str]]] = [
            ("Symbol", "symbol", ["MATIC/USDC", "ETH/USDC", "BTC/USDC"]),
            ("Algorithm", "algo", ["MeanReversion", "Momentum", "TrendFollowing", "Ensemble"]),
            ("Side", "side", ["Buy", "Sell"]),
            ("Order Type", "order_type", ["Market", "Limit"]),
        ]
        self._form_vars: Dict[str, tk.StringVar] = {}
        for label, key, opts in fields:
            row = tk.Frame(parent, bg=DARK_BG2)
            row.pack(fill="x", padx=6, pady=3)
            tk.Label(row, text=label, bg=DARK_BG2, fg=TEXT_SECONDARY,
                     font=FONT_SMALL, width=12, anchor="w").pack(side="left")
            var = tk.StringVar(value=opts[0])
            cb = ttk.Combobox(row, textvariable=var, values=opts, state="readonly", width=18)
            cb.pack(side="left")
            self._form_vars[key] = var

        # Amount
        amt_row = tk.Frame(parent, bg=DARK_BG2)
        amt_row.pack(fill="x", padx=6, pady=3)
        tk.Label(amt_row, text="Amount", bg=DARK_BG2, fg=TEXT_SECONDARY,
                 font=FONT_SMALL, width=12, anchor="w").pack(side="left")
        self._amount_var = tk.StringVar(value="100")
        tk.Entry(amt_row, textvariable=self._amount_var, bg=DARK_BG3,
                 fg=TEXT_PRIMARY, insertbackground=TEXT_PRIMARY, width=20,
                 relief="flat", font=FONT_BODY).pack(side="left")

        Divider(parent).pack(fill="x", padx=6, pady=8)
        btn_frame = tk.Frame(parent, bg=DARK_BG2)
        btn_frame.pack(fill="x", padx=6, pady=4)
        RoundedButton(btn_frame, "▶ Submit Order", self._submit_order,
                      color=SUCCESS).pack(side="left", padx=4)
        RoundedButton(btn_frame, "⟳ Run Backtest", self._run_backtest,
                      color=ACCENT).pack(side="left", padx=4)

        # Risk preview
        SectionHeader(parent, "  Risk Preview").pack(fill="x", padx=6, pady=(8, 2))
        self._risk_text = tk.Text(parent, height=4, bg=DARK_BG3, fg=TEXT_SECONDARY,
                                  font=FONT_SMALL, state="disabled", relief="flat")
        self._risk_text.pack(fill="x", padx=6, pady=2)
        self._update_risk_preview()

    def _build_signal_log(self, parent: Any) -> None:
        SectionHeader(parent, "  Signal Log").pack(fill="x", padx=6, pady=(6, 2))
        self._signal_log = scrolledtext.ScrolledText(
            parent, bg=DARK_BG3, fg=TEXT_PRIMARY,
            font=FONT_MONO, state="disabled"
        )
        self._signal_log.pack(fill="both", expand=True, padx=6, pady=4)

    def _submit_order(self) -> None:
        sym = self._form_vars["symbol"].get()
        side = self._form_vars["side"].get()
        try:
            amount = float(self._amount_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid amount.")
            return
        msg = f"Order: {side} {amount} {sym} ({self._form_vars['order_type'].get()})"
        self._log_signal(msg, SUCCESS)
        messagebox.showinfo("Order Submitted", msg)

    def _run_backtest(self) -> None:
        from ..trading.algorithms import (
            Bar, MeanReversionAlgo, MomentumAlgo, SignalAggregator, TradingMetrics
        )
        rng = np.random.default_rng(42)
        prices = np.cumsum(rng.normal(0, 0.01, 200)) + 1.0
        prices = np.exp(prices) * 0.85  # MATIC-like
        bars = [
            Bar(float(i), prices[i], prices[i] * 1.005, prices[i] * 0.995,
                prices[i], float(rng.integers(100_000, 1_000_000)))
            for i in range(len(prices))
        ]
        agg = SignalAggregator([MeanReversionAlgo(), MomentumAlgo()])
        signals = []
        for bar in bars:
            sig = agg.on_bar(bar)
            if sig:
                signals.append(sig)
        returns = np.diff(prices) / prices[:-1]
        metrics = TradingMetrics(returns).summary()
        result = (
            f"Backtest ({len(bars)} bars, {len(signals)} signals):\n"
            + "\n".join(f"  {k}: {v:.4f}" for k, v in metrics.items())
        )
        self._log_signal(result, ACCENT)

    def _log_signal(self, msg: str, color: str = TEXT_PRIMARY) -> None:
        ts = time.strftime("%H:%M:%S")
        self._signal_log.config(state="normal")
        self._signal_log.insert("end", f"[{ts}] {msg}\n")
        self._signal_log.see("end")
        self._signal_log.config(state="disabled")

    def _update_risk_preview(self) -> None:
        text = "Risk: 2% per trade  |  Max position: $10,000\nCircuit breaker: 15% drawdown"
        self._risk_text.config(state="normal")
        self._risk_text.delete("1.0", "end")
        self._risk_text.insert("end", text)
        self._risk_text.config(state="disabled")


# ---------------------------------------------------------------------------
# DeFi Panel
# ---------------------------------------------------------------------------

class DeFiPanel(DarkFrame):
    """Morpho DeFi operations: supply, borrow, repay, collateral swap."""

    def __init__(self, parent: Any, context: Dict[str, Any]) -> None:
        super().__init__(parent)
        self._ctx = context
        from ..defi.morpho import MorphoClient
        self._morpho = MorphoClient()
        self._market_var = tk.StringVar(value="WMATIC/USDC")
        self._build()

    def _build(self) -> None:
        SectionHeader(self, "  🏦  DeFi — Morpho on Polygon").pack(fill="x", padx=10, pady=(10, 4))
        Divider(self).pack(fill="x", padx=10, pady=(0, 8))

        # Market selector
        top = tk.Frame(self, bg=DARK_BG2)
        top.pack(fill="x", padx=10, pady=4)
        tk.Label(top, text="Market:", bg=DARK_BG2, fg=TEXT_SECONDARY,
                 font=FONT_SMALL).pack(side="left", padx=4)
        market_cb = ttk.Combobox(
            top, textvariable=self._market_var,
            values=["WMATIC/USDC", "WETH/USDC", "WBTC/USDC"],
            state="readonly", width=18
        )
        market_cb.pack(side="left", padx=4)
        RoundedButton(top, "🔄 Refresh Position", self._refresh_position,
                      color=DARK_BG3).pack(side="left", padx=8)

        # Position summary
        pos_frame = tk.LabelFrame(
            self, text=" Position Summary ", bg=DARK_BG2, fg=ACCENT,
            font=FONT_BODY, bd=1, relief="groove"
        )
        pos_frame.pack(fill="x", padx=10, pady=6)
        self._pos_labels: Dict[str, tk.StringVar] = {}
        for row_i, (key, label) in enumerate([
            ("collateral", "Collateral"),
            ("borrow", "Borrowed"),
            ("health_factor", "Health Factor"),
            ("liquidation_price", "Liquidation Price"),
            ("supply_apy", "Supply APY"),
            ("borrow_apy", "Borrow APY"),
        ]):
            tk.Label(pos_frame, text=f"{label}:", bg=DARK_BG2, fg=TEXT_SECONDARY,
                     font=FONT_SMALL, width=18, anchor="w").grid(
                row=row_i // 2, column=(row_i % 2) * 2, padx=8, pady=2, sticky="w"
            )
            var = tk.StringVar(value="—")
            tk.Label(pos_frame, textvariable=var, bg=DARK_BG2, fg=TEXT_PRIMARY,
                     font=FONT_BODY).grid(
                row=row_i // 2, column=(row_i % 2) * 2 + 1, padx=8, pady=2, sticky="w"
            )
            self._pos_labels[key] = var

        # Operations notebook
        ops = ttk.Notebook(self)
        ops.pack(fill="both", expand=True, padx=10, pady=6)

        supply_tab = DarkFrame(ops)
        borrow_tab = DarkFrame(ops)
        repay_tab = DarkFrame(ops)
        swap_tab = DarkFrame(ops)
        ops.add(supply_tab, text="Supply")
        ops.add(borrow_tab, text="Borrow")
        ops.add(repay_tab, text="Repay")
        ops.add(swap_tab, text="Collateral Swap")

        self._build_op_tab(supply_tab, "Supply Amount", "Supply", self._do_supply)
        self._build_op_tab(borrow_tab, "Borrow Amount", "Borrow", self._do_borrow)
        self._build_op_tab(repay_tab, "Repay Amount", "Repay (blank=full)", self._do_repay)
        self._build_swap_tab(swap_tab)

        # Transaction log
        SectionHeader(self, "  Transaction Log").pack(fill="x", padx=10, pady=(4, 2))
        self._tx_log = scrolledtext.ScrolledText(
            self, height=5, bg=DARK_BG3, fg=TEXT_PRIMARY,
            font=FONT_MONO, state="disabled"
        )
        self._tx_log.pack(fill="both", expand=False, padx=10, pady=(0, 8))

        self._refresh_position()

    def _build_op_tab(
        self, parent: Any, label: str, btn_text: str, cmd: Callable[[], None]
    ) -> None:
        frame = tk.Frame(parent, bg=DARK_BG2)
        frame.pack(padx=20, pady=12)
        tk.Label(frame, text=label + ":", bg=DARK_BG2, fg=TEXT_SECONDARY,
                 font=FONT_SMALL).grid(row=0, column=0, sticky="w", pady=4)
        var = tk.StringVar(value="100")
        entry = tk.Entry(frame, textvariable=var, bg=DARK_BG3, fg=TEXT_PRIMARY,
                         insertbackground=TEXT_PRIMARY, width=20, relief="flat",
                         font=FONT_BODY)
        entry.grid(row=0, column=1, padx=8, pady=4)
        # Attach var to parent so callback can access it
        parent._amount_var = var  # type: ignore[attr-defined]
        RoundedButton(frame, btn_text, cmd, color=ACCENT).grid(
            row=1, column=0, columnspan=2, pady=8
        )

    def _build_swap_tab(self, parent: Any) -> None:
        frame = tk.Frame(parent, bg=DARK_BG2)
        frame.pack(padx=20, pady=12)
        tk.Label(frame, text="Swap (collateral) amount:", bg=DARK_BG2,
                 fg=TEXT_SECONDARY, font=FONT_SMALL).grid(
            row=0, column=0, sticky="w", pady=4
        )
        self._swap_amount_var = tk.StringVar(value="50")
        tk.Entry(frame, textvariable=self._swap_amount_var, bg=DARK_BG3,
                 fg=TEXT_PRIMARY, insertbackground=TEXT_PRIMARY, width=20,
                 relief="flat", font=FONT_BODY).grid(row=0, column=1, padx=8, pady=4)

        dry_frame = tk.Frame(frame, bg=DARK_BG2)
        dry_frame.grid(row=1, column=0, columnspan=2, pady=4)
        self._dry_run_var = tk.BooleanVar(value=True)
        tk.Checkbutton(dry_frame, text="Dry Run", variable=self._dry_run_var,
                       bg=DARK_BG2, fg=TEXT_PRIMARY, selectcolor=DARK_BG3,
                       font=FONT_SMALL).pack(side="left")

        btn_frame = tk.Frame(frame, bg=DARK_BG2)
        btn_frame.grid(row=2, column=0, columnspan=2, pady=8)
        RoundedButton(btn_frame, "⚡ Execute Swap", self._do_swap, color=WARNING).pack(side="left", padx=4)
        RoundedButton(btn_frame, "👁 Preview", lambda: self._do_swap(force_dry=True),
                      color=DARK_BG3).pack(side="left", padx=4)

    def _market_id(self) -> str:
        from ..defi.morpho import KNOWN_MARKETS
        name = self._market_var.get()
        return KNOWN_MARKETS.get(name, list(KNOWN_MARKETS.values())[0])

    def _refresh_position(self) -> None:
        pos = self._morpho.get_position(self._market_id())
        hf = pos.health_factor
        liq_price = self._morpho.liquidation_price(self._market_id())
        self._pos_labels["collateral"].set(f"{pos.collateral:.4f}")
        self._pos_labels["borrow"].set(f"{pos.borrow_shares:.4f}")
        self._pos_labels["health_factor"].set(
            f"{hf:.2f}" if hf < 1e10 else "∞"
        )
        self._pos_labels["liquidation_price"].set(f"${liq_price:.4f}")
        self._pos_labels["supply_apy"].set(f"{pos.supply_apy:.2%}")
        self._pos_labels["borrow_apy"].set(f"{pos.borrow_apy:.2%}")
        self._ctx["health_factor"] = hf

    def _do_supply(self) -> None:
        # Find the currently active tab's amount var
        ops_widget = [w for w in self.winfo_children() if isinstance(w, ttk.Notebook)]
        if not ops_widget:
            return
        nb = ops_widget[0]
        tab_id = nb.select()
        tab_widget = nb.nametowidget(tab_id)
        try:
            amount = float(getattr(tab_widget, "_amount_var", tk.StringVar(value="0")).get())
        except (ValueError, AttributeError):
            amount = 0.0
        result = self._morpho.supply(self._market_id(), amount)
        self._log_tx("Supply", amount, result)
        self._refresh_position()

    def _do_borrow(self) -> None:
        ops_widget = [w for w in self.winfo_children() if isinstance(w, ttk.Notebook)]
        if not ops_widget:
            return
        nb = ops_widget[0]
        tab_id = nb.select()
        tab_widget = nb.nametowidget(tab_id)
        try:
            amount = float(getattr(tab_widget, "_amount_var", tk.StringVar(value="0")).get())
        except (ValueError, AttributeError):
            amount = 0.0
        result = self._morpho.borrow(self._market_id(), amount)
        self._log_tx("Borrow", amount, result)
        self._refresh_position()

    def _do_repay(self) -> None:
        ops_widget = [w for w in self.winfo_children() if isinstance(w, ttk.Notebook)]
        if not ops_widget:
            return
        nb = ops_widget[0]
        tab_id = nb.select()
        tab_widget = nb.nametowidget(tab_id)
        raw = getattr(tab_widget, "_amount_var", tk.StringVar(value="")).get().strip()
        amount = float(raw) if raw else None
        result = self._morpho.repay(self._market_id(), amount)
        self._log_tx("Repay", amount or 0.0, result)
        self._refresh_position()

    def _do_swap(self, force_dry: bool = False) -> None:
        try:
            swap_amount = float(self._swap_amount_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid swap amount.")
            return
        dry = force_dry or self._dry_run_var.get()
        result = self._morpho.collateral_swap(self._market_id(), swap_amount, dry_run=dry)
        if result.get("dry_run"):
            msg = (
                f"DRY RUN Collateral Swap:\n"
                f"  Swap: {result['swap_amount']:.4f}\n"
                f"  Received: {result['received']:.4f}\n"
                f"  Repay: {result['repay_amount']:.4f}\n"
                f"  Projected borrow after: {result['projected_borrow_after']:.4f}"
            )
            messagebox.showinfo("Dry Run Preview", msg)
        elif result.get("success"):
            self._log_tx("CollateralSwap", swap_amount,
                         type("R", (), {"success": True, "tx_hash": "0x…", "error": ""})())
            self._refresh_position()
        else:
            messagebox.showerror("Swap Failed", result.get("error", "Unknown error"))

    def _log_tx(self, op: str, amount: float, result: Any) -> None:
        ts = time.strftime("%H:%M:%S")
        status = "✔" if result.success else "✘"
        color = SUCCESS if result.success else ERROR
        msg = f"[{ts}] {status} {op} {amount:.4f}  tx={getattr(result, 'tx_hash', '')[:12]}\n"
        self._tx_log.config(state="normal")
        self._tx_log.insert("end", msg)
        self._tx_log.see("end")
        self._tx_log.config(state="disabled")


# ---------------------------------------------------------------------------
# ML Panel
# ---------------------------------------------------------------------------

class MLPanel(DarkFrame):
    """ML training, evaluation, and model management."""

    def __init__(self, parent: Any, context: Dict[str, Any]) -> None:
        super().__init__(parent)
        self._ctx = context
        self._training_thread: Optional[threading.Thread] = None
        self._build()

    def _build(self) -> None:
        SectionHeader(self, "  🤖  ML — Supervised Learning").pack(fill="x", padx=10, pady=(10, 4))
        Divider(self).pack(fill="x", padx=10, pady=(0, 8))

        config_frame = tk.LabelFrame(
            self, text=" Training Configuration ", bg=DARK_BG2, fg=ACCENT,
            font=FONT_BODY, bd=1, relief="groove"
        )
        config_frame.pack(fill="x", padx=10, pady=4)

        fields = [
            (
                "Model Type",
                "model_type",
                [
                    "LinearRegression",
                    "RandomForest",
                    "GradientBoosting",
                    "NeuralNetwork",
                    "EquityHealthEnsemble",
                    "Ensemble",
                ],
            ),
            ("HPO Method", "hpo", ["None", "RandomSearch", "BayesianOptimisation"]),
        ]
        self._config_vars: Dict[str, tk.StringVar] = {}
        for row_i, (label, key, opts) in enumerate(fields):
            tk.Label(config_frame, text=label + ":", bg=DARK_BG2, fg=TEXT_SECONDARY,
                     font=FONT_SMALL, width=16, anchor="w").grid(
                row=row_i, column=0, padx=8, pady=4, sticky="w"
            )
            var = tk.StringVar(value=opts[0])
            ttk.Combobox(config_frame, textvariable=var, values=opts,
                         state="readonly", width=24).grid(
                row=row_i, column=1, padx=8, pady=4, sticky="w"
            )
            self._config_vars[key] = var

        # Numeric params
        params_frame = tk.Frame(config_frame, bg=DARK_BG2)
        params_frame.grid(row=2, column=0, columnspan=2, padx=8, pady=4, sticky="w")
        numeric_params = [("Epochs", "epochs", "50"), ("LR", "lr", "0.01"), ("n_samples", "n_samples", "500")]
        self._param_vars: Dict[str, tk.StringVar] = {}
        for col_i, (label, key, default) in enumerate(numeric_params):
            tk.Label(params_frame, text=label + ":", bg=DARK_BG2, fg=TEXT_SECONDARY,
                     font=FONT_SMALL).grid(row=0, column=col_i * 2, padx=4, pady=2, sticky="w")
            var = tk.StringVar(value=default)
            tk.Entry(params_frame, textvariable=var, bg=DARK_BG3, fg=TEXT_PRIMARY,
                     insertbackground=TEXT_PRIMARY, width=8, relief="flat",
                     font=FONT_BODY).grid(row=0, column=col_i * 2 + 1, padx=4, pady=2)
            self._param_vars[key] = var

        # Buttons
        btn_f = tk.Frame(self, bg=DARK_BG2)
        btn_f.pack(fill="x", padx=10, pady=6)
        RoundedButton(btn_f, "▶ Train Model", self._start_training, color=SUCCESS).pack(side="left", padx=4)
        RoundedButton(btn_f, "⏹ Stop", self._stop_training, color=ERROR).pack(side="left", padx=4)

        # Progress
        self._progress_var = tk.DoubleVar(value=0)
        self._progress_bar = ttk.Progressbar(
            self, variable=self._progress_var, maximum=100, mode="determinate"
        )
        self._progress_bar.pack(fill="x", padx=10, pady=4)
        self._progress_label = tk.Label(
            self, text="Ready", bg=DARK_BG2, fg=TEXT_SECONDARY, font=FONT_SMALL
        )
        self._progress_label.pack(anchor="w", padx=12)

        # Metrics
        SectionHeader(self, "  Training Metrics").pack(fill="x", padx=10, pady=(8, 4))
        self._metrics_frame = tk.Frame(self, bg=DARK_BG2)
        self._metrics_frame.pack(fill="x", padx=10)
        self._metric_tiles: Dict[str, MetricTile] = {}
        for col, (key, label) in enumerate([
            ("rmse", "RMSE"), ("mae", "MAE"), ("directional_accuracy", "Dir. Accuracy")
        ]):
            t = MetricTile(self._metrics_frame, label)
            t.grid(row=0, column=col, padx=5, pady=5, sticky="ew")
            self._metrics_frame.columnconfigure(col, weight=1)
            self._metric_tiles[key] = t

        # Training log
        SectionHeader(self, "  Training Log").pack(fill="x", padx=10, pady=(8, 4))
        self._log = scrolledtext.ScrolledText(
            self, bg=DARK_BG3, fg=TEXT_PRIMARY, font=FONT_MONO, state="disabled"
        )
        self._log.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def _log_msg(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self._log.config(state="normal")
        self._log.insert("end", f"[{ts}] {msg}\n")
        self._log.see("end")
        self._log.config(state="disabled")

    def _start_training(self) -> None:
        if self._training_thread and self._training_thread.is_alive():
            messagebox.showinfo("Info", "Training already in progress.")
            return
        self._training_thread = threading.Thread(
            target=self._train_worker, daemon=True
        )
        self._training_thread.start()

    def _stop_training(self) -> None:
        self._log_msg("Stop requested (training will complete current epoch).")

    def _train_worker(self) -> None:
        from ..ml.models import (
            EquityHealthEnsembleModel,
            GradientBoostingModel,
            LinearRegressionModel,
            NeuralNetworkModel,
            RandomForestModel,
        )
        from ..ml.trainer import Trainer

        model_type = self._config_vars["model_type"].get()
        try:
            epochs = int(self._param_vars["epochs"].get())
            lr = float(self._param_vars["lr"].get())
            n = int(self._param_vars["n_samples"].get())
        except ValueError:
            self._log_msg("Invalid parameters.")
            return

        self._log_msg(f"Generating {n} synthetic samples…")
        rng = np.random.default_rng(42)
        X = rng.standard_normal((n, 10)).astype(np.float64)
        y = X @ rng.standard_normal(10) + rng.standard_normal(n) * 0.1
        y = y.astype(np.float64)

        model_map = {
            "LinearRegression": LinearRegressionModel,
            "RandomForest": RandomForestModel,
            "GradientBoosting": GradientBoostingModel,
            "NeuralNetwork": NeuralNetworkModel,
            # Default ensemble members for the equity/health-weighted option.
            "EquityHealthEnsemble": lambda: EquityHealthEnsembleModel(
                [
                    LinearRegressionModel(),
                    RandomForestModel(),
                    GradientBoostingModel(),
                    NeuralNetworkModel(),
                ]
            ),
        }
        factory = model_map.get(model_type)
        if factory is None:
            self._log_msg(f"Model '{model_type}' not yet wired in this demo.")
            return

        model = factory()
        self._log_msg(f"Training {model_type} for {epochs} epochs…")

        def on_progress(epoch: int, total: int, metrics: Dict[str, float]) -> None:
            pct = epoch / total * 100
            self._progress_var.set(pct)
            self._progress_label.config(
                text=f"Epoch {epoch}/{total}  |  RMSE: {metrics.get('rmse', 0):.4f}"
            )
            for k, tile in self._metric_tiles.items():
                v = metrics.get(k, None)
                tile.update(f"{v:.4f}" if v is not None else "—")

        trainer = Trainer(
            model,
            hyperparams={
                "epochs": epochs,
                "learning_rate": lr,
                "patience": max(3, epochs // 10),
            },
            progress_callback=on_progress,
        )
        summary = trainer.train(X, y)
        self._log_msg("Training complete!")
        for k, v in summary.items():
            self._log_msg(f"  {k}: {v}")
        self._ctx["ml_metrics"] = {k: float(v) for k, v in summary.items()
                                   if isinstance(v, (int, float))}
        acc = summary.get("directional_accuracy", None)
        if acc is not None:
            self._ctx["ml_accuracy"] = float(acc)
        self._progress_var.set(100)
        self._progress_label.config(text="Done")


# ---------------------------------------------------------------------------
# Chat Panel
# ---------------------------------------------------------------------------

class ChatPanel(DarkFrame):
    """Agentic chat interface."""

    def __init__(self, parent: Any, context: Dict[str, Any]) -> None:
        super().__init__(parent)
        self._ctx = context
        from ..agents.agents import ChatAgent
        self._agent = ChatAgent(response_callback=None, context=None)
        self._agent._ctx = context  # type: ignore[attr-defined]
        self._build()

    def _build(self) -> None:
        SectionHeader(self, "  💬  Agentic Chat").pack(fill="x", padx=10, pady=(10, 4))
        Divider(self).pack(fill="x", padx=10, pady=(0, 6))

        # Chat history
        self._chat_history = scrolledtext.ScrolledText(
            self, bg=DARK_BG3, fg=TEXT_PRIMARY, font=FONT_BODY,
            state="disabled", wrap="word"
        )
        self._chat_history.pack(fill="both", expand=True, padx=10, pady=4)
        self._chat_history.tag_configure("user", foreground=ACCENT, font=("Segoe UI", 10, "bold"))
        self._chat_history.tag_configure("assistant", foreground=SUCCESS)
        self._chat_history.tag_configure("system", foreground=TEXT_MUTED)
        self._chat_history.tag_configure("ts", foreground=TEXT_MUTED, font=FONT_SMALL)

        # Input area
        input_frame = tk.Frame(self, bg=DARK_BG2)
        input_frame.pack(fill="x", padx=10, pady=(4, 10))
        self._input_var = tk.StringVar()
        self._input_entry = tk.Entry(
            input_frame, textvariable=self._input_var,
            bg=DARK_BG3, fg=TEXT_PRIMARY, insertbackground=TEXT_PRIMARY,
            font=FONT_BODY, relief="flat"
        )
        self._input_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self._input_entry.bind("<Return>", lambda _e: self._send())
        RoundedButton(input_frame, "Send ▶", self._send, color=ACCENT).pack(side="left")

        # Quick action chips
        chips_frame = tk.Frame(self, bg=DARK_BG2)
        chips_frame.pack(fill="x", padx=10, pady=(0, 6))
        for cmd in ["/help", "/status", "/defi status", "/trade signal", "/ml metrics"]:
            btn = tk.Button(
                chips_frame, text=cmd, bg=DARK_BG3, fg=TEXT_SECONDARY,
                font=FONT_SMALL, relief="flat", bd=0, cursor="hand2",
                command=lambda c=cmd: self._send_cmd(c)
            )
            btn.pack(side="left", padx=3, pady=2)

        # Welcome message
        self._append_message(
            "system",
            "Multiplex Financials AI — Type a message or use quick commands above.",
        )

    def _send(self) -> None:
        msg = self._input_var.get().strip()
        if not msg:
            return
        self._input_var.set("")
        self._append_message("user", msg)
        # Process in background thread to keep GUI responsive
        threading.Thread(target=self._process, args=(msg,), daemon=True).start()

    def _send_cmd(self, cmd: str) -> None:
        self._input_var.set(cmd)
        self._send()

    def _process(self, message: str) -> None:
        self._agent.context = type("C", (), {"get": lambda _s, k, d=None: self._ctx.get(k, d)})()
        response = self._agent.process_message(message)
        self.after(0, lambda: self._append_message("assistant", response))

    def _append_message(self, role: str, text: str) -> None:
        ts = time.strftime("%H:%M:%S")
        prefix_map = {"user": "You", "assistant": "AI", "system": "System"}
        prefix = prefix_map.get(role, role)
        self._chat_history.config(state="normal")
        self._chat_history.insert("end", f"[{ts}] ", "ts")
        self._chat_history.insert("end", f"{prefix}: ", role)
        self._chat_history.insert("end", f"{text}\n\n")
        self._chat_history.see("end")
        self._chat_history.config(state="disabled")


# ---------------------------------------------------------------------------
# Settings Panel
# ---------------------------------------------------------------------------

class SettingsPanel(DarkFrame):
    """Configuration and settings."""

    def __init__(self, parent: Any, context: Dict[str, Any]) -> None:
        super().__init__(parent)
        self._ctx = context
        self._build()

    def _build(self) -> None:
        SectionHeader(self, "  ⚙️  Settings").pack(fill="x", padx=10, pady=(10, 4))
        Divider(self).pack(fill="x", padx=10, pady=(0, 8))

        sections = {
            "Network": [
                ("RPC URL (Polygon)", "rpc_url", "https://polygon-rpc.com"),
                ("Chain ID", "chain_id", "137"),
                ("Default Slippage (%)", "slippage", "0.5"),
            ],
            "Wallet": [
                ("Wallet Address", "wallet_address", "0x" + "00" * 20),
                ("Gas Preset", "gas_preset", "fast"),
            ],
            "Trading": [
                ("Max Position ($)", "max_position", "10000"),
                ("Risk Per Trade (%)", "risk_pct", "2.0"),
                ("Max Drawdown (%)", "max_drawdown", "15.0"),
            ],
            "ML": [
                ("Checkpoint Directory", "checkpoint_dir", "checkpoints/"),
                ("Default Epochs", "default_epochs", "50"),
                ("Validation Fraction", "val_fraction", "0.2"),
            ],
        }

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=10, pady=6)
        self._setting_vars: Dict[str, tk.StringVar] = {}

        for section_name, fields in sections.items():
            tab = DarkFrame(notebook)
            notebook.add(tab, text=section_name)
            for row_i, (label, key, default) in enumerate(fields):
                tk.Label(tab, text=label + ":", bg=DARK_BG2, fg=TEXT_SECONDARY,
                         font=FONT_SMALL, width=24, anchor="w").grid(
                    row=row_i, column=0, padx=12, pady=6, sticky="w"
                )
                var = tk.StringVar(value=str(self._ctx.get(key, default)))
                tk.Entry(tab, textvariable=var, bg=DARK_BG3, fg=TEXT_PRIMARY,
                         insertbackground=TEXT_PRIMARY, width=32, relief="flat",
                         font=FONT_BODY).grid(row=row_i, column=1, padx=8, pady=6, sticky="w")
                self._setting_vars[key] = var

        btn_f = tk.Frame(self, bg=DARK_BG2)
        btn_f.pack(fill="x", padx=10, pady=8)
        RoundedButton(btn_f, "💾 Save Settings", self._save, color=SUCCESS).pack(side="left", padx=4)
        RoundedButton(btn_f, "↺ Reset Defaults", self._reset, color=DARK_BG3).pack(side="left", padx=4)

    def _save(self) -> None:
        for key, var in self._setting_vars.items():
            self._ctx[key] = var.get()
        messagebox.showinfo("Settings", "Settings saved successfully.")

    def _reset(self) -> None:
        if messagebox.askyesno("Reset", "Reset all settings to defaults?"):
            for var in self._setting_vars.values():
                var.set("")


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------

class MainWindow:
    """Multiplex Financials main application window.

    Initialises all panels, agents, and the shared context dict,
    then starts the tkinter event loop.
    """

    def __init__(self) -> None:
        self._root = tk.Tk()
        self._ctx: Dict[str, Any] = {
            "portfolio_nav": 100_000.0,
            "pnl_today": 0.0,
            "health_factor": float("inf"),
            "active_agents": 0,
            "matic_price": 0.85,
            "eth_price": 3200.0,
            "btc_price": 65000.0,
        }
        self._configure_root()
        self._apply_theme()
        self._build_header()
        self._build_sidebar()
        self._build_notebook()
        self._build_statusbar()
        self._start_agents()
        self._start_data_simulation()

    # ------------------------------------------------------------------

    def _configure_root(self) -> None:
        self._root.title("Multiplex Financials — DEFI AI Platform")
        self._root.geometry("1440x900")
        self._root.minsize(1280, 800)
        self._root.configure(bg=DARK_BG)
        try:
            self._root.state("zoomed")
        except tk.TclError:
            pass
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _apply_theme(self) -> None:
        style = ttk.Style(self._root)
        style.theme_use("clam")
        style.configure("TNotebook", background=DARK_BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=DARK_BG3, foreground=TEXT_SECONDARY,
                        padding=(12, 6), font=FONT_BODY)
        style.map("TNotebook.Tab",
                  background=[("selected", DARK_BG2)],
                  foreground=[("selected", ACCENT)])
        style.configure("TCombobox", fieldbackground=DARK_BG3, background=DARK_BG3,
                        foreground=TEXT_PRIMARY, selectbackground=ACCENT,
                        arrowcolor=TEXT_PRIMARY)
        style.configure("Horizontal.TProgressbar", troughcolor=DARK_BG3,
                        background=ACCENT, thickness=6)
        style.configure("TScrollbar", background=DARK_BG3, troughcolor=DARK_BG,
                        arrowcolor=TEXT_MUTED, borderwidth=0)

    def _build_header(self) -> None:
        header = tk.Frame(self._root, bg=DARK_BG3, height=52)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        # Logo + title
        logo_frame = tk.Frame(header, bg=DARK_BG3)
        logo_frame.pack(side="left", padx=16)
        tk.Label(logo_frame, text="◈", bg=DARK_BG3, fg=ACCENT,
                 font=("Segoe UI", 22, "bold")).pack(side="left", padx=(0, 6))
        tk.Label(logo_frame, text="Multiplex Financials", bg=DARK_BG3,
                 fg=TEXT_PRIMARY, font=FONT_TITLE).pack(side="left")
        tk.Label(logo_frame, text="DEFI AI Platform  v0.1.0", bg=DARK_BG3,
                 fg=TEXT_MUTED, font=FONT_SMALL).pack(side="left", padx=8)

        # Right: clock, network, wallet
        right = tk.Frame(header, bg=DARK_BG3)
        right.pack(side="right", padx=16)

        self._clock_var = tk.StringVar()
        tk.Label(right, textvariable=self._clock_var, bg=DARK_BG3,
                 fg=TEXT_SECONDARY, font=FONT_SMALL).pack(side="right", padx=12)
        tk.Label(right, text="● Polygon", bg=DARK_BG3, fg=SUCCESS,
                 font=FONT_SMALL).pack(side="right", padx=8)
        self._wallet_var = tk.StringVar(value="0x0000…0000")
        tk.Label(right, textvariable=self._wallet_var, bg=DARK_BG3,
                 fg=TEXT_SECONDARY, font=FONT_SMALL).pack(side="right", padx=8)
        self._update_clock()

    def _update_clock(self) -> None:
        self._clock_var.set(time.strftime("%Y-%m-%d  %H:%M:%S"))
        self._root.after(1000, self._update_clock)

    def _build_sidebar(self) -> None:
        self._sidebar = tk.Frame(self._root, bg=DARK_BG3, width=180)
        self._sidebar.pack(side="left", fill="y")
        self._sidebar.pack_propagate(False)

        tk.Label(self._sidebar, text="Agents", bg=DARK_BG3, fg=TEXT_MUTED,
                 font=FONT_SMALL).pack(anchor="w", padx=12, pady=(12, 2))
        Divider(self._sidebar).pack(fill="x", padx=8)

        self._agent_indicators: Dict[str, tk.Label] = {}
        for agent_name in [
            "OrchestratorAgent", "TradingAgent", "DeFiAgent",
            "MLAgent", "AnalysisAgent", "ChatAgent", "RiskAgent",
        ]:
            row = tk.Frame(self._sidebar, bg=DARK_BG3)
            row.pack(fill="x", padx=8, pady=2)
            dot = tk.Label(row, text="●", bg=DARK_BG3, fg=SUCCESS, font=FONT_SMALL)
            dot.pack(side="left")
            tk.Label(row, text=agent_name.replace("Agent", ""), bg=DARK_BG3,
                     fg=TEXT_SECONDARY, font=FONT_SMALL).pack(side="left", padx=4)
            self._agent_indicators[agent_name] = dot

    def _build_notebook(self) -> None:
        self._notebook = ttk.Notebook(self._root)
        self._notebook.pack(fill="both", expand=True, side="left")

        self._dashboard = DashboardPanel(self._notebook, self._ctx)
        self._trading = TradingPanel(self._notebook, self._ctx)
        self._defi = DeFiPanel(self._notebook, self._ctx)
        self._ml = MLPanel(self._notebook, self._ctx)
        self._chat = ChatPanel(self._notebook, self._ctx)
        self._settings = SettingsPanel(self._notebook, self._ctx)

        self._notebook.add(self._dashboard, text="  📊 Dashboard  ")
        self._notebook.add(self._trading, text="  💹 Trading  ")
        self._notebook.add(self._defi, text="  🏦 DeFi  ")
        self._notebook.add(self._ml, text="  🤖 ML  ")
        self._notebook.add(self._chat, text="  💬 Chat  ")
        self._notebook.add(self._settings, text="  ⚙️ Settings  ")

        # Keyboard shortcuts
        self._root.bind("<Control-t>", lambda _: self._notebook.select(1))
        self._root.bind("<Control-d>", lambda _: self._notebook.select(2))
        self._root.bind("<Control-m>", lambda _: self._notebook.select(3))
        self._root.bind("<Control-c>", lambda _: self._notebook.select(4))

    def _build_statusbar(self) -> None:
        self._statusbar = tk.Frame(self._root, bg=DARK_BG3, height=24)
        self._statusbar.pack(fill="x", side="bottom")
        self._statusbar.pack_propagate(False)

        self._status_var = tk.StringVar(value="System ready.")
        tk.Label(self._statusbar, textvariable=self._status_var, bg=DARK_BG3,
                 fg=TEXT_SECONDARY, font=FONT_SMALL, anchor="w").pack(
            side="left", padx=12
        )
        self._block_var = tk.StringVar(value="Block: —")
        tk.Label(self._statusbar, textvariable=self._block_var, bg=DARK_BG3,
                 fg=TEXT_MUTED, font=FONT_SMALL).pack(side="right", padx=12)
        self._gas_var = tk.StringVar(value="Gas: — Gwei")
        tk.Label(self._statusbar, textvariable=self._gas_var, bg=DARK_BG3,
                 fg=TEXT_MUTED, font=FONT_SMALL).pack(side="right", padx=12)

    def _start_agents(self) -> None:
        from ..agents.agents import (
            AnalysisAgent, ChatAgent, DeFiAgent, MLAgent,
            OrchestratorAgent, RiskAgent, TradingAgent,
        )
        from ..agents.base_agent import AgentContext, AgentRegistry

        ctx = AgentContext()
        registry = AgentRegistry()
        self._orchestrator = OrchestratorAgent(context=ctx)
        sub_agents = [
            TradingAgent(context=ctx),
            DeFiAgent(context=ctx),
            MLAgent(context=ctx),
            AnalysisAgent(context=ctx),
            RiskAgent(context=ctx),
        ]
        for a in sub_agents:
            self._orchestrator.add_agent(a)
            registry.register(a)
        registry.register(self._orchestrator)
        self._orchestrator.run(background=True)
        self._ctx["active_agents"] = len(sub_agents) + 1
        self._status("Agents started.")

    def _start_data_simulation(self) -> None:
        """Simulate live market data updates."""
        self._sim_rng = np.random.default_rng(seed=int(time.time()))
        self._simulate_tick()

    def _simulate_tick(self) -> None:
        rng = self._sim_rng
        self._ctx["matic_price"] = max(0.01, self._ctx.get("matic_price", 0.85) * (
            1 + float(rng.normal(0, 0.002))
        ))
        self._ctx["eth_price"] = max(1.0, self._ctx.get("eth_price", 3200.0) * (
            1 + float(rng.normal(0, 0.001))
        ))
        self._ctx["btc_price"] = max(1.0, self._ctx.get("btc_price", 65000.0) * (
            1 + float(rng.normal(0, 0.001))
        ))
        self._ctx["gas_price"] = max(1.0, float(rng.uniform(30, 80)))
        self._ctx["pnl_today"] += float(rng.normal(0, 5))

        # Update status bar
        gas = self._ctx.get("gas_price", 0.0)
        block = self._ctx.get("block_number", 50_000_000)
        self._ctx["block_number"] = block + 1
        self._gas_var.set(f"Gas: {gas:.1f} Gwei")
        self._block_var.set(f"Block: {block:,}")

        self._root.after(3000, self._simulate_tick)

    def _status(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self._status_var.set(f"[{ts}] {msg}")

    def _on_close(self) -> None:
        if messagebox.askokcancel("Quit", "Shut down Multiplex Financials?"):
            try:
                self._orchestrator.stop()
            except Exception:
                pass
            self._root.destroy()

    def run(self) -> None:
        """Start the tkinter main loop."""
        self._root.mainloop()


def launch() -> None:
    """Entry point: create and run MainWindow."""
    window = MainWindow()
    window.run()
