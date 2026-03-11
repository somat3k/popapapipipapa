#!/usr/bin/env python3
"""Streamlit dashboard for Multiplex Financials."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable

import numpy as np
import pandas as pd
import requests
import streamlit as st


@dataclass(frozen=True)
class MetricCard:
    label: str
    value: str
    delta: str
    hint: str


def set_theme() -> None:
    st.set_page_config(
        page_title="Multiplex Studio",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');

        :root {
            --bg: #0e1117;
            --panel: #111827;
            --panel-soft: #0b1220;
            --border: #1f2937;
            --accent: #3b82f6;
            --muted: #94a3b8;
            --success: #22c55e;
            --warning: #f97316;
        }

        html, body, [class*="css"], [class*="st-"] {
            font-family: 'Roboto', sans-serif;
            font-size: 12.5px;
        }

        .stApp {
            background: var(--bg);
            color: #e2e8f0;
        }

        .block-container {
            max-width: 980px;
            padding-top: 1.25rem;
            padding-bottom: 3rem;
        }

        .card {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 18px 18px 16px 18px;
            box-shadow: 0 16px 32px rgba(2, 6, 23, 0.35);
        }

        .card.soft {
            background: var(--panel-soft);
        }

        .metric-label {
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 10px;
        }

        .metric-value {
            font-size: 20px;
            font-weight: 600;
            margin-top: 8px;
        }

        .metric-delta {
            margin-top: 4px;
            font-size: 11px;
            color: #7dd3fc;
        }

        .section-title {
            font-size: 14px;
            font-weight: 600;
            letter-spacing: 0.02em;
            margin-bottom: 6px;
        }

        .pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(59, 130, 246, 0.15);
            color: #bfdbfe;
            border-radius: 999px;
            padding: 4px 10px;
            font-size: 10px;
        }

        .muted {
            color: var(--muted);
        }

        .header-title {
            font-size: 22px;
            font-weight: 600;
        }

        .header-subtitle {
            font-size: 12px;
            color: var(--muted);
        }

        .login-shell {
            max-width: 420px;
            margin: 0 auto;
        }

        .stTextInput>div>div>input,
        .stSelectbox>div>div>div {
            background: #0f172a;
            border-radius: 12px;
            border: 1px solid #1e293b;
            color: #e2e8f0;
        }

        .stSlider>div>div>div>div {
            background: var(--accent);
        }

        .stButton>button {
            background: var(--accent);
            color: white;
            border-radius: 12px;
            border: none;
            padding: 0.4rem 1rem;
            font-size: 12px;
        }

        .stButton>button:hover {
            background: #2563eb;
        }

        .stAlert {
            border-radius: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    st.session_state.setdefault("authenticated", False)
    st.session_state.setdefault("username", "")
    st.session_state.setdefault("login_error", "")


def login_view() -> None:
    st.markdown(
        """
        <div class="login-shell card">
            <div class="pill">● Secure OAuth Studio</div>
            <div class="header-title" style="margin-top: 12px;">Multiplex Studio</div>
            <div class="header-subtitle">Sign in to access your admin dashboard.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("login-form"):
        username = st.text_input("User ID", value=st.session_state.username)
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Unlock dashboard")

    if submitted:
        if username == "1" and password == "qwerty":
            st.session_state.authenticated = True
            st.session_state.username = username
            st.session_state.login_error = ""
            st.rerun()
        else:
            st.session_state.login_error = "Invalid credentials. Try user 1 / qwerty."

    if st.session_state.login_error:
        st.error(st.session_state.login_error)

    st.caption("OAuth-style access gate for demo. Replace with production OAuth.")


def build_metrics() -> Iterable[MetricCard]:
    return (
        MetricCard("Portfolio NAV", "$1.28M", "+2.1% today", "All strategies"),
        MetricCard("Active Agents", "08", "Stable heartbeat", "All systems"),
        MetricCard("DeFi Health", "1.42", "Safe buffer", "Morpho Blue"),
        MetricCard("Model Accuracy", "93.4%", "+0.7% MoM", "ML inference"),
    )


def generate_timeseries(points: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    base = rng.normal(0.001, 0.01, points).cumsum()
    values = 1 + base
    dates = [datetime.utcnow() - timedelta(hours=points - i) for i in range(points)]
    return pd.DataFrame({"timestamp": dates, "value": values})


def render_line_chart(data: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.plot(data["timestamp"], data["value"], color="#60a5fa", linewidth=2.2)
    ax.fill_between(data["timestamp"], data["value"], color="#1d4ed8", alpha=0.18)
    ax.set_facecolor("#0b1220")
    fig.patch.set_facecolor("#0b1220")
    ax.tick_params(colors="#94a3b8", labelsize=8)
    ax.spines["bottom"].set_color("#1f2937")
    ax.spines["left"].set_color("#1f2937")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(color="#1f2937", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("", color="#94a3b8")
    ax.set_ylabel("", color="#94a3b8")
    st.pyplot(fig, use_container_width=True)


def render_allocation_chart() -> None:
    import matplotlib.pyplot as plt

    labels = ["MATIC", "ETH", "WBTC", "Stable"]
    sizes = [32, 24, 18, 26]
    colors = ["#3b82f6", "#22c55e", "#f97316", "#64748b"]

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.pie(sizes, labels=labels, colors=colors, textprops={"color": "#e2e8f0"})
    ax.set_facecolor("#0b1220")
    fig.patch.set_facecolor("#0b1220")
    st.pyplot(fig, use_container_width=True)


def maybe_send_telegram(token: str, chat_id: str, message: str) -> str:
    if not token or not chat_id:
        return "Telegram token/chat ID missing."

    payload = {"chat_id": chat_id, "text": message}
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json=payload,
            timeout=10,
        )
        if response.ok:
            return "Notification sent."
        return f"Telegram error: {response.status_code}"
    except requests.RequestException as exc:
        return f"Telegram request failed: {exc}"


def dashboard_view() -> None:
    st.sidebar.markdown("## ⚙️ Studio Controls")
    risk_guard = st.sidebar.slider("Risk guard", 0.0, 1.0, 0.65, 0.05)
    refresh_rate = st.sidebar.selectbox(
        "Auto refresh",
        ["Off", "30s", "1m", "5m"],
        index=1,
    )
    preferred_mode = st.sidebar.multiselect(
        "Execution modes",
        ["DeFi", "Spot", "Perps", "ML Signals"],
        default=["DeFi", "ML Signals"],
    )
    st.sidebar.markdown("### Telegram notifications")
    tg_token = st.sidebar.text_input("Bot token", type="password")
    tg_chat_id = st.sidebar.text_input("Chat ID")

    col_header, col_status, col_actions = st.columns([2.2, 1.2, 1])
    with col_header:
        st.markdown(
            "<div class='header-title'>Multiplex Admin Studio</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='header-subtitle'>VSCode-inspired control room for"
            " real-time oversight.</div>",
            unsafe_allow_html=True,
        )
    with col_status:
        st.markdown(
            "<div class='card'>"
            "<div class='section-title'>System status</div>"
            "<div class='muted'>● Agents stable</div>"
            "<div class='muted'>● Streams online</div>"
            "<div class='muted'>● Risk guard: "
            f"{risk_guard:.2f}</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    with col_actions:
        st.markdown(
            "<div class='card'>"
            "<div class='section-title'>Quick actions</div>"
            "<div class='muted'>Trigger calibration</div>"
            "<div class='muted'>Pause strategy</div>"
            "<div class='muted'>Review alerts</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    metric_cols = st.columns(4)
    for column, metric in zip(metric_cols, build_metrics()):
        with column:
            st.markdown(
                f"""
                <div class="card">
                    <div class="metric-label">{metric.label}</div>
                    <div class="metric-value">{metric.value}</div>
                    <div class="metric-delta">{metric.delta}</div>
                    <div class="muted">{metric.hint}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    left_col, right_col = st.columns([1.6, 1])
    with left_col:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Performance heartbeat</div>",
                    unsafe_allow_html=True)
        st.markdown(
            "<div class='muted'>Hourly PnL trend with strategy overlay.</div>",
            unsafe_allow_html=True,
        )
        render_line_chart(generate_timeseries())
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card' style='margin-top:16px;'>",
                    unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Future action planner</div>",
                    unsafe_allow_html=True)
        st.markdown(
            "<div class='muted'>Tune upcoming actions based on requirements.</div>",
            unsafe_allow_html=True,
        )
        st.checkbox("Enable collateral swap automation", value=True)
        st.checkbox("Promote top-performing model", value=False)
        st.checkbox("Throttle high-volatility pairs", value=True)
        st.slider("Target health factor", 1.1, 2.5, 1.6, 0.1)
        st.selectbox("Rebalance cadence", ["Hourly", "Daily", "Weekly"],
                     index=1)
        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Allocation mix</div>",
                    unsafe_allow_html=True)
        render_allocation_chart()
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card' style='margin-top:16px;'>",
                    unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Signal telemetry</div>",
                    unsafe_allow_html=True)
        signals = pd.DataFrame(
            {
                "Stream": ["Mean reversion", "Momentum", "Risk filter"],
                "Score": ["0.71", "0.63", "0.89"],
                "Status": ["Stable", "Watching", "Optimal"],
            }
        )
        st.dataframe(signals, use_container_width=True, height=150)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card' style='margin-top:16px;'>",
                    unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Notifications</div>",
                    unsafe_allow_html=True)
        st.markdown(
            "<div class='muted'>Optional Telegram alert channel.</div>",
            unsafe_allow_html=True,
        )
        if st.button("Send Telegram ping"):
            status = maybe_send_telegram(
                tg_token,
                tg_chat_id,
                "Multiplex Studio: system heartbeat OK.",
            )
            st.info(status)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card' style='margin-top:16px;'>",
                unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Activity feed</div>",
                unsafe_allow_html=True)
    feed = [
        "✅ Risk guard recalibrated for low volatility session.",
        "🟦 Model inference batch processed (latency 82ms).",
        "🔕 Alert throttle engaged for stable markets.",
        "🔵 DeFi position health factor improved to 1.42.",
    ]
    for item in feed:
        st.markdown(f"- {item}")
    st.markdown(
        f"<div class='muted'>Refresh: {refresh_rate} · Modes: "
        f"{', '.join(preferred_mode)}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.caption("Multiplex Studio — mobile-first admin vision in a single,"
               " elegant pane.")


def main() -> None:
    set_theme()
    init_state()

    if not st.session_state.authenticated:
        login_view()
    else:
        dashboard_view()


if __name__ == "__main__":
    main()
