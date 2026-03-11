#!/usr/bin/env python3
"""Streamlit dashboard for Multiplex Financials."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging
import secrets
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

logger = logging.getLogger(__name__)

DEMO_SEED = 42
# Demo credentials are intentionally weak and only used when demo mode is enabled.
DEMO_USER = "1"
DEMO_PASSWORD = "qwerty"
TELEGRAM_REQUEST_TIMEOUT = 10


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


def resolve_secret(env_key: str, section: str, key: str, fallback: str) -> str:
    """Resolve secrets with environment variables taking priority.

    Args:
        env_key: Environment variable name to check first.
        section: Streamlit secrets section name.
        key: Key within the Streamlit secrets section.
        fallback: Value to return when no secrets are configured.

    Returns:
        The resolved secret value based on env → secrets → fallback precedence.
    """
    env_value = os.getenv(env_key)
    if env_value:
        return env_value
    try:
        section_data = st.secrets.get(section) or {}
        return section_data.get(key, fallback)
    except (AttributeError, KeyError, TypeError, StreamlitSecretNotFoundError):
        return fallback


def is_secret_configured(env_key: str, section: str, key: str) -> bool:
    if os.getenv(env_key):
        return True
    try:
        section_data = st.secrets.get(section) or {}
        return key in section_data
    except (AttributeError, KeyError, TypeError, StreamlitSecretNotFoundError):
        return False


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

    auth_user = resolve_secret(
        "MULTIPLEX_AUTH_USER",
        "auth",
        "user",
        DEMO_USER,
    )
    auth_password = resolve_secret(
        "MULTIPLEX_AUTH_PASSWORD",
        "auth",
        "password",
        DEMO_PASSWORD,
    )
    auth_configured = (
        is_secret_configured("MULTIPLEX_AUTH_USER", "auth", "user")
        and is_secret_configured("MULTIPLEX_AUTH_PASSWORD", "auth", "password")
    )
    demo_mode_enabled = os.getenv("MULTIPLEX_ENABLE_DEMO_MODE", "false").lower() == "true"
    if not auth_configured and not demo_mode_enabled:
        st.error(
            "Demo mode is disabled. Configure MULTIPLEX_AUTH_USER/"
            "MULTIPLEX_AUTH_PASSWORD or Streamlit secrets to continue."
        )
        st.stop()
    demo_mode = demo_mode_enabled and not auth_configured
    if demo_mode:
        st.warning(
            "Demo credentials are active. Configure MULTIPLEX_AUTH_USER/"
            "MULTIPLEX_AUTH_PASSWORD or Streamlit secrets before production."
        )
    query_params = getattr(st, "query_params", {})
    auto_login_value = query_params.get("autologin", "false")
    if isinstance(auto_login_value, list):
        auto_login_value = auto_login_value[0]
    auto_login = str(auto_login_value).lower() == "true"
    if demo_mode and auto_login:
        st.session_state.authenticated = True
        st.session_state.username = DEMO_USER

    def handle_login() -> None:
        username_value = st.session_state.get("login_user", "")
        password_value = st.session_state.get("login_password", "")
        if secrets.compare_digest(username_value, auth_user) and secrets.compare_digest(
            password_value,
            auth_password,
        ):
            st.session_state.authenticated = True
            st.session_state.username = username_value
            st.session_state.login_error = ""
        else:
            st.session_state.login_error = "Invalid credentials. Please try again."

    st.text_input(
        "User ID",
        value=st.session_state.username,
        key="login_user",
    )
    st.text_input(
        "Password",
        type="password",
        key="login_password",
    )
    st.button("Unlock dashboard", on_click=handle_login)

    if (
        not st.session_state.authenticated
        and secrets.compare_digest(st.session_state.get("login_user", ""), auth_user)
        and secrets.compare_digest(
            st.session_state.get("login_password", ""),
            auth_password,
        )
    ):
        st.session_state.authenticated = True
        st.session_state.username = st.session_state.get("login_user", "")
        st.session_state.login_error = ""

    if st.session_state.authenticated:
        st.rerun()

    if st.session_state.login_error:
        st.error(st.session_state.login_error)

    st.caption("OAuth-style access gate for demo. Replace with production OAuth.")


def build_metrics() -> Iterable[MetricCard]:
    """Return demo KPI cards for the dashboard preview."""
    return (
        MetricCard("Portfolio NAV", "$1.28M", "+2.1% today", "All strategies"),
        MetricCard("Active Agents", "08", "Stable heartbeat", "All systems"),
        MetricCard("DeFi Health", "1.42", "Safe buffer", "Morpho Blue"),
        MetricCard("Model Accuracy", "93.4%", "+0.7% MoM", "ML inference"),
    )


def generate_timeseries(points: int = 30) -> pd.DataFrame:
    """Generate synthetic performance data for the demo charts."""
    rng = np.random.default_rng(DEMO_SEED)
    base = rng.normal(0.001, 0.01, points).cumsum()
    values = 1 + base
    now = datetime.now(timezone.utc)
    dates = [now - timedelta(hours=points - i) for i in range(points)]
    return pd.DataFrame({"timestamp": dates, "value": values})


def render_line_chart(data: pd.DataFrame) -> None:
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
    labels = ["MATIC", "ETH", "WBTC", "Stable"]
    sizes = [32, 24, 18, 26]
    colors = ["#3b82f6", "#22c55e", "#f97316", "#64748b"]

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.pie(sizes, labels=labels, colors=colors, textprops={"color": "#e2e8f0"})
    ax.set_facecolor("#0b1220")
    fig.patch.set_facecolor("#0b1220")
    st.pyplot(fig, use_container_width=True)


def send_telegram_notification(token: str, chat_id: str, message: str) -> str:
    """Send a Telegram notification and return a status string."""
    if not token or not chat_id:
        return "Telegram token/chat ID missing."

    payload = {"chat_id": chat_id, "text": message}
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json=payload,
            timeout=TELEGRAM_REQUEST_TIMEOUT,
        )
        if response.ok:
            return "Notification sent."
        return f"Telegram error: {response.status_code}"
    except requests.RequestException as exc:
        logger.warning(
            "Telegram notification failed: %s",
            exc,
            exc_info=True,
        )
        return "Telegram request failed. Verify connectivity and credentials."


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
    tg_token_default = resolve_secret(
        "MULTIPLEX_TG_TOKEN",
        "telegram",
        "bot_token",
        "",
    )
    tg_chat_default = resolve_secret(
        "MULTIPLEX_TG_CHAT_ID",
        "telegram",
        "chat_id",
        "",
    )
    override_tg = st.sidebar.checkbox("Override Telegram credentials")
    if override_tg:
        tg_token = st.sidebar.text_input(
            "Bot token",
            value=tg_token_default,
            type="password",
        )
        tg_chat_id = st.sidebar.text_input("Chat ID", value=tg_chat_default)
    else:
        tg_token = tg_token_default
        tg_chat_id = tg_chat_default
        st.sidebar.caption("Loaded from secrets/environment when available.")

    col_header, col_status, col_actions = st.columns([2.2, 1.2, 1])
    with col_header:
        st.markdown(
            "<div class='header-title'>Multiplex Admin Studio</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='header-subtitle'>VSCode-inspired control room for "
            "real-time oversight.</div>",
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
        st.markdown(
            "<div class='section-title'>Performance heartbeat</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='muted'>Hourly PnL trend with strategy overlay.</div>",
            unsafe_allow_html=True,
        )
        render_line_chart(generate_timeseries())
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            "<div class='card' style='margin-top:16px;'>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='section-title'>Future action planner</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='muted'>Tune upcoming actions based on requirements.</div>",
            unsafe_allow_html=True,
        )
        st.checkbox("Enable collateral swap automation", value=True)
        st.checkbox("Promote top-performing model", value=False)
        st.checkbox("Throttle high-volatility pairs", value=True)
        st.slider("Target health factor", 1.1, 2.5, 1.6, 0.1)
        st.selectbox(
            "Rebalance cadence",
            ["Hourly", "Daily", "Weekly"],
            index=1,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-title'>Allocation mix</div>",
            unsafe_allow_html=True,
        )
        render_allocation_chart()
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            "<div class='card' style='margin-top:16px;'>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='section-title'>Signal telemetry</div>",
            unsafe_allow_html=True,
        )
        signals = pd.DataFrame(
            {
                "Stream": ["Mean reversion", "Momentum", "Risk filter"],
                "Score": ["0.71", "0.63", "0.89"],
                "Status": ["Stable", "Watching", "Optimal"],
            }
        )
        st.dataframe(signals, use_container_width=True, height=150)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            "<div class='card' style='margin-top:16px;'>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='section-title'>Notifications</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='muted'>Optional Telegram alert channel.</div>",
            unsafe_allow_html=True,
        )
        if st.button("Send Telegram ping"):
            status = send_telegram_notification(
                tg_token,
                tg_chat_id,
                "Multiplex Studio: system heartbeat OK.",
            )
            st.info(status)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='card' style='margin-top:16px;'>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='section-title'>Activity feed</div>",
        unsafe_allow_html=True,
    )
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

    st.caption("Multiplex Studio — mobile-first admin vision in a single elegant pane.")


def main() -> None:
    set_theme()
    init_state()

    if not st.session_state.authenticated:
        login_view()
    else:
        dashboard_view()


if __name__ == "__main__":
    main()
