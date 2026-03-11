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
        page_title="AbbiTower Command Center",
        page_icon="🟢",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&family=Roboto+Mono:wght@300;400;600&display=swap');

        :root {
            --bg: #05070b;
            --panel: #0d1117;
            --panel-soft: #0b0f16;
            --border: #1f2937;
            --accent: #b6ff4d;
            --accent-strong: #d1ff7a;
            --muted: #9aa4b2;
            --success: #22c55e;
            --warning: #f59e0b;
            --danger: #ef4444;
        }

        html, body, [class*="css"], [class*="st-"] {
            font-family: 'Roboto', sans-serif;
            font-size: 12px;
        }

        .stApp {
            background: var(--bg);
            color: #e5e7eb;
        }

        .block-container {
            max-width: 1024px;
            padding-top: 1.3rem;
            padding-bottom: 3rem;
            position: relative;
            z-index: 2;
        }

        .bg-grid {
            position: fixed;
            inset: 0;
            background:
                radial-gradient(circle at 20% 20%, rgba(182, 255, 77, 0.12), transparent 40%),
                radial-gradient(circle at 80% 20%, rgba(182, 255, 77, 0.08), transparent 45%),
                radial-gradient(circle at 20% 80%, rgba(148, 163, 184, 0.08), transparent 40%),
                linear-gradient(140deg, rgba(17, 24, 39, 0.85), rgba(5, 7, 11, 0.95));
            z-index: 0;
            pointer-events: none;
            animation: drift 24s ease-in-out infinite;
        }

        .bg-grid::before {
            content: "";
            position: absolute;
            inset: 0;
            background-image:
                linear-gradient(rgba(182, 255, 77, 0.08) 1px, transparent 1px),
                linear-gradient(90deg, rgba(182, 255, 77, 0.06) 1px, transparent 1px);
            background-size: 48px 48px;
            opacity: 0.25;
            animation: scan 16s linear infinite;
        }

        .bg-grid .node {
            position: absolute;
            border: 1px solid rgba(182, 255, 77, 0.25);
            border-radius: 999px;
            box-shadow: 0 0 24px rgba(182, 255, 77, 0.3);
            animation: pulse 6s ease-in-out infinite;
        }

        .bg-grid .node-1 {
            width: 140px;
            height: 140px;
            top: 10%;
            left: 12%;
        }

        .bg-grid .node-2 {
            width: 220px;
            height: 220px;
            top: 55%;
            left: 68%;
            animation-delay: 1.5s;
        }

        .bg-grid .node-3 {
            width: 120px;
            height: 120px;
            top: 72%;
            left: 18%;
            animation-delay: 3s;
        }

        .card {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 18px 18px 16px 18px;
            box-shadow: 0 24px 48px rgba(2, 6, 23, 0.35);
            backdrop-filter: blur(16px);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 28px 60px rgba(2, 6, 23, 0.4);
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
            color: var(--accent-strong);
            font-family: 'Roboto Mono', monospace;
        }

        .metric-delta {
            margin-top: 4px;
            font-size: 11px;
            color: #9efc6f;
        }

        .mono {
            font-family: 'Roboto Mono', monospace;
        }

        .hero-line {
            height: 2px;
            width: 100%;
            margin: 14px 0 6px 0;
            background: linear-gradient(90deg, transparent, rgba(182, 255, 77, 0.6), transparent);
            animation: glow 4s ease-in-out infinite;
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
            background: rgba(182, 255, 77, 0.16);
            color: #c9ff8c;
            border-radius: 999px;
            padding: 4px 10px;
            font-size: 10px;
            border: 1px solid rgba(182, 255, 77, 0.4);
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
            background: #0f141d;
            border-radius: 12px;
            border: 1px solid #1f2937;
            color: #e5e7eb;
        }

        .stSlider>div>div>div>div {
            background: var(--accent);
        }

        .stButton>button {
            background: linear-gradient(135deg, rgba(182, 255, 77, 0.95), rgba(127, 255, 212, 0.9));
            color: #05070b;
            border-radius: 12px;
            border: none;
            padding: 0.4rem 1rem;
            font-size: 12px;
            font-weight: 600;
        }

        .stButton>button:hover {
            background: linear-gradient(135deg, rgba(182, 255, 77, 1), rgba(104, 255, 182, 1));
        }

        .stAlert {
            border-radius: 12px;
        }

        section[data-testid="stSidebar"] {
            background: rgba(7, 10, 15, 0.85);
            border-right: 1px solid rgba(182, 255, 77, 0.2);
        }

        div[data-testid="stProgress"] > div > div > div {
            background: var(--accent);
        }

        .stTabs [data-baseweb="tab"] {
            color: var(--muted);
            font-size: 12px;
        }

        .stTabs [aria-selected="true"] {
            color: var(--accent-strong);
        }

        @keyframes drift {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-12px); }
        }

        @keyframes scan {
            0% { transform: translateY(0); }
            100% { transform: translateY(48px); }
        }

        @keyframes pulse {
            0%, 100% { opacity: 0.4; transform: scale(1); }
            50% { opacity: 0.8; transform: scale(1.06); }
        }

        @keyframes glow {
            0%, 100% { opacity: 0.4; }
            50% { opacity: 1; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="bg-grid">
            <span class="node node-1"></span>
            <span class="node node-2"></span>
            <span class="node node-3"></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    st.session_state.setdefault("authenticated", False)
    st.session_state.setdefault("username", "")
    st.session_state.setdefault("login_error", "")


def has_non_whitespace_content(value: object | None) -> bool:
    if value is None:
        return False
    return str(value).strip() != ""


def resolve_secret(
    env_key: str,
    section: str,
    key: str,
    fallback: str | None,
) -> str:
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
    if has_non_whitespace_content(env_value):
        return str(env_value)
    try:
        section_data = st.secrets.get(section) or {}
        value = section_data.get(key, fallback)
        if value is None:
            return ""
        return value if isinstance(value, str) else str(value)
    except (AttributeError, KeyError, TypeError, StreamlitSecretNotFoundError):
        if fallback is None:
            return ""
        return fallback if isinstance(fallback, str) else str(fallback)


def is_secret_configured(env_key: str, section: str, key: str) -> bool:
    env_value = os.getenv(env_key)
    if has_non_whitespace_content(env_value):
        return True
    try:
        section_data = st.secrets.get(section) or {}
        value = section_data.get(key)
        return has_non_whitespace_content(value)
    except (AttributeError, KeyError, TypeError, StreamlitSecretNotFoundError):
        return False


def login_view() -> None:
    st.markdown(
        """
        <div class="login-shell card">
            <div class="pill">● AbbiTower Secure Gate</div>
            <div class="header-title" style="margin-top: 12px;">AbbiTower Command Center</div>
            <div class="header-subtitle">Secure access to the AI financial operations center.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    demo_mode_enabled = os.getenv("MULTIPLEX_ENABLE_DEMO_MODE", "false").lower() == "true"
    preview_mode_enabled = (
        os.getenv("MULTIPLEX_ENABLE_PREVIEW_MODE", "false").lower() == "true"
    )
    if demo_mode_enabled and not preview_mode_enabled:
        st.error(
            "Demo mode requires preview mode approval. Set "
            "MULTIPLEX_ENABLE_PREVIEW_MODE=true once the production build is "
            "ready for demo access."
        )
        st.stop()
    user_configured = is_secret_configured("MULTIPLEX_AUTH_USER", "auth", "user")
    password_configured = is_secret_configured(
        "MULTIPLEX_AUTH_PASSWORD",
        "auth",
        "password",
    )
    if user_configured != password_configured:
        st.error(
            "Partial authentication configuration detected. Configure both "
            "MULTIPLEX_AUTH_USER and MULTIPLEX_AUTH_PASSWORD (or corresponding "
            "Streamlit secrets), or enable demo mode by setting "
            "MULTIPLEX_ENABLE_DEMO_MODE=true (case-insensitive)."
        )
        st.stop()

    auth_configured = user_configured and password_configured
    if not auth_configured and not demo_mode_enabled:
        st.error(
            "Demo mode is disabled. Configure MULTIPLEX_AUTH_USER/"
            "MULTIPLEX_AUTH_PASSWORD or Streamlit secrets to continue."
        )
        st.stop()

    demo_mode = demo_mode_enabled and not auth_configured
    if demo_mode:
        auth_user = DEMO_USER
        auth_password = DEMO_PASSWORD
        st.warning(
            "Demo credentials are active. Configure MULTIPLEX_AUTH_USER/"
            "MULTIPLEX_AUTH_PASSWORD or Streamlit secrets before production."
        )
    else:
        auth_user = resolve_secret(
            "MULTIPLEX_AUTH_USER",
            "auth",
            "user",
            None,
        )
        auth_password = resolve_secret(
            "MULTIPLEX_AUTH_PASSWORD",
            "auth",
            "password",
            None,
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

    with st.form("login_form"):
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
        submitted = st.form_submit_button("Unlock dashboard")
        if submitted:
            handle_login()

    if st.session_state.authenticated:
        st.rerun()

    if st.session_state.login_error:
        st.error(st.session_state.login_error)

    st.caption(
        "Basic username/password access gate (demo-friendly). Replace with a "
        "production-grade auth mechanism (e.g., OAuth/SSO)."
    )


def build_metrics() -> Iterable[MetricCard]:
    """Return demo KPI cards for the dashboard preview."""
    return (
        MetricCard("Portfolio NAV", "$3.42M", "+3.8% 24h", "Financial growth"),
        MetricCard("Active Models", "14", "2 staging", "Active deployments"),
        MetricCard("Inference SLA", "98.7%", "Stable", "Latency target"),
        MetricCard("DeFi Health Factor", "1.62", "Above target", "Portfolio safety"),
        MetricCard("AI Throughput", "9.4k/s", "+6% WoW", "Signal throughput"),
    )


def generate_timeseries(points: int = 30) -> pd.DataFrame:
    """Generate synthetic performance data for the demo charts."""
    rng = np.random.default_rng(DEMO_SEED)
    base = rng.normal(0.001, 0.01, points).cumsum()
    values = 1 + base
    now = datetime.now(timezone.utc)
    dates = [now - timedelta(hours=points - i) for i in range(points)]
    return pd.DataFrame({"timestamp": dates, "value": values})


def render_line_chart(
    data: pd.DataFrame,
    line_color: str = "#9efc6f",
    fill_color: str = "#1f8b4c",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.plot(data["timestamp"], data["value"], color=line_color, linewidth=2.2)
    ax.fill_between(data["timestamp"], data["value"], color=fill_color, alpha=0.18)
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
    plt.close(fig)


def render_allocation_chart() -> None:
    labels = ["MATIC", "ETH", "WBTC", "Stable"]
    sizes = [32, 24, 18, 26]
    colors = ["#b6ff4d", "#86efac", "#facc15", "#64748b"]

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.pie(sizes, labels=labels, colors=colors, textprops={"color": "#e2e8f0"})
    ax.set_facecolor("#0b1220")
    fig.patch.set_facecolor("#0b1220")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


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
    st.sidebar.markdown("## ⚙️ Ops Controls")
    ops_profile = st.sidebar.selectbox(
        "Ops profile",
        ["Growth", "Balanced", "Shield"],
        index=0,
    )
    risk_guard = st.sidebar.slider("Risk guard", 0.0, 1.0, 0.68, 0.05)
    latency_budget = st.sidebar.slider("Latency budget (ms)", 40, 180, 90, 5)
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
    st.sidebar.markdown("### Model orchestration")
    adaptive_hyperparams = st.sidebar.checkbox(
        "Adaptive hyperparameters",
        value=True,
    )
    auto_heal = st.sidebar.checkbox("Auto-heal pipelines", value=True)
    liquidity_guardian = st.sidebar.checkbox("Liquidity guardian", value=True)
    st.sidebar.progress(0.78)
    st.sidebar.caption("Compute utilization 78%")

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

    ops_focus = {
        "Growth": "Aggressive growth bias",
        "Balanced": "Balanced risk posture",
        "Shield": "Capital preservation mode",
    }[ops_profile]
    if latency_budget <= 80:
        latency_label = "Tight latency budget"
    elif latency_budget <= 120:
        latency_label = "Balanced latency budget"
    else:
        latency_label = "Relaxed latency budget"

    col_header, col_status, col_actions = st.columns([2.2, 1.2, 1])
    with col_header:
        st.markdown(
            "<div class='header-title'>AbbiTower AI Command Center</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='header-subtitle'>Advanced algorithmic intelligence for "
            "financial growth.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='hero-line'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='muted'>Monitoring, orchestration, and AI governance in a "
            "single highly productive view.</div>",
            unsafe_allow_html=True,
        )
    with col_status:
        st.markdown(
            "<div class='card'>"
            "<div class='section-title'>System status</div>"
            "<div class='muted'>● Ops focus: "
            f"{ops_focus}</div>"
            "<div class='muted'>● Risk guard: "
            f"{risk_guard:.2f}</div>"
            "<div class='muted'>● "
            f"{latency_label} ({latency_budget} ms)</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    with col_actions:
        st.markdown(
            "<div class='card'>"
            "<div class='section-title'>Command actions</div>"
            "<div class='muted'>Recalibrate models</div>"
            "<div class='muted'>Deploy shadow version</div>"
            "<div class='muted'>Review guardrails</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    metrics = list(build_metrics())
    metric_cols = st.columns(len(metrics))
    for column, metric in zip(metric_cols, metrics):
        with column:
            st.markdown(
                f"""
                <div class="card">
                    <div class="metric-label">{metric.label}</div>
                    <div class="metric-value mono">{metric.value}</div>
                    <div class="metric-delta">{metric.delta}</div>
                    <div class="muted">{metric.hint}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    overview_tab, model_tab, risk_tab = st.tabs(
        ["Overview", "Model Ops", "Risk & Compliance"]
    )

    with overview_tab:
        left_col, right_col = st.columns([1.6, 1])
        with left_col:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(
                "<div class='section-title'>Growth velocity</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div class='muted'>AI-driven capital momentum and adaptive alpha "
                "(risk-adjusted signal quality).</div>",
                unsafe_allow_html=True,
            )
            render_line_chart(
                generate_timeseries(),
                line_color="#b6ff4d",
                fill_color="#1f8b4c",
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown(
                "<div class='card' style='margin-top:16px;'>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div class='section-title'>Action orchestration</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div class='muted'>Tune next-cycle decisions from the command hub."
                "</div>",
                unsafe_allow_html=True,
            )
            adaptive_liquidity = st.checkbox(
                "Enable adaptive liquidity routing",
                value=True,
            )
            promote_model = st.checkbox("Promote top-performing model", value=True)
            throttle_pairs = st.checkbox(
                "Throttle high-volatility pairs",
                value=True,
            )
            target_confidence = st.slider(
                "Target AI confidence",
                0.6,
                0.98,
                0.84,
                0.02,
            )
            rebalance_cadence = st.selectbox(
                "Rebalance cadence",
                ["Hourly", "Daily", "Weekly"],
                index=1,
            )
            directive_summary = (
                f"{'Adaptive liquidity' if adaptive_liquidity else 'Manual liquidity'} · "
                f"{'Promote models' if promote_model else 'Hold models'} · "
                f"{'Throttle pairs' if throttle_pairs else 'Full speed'}"
            )
            st.markdown(
                "<div class='muted mono'>"
                f"Directives: {directive_summary} · "
                f"Confidence {target_confidence:.2f} · "
                f"{rebalance_cadence} cadence"
                "</div>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with right_col:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(
                "<div class='section-title'>Capital mix</div>",
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
                    "Stream": ["Atlas-Lime", "Momentum-Flow", "Sentinel-Prime"],
                    "Score": ["0.82", "0.71", "0.94"],
                    "Status": ["Optimal", "Stable", "Watching"],
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
                    "✅ AbbiTower heartbeat OK — systems stable.",
                )
                st.info(status)
            st.markdown("</div>", unsafe_allow_html=True)

    with model_tab:
        model_fleet = pd.DataFrame(
            {
                "Model": [
                    "Atlas-Lime-V2",
                    "Quantis-V3",
                    "Sentinel-Prime-V1",
                    "Nova-Pulse-V2",
                ],
                "Stage": ["Production", "Production", "Shadow", "Staging"],
                "Drift": ["0.7%", "1.1%", "2.4%", "0.5%"],
                "Latency": ["82 ms", "95 ms", "101 ms", "76 ms"],
                "Status": ["Optimal", "Stable", "Watch", "Deploying"],
            }
        )
        artifact_vault = pd.DataFrame(
            {
                "Artifact": ["Feature store v8", "Signal bundle 42", "Risk pack Q4"],
                "Owner": ["ML Ops", "Strategy Lab", "Risk Core"],
                "Updated": ["2h ago", "5h ago", "1d ago"],
            }
        )
        col_one, col_two = st.columns([1.4, 1])
        with col_one:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(
                "<div class='section-title'>Model fleet status</div>",
                unsafe_allow_html=True,
            )
            st.dataframe(model_fleet, use_container_width=True, height=220)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown(
                "<div class='card' style='margin-top:16px;'>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div class='section-title'>Artifact vault</div>",
                unsafe_allow_html=True,
            )
            st.dataframe(artifact_vault, use_container_width=True, height=160)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_two:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(
                "<div class='section-title'>Training pipeline</div>",
                unsafe_allow_html=True,
            )
            st.progress(0.78)
            st.markdown(
                "<div class='muted'>Epoch 42/54 • Warm restart scheduled</div>",
                unsafe_allow_html=True,
            )
            st.metric("Inference latency", "82 ms", "-4 ms")
            st.metric("Retraining queue", "3 jobs", "+1 staged")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown(
                "<div class='card' style='margin-top:16px;'>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div class='section-title'>Deployment health</div>",
                unsafe_allow_html=True,
            )
            st.progress(0.92)
            st.markdown(
                "<div class='muted'>Blue/green rollout: 92% stable</div>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

    with risk_tab:
        risk_channels = pd.DataFrame(
            {
                "Channel": ["Liquidity", "Volatility", "Leverage", "Correlation"],
                "Budget": ["$420k", "$280k", "$190k", "$150k"],
                "Utilization": ["62%", "48%", "39%", "52%"],
                "Status": ["Healthy", "Healthy", "Guarded", "Stable"],
            }
        )
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-title'>Risk governance matrix</div>",
            unsafe_allow_html=True,
        )
        st.dataframe(risk_channels, use_container_width=True, height=200)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            "<div class='card' style='margin-top:16px;'>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='section-title'>Compliance pulse</div>",
            unsafe_allow_html=True,
        )
        st.progress(0.88)
        st.markdown(
            "<div class='muted'>Policy checks: 128 passed • 2 under review</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='card' style='margin-top:16px;'>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='section-title'>Activity feed</div>",
        unsafe_allow_html=True,
    )
    orchestration_status = (
        f"⚙️ Ops focus: {ops_focus} · "
        f"Hyperparams {'On' if adaptive_hyperparams else 'Off'} · "
        f"Auto-heal {'On' if auto_heal else 'Off'} · "
        f"Liquidity guardian {'On' if liquidity_guardian else 'Off'}."
    )
    feed = [
        orchestration_status,
        "✅ AbbiTower guardrails recalibrated for macro stability.",
        "🟩 Atlas-Lime-V2 model refreshed with low-latency weights.",
        "🟨 Sentinel-Prime-V1 flagged correlation drift in cross-asset basket.",
        "🟢 Liquidity guardian raised execution confidence to 0.84.",
    ]
    for item in feed:
        st.markdown(f"- {item}")
    st.markdown(
        f"<div class='muted'>Refresh: {refresh_rate} · Modes: "
        f"{', '.join(preferred_mode)}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.caption(
        "AbbiTower — production-grade AI operations for financial growth."
    )


def main() -> None:
    set_theme()
    init_state()

    if not st.session_state.authenticated:
        login_view()
    else:
        dashboard_view()


if __name__ == "__main__":
    main()
