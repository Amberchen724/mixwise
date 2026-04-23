import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from scipy import stats
import io

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MixWise",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap" rel="stylesheet">

<style>
/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d0f14;
    border-right: 1px solid #1e2330;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'DM Serif Display', serif;
    color: #e8eaf0;
}

/* ── Main content ── */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

/* ── Headings ── */
h1, h2, h3 {
    font-family: 'DM Serif Display', serif !important;
    color: #e8eaf0 !important;
}
h4, h5, h6 {
    font-family: 'DM Sans', sans-serif !important;
    color: #b0b4c4 !important;
    font-weight: 500 !important;
}

/* ── Metric cards ── */
.metric-card {
    background: #13161e;
    border: 1px solid #1e2330;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.metric-card .label {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}
.metric-card .value {
    font-family: 'DM Serif Display', serif;
    font-size: 32px;
    color: #e8eaf0;
    line-height: 1.1;
}
.metric-card .sub {
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    color: #5b8dee;
    margin-top: 4px;
}

/* ── Info / surface boxes ── */
.surface-box {
    background: #13161e;
    border: 1px solid #1e2330;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 12px;
}

/* ── Tier banners ── */
.tier-banner {
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 20px;
    display: flex;
    align-items: flex-start;
    gap: 14px;
}
.tier-banner.tier1 { background: rgba(34,197,94,0.08); border: 1px solid rgba(34,197,94,0.3); }
.tier-banner.tier2 { background: rgba(249,115,22,0.08); border: 1px solid rgba(249,115,22,0.3); }
.tier-banner.tier3 { background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.3); }

.tier-pill {
    display: inline-block;
    border-radius: 20px;
    padding: 3px 12px;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
}
.tier-pill.tier1 { background: rgba(34,197,94,0.2); color: #22c55e; border: 1px solid rgba(34,197,94,0.4); }
.tier-pill.tier2 { background: rgba(249,115,22,0.2); color: #f97316; border: 1px solid rgba(249,115,22,0.4); }
.tier-pill.tier3 { background: rgba(239,68,68,0.2); color: #ef4444; border: 1px solid rgba(239,68,68,0.4); }

/* ── Road to MMM cards ── */
.unlock-card {
    background: #13161e;
    border: 1px solid #1e2330;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
    margin: 4px;
}
.unlock-card.done { border-color: rgba(34,197,94,0.4); }
.unlock-card.active { border-color: rgba(249,115,22,0.4); }
.unlock-card.locked { opacity: 0.5; }

/* ── Table styling ── */
.styled-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
}
.styled-table th {
    background: #1a1d28;
    border-bottom: 1px solid #1e2330;
    padding: 10px 14px;
    text-align: left;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.styled-table td {
    padding: 10px 14px;
    border-bottom: 1px solid #1a1d28;
    color: #e8eaf0;
}
.styled-table tr:hover td { background: #161922; }

/* ── Progress mini bar ── */
.mini-bar-bg {
    background: #1e2330;
    border-radius: 4px;
    height: 6px;
    width: 100%;
    margin-top: 4px;
}
.mini-bar-fill {
    background: #5b8dee;
    border-radius: 4px;
    height: 6px;
}

/* ── Buttons ── */
.stButton > button {
    background: #5b8dee !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    padding: 8px 20px !important;
    transition: background 0.15s !important;
}
.stButton > button:hover {
    background: #4a7de0 !important;
}

/* ── Sidebar nav ── */
.nav-item {
    display: block;
    padding: 10px 14px;
    margin: 2px 0;
    border-radius: 8px;
    cursor: pointer;
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    font-weight: 500;
    color: #9ca3af;
    transition: all 0.15s;
    text-decoration: none;
}
.nav-item:hover { background: #1a1d28; color: #e8eaf0; }
.nav-item.active { background: rgba(91,141,238,0.15); color: #5b8dee; border-left: 2px solid #5b8dee; }

/* ── Sidebar status dots ── */
.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
}
.status-dot.on { background: #22c55e; }
.status-dot.off { background: #374151; }

/* ── Checklist ── */
.check-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 8px 0;
    border-bottom: 1px solid #1a1d28;
    font-size: 14px;
}
.check-icon { font-size: 16px; flex-shrink: 0; margin-top: 1px; }
.check-label { color: #e8eaf0; }
.check-tip { color: #6b7280; font-size: 12px; margin-top: 2px; }

/* ── Waterfall chart colors ── */
.wf-positive { color: #22c55e; }
.wf-negative { color: #ef4444; }

/* ── Tooltip ── */
.tooltip-box {
    background: #1a1d28;
    border: 1px solid #1e2330;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 13px;
    color: #b0b4c4;
    margin-top: 8px;
}

/* ── Coming soon overlay ── */
.coming-soon-overlay {
    background: rgba(13,15,20,0.85);
    border-radius: 12px;
    padding: 40px;
    text-align: center;
    border: 1px solid #1e2330;
}
.coming-soon-title {
    font-family: 'DM Serif Display', serif;
    font-size: 28px;
    color: #e8eaf0;
    margin-bottom: 12px;
}

/* ── Monospace data ── */
.mono { font-family: 'DM Mono', monospace; }

/* ── Arrow indicators ── */
.arrow-up { color: #22c55e; }
.arrow-down { color: #ef4444; }

/* ── Section header ── */
.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 22px;
    color: #e8eaf0;
    margin-bottom: 4px;
    margin-top: 24px;
}
.section-sub {
    font-size: 13px;
    color: #6b7280;
    margin-bottom: 20px;
    font-family: 'DM Sans', sans-serif;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# PLOTLY THEME DEFAULTS
# ──────────────────────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0d0f14",
    plot_bgcolor="#13161e",
    font=dict(color="#e8eaf0", family="DM Sans"),
    margin=dict(l=16, r=16, t=40, b=16),
    colorway=["#5b8dee", "#f97316", "#22c55e", "#a855f7", "#06b6d4", "#f59e0b", "#ef4444"],
    xaxis=dict(gridcolor="#1e2330", zerolinecolor="#1e2330", tickfont=dict(color="#9ca3af")),
    yaxis=dict(gridcolor="#1e2330", zerolinecolor="#1e2330", tickfont=dict(color="#9ca3af")),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e2330"),
    hoverlabel=dict(bgcolor="#1a1d28", bordercolor="#1e2330", font=dict(color="#e8eaf0")),
)
CHANNEL_COLORS = {
    "tv_spend": "#5b8dee",
    "paid_search_spend": "#f97316",
    "social_spend": "#22c55e",
    "display_spend": "#a855f7",
}
CHANNEL_LABELS = {
    "tv_spend": "TV",
    "paid_search_spend": "Paid Search",
    "social_spend": "Social",
    "display_spend": "Display",
}


# ──────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ──────────────────────────────────────────────────────────────────────────────
defaults = {
    "page": "Upload & Clean",
    "data_loaded": False,
    "model_run": False,
    "df": None,
    "tier": None,
    "model_results": None,
    "adstock_params": {},
    "transformed_results": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ──────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATA GENERATOR
# ──────────────────────────────────────────────────────────────────────────────
def generate_demo_data() -> pd.DataFrame:
    np.random.seed(42)
    n = 104
    dates = pd.date_range("2023-01-01", periods=n, freq="W")
    t = np.arange(n)

    # Seasonality: sine wave, Q4 peak (around week 44–52)
    seasonality_index = 0.5 + 0.5 * np.sin(2 * np.pi * (t - 10) / 52)

    # Promo flag: ~8 weeks per year
    promo_flag = np.zeros(n, dtype=int)
    promo_weeks = np.random.choice(np.arange(n), size=16, replace=False)
    promo_flag[promo_weeks] = 1

    # TV spend: high variance, some zero weeks
    tv_base = np.random.exponential(30000, n)
    tv_zero = np.random.random(n) < 0.12
    tv_spend = np.where(tv_zero, 0, np.clip(tv_base, 10000, 80000))

    # Paid search: steady 5k–25k
    paid_search_spend = np.random.uniform(5000, 25000, n)

    # Social: 3k–20k, trending up
    social_trend = np.linspace(3000, 20000, n)
    social_spend = np.clip(social_trend + np.random.normal(0, 2000, n), 3000, 20000)

    # Display: low 1k–8k
    display_spend = np.random.uniform(1000, 8000, n)

    # Revenue: correlated with spend + seasonality + noise
    revenue = (
        50000
        + 0.8 * tv_spend
        + 1.5 * paid_search_spend
        + 1.2 * social_spend
        + 0.6 * display_spend
        + 40000 * seasonality_index
        + 20000 * promo_flag
        + np.random.normal(0, 8000, n)
    )
    revenue = np.clip(revenue, 50000, 300000)

    return pd.DataFrame({
        "date": dates,
        "tv_spend": tv_spend.round(2),
        "paid_search_spend": paid_search_spend.round(2),
        "social_spend": social_spend.round(2),
        "display_spend": display_spend.round(2),
        "revenue": revenue.round(2),
        "seasonality_index": seasonality_index.round(4),
        "promo_flag": promo_flag,
    })


# ──────────────────────────────────────────────────────────────────────────────
# DATA MATURITY DETECTOR
# ──────────────────────────────────────────────────────────────────────────────
SPEND_COLUMNS = ["tv_spend", "paid_search_spend", "social_spend", "display_spend"]


def detect_tier(df: pd.DataFrame) -> int:
    spend_cols = [c for c in SPEND_COLUMNS if c in df.columns]
    n_weeks = len(df)
    n_channels = len(spend_cols)

    if n_weeks >= 104 and n_channels >= 3:
        return 1
    elif n_weeks >= 26 or n_channels >= 2:
        return 2
    else:
        return 3


def get_spend_cols(df: pd.DataFrame):
    return [c for c in SPEND_COLUMNS if c in df.columns]


# ──────────────────────────────────────────────────────────────────────────────
# ADSTOCK TRANSFORM
# ──────────────────────────────────────────────────────────────────────────────
def apply_adstock(series: np.ndarray, decay: float) -> np.ndarray:
    result = np.zeros_like(series, dtype=float)
    result[0] = series[0]
    for i in range(1, len(series)):
        result[i] = series[i] + decay * result[i - 1]
    return result


def hill_saturation(x: np.ndarray, alpha: float = 2.0, gamma: float = 0.5) -> np.ndarray:
    return x ** alpha / (x ** alpha + gamma ** alpha)


# ──────────────────────────────────────────────────────────────────────────────
# RUN MODEL
# ──────────────────────────────────────────────────────────────────────────────
def run_model(df, dep_var, channels, controls, ridge=False, adstock_params=None, use_saturation=None):
    X_df = df[channels].copy()

    if adstock_params:
        for ch in channels:
            if ch in adstock_params:
                decay = adstock_params[ch]["decay"]
                sat = adstock_params.get(ch, {}).get("saturation", False)
                transformed = apply_adstock(X_df[ch].values, decay)
                if sat and use_saturation and use_saturation.get(ch, False):
                    max_val = transformed.max() if transformed.max() > 0 else 1
                    transformed = hill_saturation(transformed / max_val) * max_val
                X_df[ch] = transformed

    if controls:
        for c in controls:
            if c in df.columns:
                X_df[c] = df[c].values

    y = df[dep_var].values
    X = X_df.values

    model = Ridge(alpha=1.0) if ridge else LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    n, p = X.shape
    # Compute p-values via OLS t-test (approximate even for Ridge)
    if not ridge and n > p + 1:
        residuals = y - y_pred
        mse = np.sum(residuals ** 2) / (n - p - 1)
        XtX_inv = np.linalg.pinv(X.T @ X)
        se = np.sqrt(mse * np.diag(XtX_inv))
        t_stats = model.coef_ / (se + 1e-10)
        p_values = [2 * (1 - stats.t.cdf(abs(t), df=n - p - 1)) for t in t_stats]
    else:
        p_values = [None] * len(channels)
        if controls:
            p_values = [None] * (len(channels) + len([c for c in controls if c in df.columns]))

    feature_names = list(X_df.columns)
    coef_table = pd.DataFrame({
        "Channel": feature_names,
        "Coefficient": model.coef_,
        "P-Value": p_values,
    })

    return {
        "model": model,
        "coef_table": coef_table,
        "r2": r2,
        "y_pred": y_pred,
        "y_actual": y,
        "intercept": model.intercept_,
        "feature_names": feature_names,
        "channels": channels,
    }


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding: 16px 0 24px 0;">
            <div style="font-family:'DM Serif Display',serif; font-size:26px; color:#5b8dee; letter-spacing:-0.02em;">MixWise</div>
            <div style="font-family:'DM Mono',monospace; font-size:11px; color:#6b7280; margin-top:2px;">Marketing Mix Modeling</div>
        </div>
        """, unsafe_allow_html=True)

        # Status indicators
        dl_dot = "on" if st.session_state.data_loaded else "off"
        mr_dot = "on" if st.session_state.model_run else "off"
        st.markdown(f"""
        <div style="margin-bottom:20px; padding:12px; background:#13161e; border:1px solid #1e2330; border-radius:8px;">
            <div style="font-family:'DM Mono',monospace; font-size:11px; color:#6b7280; margin-bottom:8px; text-transform:uppercase; letter-spacing:0.06em;">Status</div>
            <div style="display:flex; align-items:center; margin-bottom:6px; font-size:13px; color:#9ca3af;">
                <span class="status-dot {dl_dot}"></span>
                {"<span style='color:#22c55e'>Data Loaded</span>" if st.session_state.data_loaded else "Data Loaded"}
            </div>
            <div style="display:flex; align-items:center; font-size:13px; color:#9ca3af;">
                <span class="status-dot {mr_dot}"></span>
                {"<span style='color:#22c55e'>Model Run</span>" if st.session_state.model_run else "Model Run"}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Tier badge
        if st.session_state.tier:
            tier = st.session_state.tier
            tier_info = {
                1: ("#22c55e", "rgba(34,197,94,0.15)", "Tier 1 · Full MMM"),
                2: ("#f97316", "rgba(249,115,22,0.15)", "Tier 2 · Lite MMM"),
                3: ("#ef4444", "rgba(239,68,68,0.15)", "Tier 3 · Incrementality"),
            }
            color, bg, label = tier_info[tier]
            st.markdown(f"""
            <div style="margin-bottom:20px; padding:8px 12px; background:{bg}; border:1px solid {color}40; border-radius:8px;
                        font-family:'DM Mono',monospace; font-size:11px; color:{color}; text-align:center; letter-spacing:0.04em;">
                {label}
            </div>
            """, unsafe_allow_html=True)

        # Navigation
        pages = {
            "Upload & Clean": "📂",
            "Model Builder": "🔬",
            "Priors & Adstock": "⚙️",
            "ROAS Dashboard": "📈",
            "A/B Testing": "🧪",
        }
        if st.session_state.tier == 3:
            pages["Model Builder"] = "🔬"
            model_label = "Lift Calculator"
        else:
            model_label = "Model Builder"

        st.markdown('<div style="font-family:\'DM Mono\',monospace; font-size:11px; color:#6b7280; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:8px;">Navigation</div>', unsafe_allow_html=True)

        page_names = list(pages.keys())
        for page in page_names:
            display_name = model_label if page == "Model Builder" else page
            suffix = " (Lite)" if page == "Model Builder" and st.session_state.tier == 2 else ""
            is_active = st.session_state.page == page
            is_locked = (page in ["Priors & Adstock", "ROAS Dashboard"] and not st.session_state.model_run)

            if is_locked:
                st.markdown(f"""
                <div style="padding:10px 14px; margin:2px 0; border-radius:8px; font-size:14px; color:#374151;
                            font-family:'DM Sans',sans-serif; display:flex; align-items:center; gap:8px;">
                    <span>{pages[page]}</span>
                    <span>{display_name}{suffix}</span>
                    <span style="margin-left:auto; font-size:11px;">🔒</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                active_style = "background:rgba(91,141,238,0.15); color:#5b8dee; border-left:2px solid #5b8dee;" if is_active else "color:#9ca3af;"
                if st.button(f"{pages[page]} {display_name}{suffix}", key=f"nav_{page}", use_container_width=True):
                    st.session_state.page = page
                    st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# TIER BANNER
# ──────────────────────────────────────────────────────────────────────────────
def render_tier_banner(tier: int, n_weeks: int, n_channels: int):
    configs = {
        1: {
            "class": "tier1",
            "pill_class": "tier1",
            "pill": "Tier 1 · Full MMM",
            "title": "Your data is MMM-ready. Full model available.",
            "why": "With 104+ weeks of data and 3+ channels, we have enough statistical power to accurately decompose revenue into channel contributions, measure carry-over effects, and estimate saturation curves.",
        },
        2: {
            "class": "tier2",
            "pill_class": "tier2",
            "pill": "Tier 2 · Lite MMM",
            "title": "Lite MMM mode. Stronger industry priors applied to compensate for limited history.",
            "why": "With fewer than 104 weeks or fewer than 3 channels, the model has less data to learn from. We compensate by applying tighter industry priors on adstock decay and saturation parameters.",
        },
        3: {
            "class": "tier3",
            "pill_class": "tier3",
            "pill": "Tier 3 · Incrementality",
            "title": "Not enough history for MMM. Switched to Incrementality Mode.",
            "why": "Marketing Mix Modeling needs at least 26 weeks of consistent data to be reliable. With fewer weeks, we use a Pre/Post Lift Calculator to estimate campaign effectiveness instead.",
        },
    }
    c = configs[tier]
    st.markdown(f"""
    <div class="tier-banner {c['class']}">
        <div style="flex:1;">
            <div class="tier-pill {c['pill_class']}">{c['pill']}</div>
            <div style="font-size:15px; font-weight:600; color:#e8eaf0; margin:4px 0;">{c['title']}</div>
            <div style="font-size:13px; color:#9ca3af;">
                {n_weeks} weeks · {n_channels} channels detected
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Why does this matter?"):
        st.markdown(f"""<div class="tooltip-box">{c['why']}</div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# ROAD TO MMM CARD
# ──────────────────────────────────────────────────────────────────────────────
def render_road_to_mmm(df: pd.DataFrame, tier: int):
    n_weeks = len(df)
    spend_cols = get_spend_cols(df)
    n_channels = len(spend_cols)

    st.markdown('<div class="section-header">Road to Full MMM</div>', unsafe_allow_html=True)

    # Progress bar
    progress = min(n_weeks / 104, 1.0)
    pct = int(progress * 100)
    milestones = [
        (0, "0w", "#374151"),
        (25, "26w Lite MMM", "#f97316"),
        (100, "104w Full MMM", "#22c55e"),
    ]
    weeks_to_full = max(0, 104 - n_weeks)

    st.markdown(f"""
    <div class="surface-box">
        <div style="display:flex; justify-content:space-between; font-family:'DM Mono',monospace; font-size:11px; color:#6b7280; margin-bottom:6px;">
            <span>Incrementality</span>
            <span>Lite MMM (26w)</span>
            <span>Full MMM (104w)</span>
        </div>
        <div style="background:#1e2330; border-radius:6px; height:12px; position:relative;">
            <div style="background:linear-gradient(90deg, #22c55e 0%, #f97316 50%, #5b8dee 100%);
                        border-radius:6px; height:12px; width:{pct}%; transition:width 0.3s;"></div>
            <div style="position:absolute; top:-4px; left:{pct}%; transform:translateX(-50%);
                        width:20px; height:20px; border-radius:50%; background:#e8eaf0; border:3px solid #5b8dee; z-index:1;"></div>
        </div>
        <div style="text-align:right; font-family:'DM Mono',monospace; font-size:12px; color:#5b8dee; margin-top:8px;">
            {weeks_to_full} more weeks to Full MMM
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Three unlock cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="unlock-card done">
            <div style="font-size:24px; margin-bottom:8px;">✅</div>
            <div style="font-family:'DM Sans',sans-serif; font-weight:600; color:#22c55e; font-size:14px;">Incrementality</div>
            <div style="font-size:12px; color:#6b7280; margin-top:4px;">Always available</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        state = "done" if tier <= 2 else "locked"
        icon = "🟠" if tier == 2 else ("✅" if tier == 1 else "🔒")
        color = "#f97316" if tier == 2 else ("#22c55e" if tier == 1 else "#374151")
        label_color = color
        st.markdown(f"""
        <div class="unlock-card {state}">
            <div style="font-size:24px; margin-bottom:8px;">{icon}</div>
            <div style="font-family:'DM Sans',sans-serif; font-weight:600; color:{label_color}; font-size:14px;">Lite MMM</div>
            <div style="font-size:12px; color:#6b7280; margin-top:4px;">Requires 26+ weeks</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        state = "done" if tier == 1 else "locked"
        icon = "✅" if tier == 1 else "🔒"
        color = "#22c55e" if tier == 1 else "#374151"
        st.markdown(f"""
        <div class="unlock-card {state}">
            <div style="font-size:24px; margin-bottom:8px;">{icon}</div>
            <div style="font-family:'DM Sans',sans-serif; font-weight:600; color:{color}; font-size:14px;">Full MMM</div>
            <div style="font-size:12px; color:#6b7280; margin-top:4px;">Requires 104+ weeks & 3+ channels</div>
        </div>
        """, unsafe_allow_html=True)

    # Tracking checklist
    st.markdown('<div style="margin-top:24px;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header" style="font-size:18px;">Data Checklist</div>', unsafe_allow_html=True)

    checks = [
        (
            any(c in df.columns for c in SPEND_COLUMNS),
            "Weekly spend by channel",
            "Add columns named tv_spend, paid_search_spend, social_spend, or display_spend",
        ),
        (
            "revenue" in df.columns,
            "Weekly revenue or conversions",
            "Add a 'revenue' or 'conversions' column",
        ),
        (
            "promo_flag" in df.columns,
            "Holiday & promotion flags",
            "Add a binary 'promo_flag' column (1 = promo week)",
        ),
        (
            n_channels >= 3,
            "3+ channels",
            f"Currently have {n_channels} channel(s). Add more spend columns.",
        ),
        (
            "seasonality_index" in df.columns,
            "Seasonality index",
            "Add a 'seasonality_index' column (e.g. sine wave or index 0–1)",
        ),
    ]

    for ok, label, tip in checks:
        icon = "✅" if ok else "⬜"
        tip_line = f"<br><span style='color:#6b7280; font-size:12px;'>💡 {tip}</span>" if not ok else ""
        st.markdown(
            f"<div style='display:flex; align-items:flex-start; gap:10px; padding:8px 0; "
            f"border-bottom:1px solid #1a1d28; font-size:14px;'>"
            f"<span style='font-size:16px; flex-shrink:0; margin-top:1px;'>{icon}</span>"
            f"<span style='color:#e8eaf0;'>{label}{tip_line}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 1 — UPLOAD & CLEAN
# ──────────────────────────────────────────────────────────────────────────────
def page_upload():
    st.markdown('<h1 style="margin-bottom:4px;">Upload & Clean</h1>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Load your media spend and revenue data to get started.</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    if uploaded:
        df = pd.read_csv(uploaded)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        demo = False
    else:
        st.markdown("""
        <div style="background:rgba(91,141,238,0.08); border:1px solid rgba(91,141,238,0.3); border-radius:10px;
                    padding:14px 18px; margin-bottom:20px; font-size:14px; color:#5b8dee;">
            ℹ️ No file uploaded — using the built-in demo dataset (104 weeks of synthetic marketing data).
        </div>
        """, unsafe_allow_html=True)
        df = generate_demo_data()
        demo = True

    # Clear any previously-run model results so stale data can't carry over
    if st.session_state.get("df") is None or (uploaded and not demo):
        st.session_state.model_run = False
        st.session_state.model_results = None
        st.session_state.transformed_results = None
        st.session_state.lift_results = None

    st.session_state.df = df
    st.session_state.data_loaded = True
    tier = detect_tier(df)
    st.session_state.tier = tier

    spend_cols = get_spend_cols(df)
    n_weeks = len(df)
    n_channels = len(spend_cols)

    # ── Tier banner
    render_tier_banner(tier, n_weeks, n_channels)

    # ── Road to MMM (Tier 2 and 3)
    if tier in [2, 3]:
        render_road_to_mmm(df, tier)

    # ── Data Quality Report
    st.markdown('<div class="section-header">Data Quality Report</div>', unsafe_allow_html=True)

    # Stats row
    cols = st.columns(4)
    stats_items = [
        ("Rows", f"{len(df):,}"),
        ("Columns", f"{len(df.columns)}"),
        ("Date Range", f"{df['date'].min().strftime('%b %Y') if 'date' in df.columns else 'N/A'} – {df['date'].max().strftime('%b %Y') if 'date' in df.columns else 'N/A'}"),
        ("Null Values", f"{df.isnull().sum().sum():,}"),
    ]
    for col, (label, val) in zip(cols, stats_items):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">{label}</div>
                <div class="value" style="font-size:22px;">{val}</div>
            </div>
            """, unsafe_allow_html=True)

    # Date gap detection
    if "date" in df.columns:
        df_sorted = df.sort_values("date")
        diffs = df_sorted["date"].diff().dropna()
        expected = pd.Timedelta("7 days")
        gaps = diffs[diffs > expected * 1.5]
        if len(gaps) > 0:
            st.markdown(f"""
            <div style="background:rgba(249,115,22,0.08); border:1px solid rgba(249,115,22,0.3); border-radius:8px;
                        padding:12px 16px; margin-bottom:12px; font-size:13px; color:#f97316;">
                ⚠️ {len(gaps)} date gap(s) detected in the time series. Consider filling missing weeks.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:rgba(34,197,94,0.08); border:1px solid rgba(34,197,94,0.3); border-radius:8px;
                        padding:12px 16px; margin-bottom:12px; font-size:13px; color:#22c55e;">
                ✅ No date gaps detected. Time series is continuous.
            </div>
            """, unsafe_allow_html=True)

    # Null counts table
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        st.markdown("**Null counts by column:**")
        null_df = null_counts[null_counts > 0].reset_index()
        null_df.columns = ["Column", "Null Count"]
        st.dataframe(null_df, use_container_width=True, hide_index=True)

    # Preview table
    st.markdown('<div class="section-header" style="font-size:18px;">Data Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)

    # ── Correlation heatmap
    st.markdown('<div class="section-header" style="font-size:18px;">Correlation Heatmap</div>', unsafe_allow_html=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale=[[0, "#ef4444"], [0.5, "#1e2330"], [1, "#5b8dee"]],
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            showscale=True,
            colorbar=dict(tickfont=dict(color="#9ca3af")),
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=420, title="Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)

    # ── Data cleaning buttons
    st.markdown('<div class="section-header" style="font-size:18px;">Data Cleaning</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Drop Nulls", use_container_width=True):
            before = len(st.session_state.df)
            st.session_state.df = st.session_state.df.dropna()
            after = len(st.session_state.df)
            st.success(f"Dropped {before - after} rows with nulls. {after} rows remain.")
    with c2:
        if st.button("Fill with Median", use_container_width=True):
            num_cols = st.session_state.df.select_dtypes(include=[np.number]).columns
            st.session_state.df[num_cols] = st.session_state.df[num_cols].fillna(st.session_state.df[num_cols].median())
            st.success("Filled numeric nulls with column medians.")
    with c3:
        if st.button("Normalize Spend (Min-Max)", use_container_width=True):
            spend_cols_present = get_spend_cols(st.session_state.df)
            if spend_cols_present:
                scaler = MinMaxScaler()
                st.session_state.df[spend_cols_present] = scaler.fit_transform(st.session_state.df[spend_cols_present])
                st.success(f"Normalized {len(spend_cols_present)} spend columns to [0, 1].")
            else:
                st.warning("No spend columns found to normalize.")


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 2 — MODEL BUILDER / LIFT CALCULATOR
# ──────────────────────────────────────────────────────────────────────────────
def page_model_builder():
    if not st.session_state.data_loaded:
        st.warning("Please upload or load data first on the Upload & Clean page.")
        return

    df = st.session_state.df
    tier = st.session_state.tier

    if tier == 3:
        # LIFT CALCULATOR
        st.markdown('<h1 style="margin-bottom:4px;">Lift Calculator</h1>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Estimate campaign effectiveness using Pre/Post analysis.</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="tier-banner tier3">
            <div>
                <div class="tier-pill tier3">Incrementality Mode</div>
                <div style="font-size:14px; color:#9ca3af; margin-top:4px;">
                    Not enough historical data for MMM. Use this calculator to measure lift from individual campaigns.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            campaign_start = st.date_input("Campaign Start Date", value=pd.Timestamp("2023-06-01"))
            kpi_before = st.number_input("KPI Average (Pre-campaign period)", min_value=0.0, value=50000.0, step=1000.0)
        with col2:
            campaign_name = st.text_input("Campaign Name", value="Summer Campaign")
            kpi_after = st.number_input("KPI Average (During/Post-campaign period)", min_value=0.0, value=65000.0, step=1000.0)

        n_before = st.slider("Number of weeks in pre-period", 2, 26, 8)
        n_after = st.slider("Number of weeks in post-period", 2, 26, 4)

        if st.button("Calculate Lift", use_container_width=False):
            if kpi_before > 0 and kpi_after >= 0:
                lift_pct = ((kpi_after - kpi_before) / kpi_before) * 100
                std_before = kpi_before * 0.15
                std_after = kpi_after * 0.15
                t_stat, p_val = stats.ttest_ind_from_stats(
                    mean1=kpi_before, std1=std_before, nobs1=n_before,
                    mean2=kpi_after, std2=std_after, nobs2=n_after,
                )
                st.session_state.lift_results = {
                    "lift_pct": lift_pct, "p_val": p_val,
                    "t_stat": t_stat, "campaign_name": campaign_name,
                }
                st.session_state.model_run = True
                st.rerun()
            else:
                st.error("KPI before must be greater than 0.")

        # ── Always render lift results when available
        lift_res = st.session_state.get("lift_results")
        if lift_res:
            lift_pct = lift_res["lift_pct"]
            p_val = lift_res["p_val"]
            t_stat = lift_res["t_stat"]
            campaign_name_res = lift_res["campaign_name"]

            st.markdown('<div class="section-header">Results</div>', unsafe_allow_html=True)
            r1, r2, r3 = st.columns(3)
            with r1:
                st.metric("Lift", f"{lift_pct:+.1f}%")
            with r2:
                st.metric("P-Value", f"{p_val:.4f}",
                          delta="Significant" if p_val < 0.05 else "Not significant")
            with r3:
                st.metric("T-Statistic", f"{t_stat:.3f}")

            if p_val < 0.05:
                interpretation = (
                    f"The {campaign_name_res} drove a statistically significant {lift_pct:.1f}% increase in your KPI. The result is unlikely to be due to chance (p={p_val:.4f})."
                    if lift_pct > 0 else
                    f"The {campaign_name_res} was associated with a statistically significant {abs(lift_pct):.1f}% decrease in your KPI. This is worth investigating (p={p_val:.4f})."
                )
            else:
                interpretation = f"We cannot conclude that the {campaign_name_res} had a significant effect on your KPI. The {lift_pct:+.1f}% change could be due to random variation (p={p_val:.4f}). Consider running the campaign longer or with a larger sample."

            st.markdown(f"""
            <div style="background:rgba(91,141,238,0.08); border:1px solid rgba(91,141,238,0.3); border-radius:10px;
                        padding:18px 22px; margin-top:16px;">
                <div style="font-family:'DM Mono',monospace; font-size:11px; color:#5b8dee; margin-bottom:8px; text-transform:uppercase; letter-spacing:0.06em;">Plain English</div>
                <div style="font-size:15px; color:#e8eaf0; line-height:1.6;">{interpretation}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        # FULL / LITE MODEL BUILDER
        title = "Model Builder"
        if tier == 2:
            title += " (Lite)"
        st.markdown(f'<h1 style="margin-bottom:4px;">{title}</h1>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Configure and run your Marketing Mix Model.</div>', unsafe_allow_html=True)

        if tier == 2:
            st.markdown("""
            <div class="tier-banner tier2">
                <div>
                    <div class="tier-pill tier2">Lite MMM</div>
                    <div style="font-size:14px; color:#9ca3af; margin-top:4px;">
                        Stronger industry priors are applied. Results are directionally valid but less precise than Full MMM.
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        spend_cols = get_spend_cols(df)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_spend = [c for c in numeric_cols if c not in spend_cols and c != "revenue"]

        dep_var = "revenue" if "revenue" in df.columns else (numeric_cols[0] if numeric_cols else None)
        st.markdown(f'<div style="font-family:\'DM Mono\',monospace; font-size:12px; color:#6b7280; margin-bottom:4px;">DEPENDENT VARIABLE</div><div style="background:#13161e; border:1px solid #1e2330; border-radius:8px; padding:10px 14px; font-size:14px; color:#5b8dee; margin-bottom:16px; font-family:\'DM Sans\',sans-serif;">{dep_var}</div>', unsafe_allow_html=True)
        model_type = st.radio("Model Type", ["OLS (Linear Regression)", "Ridge Regression"], horizontal=True)

        channels = st.multiselect("Media Channels", spend_cols, default=spend_cols)
        controls = st.multiselect("Control Variables (optional)", non_spend, default=[c for c in ["seasonality_index", "promo_flag"] if c in non_spend])

        if st.button("Run Model", use_container_width=False):
            if not channels:
                st.error("Select at least one media channel.")
            else:
                ridge = "Ridge" in model_type
                with st.spinner("Fitting model..."):
                    results = run_model(df, dep_var, channels, controls, ridge=ridge)
                st.session_state.model_results = results
                st.session_state.model_run = True
                # Rerun so the sidebar immediately reflects the newly unlocked tabs
                st.rerun()

        # ── Always render results when available (survives rerun)
        results = st.session_state.get("model_results")
        if results:
            st.markdown('<div class="section-header">Model Results</div>', unsafe_allow_html=True)
            r2_val = results["r2"]
            r2_color = "#22c55e" if r2_val > 0.7 else ("#f97316" if r2_val > 0.4 else "#ef4444")
            st.metric("R² (Model Fit)", f"{r2_val:.4f}",
                      delta="Excellent fit" if r2_val > 0.8 else "Good fit" if r2_val > 0.6 else "Moderate fit" if r2_val > 0.4 else "Weak fit")

            coef_df = results["coef_table"].copy()
            coef_df["Coefficient"] = coef_df["Coefficient"].round(4)
            if coef_df["P-Value"].notna().any():
                coef_df["Significant"] = coef_df["P-Value"].apply(
                    lambda p: "✅ Yes" if p is not None and p < 0.05 else "❌ No" if p is not None else "—"
                )
                coef_df["P-Value"] = coef_df["P-Value"].apply(
                    lambda p: f"{p:.4f}" if p is not None else "—"
                )
            st.dataframe(coef_df, use_container_width=True, hide_index=True)

            # ── Channel coefficient bar chart
            st.markdown('<div class="section-header" style="font-size:18px;">Channel Coefficients</div>', unsafe_allow_html=True)
            run_channels = results["channels"]
            coef_channel = results["coef_table"][results["coef_table"]["Channel"].isin(run_channels)]
            fig_coef = go.Figure(go.Bar(
                x=coef_channel["Channel"].map(lambda x: CHANNEL_LABELS.get(x, x)),
                y=coef_channel["Coefficient"],
                marker_color=[CHANNEL_COLORS.get(c, "#5b8dee") for c in coef_channel["Channel"]],
                text=coef_channel["Coefficient"].round(4),
                textposition="outside",
            ))
            fig_coef.update_layout(**PLOTLY_LAYOUT, height=350,
                                   title="Revenue per Unit Spend by Channel",
                                   yaxis_title="Coefficient")
            st.plotly_chart(fig_coef, use_container_width=True)

            # ── Predicted vs Actual
            st.markdown('<div class="section-header" style="font-size:18px;">Predicted vs Actual</div>', unsafe_allow_html=True)
            date_col = df["date"] if "date" in df.columns else pd.RangeIndex(len(df))
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(
                x=date_col, y=results["y_actual"],
                name="Actual", line=dict(color="#e8eaf0", width=2),
            ))
            fig_pred.add_trace(go.Scatter(
                x=date_col, y=results["y_pred"],
                name="Predicted", line=dict(color="#5b8dee", width=2, dash="dot"),
            ))
            fig_pred.update_layout(**PLOTLY_LAYOUT, height=380,
                                   title="Model Predicted vs Actual Revenue",
                                   yaxis_title="Revenue ($)")
            st.plotly_chart(fig_pred, use_container_width=True)

            # ── Quick-nav prompt
            st.markdown("""
            <div style="background:rgba(91,141,238,0.08); border:1px solid rgba(91,141,238,0.3);
                        border-radius:10px; padding:14px 18px; margin-top:20px; font-size:14px; color:#5b8dee;">
                ✅ Model run complete — <strong>Priors &amp; Adstock</strong> and <strong>ROAS Dashboard</strong>
                are now unlocked in the sidebar.
            </div>
            """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 3 — PRIORS & ADSTOCK
# ──────────────────────────────────────────────────────────────────────────────
def page_priors():
    if not st.session_state.model_run:
        st.warning("Please run a model first on the Model Builder page.")
        return

    df = st.session_state.df
    tier = st.session_state.tier
    results = st.session_state.model_results

    if tier == 3 or results is None:
        st.info("Adstock settings are not available in Incrementality Mode.")
        return

    st.markdown('<h1 style="margin-bottom:4px;">Priors & Adstock</h1>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Configure adstock decay and saturation curves per channel.</div>', unsafe_allow_html=True)

    if tier == 2:
        st.markdown("""
        <div class="surface-box" style="border-color:rgba(249,115,22,0.3);">
            <div style="color:#f97316; font-weight:600; margin-bottom:4px;">⚠️ Lite MMM: Industry Priors Active</div>
            <div style="font-size:13px; color:#9ca3af;">Priors carry more weight than usual — default values are set to industry benchmarks to compensate for limited data history.</div>
        </div>
        """, unsafe_allow_html=True)

    channels = results["channels"]

    # Defaults
    tier1_defaults = {
        "tv_spend": 0.7,
        "paid_search_spend": 0.3,
        "social_spend": 0.45,
        "display_spend": 0.2,
    }
    tier2_defaults = {
        "tv_spend": 0.65,
        "paid_search_spend": 0.25,
        "social_spend": 0.40,
        "display_spend": 0.15,
    }
    defaults = tier2_defaults if tier == 2 else tier1_defaults

    adstock_params = {}
    use_saturation = {}

    # Adstock explanation
    with st.expander("What is Adstock?"):
        st.markdown("""
        <div class="tooltip-box">
            <strong>Adstock</strong> captures the carry-over effect of advertising — the idea that ads don't just impact sales in the week they run, 
            but continue to influence consumers in future weeks as well. A decay of 0.7 means 70% of this week's ad effect carries over to next week.
            <br><br>
            Higher decay = longer lasting ad effects (common for TV, brand campaigns).
            Lower decay = quick fade (common for paid search, direct response).
        </div>
        """, unsafe_allow_html=True)

    with st.expander("What is Saturation?"):
        st.markdown("""
        <div class="tooltip-box">
            <strong>Saturation</strong> (Hill function) models diminishing returns — the idea that doubling your ad spend doesn't double your results. 
            The first dollar is more effective than the thousandth. Enabling this applies an S-curve transformation to the spend data before modeling.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Channel Settings</div>', unsafe_allow_html=True)

    for ch in channels:
        label = CHANNEL_LABELS.get(ch, ch)
        default_decay = defaults.get(ch, 0.3)
        industry_label = " (industry prior)" if tier == 2 else ""

        with st.container():
            st.markdown(f"""
            <div class="surface-box">
                <div style="font-family:'DM Sans',sans-serif; font-weight:600; color:#e8eaf0; margin-bottom:12px;">
                    {label}
                </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            col1, col2 = st.columns([3, 1])
            with col1:
                decay = st.slider(
                    f"Adstock Decay — {label}{industry_label}",
                    min_value=0.0, max_value=1.0,
                    value=default_decay, step=0.05,
                    key=f"decay_{ch}",
                    help=f"Default: {default_decay}" + (" (Industry benchmark)" if tier == 2 else ""),
                )
            with col2:
                sat = st.checkbox(f"Hill Saturation", key=f"sat_{ch}", value=False)

            adstock_params[ch] = {"decay": decay, "saturation": sat}
            use_saturation[ch] = sat

    st.session_state.adstock_params = adstock_params

    # Re-run button
    st.markdown('<div style="margin-top:24px;"></div>', unsafe_allow_html=True)
    if st.button("Re-run with Transformations", use_container_width=False):
        dep_var = "revenue" if "revenue" in df.columns else df.select_dtypes(include=[np.number]).columns[0]
        controls = [c for c in ["seasonality_index", "promo_flag"] if c in df.columns]

        with st.spinner("Re-running model with adstock & saturation transformations..."):
            new_results = run_model(
                df, dep_var, channels, controls,
                ridge=False,
                adstock_params=adstock_params,
                use_saturation=use_saturation,
            )
            st.session_state.transformed_results = new_results

        # Before/after comparison
        st.markdown('<div class="section-header">Before vs After Transformations</div>', unsafe_allow_html=True)

        orig = results["coef_table"].set_index("Channel")["Coefficient"]
        new = new_results["coef_table"].set_index("Channel")["Coefficient"]

        compare_rows = []
        for ch in channels:
            label = CHANNEL_LABELS.get(ch, ch)
            before_val = orig.get(ch, 0)
            after_val = new.get(ch, 0)
            delta = after_val - before_val
            arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
            color = "#22c55e" if delta > 0 else ("#ef4444" if delta < 0 else "#9ca3af")
            compare_rows.append({
                "Channel": label,
                "Coef Before": f"{before_val:.4f}",
                "Coef After": f"{after_val:.4f}",
                "Change": f'<span style="color:{color}">{arrow} {delta:+.4f}</span>',
            })

        # R² comparison
        r1_before = results["r2"]
        r1_after = new_results["r2"]
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">R² Before</div>
                <div class="value" style="font-size:24px;">{r1_before:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            delta_r2 = r1_after - r1_before
            color = "#22c55e" if delta_r2 > 0 else "#ef4444"
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">R² After</div>
                <div class="value" style="font-size:24px; color:{color};">{r1_after:.4f}</div>
                <div class="sub">{delta_r2:+.4f} change</div>
            </div>
            """, unsafe_allow_html=True)

        # Comparison table
        table_html = """
        <table class="styled-table">
            <thead><tr>
                <th>Channel</th><th>Coef Before</th><th>Coef After</th><th>Change</th>
            </tr></thead>
            <tbody>
        """
        for row in compare_rows:
            table_html += f"""<tr>
                <td>{row['Channel']}</td>
                <td class="mono">{row['Coef Before']}</td>
                <td class="mono">{row['Coef After']}</td>
                <td>{row['Change']}</td>
            </tr>"""
        table_html += "</tbody></table>"
        st.markdown(table_html, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 4 — ROAS DASHBOARD
# ──────────────────────────────────────────────────────────────────────────────
def page_roas():
    if not st.session_state.model_run:
        st.warning("Please run a model first on the Model Builder page.")
        return

    df = st.session_state.df
    tier = st.session_state.tier
    results = st.session_state.transformed_results or st.session_state.model_results

    if tier == 3 or results is None:
        st.info("ROAS Dashboard is not available in Incrementality Mode.")
        return

    st.markdown('<h1 style="margin-bottom:4px;">ROAS Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Revenue attribution, return on ad spend, and budget optimization.</div>', unsafe_allow_html=True)

    channels = results["channels"]

    # ── Detect normalized spend (min-max scaled → max ≤ 1)
    spend_max = max((df[ch].max() for ch in channels if ch in df.columns), default=1)
    if spend_max <= 1.05:
        st.markdown("""
        <div style="background:rgba(239,68,68,0.10); border:1px solid rgba(239,68,68,0.4);
                    border-radius:10px; padding:14px 18px; margin-bottom:20px;">
            <div style="font-weight:600; color:#ef4444; margin-bottom:4px;">⚠️ Normalized Spend Detected</div>
            <div style="font-size:13px; color:#9ca3af; line-height:1.6;">
                Your spend columns have been scaled to [0, 1]. ROAS values are not meaningful on normalized data
                because all dollar amounts are near zero.<br><br>
                <strong style="color:#e8eaf0;">To fix:</strong> go back to <strong>Upload &amp; Clean</strong>,
                reload the data (or re-upload your CSV), and run the model again without normalizing spend.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("↩ Reset Data & Model", key="roas_reset"):
            st.session_state.model_run = False
            st.session_state.model_results = None
            st.session_state.transformed_results = None
            st.session_state.df = None
            st.session_state.data_loaded = False
            st.session_state.page = "Upload & Clean"
            st.rerun()

    coef_df = results["coef_table"].set_index("Channel")

    # Compute attributed revenue & ROAS
    channel_data = []
    for ch in channels:
        if ch not in df.columns:
            continue
        coef = coef_df.loc[ch, "Coefficient"] if ch in coef_df.index else 0
        spend = df[ch].sum()
        attributed_rev = max(coef * spend, 0)
        roas = attributed_rev / spend if spend > 0 else 0
        channel_data.append({
            "channel": ch,
            "label": CHANNEL_LABELS.get(ch, ch),
            "spend": spend,
            "attributed_revenue": attributed_rev,
            "roas": roas,
            "color": CHANNEL_COLORS.get(ch, "#5b8dee"),
        })

    total_revenue = df["revenue"].sum() if "revenue" in df.columns else sum(d["attributed_revenue"] for d in channel_data)
    total_spend = sum(d["spend"] for d in channel_data)
    total_attributed = sum(d["attributed_revenue"] for d in channel_data)
    blended_roas = total_attributed / total_spend if total_spend > 0 else 0
    best_channel = max(channel_data, key=lambda x: x["roas"]) if channel_data else None

    # ── 4 Metric Cards
    best_label = best_channel["label"] if best_channel else "—"
    best_roas = best_channel["roas"] if best_channel else 0
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Revenue", f"${total_revenue/1e6:.2f}M")
    with c2:
        st.metric("Blended ROAS", f"{blended_roas:.2f}x")
    with c3:
        st.metric("Best Channel", best_label, f"{best_roas:.2f}x ROAS")
    with c4:
        st.metric("Total Spend", f"${total_spend/1e6:.2f}M")

    # ── Stacked bar chart: revenue contribution by channel over time
    st.markdown('<div class="section-header">Revenue Attribution Over Time</div>', unsafe_allow_html=True)
    date_col = df["date"] if "date" in df.columns else pd.RangeIndex(len(df))
    baseline = results["intercept"]
    fig_stack = go.Figure()

    # Baseline contribution — intercept is already a per-week value
    baseline_arr = np.full(len(df), max(baseline, 0))
    fig_stack.add_trace(go.Bar(
        x=date_col, y=baseline_arr,
        name="Baseline", marker_color="#374151",
    ))

    for cd in channel_data:
        ch = cd["channel"]
        coef = coef_df.loc[ch, "Coefficient"] if ch in coef_df.index else 0
        contrib = np.maximum(df[ch].values * coef, 0)
        fig_stack.add_trace(go.Bar(
            x=date_col, y=contrib,
            name=cd["label"], marker_color=cd["color"],
        ))

    fig_stack.update_layout(
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k != "legend"},
        barmode="stack",
        height=380,
        title="Stacked Revenue Attribution by Channel",
        yaxis_title="Revenue ($)",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0)", bordercolor="#1e2330",
        ),
    )
    st.plotly_chart(fig_stack, use_container_width=True)

    # ── ROAS Table (Plotly native — avoids HTML sanitizer issues)
    st.markdown('<div class="section-header">ROAS by Channel</div>', unsafe_allow_html=True)

    sorted_cd = sorted(channel_data, key=lambda x: x["roas"], reverse=True)
    tbl_channels   = [cd["label"] for cd in sorted_cd]
    tbl_spend      = [f"${cd['spend']:,.0f}" for cd in sorted_cd]
    tbl_rev        = [f"${cd['attributed_revenue']:,.0f}" for cd in sorted_cd]
    tbl_roas       = [f"{cd['roas']:.2f}x" for cd in sorted_cd]
    tbl_share      = [
        f"{cd['attributed_revenue'] / total_attributed * 100:.1f}%" if total_attributed > 0 else "—"
        for cd in sorted_cd
    ]
    tbl_roas_colors = [
        "#22c55e" if cd["roas"] > 2 else ("#f97316" if cd["roas"] > 1 else "#ef4444")
        for cd in sorted_cd
    ]
    row_fill = ["#13161e" if i % 2 == 0 else "#161922" for i in range(len(sorted_cd))]

    fig_tbl = go.Figure(go.Table(
        columnwidth=[2, 2, 2, 1.5, 1.5],
        header=dict(
            values=["<b>Channel</b>", "<b>Total Spend</b>", "<b>Attributed Revenue</b>", "<b>ROAS</b>", "<b>Rev Share</b>"],
            fill_color="#1a1d28",
            font=dict(color="#9ca3af", size=12, family="DM Sans"),
            align="left",
            height=36,
            line_color="#1e2330",
        ),
        cells=dict(
            values=[tbl_channels, tbl_spend, tbl_rev, tbl_roas, tbl_share],
            fill_color=[
                row_fill,
                row_fill,
                row_fill,
                row_fill,
                row_fill,
            ],
            font=dict(
                color=[
                    ["#e8eaf0"] * len(sorted_cd),
                    ["#e8eaf0"] * len(sorted_cd),
                    ["#e8eaf0"] * len(sorted_cd),
                    tbl_roas_colors,
                    ["#9ca3af"] * len(sorted_cd),
                ],
                size=13,
                family="DM Mono",
            ),
            align="left",
            height=38,
            line_color="#1e2330",
        ),
    ))
    fig_tbl.update_layout(
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis", "margin")},
        height=44 + 38 * len(sorted_cd),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    st.plotly_chart(fig_tbl, use_container_width=True)

    # ── Waterfall chart
    st.markdown('<div class="section-header">Revenue Waterfall</div>', unsafe_allow_html=True)
    waterfall_labels = ["Baseline"] + [cd["label"] for cd in channel_data] + ["Total"]
    waterfall_values = [max(baseline * len(df), 0)] + [cd["attributed_revenue"] for cd in channel_data] + [0]
    waterfall_measures = ["absolute"] + ["relative"] * len(channel_data) + ["total"]
    waterfall_colors = ["#374151"] + [cd["color"] for cd in channel_data] + ["#5b8dee"]

    fig_wf = go.Figure(go.Waterfall(
        name="Revenue",
        orientation="v",
        measure=waterfall_measures,
        x=waterfall_labels,
        y=waterfall_values,
        connector={"line": {"color": "#1e2330"}},
        increasing={"marker": {"color": "#22c55e"}},
        decreasing={"marker": {"color": "#ef4444"}},
        totals={"marker": {"color": "#5b8dee"}},
    ))
    fig_wf.update_layout(**PLOTLY_LAYOUT, height=380, title="Revenue Attribution Waterfall")
    st.plotly_chart(fig_wf, use_container_width=True)

    # ── Budget Optimizer
    st.markdown('<div class="section-header">Budget Optimizer</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Allocate a total budget across channels proportional to ROAS.</div>', unsafe_allow_html=True)

    total_budget = st.slider("Total Budget ($)", min_value=10000, max_value=500000, value=100000, step=5000, format="$%d")

    total_roas = sum(cd["roas"] for cd in channel_data)
    if total_roas > 0:
        alloc_rows = []
        for cd in channel_data:
            alloc_pct = cd["roas"] / total_roas
            alloc_dollars = alloc_pct * total_budget
            expected_rev = alloc_dollars * cd["roas"]
            alloc_rows.append({
                "Channel": cd["label"],
                "ROAS": f"{cd['roas']:.2f}x",
                "Suggested Allocation": f"${alloc_dollars:,.0f}",
                "Allocation %": f"{alloc_pct*100:.1f}%",
                "Expected Revenue": f"${expected_rev:,.0f}",
            })

        alloc_df = pd.DataFrame(alloc_rows)
        st.dataframe(alloc_df, use_container_width=True, hide_index=True)

    # ── Export button
    st.markdown('<div style="margin-top:20px;"></div>', unsafe_allow_html=True)
    export_data = []
    for cd in channel_data:
        export_data.append({
            "Channel": cd["label"],
            "Total Spend": cd["spend"],
            "Attributed Revenue": cd["attributed_revenue"],
            "ROAS": cd["roas"],
        })
    export_df = pd.DataFrame(export_data)
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Export ROAS Table (CSV)",
        data=csv_bytes,
        file_name="mixwise_roas_export.csv",
        mime="text/csv",
    )


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 5 — A/B TESTING (STUB)
# ──────────────────────────────────────────────────────────────────────────────
def page_ab():
    st.markdown('<h1 style="margin-bottom:4px;">A/B Testing</h1>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Advanced experimentation tools — coming soon.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="coming-soon-overlay" style="margin-bottom:24px;">
        <div style="font-size:48px; margin-bottom:16px;">🧪</div>
        <div class="coming-soon-title">Experimentation Suite</div>
        <div style="font-size:15px; color:#6b7280; max-width:500px; margin:0 auto; line-height:1.7;">
            We're building a full suite of causal inference tools to help you run and measure marketing experiments with statistical rigor.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    features = [
        ("📐", "Difference-in-Differences", "Measure causal impact by comparing treated vs control groups over time, controlling for pre-existing trends."),
        ("⚖️", "Propensity Score Matching", "Construct a valid counterfactual control group by matching experimental units on their likelihood of treatment."),
        ("📊", "Lift & Significance Testing", "Compute lift percentage, confidence intervals, p-values, and minimum detectable effects for any experiment."),
    ]
    for col, (icon, title, desc) in zip([col1, col2, col3], features):
        with col:
            st.markdown(f"""
            <div style="background:#13161e; border:1px solid #1e2330; border-radius:12px; padding:24px; opacity:0.6; text-align:center; height:220px;">
                <div style="font-size:32px; margin-bottom:12px;">{icon}</div>
                <div style="font-family:'DM Serif Display',serif; font-size:16px; color:#e8eaf0; margin-bottom:8px;">{title}</div>
                <div style="font-size:13px; color:#6b7280; line-height:1.5;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

    # Mock locked UI
    st.markdown("""
    <div style="background:#13161e; border:1px dashed #1e2330; border-radius:12px; padding:24px; opacity:0.4; pointer-events:none;">
        <div style="font-family:'DM Mono',monospace; font-size:11px; color:#6b7280; margin-bottom:12px; text-transform:uppercase; letter-spacing:0.06em;">Experiment Configuration (Preview)</div>
        <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px;">
            <div style="background:#1a1d28; border-radius:6px; padding:10px; font-size:13px; color:#374151;">Control Group ▸</div>
            <div style="background:#1a1d28; border-radius:6px; padding:10px; font-size:13px; color:#374151;">Treatment Group ▸</div>
            <div style="background:#1a1d28; border-radius:6px; padding:10px; font-size:13px; color:#374151;">Metric ▸</div>
        </div>
        <div style="margin-top:12px; background:#1a1d28; border-radius:6px; padding:16px; text-align:center; color:#374151; font-size:13px;">
            🔒 Results will appear here
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:32px;'></div>", unsafe_allow_html=True)

    col_btn, col_right = st.columns([1, 3])
    with col_btn:
        if st.button("Request Early Access", use_container_width=True):
            st.toast("You're on the list! We'll be in touch when A/B Testing launches.", icon="✅")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN ROUTER
# ──────────────────────────────────────────────────────────────────────────────
render_sidebar()

page = st.session_state.page
if page == "Upload & Clean":
    page_upload()
elif page == "Model Builder":
    page_model_builder()
elif page == "Priors & Adstock":
    page_priors()
elif page == "ROAS Dashboard":
    page_roas()
elif page == "A/B Testing":
    page_ab()
