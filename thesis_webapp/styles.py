from __future__ import annotations

import streamlit as st


APP_CSS = """
<style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(220, 38, 38, 0.10), transparent 30%),
            radial-gradient(circle at top right, rgba(37, 99, 235, 0.08), transparent 26%),
            linear-gradient(180deg, #f8f5f2 0%, #f2ede8 45%, #fcfbfa 100%);
        color: #18212f;
    }
    #MainMenu,
    header,
    footer,
    .stAppHeader,
    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"] {
        visibility: hidden !important;
        display: none !important;
        height: 0 !important;
    }
    .block-container { padding-top: 0 !important; }
    [data-testid="stAppViewContainer"] > .main {
        padding-top: 0 !important;
    }
    [data-testid="stAppViewContainer"] .main .block-container {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    .landing-hero {
        background: linear-gradient(145deg, #0f172a 0%, #7f1d1d 55%, #1e3a5f 100%);
        color: #fff;
        padding: 2rem 2rem 5.75rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        border-radius: 0 0 56px 56px;
        margin: -2.75rem -1rem 2.6rem;
        min-height: 29rem;
    }
    .landing-hero::before {
        content: "";
        position: absolute;
        inset: 0;
        background:
            radial-gradient(circle at 18% 45%, rgba(220,38,38,0.28) 0%, transparent 52%),
            radial-gradient(circle at 82% 18%, rgba(37,99,235,0.22) 0%, transparent 44%);
        pointer-events: none;
    }
    .landing-hero h1 {
        font-size: 2.5rem; font-weight: 800; letter-spacing: -0.025em;
        margin: 0 0 0.9rem; position: relative;
        line-height: 1.2;
    }
    .landing-badge {
        display: inline-block;
        background: rgba(255,255,255,0.13);
        border: 1px solid rgba(255,255,255,0.24);
        border-radius: 999px;
        padding: 0.3rem 1.1rem;
        font-size: 0.82rem; letter-spacing: 0.12em; text-transform: uppercase;
        margin-bottom: 1.4rem; position: relative;
    }
    .landing-sub {
        font-size: 1.08rem; color: rgba(255,255,255,0.90);
        max-width: 760px; margin: 0 auto; line-height: 1.72; position: relative;
        text-align: center;
        display: block;
        width: 100%;
    }
    .feature-cards {
        display: flex; gap: 1.2rem; justify-content: center;
        flex-wrap: wrap; padding: 2.4rem 1rem 0.4rem;
    }
    .feature-card {
        background: rgba(255,255,255,0.82);
        border: 1px solid rgba(24,33,47,0.09);
        border-radius: 20px; padding: 1.55rem 1.6rem; width: 210px;
        box-shadow: 0 8px 30px rgba(15,23,42,0.07); text-align: center;
    }
    .feature-card .fc-icon { font-size: 2rem; margin-bottom: 0.55rem; }
    .feature-card .fc-title { font-weight: 700; font-size: 0.96rem; color: #0f172a; margin-bottom: 0.3rem; }
    .feature-card .fc-desc { font-size: 0.83rem; color: #475569; line-height: 1.5; }
    .landing-disclaimer {
        text-align: center; color: #94a3b8; font-size: 0.78rem;
        margin-top: 1.1rem; padding-bottom: 0.2rem;
    }
    .glass-card {
        padding: 1rem 1.1rem; border-radius: 22px;
        border: 1px solid rgba(24,33,47,0.08);
        background: rgba(255,255,255,0.78);
        box-shadow: 0 18px 40px rgba(15,23,42,0.07);
        margin-bottom: 1rem;
    }
    .section-header {
        padding: 1.6rem 0 0.4rem;
        border-bottom: 2px solid rgba(220,38,38,0.18);
        margin-bottom: 1.3rem;
    }
    .section-header h2 { font-size: 1.55rem; font-weight: 700; margin: 0; color: #0f172a; }
    .section-header p { color: #64748b; margin: 0.25rem 0 0; font-size: 0.93rem; }
    .output-hero {
        background: linear-gradient(135deg, rgba(17,24,39,0.94), rgba(127,29,29,0.88));
        border-radius: 22px; padding: 1.4rem 1.8rem; color: #fff; margin-bottom: 0;
    }
    .output-hero .oh-label { font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.15em; color: rgba(255,255,255,0.62); }
    .output-hero .oh-value { font-size: 2.1rem; font-weight: 800; margin-top: 0.1rem; }
    .output-hero .oh-sub { font-size: 0.87rem; color: rgba(255,255,255,0.70); margin-top: 0.15rem; }
    .risk-badge { display: inline-block; border-radius: 999px; padding: 0.32rem 1rem; font-size: 0.95rem; font-weight: 700; }
    .risk-low        { background: #dcfce7; color: #15803d; }
    .risk-medium     { background: #fef9c3; color: #92400e; }
    .risk-high       { background: #fee2e2; color: #b91c1c; }
    .risk-at-risk    { background: #fee2e2; color: #b91c1c; }
    .risk-not-at-risk{ background: #dcfce7; color: #15803d; }
    .kpi-title { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.18em; color: #64748b; }
    .kpi-value { font-size: 1.8rem; font-weight: 700; margin-top: 0.15rem; color: #0f172a; }
    .kpi-subtitle { color: #64748b; font-size: 0.92rem; margin-top: 0.18rem; }
    @media (max-width: 768px) {
        .output-hero {
            margin-bottom: 0.85rem;
        }
    }
    button[data-testid="stNumberInputStepDown"],
    button[data-testid="stNumberInputStepUp"] {
        display: none !important;
    }
    section[data-testid="stSidebar"] {
        background: rgba(255,255,255,0.82);
        border-right: 1px solid rgba(15,23,42,0.08);
    }
</style>
"""


def apply_global_styles() -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)
