"""
app.py — AI Job Recommender v3
Premium dark-theme SaaS-style Streamlit frontend.
Run: python -m streamlit run app.py
"""

import streamlit as st
from pathlib import Path
from collections import Counter
import pandas as pd

from utils import extract_text_from_pdf, detect_experience_level, extract_skills_from_text
from model import (
    load_and_clean_data, build_models,
    build_skill_vocabulary, recommend_jobs,
)

# ═══════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ResumeAI · Job Matcher",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════
# DESIGN SYSTEM — CSS
# ═══════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

/* ── Root palette ── */
:root {
    --bg:         #080E1A;
    --surface:    #0F1826;
    --card:       #111E2E;
    --card-hover: #162336;
    --border:     #1E3048;
    --primary:    #4F8EF7;
    --accent:     #7C5CF7;
    --success:    #10CBA0;
    --warning:    #F5A623;
    --danger:     #F55F5F;
    --text:       #E8EFF8;
    --muted:      #6B8299;
    --dim:        #3A5068;
}

/* ── Global reset ── */
* { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text) !important;
}

/* ── Hide Streamlit chrome ── */
header[data-testid="stHeader"],
[data-testid="collapsedControl"],
footer { display: none !important; }

/* ── Main content area ── */
.block-container {
    padding: 2rem 2.5rem 4rem 2.5rem !important;
    max-width: 1280px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1.2rem !important;
}

/* ── Streamlit widgets ── */
.stSelectbox > div > div,
.stFileUploader > div {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}

.stSelectbox label, .stFileUploader label,
.stSlider label, .stToggle label {
    color: var(--muted) !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--primary), var(--accent)) !important;
    border-radius: 4px !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    border: 1px solid var(--border) !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    padding: 0.4rem 1.2rem !important;
}
.stTabs [aria-selected="true"] {
    background: var(--card) !important;
    color: var(--text) !important;
}

/* ── Expanders ── */
.streamlit-expanderHeader {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-weight: 600 !important;
}
.streamlit-expanderContent {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
}

/* ── Dataframe ── */
.stDataFrame { border-radius: 10px !important; overflow: hidden !important; }

/* ── Metric ── */
[data-testid="stMetric"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.72rem !important; text-transform: uppercase !important; letter-spacing: 0.08em !important; }
[data-testid="stMetricValue"] { color: var(--text) !important; font-size: 1.6rem !important; font-weight: 700 !important; }

/* ── Dividers ── */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* ── Caption / small text ── */
.stCaption { color: var(--muted) !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--primary) !important; }

/* ── Warning/error/info ── */
.stAlert { border-radius: 10px !important; }

/* ══════════════════════════════════════════════════════════
   CUSTOM COMPONENTS
══════════════════════════════════════════════════════════ */

/* Hero section */
.hero {
    background: linear-gradient(135deg, #0D1B2E 0%, #0F1826 60%, #13102B 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(79,142,247,0.12) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -80px; left: -40px;
    width: 240px; height: 240px;
    background: radial-gradient(circle, rgba(124,92,247,0.10) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(79,142,247,0.12);
    border: 1px solid rgba(79,142,247,0.3);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--primary);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 1rem;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    color: var(--text);
    margin: 0 0 0.6rem 0;
    line-height: 1.2;
}
.hero-title span {
    background: linear-gradient(90deg, var(--primary), var(--accent));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    color: var(--muted);
    font-size: 1rem;
    max-width: 540px;
    line-height: 1.6;
    margin: 0;
}
.hero-stats {
    display: flex;
    gap: 2rem;
    margin-top: 1.8rem;
    flex-wrap: wrap;
}
.hero-stat-item {
    display: flex;
    flex-direction: column;
    gap: 2px;
}
.hero-stat-num {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text);
    font-family: 'DM Mono', monospace;
}
.hero-stat-label {
    font-size: 0.68rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Upload zone */
.upload-zone {
    background: var(--card);
    border: 2px dashed var(--border);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    transition: border-color 0.2s;
}
.upload-zone:hover { border-color: var(--primary); }
.upload-icon { font-size: 2rem; margin-bottom: 0.5rem; }
.upload-title { font-size: 1rem; font-weight: 600; color: var(--text); margin: 0; }
.upload-sub { font-size: 0.8rem; color: var(--muted); margin: 4px 0 0; }

/* Section label */
.section-label {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
    margin-bottom: 0.5rem;
}

/* Job card */
.job-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    position: relative;
    transition: border-color 0.2s, transform 0.15s;
}
.job-card:hover {
    border-color: var(--primary);
    transform: translateY(-1px);
}
.job-card-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 1rem;
    margin-bottom: 1rem;
}
.job-rank {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: var(--dim);
    font-weight: 500;
    margin-bottom: 4px;
}
.job-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text);
    margin: 0 0 4px 0;
    line-height: 1.3;
}
.job-category {
    font-size: 0.72rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
}
.score-circle {
    flex-shrink: 0;
    text-align: right;
}
.score-num {
    font-size: 2rem;
    font-weight: 800;
    line-height: 1;
    font-family: 'DM Mono', monospace;
}
.score-label {
    font-size: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
    font-weight: 600;
}

/* Tier badge */
.tier-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 6px;
}

/* Progress bar */
.prog-wrap {
    margin: 0.8rem 0;
    height: 5px;
    background: var(--border);
    border-radius: 3px;
    overflow: hidden;
}
.prog-fill {
    height: 100%;
    border-radius: 3px;
    background: linear-gradient(90deg, var(--primary), var(--accent));
    transition: width 0.6s ease;
}

/* Score breakdown bars */
.score-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 5px 0;
    font-size: 0.78rem;
    color: var(--muted);
}
.score-bar-bg {
    flex: 1;
    height: 4px;
    background: var(--border);
    border-radius: 2px;
    overflow: hidden;
}
.score-bar-fill { height: 100%; border-radius: 2px; }
.score-val {
    width: 42px;
    text-align: right;
    color: var(--text);
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
}

/* Skill section inside card */
.skill-section { margin-top: 0.8rem; }
.skill-section-title {
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 6px;
}
.skill-chips { display: flex; flex-wrap: wrap; gap: 5px; }
.chip {
    font-size: 0.72rem;
    font-weight: 500;
    padding: 3px 10px;
    border-radius: 20px;
    display: inline-block;
    font-family: 'DM Mono', monospace;
    letter-spacing: -0.01em;
}
.chip-match   { background: rgba(16,203,160,0.12); color: #10CBA0; border: 1px solid rgba(16,203,160,0.25); }
.chip-missing { background: rgba(245,95,95,0.10);  color: #F55F5F; border: 1px solid rgba(245,95,95,0.22); }
.chip-learn   { background: rgba(245,166,35,0.10);  color: #F5A623; border: 1px solid rgba(245,166,35,0.22); }
.chip-more    { background: var(--border); color: var(--muted); border: 1px solid transparent; }

/* Resume panel */
.resume-panel {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.5rem;
}
.resume-panel-title {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
    margin-bottom: 1rem;
}

/* Experience badge */
.exp-badge {
    display: inline-block;
    font-size: 0.8rem;
    font-weight: 700;
    padding: 5px 14px;
    border-radius: 8px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.exp-senior { background: rgba(124,92,247,0.15); color: #9B7EFF; border: 1px solid rgba(124,92,247,0.3); }
.exp-mid    { background: rgba(79,142,247,0.12); color: #4F8EF7; border: 1px solid rgba(79,142,247,0.3); }
.exp-entry  { background: rgba(16,203,160,0.12); color: #10CBA0; border: 1px solid rgba(16,203,160,0.25); }

/* Summary bar */
.summary-bar {
    display: flex;
    gap: 0;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    overflow: hidden;
    margin-bottom: 1.5rem;
}
.summary-item {
    flex: 1;
    padding: 1rem 1.2rem;
    border-right: 1px solid var(--border);
    text-align: center;
}
.summary-item:last-child { border-right: none; }
.summary-val {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text);
    font-family: 'DM Mono', monospace;
    line-height: 1;
}
.summary-lbl {
    font-size: 0.64rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.09em;
    font-weight: 600;
    margin-top: 4px;
}

/* How it works cards */
.how-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem;
    text-align: center;
}
.how-num {
    width: 32px; height: 32px;
    border-radius: 8px;
    background: rgba(79,142,247,0.12);
    border: 1px solid rgba(79,142,247,0.25);
    color: var(--primary);
    font-size: 0.85rem;
    font-weight: 700;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 0.7rem;
    font-family: 'DM Mono', monospace;
}
.how-title { font-size: 0.88rem; font-weight: 600; color: var(--text); margin: 0 0 4px; }
.how-desc  { font-size: 0.75rem; color: var(--muted); margin: 0; line-height: 1.5; }

/* Sidebar filter section header */
.sidebar-section {
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--dim);
    margin: 1.2rem 0 0.5rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--border);
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# LOAD MODEL (cached)
# ═══════════════════════════════════════════════════════════
DATA_PATH = Path(__file__).parent / "all_job_post.csv"

@st.cache_resource(show_spinner=False)
def load_everything():
    df = load_and_clean_data(str(DATA_PATH))
    skill_vec, skill_mat, desc_vec, desc_mat = build_models(df)
    skill_vocab, skill_freq = build_skill_vocabulary(df)
    return df, skill_vec, skill_mat, desc_vec, desc_mat, skill_vocab, skill_freq

with st.spinner("Initializing model…"):
    try:
        df, skill_vec, skill_mat, desc_vec, desc_mat, skill_vocab, skill_freq = load_everything()
    except Exception as e:
        st.error(f"❌ Model load failed: {e}")
        st.stop()


# ═══════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════
def score_color(pct: float) -> str:
    if pct >= 70: return "#10CBA0"
    if pct >= 45: return "#F5A623"
    if pct >= 25: return "#F97316"
    return "#F55F5F"

def tier_badge_html(tier: str, icon: str, color: str) -> str:
    bg  = color + "18"
    bdr = color + "40"
    return (f'<span class="tier-badge" style="background:{bg};color:{color};border:1px solid {bdr};">'
            f'{icon} {tier}</span>')

def chip_html(skill: str, cls: str) -> str:
    return f'<span class="chip {cls}">{skill.title()}</span>'

def chips_row(skills: list, cls: str, limit: int = 10) -> str:
    if not skills:
        return '<span style="font-size:0.75rem;color:var(--dim);">None</span>'
    shown = [chip_html(s, cls) for s in skills[:limit]]
    if len(skills) > limit:
        shown.append(f'<span class="chip chip-more">+{len(skills)-limit} more</span>')
    return '<div class="skill-chips">' + "".join(shown) + '</div>'

def score_bar(label: str, value: float, color: str) -> str:
    return f"""
    <div class="score-row">
        <span style="width:100px;flex-shrink:0;">{label}</span>
        <div class="score-bar-bg">
            <div class="score-bar-fill" style="width:{min(value,100):.1f}%;background:{color};"></div>
        </div>
        <span class="score-val">{value:.1f}%</span>
    </div>"""

def exp_badge_html(level: str) -> str:
    cls = {"Senior": "exp-senior", "Mid": "exp-mid", "Entry": "exp-entry"}.get(level, "exp-mid")
    icons = {"Senior": "⬆", "Mid": "➡", "Entry": "⬇"}
    return f'<span class="exp-badge {cls}">{icons.get(level,"")} {level}</span>'


# ═══════════════════════════════════════════════════════════
# SIDEBAR — CONTROLS
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:1.5rem;">
        <span style="font-size:1.4rem;">✦</span>
        <div>
            <div style="font-size:1rem;font-weight:700;color:var(--text);">ResumeAI</div>
            <div style="font-size:0.7rem;color:var(--muted);">Job Matcher v3</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Upload ──────────────────────────────────────────
    st.markdown('<div class="sidebar-section">Resume Upload</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload Resume (PDF)",
        type=["pdf"],
        help="Text-based PDF only. Scanned images cannot be parsed.",
        label_visibility="collapsed",
    )
    if uploaded_file:
        st.success(f"✓ {uploaded_file.name}", icon=None)
    else:
        st.markdown("""
        <div style="background:var(--card);border:1px dashed var(--border);border-radius:10px;
                    padding:1rem;text-align:center;margin-top:0.4rem;">
            <div style="font-size:1.4rem;">📄</div>
            <div style="font-size:0.78rem;color:var(--muted);margin-top:4px;">
                Drop your PDF resume here<br>or click to browse
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Filters ─────────────────────────────────────────
    st.markdown('<div class="sidebar-section">Filters</div>', unsafe_allow_html=True)

    cats = ["All"] + sorted(df["category"].unique().tolist())
    category = st.selectbox("Category", cats, help="Filter results to a specific industry")

    top_n = st.select_slider(
        "Number of Results",
        options=[3, 5, 7, 10],
        value=5,
    )

    show_scores = st.toggle("Show Score Breakdown", value=False,
                             help="Display skill, description, and hybrid scores per job")

    # ── Dataset info ────────────────────────────────────
    st.markdown('<div class="sidebar-section">Dataset</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:0.75rem;color:var(--muted);line-height:1.8;">
        🗂 <b style="color:var(--text);">{len(df):,}</b> unique jobs<br>
        🔑 <b style="color:var(--text);">{len(skill_vocab):,}</b> unique skills<br>
        🏷 <b style="color:var(--text);">{df['category'].nunique()}</b> categories<br>
        ⚙️ Hybrid TF-IDF + Cosine
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# HERO SECTION
# ═══════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero">
    <div class="hero-eyebrow">✦ AI-Powered · Hybrid ML · Skill Gap Analysis</div>
    <h1 class="hero-title">Find Your <span>Perfect Role</span><br>With AI Precision</h1>
    <p class="hero-sub">
        Upload your resume and our hybrid ML engine matches your skills against 
        real job postings — with detailed gap analysis and learning recommendations.
    </p>
    <div class="hero-stats">
        <div class="hero-stat-item">
            <span class="hero-stat-num">{len(df):,}</span>
            <span class="hero-stat-label">Jobs Indexed</span>
        </div>
        <div class="hero-stat-item">
            <span class="hero-stat-num">{len(skill_vocab):,}</span>
            <span class="hero-stat-label">Unique Skills</span>
        </div>
        <div class="hero-stat-item">
            <span class="hero-stat-num">70/30</span>
            <span class="hero-stat-label">Skill / Desc Weight</span>
        </div>
        <div class="hero-stat-item">
            <span class="hero-stat-num">{df['category'].nunique()}</span>
            <span class="hero-stat-label">Categories</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# EMPTY STATE — No resume uploaded
# ═══════════════════════════════════════════════════════════
if uploaded_file is None:
    st.markdown('<p style="color:var(--muted);font-size:0.9rem;">👈 Upload your resume in the sidebar to get started.</p>', unsafe_allow_html=True)

    cols = st.columns(5)
    steps = [
        ("01", "Upload PDF", "Your resume is read with PyMuPDF text extraction"),
        ("02", "Extract Skills", "Skills matched against 5,000+ vocabulary entries"),
        ("03", "Detect Level", "Experience level auto-detected from text patterns"),
        ("04", "Hybrid Match", "Skill (70%) + Description (30%) cosine similarity"),
        ("05", "Gap Analysis", "Matched, missing & recommended skills per job"),
    ]
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div class="how-card">
                <div class="how-num">{num}</div>
                <p class="how-title">{title}</p>
                <p class="how-desc">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### 📂 Categories in Dataset")
    cat_counts = df["category"].value_counts()
    cat_df = pd.DataFrame({"Category": cat_counts.index, "Jobs": cat_counts.values})
    st.bar_chart(cat_df.set_index("Category"), color="#4F8EF7", height=220)

    st.stop()


# ═══════════════════════════════════════════════════════════
# EXTRACT RESUME
# ═══════════════════════════════════════════════════════════
with st.spinner("Reading resume…"):
    try:
        resume_text = extract_text_from_pdf(uploaded_file)
    except RuntimeError as e:
        st.error(str(e))
        st.stop()

if not resume_text.strip():
    st.error("⚠ No text found. Please upload a text-based PDF (not a scanned image).")
    st.stop()

resume_skills = extract_skills_from_text(resume_text, skill_vocab)
exp_level     = detect_experience_level(resume_text)


# ═══════════════════════════════════════════════════════════
# RESUME ANALYSIS PANEL
# ═══════════════════════════════════════════════════════════
st.markdown('<div class="resume-panel">', unsafe_allow_html=True)
st.markdown('<div class="resume-panel-title">📄 Resume Analysis</div>', unsafe_allow_html=True)

col_skills, col_exp = st.columns([4, 1])

with col_skills:
    st.markdown('<div class="section-label">Detected Skills</div>', unsafe_allow_html=True)
    if resume_skills:
        st.markdown(chips_row(resume_skills, "chip-match", limit=20), unsafe_allow_html=True)
    else:
        st.warning("No skills matched the job dataset vocabulary. "
                   "Add an explicit **Skills** section to your resume for better results.")

with col_exp:
    st.markdown('<div class="section-label">Level</div>', unsafe_allow_html=True)
    st.markdown(exp_badge_html(exp_level), unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:0.68rem;color:var(--muted);margin-top:6px;">Auto-detected</div>',
                unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# RUN RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════
with st.spinner(f"Matching against {len(df):,} jobs…"):
    results = recommend_jobs(
        resume_text    = resume_text,
        df             = df,
        skill_vec      = skill_vec,
        skill_matrix   = skill_mat,
        desc_vec       = desc_vec,
        desc_matrix    = desc_mat,
        skill_vocab    = skill_vocab,
        skill_freq     = skill_freq,
        category_filter= category,
        top_n          = top_n,
    )

if not results:
    st.warning("No matches found. Try selecting **All** categories, or add a Skills section to your resume.")
    st.stop()


# ═══════════════════════════════════════════════════════════
# SUMMARY BAR
# ═══════════════════════════════════════════════════════════
best = results[0]
avg_match = round(sum(r["match_pct"] for r in results) / len(results), 1)
n_matched_unique = len(set(s for r in results for s in r["matched_skills"]))
n_missing_unique = len(set(s for r in results for s in r["missing_skills"]))

st.markdown(f"""
<div class="summary-bar">
    <div class="summary-item">
        <div class="summary-val" style="color:{score_color(best['match_pct'])};">{best['match_pct']}%</div>
        <div class="summary-lbl">Best Match</div>
    </div>
    <div class="summary-item">
        <div class="summary-val">{avg_match}%</div>
        <div class="summary-lbl">Avg Match</div>
    </div>
    <div class="summary-item">
        <div class="summary-val">{len(resume_skills)}</div>
        <div class="summary-lbl">Resume Skills</div>
    </div>
    <div class="summary-item">
        <div class="summary-val" style="color:#10CBA0;">{n_matched_unique}</div>
        <div class="summary-lbl">Skills Matched</div>
    </div>
    <div class="summary-item">
        <div class="summary-val" style="color:#F55F5F;">{n_missing_unique}</div>
        <div class="summary-lbl">Skills to Gain</div>
    </div>
    <div class="summary-item">
        <div class="summary-val">{len(results)}</div>
        <div class="summary-lbl">Jobs Found</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# TABS — Results / Skill Gap / Analysis
# ═══════════════════════════════════════════════════════════
tab_results, tab_gap, tab_analysis = st.tabs(["🎯 Recommendations", "📊 Skill Gap", "🔬 Analysis"])


# ─────────────────────────────────────────────────────────
# TAB 1 — JOB RECOMMENDATION CARDS
# ─────────────────────────────────────────────────────────
with tab_results:
    for rank, job in enumerate(results, 1):
        pct   = job["match_pct"]
        color = score_color(pct)

        # Card open
        st.markdown('<div class="job-card">', unsafe_allow_html=True)

        # ── Header ──────────────────────────────────────
        st.markdown(f"""
        <div class="job-card-header">
            <div>
                <div class="job-rank">#{rank:02d}</div>
                <div class="job-title">{job['job_title']}</div>
                <div class="job-category">{job['category'].replace('-',' ').title()}</div>
                {tier_badge_html(job['tier'], job['tier_icon'], job['tier_color'])}
            </div>
            <div class="score-circle">
                <div class="score-num" style="color:{color};">{pct}%</div>
                <div class="score-label">Match</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Match progress bar ──────────────────────────
        st.markdown(f"""
        <div class="prog-wrap">
            <div class="prog-fill" style="width:{min(pct,100):.1f}%;
                 background:linear-gradient(90deg,{color},{color}88);"></div>
        </div>
        """, unsafe_allow_html=True)

        # ── Score breakdown (optional) ──────────────────
        if show_scores:
            st.markdown(
                score_bar("Skill Score",  job["skill_score"],  "#4F8EF7") +
                score_bar("Desc Score",   job["desc_score"],   "#7C5CF7") +
                score_bar("Hybrid Score", job["hybrid_score"], "#10CBA0"),
                unsafe_allow_html=True,
            )

        # ── Skill gap — 3 columns ───────────────────────
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(f"""
            <div class="skill-section">
                <div class="skill-section-title" style="color:#10CBA0;">
                    ✅ Matched &nbsp;<span style="color:var(--dim);font-weight:400;">({len(job['matched_skills'])})</span>
                </div>
                {chips_row(job['matched_skills'], 'chip-match', limit=8)}
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="skill-section">
                <div class="skill-section-title" style="color:#F55F5F;">
                    ❌ Missing &nbsp;<span style="color:var(--dim);font-weight:400;">({len(job['missing_skills'])})</span>
                </div>
                {chips_row(job['missing_skills'], 'chip-missing', limit=8)}
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="skill-section">
                <div class="skill-section-title" style="color:#F5A623;">
                    🚀 Learn &nbsp;<span style="color:var(--dim);font-weight:400;">({len(job['recommended_skills'])})</span>
                </div>
                {chips_row(job['recommended_skills'], 'chip-learn', limit=8)}
            </div>
            """, unsafe_allow_html=True)

        # Card close + spacing
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# TAB 2 — SKILL GAP DASHBOARD
# ─────────────────────────────────────────────────────────
with tab_gap:
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### 📉 Top Skills You're Missing")
        st.caption("Skills most frequently required across your matched jobs")
        all_missing = [s for r in results for s in r["missing_skills"]]
        top_missing = Counter(all_missing).most_common(10)
        if top_missing:
            miss_df = pd.DataFrame(top_missing, columns=["Skill", "Jobs Requiring It"])
            miss_df["Skill"] = miss_df["Skill"].str.title()
            miss_df.index = range(1, len(miss_df) + 1)
            st.dataframe(miss_df, use_container_width=True, hide_index=False)
        else:
            st.success("You have all required skills across your matches! 🎉")

    with col_right:
        st.markdown("#### 🏆 Your Strongest Matched Skills")
        st.caption("Skills you have that appear most across matched jobs")
        all_matched = [s for r in results for s in r["matched_skills"]]
        top_matched = Counter(all_matched).most_common(10)
        if top_matched:
            match_df = pd.DataFrame(top_matched, columns=["Skill", "Jobs Where Matched"])
            match_df["Skill"] = match_df["Skill"].str.title()
            match_df.index = range(1, len(match_df) + 1)
            st.dataframe(match_df, use_container_width=True, hide_index=False)

    st.markdown("---")
    st.markdown("#### 📊 Match % by Job")
    chart_df = pd.DataFrame({
        "Job": [f"#{i+1} {r['job_title'][:30]}" for i, r in enumerate(results)],
        "Match %": [r["match_pct"] for r in results],
    }).set_index("Job")
    st.bar_chart(chart_df, color="#4F8EF7", height=280)


# ─────────────────────────────────────────────────────────
# TAB 3 — DETAILED ANALYSIS
# ─────────────────────────────────────────────────────────
with tab_analysis:
    st.markdown("#### 📋 Full Results Table")
    table_data = []
    for i, r in enumerate(results, 1):
        table_data.append({
            "Rank":        i,
            "Job Title":   r["job_title"],
            "Category":    r["category"].replace("-", " ").title(),
            "Match %":     r["match_pct"],
            "Tier":        r["tier"],
            "Skill Score": r["skill_score"],
            "Desc Score":  r["desc_score"],
            "Hybrid Score":r["hybrid_score"],
            "Matched":     len(r["matched_skills"]),
            "Missing":     len(r["missing_skills"]),
            "Total Skills":r["total_job_skills"],
        })
    st.dataframe(
        pd.DataFrame(table_data).set_index("Rank"),
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown("#### 🔬 Resume Stats")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Skills Detected",    len(resume_skills))
    c2.metric("Experience Level",   exp_level)
    c3.metric("Best Match",         f"{best['match_pct']}%")
    c4.metric("Avg Match",          f"{avg_match}%")

    if resume_skills:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**All detected resume skills:**")
        st.markdown(chips_row(resume_skills, "chip-match", limit=50), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:center;
            color:var(--muted);font-size:0.72rem;flex-wrap:wrap;gap:8px;">
    <span>✦ ResumeAI v3 · Hybrid TF-IDF (Skill 70% + Desc 30%) · Cosine Similarity</span>
    <span>{len(df):,} jobs · {len(skill_vocab):,} skills · Built with Streamlit</span>
</div>
""", unsafe_allow_html=True)
