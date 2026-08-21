"""Packet Pulse visual identity for the Streamlit app -- same tokens as
website/index.html (ink/surface/line/signal/pulse), applied via injected CSS
since Streamlit's own theme config only covers app chrome, not every widget.
"""
import streamlit as st

BRAND_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Big+Shoulders+Display:wght@600;700;800&family=Public+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root {
  --ink: #0b0e13; --surface: #12161d; --surface-raised: #161b23;
  --line: #232a35; --line-strong: #3a4452; --fog: #8a94a3; --fog-dim: #5c6472;
  --paper: #edeff2; --signal: #ffb020; --pulse: #3fd0c9; --alert: #e5484d;
  --font-display: "Big Shoulders Display", "Arial Narrow", sans-serif;
  --font-body: "Public Sans", -apple-system, "Segoe UI", sans-serif;
  --font-mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
html, body, [class*="css"] { font-family: var(--font-body) !important; }
[data-testid="stAppViewContainer"], [data-testid="stHeader"] { background: var(--ink); }
[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--line); }
[data-testid="stSidebar"] * { color: var(--paper); }
h1, h2, h3 { font-family: var(--font-display) !important; font-weight: 700 !important; letter-spacing: -0.01em; }
.pp-eyebrow {
  font-family: var(--font-mono); font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--signal); display: flex; align-items: center; gap: 0.6em; margin-bottom: 6px;
}
.pp-eyebrow::before { content: ""; width: 1.4em; height: 1px; background: var(--signal); display: inline-block; }
.pp-header {
  display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;
  padding-bottom: 18px; margin-bottom: 6px; border-bottom: 1px solid var(--line);
}
.pp-brand { display: flex; align-items: center; gap: 10px; font-family: var(--font-display);
  font-size: 1.35rem; font-weight: 700; color: var(--paper); }
.pp-live-pill {
  font-family: var(--font-mono); font-size: 0.72rem; color: var(--pulse);
  border: 1px solid #1f6b66; background: rgba(63,208,201,0.08);
  padding: 5px 12px; border-radius: 20px; display: inline-flex; align-items: center; gap: 6px;
}
.pp-live-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--pulse); animation: pp-blink 1.6s ease-in-out infinite; }
@keyframes pp-blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.25; } }
@media (prefers-reduced-motion: reduce) { .pp-live-dot { animation: none; } }
[data-testid="stMetric"] {
  background: var(--surface); border: 1px solid var(--line); border-top: 2px solid var(--signal);
  padding: 14px 16px 10px; border-radius: 3px;
}
[data-testid="stMetricValue"] { font-family: var(--font-mono) !important; color: var(--paper) !important; font-variant-numeric: tabular-nums; }
[data-testid="stMetricLabel"] { color: var(--fog) !important; font-size: 0.82rem !important; }
[data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--line); }
[data-baseweb="tab"] { font-family: var(--font-mono); font-size: 0.85rem; color: var(--fog) !important; }
[data-baseweb="tab"][aria-selected="true"] { color: var(--signal) !important; }
[data-baseweb="tab-highlight"] { background-color: var(--signal) !important; }
[data-baseweb="tab-border"] { background-color: var(--line) !important; }
.stButton > button, .stDownloadButton > button {
  background: linear-gradient(180deg, #ffc350, var(--signal)) !important; color: #1a1204 !important;
  border: none !important; font-weight: 600 !important; border-radius: 3px !important;
  box-shadow: 0 8px 20px -10px rgba(255,176,32,0.65) !important;
}
.stButton > button:hover, .stDownloadButton > button:hover { filter: brightness(1.05); }
.stButton > button[kind="primary"] { background: linear-gradient(180deg, #ffc350, var(--signal)) !important; }
[data-baseweb="slider"] [role="slider"] { background-color: var(--signal) !important; }
div[data-testid="stSlider"] div[style*="background-color: rgb(255, 75, 75)"] { background-color: var(--signal) !important; }
[data-testid="stDataFrame"] { border: 1px solid var(--line) !important; border-radius: 3px; }
code, .stCodeBlock, [data-testid="stCaptionContainer"] code { font-family: var(--font-mono) !important; }
[data-testid="stExpander"] { border: 1px solid var(--line) !important; background: var(--surface); border-radius: 3px; }
[data-baseweb="select"] { font-family: var(--font-body); }
</style>
"""

HEADER_HTML = """
<div class="pp-header">
  <div class="pp-brand">
    <svg width="24" height="24" viewBox="0 0 26 26" fill="none">
      <rect x="0.5" y="0.5" width="25" height="25" rx="3" stroke="#3a4452"/>
      <path d="M3 15L7 15L9.5 8L13 20L15.5 11L17.5 15L23 15" stroke="#ffb020" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
    Packet Pulse
  </div>
  <span class="pp-live-pill"><span class="pp-live-dot"></span>4 models scoring live</span>
</div>
"""


def inject():
    st.markdown(BRAND_CSS, unsafe_allow_html=True)


def header():
    st.markdown(HEADER_HTML, unsafe_allow_html=True)
