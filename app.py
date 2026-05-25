"""
app.py  ·  Beyond the One Number  ·  v3.0
------------------------------------------
Premium NILM Streamlit app with a working dark/light mode toggle.

Architecture rule: every `st.markdown(unsafe_allow_html=True)` call must contain
a COMPLETE, self-closing HTML block. Never open a <div> in one call and close it
in another — Streamlit wraps each call in its own DOM element.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import json
import time
from collections import Counter
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from pdf_generator import generate_energy_report
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Constants
SAVINGS_PCT = 0.20
WINDOW_SIZE = 5

APPLIANCE_COLORS = {
    "Refrigerator":   "#00C9FF",
    "AC":             "#FF6B6B",
    "WashingMachine": "#7B68EE",
    "Microwave":      "#FFD700",
    "TV":             "#00E676",
    "WaterHeater":    "#FF9800",
    "Standby":        "#78909C",
}

# Page config
st.set_page_config(
    page_title="Beyond the One Number",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state ─────────────────────────────────────────────────────────────
if "dark_mode"    not in st.session_state: st.session_state["dark_mode"]    = False
if "compare_mode" not in st.session_state: st.session_state["compare_mode"] = False
if "lang"         not in st.session_state: st.session_state["lang"]         = "en"
if "currency"     not in st.session_state: st.session_state["currency"]     = "USD"

DK = st.session_state["dark_mode"]
lang = st.session_state["lang"]
currency = st.session_state["currency"]

from translations import TRANSLATIONS, LANGUAGES, RTL_LANGUAGES
from currencies import CURRENCIES

TR = TRANSLATIONS[lang]
CURR = CURRENCIES[currency]
symbol = CURR["symbol"]
code = currency
mult = CURR["usd_multiplier"]

# THEME — every colour token lives here; one boolean flips the whole UI
T = {
    # backgrounds
    "bg_app":         "#0D1117"   if DK else "#F0F4F8",
    "bg_card":        "#161B22"   if DK else "#FFFFFF",
    "bg_card2":       "#1C2333"   if DK else "#F8FAFC",
    "bg_input":       "#21262D"   if DK else "#FFFFFF",
    # text
    "tx_primary":     "#E6EDF3"   if DK else "#1A202C",
    "tx_secondary":   "#8B949E"   if DK else "#64748B",
    "tx_heading":     "#F0F6FF"   if DK else "#0A1929",
    # borders
    "border":         "#30363D"   if DK else "#E2E8F0",
    # header
    "hdr_g1":         "#0D1117"   if DK else "#0A1929",
    "hdr_g2":         "#1F2D3D"   if DK else "#15305B",
    "hdr_wave":       "#0D1117"   if DK else "#F0F4F8",
    # sidebar
    "sb_bg":          "#0D1117"   if DK else "#0A1929",
    # plotly
    "plt_bg":         "#161B22"   if DK else "#F8FAFC",
    "plt_grid":       "#21262D"   if DK else "#E2E8F0",
    "plt_font":       "#C9D1D9"   if DK else "#334155",
    # accents (unchanged between modes)
    "blue":           "#2196F3",
    "green":          "#00C853",
    "orange":         "#FF9800",
    "red":            "#FF5252",
    # badge variants
    "badge_g_bg":     "#1B3A2A"   if DK else "#E8F5E9",
    "badge_g_tx":     "#4CAF50"   if DK else "#2E7D32",
    "badge_o_bg":     "#3A2010"   if DK else "#FFF3E0",
    "badge_o_tx":     "#FF9800"   if DK else "#E65100",
    "badge_b_bg":     "#0D2137"   if DK else "#E3F2FD",
    "badge_b_tx":     "#64B5F6"   if DK else "#0D47A1",
    "badge_p_bg":     "#1E1035"   if DK else "#F3E5F5",
    "badge_p_tx":     "#CE93D8"   if DK else "#6A1B9A",
    # appliance rows
    "row_bg":         "#1C2333"   if DK else "#F8FAFC",
    "row_bdr":        "#30363D"   if DK else "#E2E8F0",
    "row_name":       "#C9D1D9"   if DK else "#334155",
    "row_pct":        "#E6EDF3"   if DK else "#0A1929",
    "row_kwh":        "#8B949E"   if DK else "#64748B",
    "row_track":      "#30363D"   if DK else "#E2E8F0",
    # success
    "suc_bg":         "#1B3A2A"   if DK else "#E8F5E9",
    "suc_bdr":        "#4CAF50"   if DK else "#00C853",
    "suc_title":      "#81C784"   if DK else "#1B5E20",
    "suc_sub":        "#66BB6A"   if DK else "#2E7D32",
    # toggle
    "tog_bg":         "#21262D"   if DK else "#EEF2F7",
    "tog_tx":         "#E6EDF3"   if DK else "#334155",
    "tog_track":      "#2196F3"   if DK else "#CBD5E1",
    "tog_thumb_pos":  "flex-end"  if DK else "flex-start",
    "tog_label":      TR["light_mode_label"] if DK else TR["dark_mode_label"],
    # misc
    "footer_bdr":     "#30363D"   if DK else "#E2E8F0",
    "empty_bdr":      "#30363D"   if DK else "#CBD5E1",
    "compare_bg":     "#0D2137"   if DK else "#EBF5FF",
    "compare_bdr":    "#1565C0",
    "compare_tx":     "#64B5F6"   if DK else "#1565C0",
    "tip_action":     "#2196F3",
}

rtl_css = ""
if lang in RTL_LANGUAGES:
    rtl_css = f"""
    .stApp, .hdr, .card, .mbox, .appl-row, .tip-card, .suc-banner, .ftr, section[data-testid="stSidebar"] {{
        direction: rtl !important;
        text-align: right !important;
    }}
    .appl-row {{
        flex-direction: row-reverse !important;
    }}
    .appl-pct, .appl-kwh, .cost-val {{
        text-align: left !important;
    }}
    .cost-row, .tog-wrap {{
        flex-direction: row-reverse !important;
    }}
    .ftr > div {{
        flex-direction: row-reverse !important;
    }}
    .suc-banner {{
        border-left: none !important;
        border-right: 5px solid {T['suc_bdr']} !important;
    }}
    .mbox-lbl, .tip-badge {{
        left: 14px !important;
        right: auto !important;
    }}
    .tip-title {{
        padding-left: 70px !important;
        padding-right: 0 !important;
    }}
    """

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS  (all tokens resolved at render time from T dict)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@500;600;700;800&display=swap');

/* reset */
*,*::before,*::after{{box-sizing:border-box;}}

html,body,[class*="css"]{{
    font-family:'Inter',sans-serif;
    color:{T['tx_primary']};
    transition:color .3s,background-color .3s;
}}
h1,h2,h3,h4,h5,h6{{font-family:'Poppins',sans-serif;color:{T['tx_heading']};}}

/* hide streamlit chrome and deploy buttons */
#MainMenu,footer{{visibility:hidden;}}
.stDeployButton,[data-testid="stDeployButton"],header{{display:none!important;}}

/* scrollbar */
::-webkit-scrollbar{{width:6px;height:6px;}}
::-webkit-scrollbar-track{{background:transparent;border-radius:10px;}}
::-webkit-scrollbar-thumb{{background:{T['blue']};border-radius:10px;}}

/* app background */
.stApp{{
    background:{T['bg_app']}!important;
    transition:background .35s ease;
}}

/* sidebar */
section[data-testid="stSidebar"]{{
    background:{T['sb_bg']}!important;
    border-right:1px solid {T['border']};
}}
section[data-testid="stSidebar"] *{{color:#C9D1D9;}}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4{{color:#FFFFFF;}}

/* header banner */
.hdr{{
    background:linear-gradient(135deg,{T['hdr_g1']} 0%,{T['hdr_g2']} 55%,{T['hdr_g1']} 100%);
    padding:32px 40px 52px 40px;
    border-radius:0 0 32px 32px;
    color:#FFF;
    box-shadow:0 8px 30px rgba(0,0,0,{'.45' if DK else '.15'});
    position:relative;
    overflow:hidden;
    margin:-80px -52px 28px -52px;
}}
.hdr::before{{
    content:"";position:absolute;
    width:300px;height:300px;border-radius:50%;
    background:radial-gradient(circle,rgba(33,150,243,.18) 0%,transparent 70%);
    top:-110px;right:-50px;
}}
.hdr-title{{
    font-family:'Poppins',sans-serif;font-size:2.55rem;font-weight:800;
    background:linear-gradient(90deg,#fff 0%,#B3E5FC 60%,#80DEEA 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    line-height:1.15;margin-bottom:8px;
}}
.hdr-tag{{font-size:.98rem;color:rgba(255,255,255,.72);font-weight:400;}}
.hdr-wave{{position:absolute;bottom:0;left:0;width:100%;line-height:0;}}
.hdr-wave svg{{display:block;width:100%;height:28px;}}

/* cards */
.card{{
    background:{T['bg_card']};
    border:1px solid {T['border']};
    border-radius:18px;
    padding:22px;
    box-shadow:0 4px 18px rgba(0,0,0,{'.28' if DK else '.04'});
    transition:transform .25s,box-shadow .25s,border-color .25s;
    margin-bottom:18px;
}}
.card:hover{{
    transform:translateY(-4px);
    box-shadow:0 14px 36px rgba(0,0,0,{'.45' if DK else '.08'});
    border-color:{T['blue']}55;
}}
.card-title{{
    font-family:'Poppins',sans-serif;font-size:1.1rem;font-weight:700;
    color:{T['tx_heading']};margin-bottom:4px;
}}
.card-sub{{font-size:.8rem;color:{T['tx_secondary']};margin-bottom:16px;}}

/* stat numbers */
.stat-val{{
    font-family:'Poppins',sans-serif;font-size:1.9rem;font-weight:800;
    color:{T['blue']};line-height:1.1;
}}
.stat-val.g{{color:{T['green']};}}
.stat-lbl{{font-size:.74rem;color:{T['tx_secondary']};font-weight:600;
           text-transform:uppercase;letter-spacing:.6px;margin-top:6px;}}

/* metric box (results) */
.mbox{{
    background:{T['bg_card']};
    border:1px solid {T['border']};
    border-radius:14px;
    padding:18px;
    transition:all .25s;
    position:relative;
}}
.mbox:hover{{border-color:{T['blue']}55;transform:translateY(-3px);
             box-shadow:0 8px 24px rgba(0,0,0,{'.3' if DK else '.06'});}}
.mbox-lbl{{font-size:.76rem;color:{T['tx_secondary']};font-weight:600;margin-bottom:8px;}}
.mbox-val{{
    font-family:'Poppins',sans-serif;font-size:2rem;font-weight:800;color:{T['blue']};
}}
.mbox-val.g{{color:{T['green']};}}

/* appliance rows */
.appl-row{{
    display:flex;align-items:center;gap:12px;
    padding:10px 13px;border-radius:10px;
    background:{T['row_bg']};border:1px solid {T['row_bdr']};
    margin-bottom:9px;
    transition:transform .18s,border-color .18s,background .18s;
}}
.appl-row:hover{{
    transform:translateX(5px);
    border-color:{T['blue']}55;
    background:{T['bg_card2']};
}}
.appl-dot{{
    width:12px;height:12px;border-radius:50%;flex-shrink:0;
}}
.appl-name{{
    font-weight:600;font-size:.88rem;
    color:{T['row_name']};min-width:120px;
}}
.appl-track{{
    flex-grow:1;height:8px;border-radius:8px;
    background:{T['row_track']};overflow:hidden;
}}
.appl-fill{{height:100%;border-radius:8px;}}
.appl-pct{{
    font-family:'Poppins',sans-serif;font-weight:700;
    font-size:.88rem;color:{T['row_pct']};
    min-width:44px;text-align:right;
}}
.appl-kwh{{
    font-size:.72rem;color:{T['row_kwh']};
    min-width:60px;text-align:right;
}}

/* smart tip cards */
.tip-card{{
    background:{T['bg_card']};border:1.5px solid {T['border']};
    border-radius:16px;padding:20px;
    transition:transform .25s,box-shadow .25s;
    position:relative;overflow:hidden;height:100%;
}}
.tip-card:hover{{
    transform:translateY(-5px);
    box-shadow:0 12px 28px rgba(0,0,0,{'.4' if DK else '.07'});
    border-color:{T['blue']}55;
}}
.tip-badge{{
    position:absolute;top:14px;right:14px;
    font-size:.67rem;font-weight:700;padding:4px 10px;
    border-radius:20px;text-transform:uppercase;letter-spacing:.5px;
}}
.badge-g{{background:{T['badge_g_bg']};color:{T['badge_g_tx']};}}
.badge-o{{background:{T['badge_o_bg']};color:{T['badge_o_tx']};}}
.badge-b{{background:{T['badge_b_bg']};color:{T['badge_b_tx']};}}
.badge-p{{background:{T['badge_p_bg']};color:{T['badge_p_tx']};}}
.tip-title{{
    font-family:'Poppins',sans-serif;font-weight:700;font-size:.95rem;
    color:{T['tx_heading']};margin:10px 0 8px 0;padding-right:70px;
}}
.tip-desc{{font-size:.82rem;color:{T['tx_secondary']};line-height:1.55;}}
.tip-cta{{
    font-size:.77rem;font-weight:700;color:{T['tip_action']};
    margin-top:12px;display:block;
}}

/* toggle pill */
.tog-wrap{{
    display:flex;align-items:center;gap:10px;
    background:{T['tog_bg']};border:1px solid {T['border']};
    border-radius:12px;padding:9px 14px;margin-bottom:12px;
    transition:all .2s;
}}
.tog-track{{
    width:42px;height:22px;border-radius:50px;
    background:{T['tog_track']};padding:3px;
    display:flex;align-items:center;
    justify-content:{T['tog_thumb_pos']};
    flex-shrink:0;transition:background .3s;
}}
.tog-thumb{{
    width:16px;height:16px;border-radius:50%;
    background:#FFF;box-shadow:0 1px 4px rgba(0,0,0,.25);
}}
.tog-label{{
    font-size:.82rem;font-weight:600;
    color:{T['tog_tx']};flex:1;
}}
.tog-mode{{
    font-size:.7rem;color:{T['tx_secondary']};
    text-transform:uppercase;letter-spacing:.5px;
}}

/* buttons */
.stButton>button{{
    background:linear-gradient(135deg,{T['blue']} 0%,{T['green']} 100%)!important;
    color:#FFF!important;font-family:'Poppins',sans-serif!important;
    font-weight:700!important;border:none!important;border-radius:12px!important;
    padding:12px 28px!important;
    box-shadow:0 4px 14px rgba(33,150,243,.38)!important;
    letter-spacing:.5px!important;transition:all .2s ease!important;
}}
.stButton>button:hover{{
    transform:scale(1.025)!important;
    box-shadow:0 8px 22px rgba(33,150,243,.55)!important;
}}
.stDownloadButton>button{{
    background:{T['bg_card2']}!important;color:{T['blue']}!important;
    border:2px solid {T['blue']}!important;font-weight:600!important;
    border-radius:12px!important;transition:all .2s!important;
}}
.stDownloadButton>button:hover{{
    background:{T['blue']}!important;color:#FFF!important;transform:scale(1.02)!important;
}}

/* file uploader */
[data-testid="stFileUploader"]{{
    background:{T['bg_card2']};border:2px dashed {T['border']};
    border-radius:14px;padding:8px;transition:border-color .2s;
}}
[data-testid="stFileUploader"]:hover{{border-color:{T['blue']};}}

/* expanders */
[data-testid="stExpander"]{{
    background:{T['bg_card']}!important;border:1px solid {T['border']}!important;
    border-radius:12px!important;
}}

/* dataframe */
.stDataFrame{{border-radius:12px;overflow:hidden;}}

/* success banner */
.suc-banner{{
    background:{T['suc_bg']};border-left:5px solid {T['suc_bdr']};
    border-radius:12px;padding:18px 22px;margin-bottom:20px;
    animation:fadeSlide .45s ease forwards;
}}
@keyframes fadeSlide{{
    from{{opacity:0;transform:translateY(8px);}}
    to  {{opacity:1;transform:translateY(0);}}
}}
.suc-title{{font-weight:700;color:{T['suc_title']};font-size:1.05rem;}}
.suc-sub  {{font-size:.84rem;color:{T['suc_sub']};margin-top:4px;}}

/* tooltip icon */
.tip-i{{
    display:inline-flex;align-items:center;justify-content:center;
    background:{T['bg_card2']};color:{T['tx_secondary']};
    border:1px solid {T['border']};border-radius:50%;
    width:15px;height:15px;font-size:.6rem;font-weight:bold;
    margin-left:5px;vertical-align:middle;cursor:help;
}}

/* comparison box */
.cmp-box{{
    background:{T['compare_bg']};border:1px solid {T['compare_bdr']};
    border-radius:10px;padding:13px 16px;margin-top:12px;
    font-size:.84rem;color:{T['compare_tx']};
}}

/* footer */
.ftr{{
    margin-top:56px;border-top:1px solid {T['footer_bdr']};
    padding:36px 0 20px 0;color:{T['tx_secondary']};
}}
.ftr-logo{{
    font-family:'Poppins',sans-serif;font-weight:700;
    color:{T['tx_heading']};font-size:1.15rem;
}}
.tech-chip{{
    background:{T['bg_card2']};color:{T['tx_secondary']};
    border:1px solid {T['border']};padding:4px 10px;
    border-radius:20px;font-size:.7rem;font-weight:600;
    display:inline-block;margin:3px 3px 3px 0;
}}

/* hero text */
.hero-h{{
    font-family:'Poppins',sans-serif;font-size:2.05rem;font-weight:800;
    color:{T['tx_heading']};line-height:1.25;margin-bottom:14px;
}}
.hero-p{{font-size:.97rem;color:{T['tx_secondary']};line-height:1.65;margin-bottom:20px;}}

/* pulse on savings number */
@keyframes pulse-g{{
    0%,100%{{text-shadow:0 0 0px {T['green']};}}
    50%     {{text-shadow:0 0 16px {T['green']};}}
}}
.pulse{{animation:pulse-g 2.8s ease-in-out infinite;}}

/* section divider label */
.sec-lbl{{
    font-family:'Poppins',sans-serif;font-size:1.15rem;font-weight:700;
    color:{T['tx_heading']};margin-bottom:4px;
}}
.sec-sub{{font-size:.81rem;color:{T['tx_secondary']};margin-bottom:14px;}}

/* inline cost summary under donut */
.cost-row{{
    display:flex;justify-content:space-between;align-items:center;
    padding:8px 12px;border-radius:8px;
    background:{T['row_bg']};border:1px solid {T['row_bdr']};
    margin-bottom:7px;font-size:.83rem;
}}
.cost-name{{color:{T['tx_primary']};font-weight:500;}}
.cost-val {{color:{T['blue']};font-weight:700;}}

/* hero link button */
.hero-btn-link {{
    display: inline-block;
    background: linear-gradient(135deg, {T['blue']} 0%, {T['green']} 100%);
    color: #FFF !important;
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    border-radius: 12px;
    padding: 12px 28px;
    box-shadow: 0 4px 14px rgba(33,150,243,0.38);
    letter-spacing: 0.5px;
    transition: all 0.2s ease;
    text-align: center;
    text-decoration: none !important;
    border: none !important;
    cursor: pointer;
}}
.hero-btn-link:hover {{
    transform: scale(1.025);
    box-shadow: 0 8px 22px rgba(33,150,243,0.55);
    color: #FFF !important;
}}

/* Style Streamlit toggle */
div[data-testid="stToggle"], div[data-testid="stCheckbox"] {{
    background: {T['tog_bg']};
    border: 1px solid {T['border']};
    border-radius: 12px;
    padding: 10px 14px;
    transition: all 0.25s;
    margin-bottom: 12px;
}}
div[data-testid="stToggle"]:hover, div[data-testid="stCheckbox"]:hover {{
    border-color: {T['blue']}55;
}}
div[data-testid="stToggle"] label, div[data-testid="stCheckbox"] label {{
    font-family: 'Poppins', sans-serif;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    color: {T['tog_tx']} !important;
}}

{rtl_css}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ML & DATA UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def extract_features(window: np.ndarray) -> np.ndarray:
    mn=np.mean(window); sd=np.std(window)
    mx=np.max(window);  mi=np.min(window)
    med=np.median(window)
    q75=np.percentile(window,75); q25=np.percentile(window,25)
    slope=np.polyfit(np.arange(len(window),dtype=float),window,1)[0]
    return np.array([mn,sd,mx,mi,mx-mi,med,q75,q25,slope,np.sum(window)/60/1000])

def majority_vote_smooth(labels:list[str], window:int=7)->list[str]:
    h=window//2
    return [Counter(labels[max(0,i-h):min(len(labels),i+h+1)]).most_common(1)[0][0]
            for i in range(len(labels))]

@st.cache_resource(show_spinner=False)
def load_model_and_scaler():
    m=joblib.load("nilm_model.joblib"); sc=joblib.load("nilm_scaler.joblib")
    with open("label_map.json") as f: lmap=json.load(f)
    return m, sc, {int(k):v for k,v in lmap["idx_to_label"].items()}

def run_nilm_inference(df, model, scaler, i2l):
    pw = df["total_power_watts"].values
    n = len(pw)
    if n < WINDOW_SIZE:
        r = df.copy()
        r["appliance"] = "Standby"
        return r
    
    # Vectorized 2D sliding window view
    windows = np.lib.stride_tricks.sliding_window_view(pw, WINDOW_SIZE)
    
    # Parallelized feature statistics
    mn = np.mean(windows, axis=1)
    sd = np.std(windows, axis=1)
    mx = np.max(windows, axis=1)
    mi = np.min(windows, axis=1)
    rng = mx - mi
    med = np.median(windows, axis=1)
    q75 = np.percentile(windows, 75, axis=1)
    q25 = np.percentile(windows, 25, axis=1)
    
    # Matrix-vector linear regression slope
    x_indices = np.arange(WINDOW_SIZE, dtype=float)
    x_dev = x_indices - np.mean(x_indices)
    slope = np.dot(windows, x_dev) / np.sum(x_dev**2)
    
    energy = np.sum(windows, axis=1) / 60.0 / 1000.0
    
    # Bulk feature scaling and prediction
    X = np.column_stack([mn, sd, mx, mi, rng, med, q75, q25, slope, energy])
    X_scaled = scaler.transform(X)
    pred_indices = model.predict(X_scaled)
    pred_labels = [i2l[idx] for idx in pred_indices]
    
    # Align and pad predictions
    half = WINDOW_SIZE // 2
    lbl = ["Standby"] * half + pred_labels + ["Standby"] * (n - len(pred_labels) - half)
    if len(lbl) < n:
        lbl += ["Standby"] * (n - len(lbl))
    elif len(lbl) > n:
        lbl = lbl[:n]
        
    r = df.copy()
    r["appliance"] = majority_vote_smooth(lbl)
    return r

def compute_breakdown(rdf:pd.DataFrame)->dict[str,float]:
    bd={(a,grp["total_power_watts"].sum()/60.0) for a,grp in rdf.groupby("appliance")}
    return dict(sorted({a:w for a,w in bd if w>0}.items(),key=lambda x:-x[1]))

def gen_tips(bd:dict[str,float], rdf:pd.DataFrame, TR:dict)->list[dict]:
    tips=[]
    def get_t(key, default):
        return TR.get(key, default)

    if "AC" in rdf["appliance"].values:
        ts=pd.to_datetime(rdf.loc[rdf["appliance"]=="AC","timestamp"])
        if (ts.dt.hour>=20).sum()>5:
            tips.append({
                "badge": get_t("badge_evening", "Evening Alert"),
                "bc": "badge-o",
                "title": get_t("tip_ac_title", "Evening AC Usage"),
                "desc": get_t("tip_ac_desc", "Heavy after-8PM cooling detected. Raising the setpoint by 4°C saves up to 30% on cooling costs with negligible comfort impact.")
            })
    if "WaterHeater" in rdf["appliance"].values:
        ts=pd.to_datetime(rdf.loc[rdf["appliance"]=="WaterHeater","timestamp"])
        if (ts.dt.hour<8).sum()>5:
            tips.append({
                "badge": get_t("badge_timer", "Timer Tip"),
                "bc": "badge-b",
                "title": get_t("tip_wh_title", "Water Heater Scheduling"),
                "desc": get_t("tip_wh_desc", "Pre-heating water exactly 15 min before your shower eliminates tank standby losses.")
            })
    if bd.get("Refrigerator",0)>700:
        tips.append({
            "badge": get_t("badge_health", "Health"),
            "bc": "badge-g",
            "title": get_t("tip_ref_title", "Fridge Maintenance"),
            "desc": get_t("tip_ref_desc", "Higher-than-typical fridge energy. Clean condenser coils and check door seals to recover 15–20% efficiency.")
        })
    total=sum(bd.values()) or 1
    ac_s=bd.get("AC",0)/total
    if ac_s>0.40:
        pct_val = f"{ac_s*100:.0f}"
        title_str = get_t("tip_ac_share_title", "AC = {pct}% of Your Bill").replace("{pct}", pct_val)
        tips.append({
            "badge": get_t("badge_load", "Load Shift"),
            "bc": "badge-p",
            "title": title_str,
            "desc": get_t("tip_ac_share_desc", "Ceiling fans alongside AC allow raising the thermostat 2°C with identical perceived comfort — 9% savings per degree.")
        })
    if bd.get("WashingMachine",0)>0:
        tips.append({
            "badge": get_t("badge_peak", "Peak Avoid"),
            "bc": "badge-p",
            "title": get_t("tip_wm_title", "Shift Laundry Off-Peak"),
            "desc": get_t("tip_wm_desc", "Running the washer before 7 AM or after 9 PM cuts costs by up to 40% on Time-of-Use tariffs.")
        })
    if not tips:
        tips.append({
            "badge": get_t("badge_good", "All Good"),
            "bc": "badge-g",
            "title": get_t("tip_ok_title", "Profile Looks Efficient"),
            "desc": get_t("tip_ok_desc", "No anomalies detected. Keep monitoring daily to catch baseline changes early.")
        })
    return tips

def sample_csv()->bytes:
    np.random.seed(42)
    from datetime import datetime,timedelta
    START=datetime(2024,1,15); N=1440
    ts=[START+timedelta(minutes=i) for i in range(N)]
    fr=np.array([150. if(i%50)<30 else 0. for i in range(N)])
    ac=np.zeros(N);ac[780:960]=2000.
    ws=np.zeros(N);ws[540:600]=500.
    mw=np.zeros(N);mw[480:485]=1200.;mw[1140:1145]=1200.
    tv=np.zeros(N);tv[1200:1380]=100.
    wh=np.zeros(N);wh[420:450]=1500.;wh[1260:1290]=1500.
    tot=np.clip(fr+ac+ws+mw+tv+wh+np.random.normal(0,20,N),0,None)
    rows=[f"{t.strftime('%Y-%m-%d %H:%M:%S')},{round(p,2)}" for t,p in zip(ts,tot)]
    return ("timestamp,total_power_watts\n"+"\n".join(rows)).encode()


with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center;padding:8px 0 20px 0;
                border-bottom:1px solid {T['border']};">
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin-bottom:10px;">
            <path d="M13 2L3 14H12L11 22L21 10H12L13 2Z" fill="#2196F3" stroke="#2196F3" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        <div style="font-family:'Poppins',sans-serif;font-size:1.05rem;
                    font-weight:700;color:#FFF;margin-top:4px;">
            {TR['title'].upper()}</div>
        <div style="font-size:.68rem;color:#8B949E;letter-spacing:.6px;margin-top:3px;">
            {TR['sidebar_billing']}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # LANGUAGE SELECTOR
    selected_lang_name = st.selectbox(
        "Language / भाषा / اللغة",
        options=list(LANGUAGES.values()),
        index=list(LANGUAGES.keys()).index(st.session_state.get("lang", "en"))
    )
    lang_code = [k for k, v in LANGUAGES.items() if v == selected_lang_name][0]
    if lang_code != st.session_state["lang"]:
        st.session_state["lang"] = lang_code
        st.rerun()

    # CURRENCY SELECTOR
    currency_options = [f"{k} - {v['name']} ({v['symbol']})" for k, v in CURRENCIES.items()]
    current_curr_str = f"{st.session_state['currency']} - {CURRENCIES[st.session_state['currency']]['name']} ({CURRENCIES[st.session_state['currency']]['symbol']})"
    selected_curr_str = st.selectbox(
        "Currency",
        options=currency_options,
        index=currency_options.index(current_curr_str)
    )
    curr_code = selected_curr_str.split(" - ")[0]
    if curr_code != st.session_state["currency"]:
        st.session_state["currency"] = curr_code
        st.rerun()

    st.markdown("---")

    # DISPLAY MODE TOGGLE
    st.markdown(f"""
    <div style="font-size:.68rem;color:#8B949E;text-transform:uppercase;
                letter-spacing:.8px;font-weight:600;margin-bottom:8px;">
        {TR['display_mode']}</div>
    """, unsafe_allow_html=True)

    current_mode = st.session_state["dark_mode"]
    new_mode = st.toggle(
        TR["light_mode_label"] if current_mode else TR["dark_mode_label"],
        value=current_mode,
        key="dark_mode_toggle_switch"
    )
    if new_mode != current_mode:
        st.session_state["dark_mode"] = new_mode
        st.rerun()

    st.markdown("---")

    # CONFIGURATION
    st.markdown(f"""<div style="font-size:.68rem;color:#8B949E;text-transform:uppercase;
        letter-spacing:.8px;font-weight:600;margin-bottom:8px;">{TR['config']}</div>""",
        unsafe_allow_html=True)

    default_rate = round(0.15 * mult, 2)
    min_rate = round(0.01 * mult, 2)
    max_rate = round(2.00 * mult, 2)
    step_rate = 0.01
    if mult >= 10:
        step_rate = 0.1
    if mult >= 500:
        step_rate = 1.0

    RATE = st.number_input(f"{TR['rate_lbl']} ({symbol}/kWh)", min_value=min_rate, max_value=max_rate, value=default_rate, step=step_rate,
                           help=TR['rate_help'])

    st.markdown("---")
    with st.expander(TR['how_works']):
        st.markdown(f"""<div style="font-size:.81rem;line-height:1.65;color:#C9D1D9;">
        {TR['how_works_desc']}</div>""", unsafe_allow_html=True)
    with st.expander(TR['tech_stack']):
        for s in ["Streamlit","scikit-learn","NumPy / Pandas","Plotly","FPDF2"]:
            st.markdown(f"<div style='font-size:.81rem;color:#C9D1D9;'>• {s}</div>",
                        unsafe_allow_html=True)
    with st.expander(TR['faq']):
        st.markdown(f"""<div style="font-size:.81rem;line-height:1.7;color:#C9D1D9;">
        <b>{TR['faq_q1'].split('?')[0]}?</b> {TR['faq_q1'].split('--')[1]}<br><br>
        <b>{TR['faq_q2'].split('?')[0]}?</b> {TR['faq_q2'].split('--')[1]}
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""<div style="text-align:center;font-size:.72rem;color:#8B949E;">
        Beyond the One Number · Hackathon Edition<br>
        <a href="https://huggingface.co" target="_blank"
           style="color:{T['blue']};text-decoration:none;font-weight:600;">
           Hugging Face Space ↗</a></div>""", unsafe_allow_html=True)


st.markdown(f"""
<div class="hdr">
    <div class="hdr-title">{TR['title']}</div>
    <div class="hdr-tag">
        {TR['tagline']}
    </div>
    <div class="hdr-wave">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1440 28"
             preserveAspectRatio="none">
            <path fill="{T['hdr_wave']}"
                  d="M0,14 C240,28 480,0 720,14 C960,28 1200,0 1440,14 L1440,28 L0,28 Z"/>
        </svg>
    </div>
</div>
""", unsafe_allow_html=True)


hc1, hc2 = st.columns([3,2])
with hc1:
    st.markdown(f"""
    <h2 class="hero-h">{TR['hero_h']}</h2>
    <p class="hero-p">{TR['hero_p']}</p>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <a href="#upload-section" class="hero-btn-link" style="text-decoration:none;">
        {TR['get_started']} &darr;
    </a>
    """, unsafe_allow_html=True)
with hc2:
    st.markdown(f"""
    <div class="card" style="background:linear-gradient(135deg,#1565C0,#0D47A1);
         border:none;text-align:center;padding:34px 20px;">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin-bottom:10px; display:inline-block;">
            <path d="M13 2L3 14H12L11 22L21 10H12L13 2Z" fill="#FFF" stroke="#FFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        <h4 style="color:#FFF;font-family:'Poppins',sans-serif;margin:10px 0 7px 0;">
            AI Disaggregation Engine</h4>
        <p style="font-size:.83rem;color:#BBDEFB;line-height:1.55;">
            RandomForest on 10-feature statistical windows.
            100% software — any smart meter, globally.</p>
    </div>
    """, unsafe_allow_html=True)

# 4 stat cards
sc1,sc2,sc3,sc4 = st.columns(4)
for col,(svg_icon,val,lbl,g) in zip(
    [sc1,sc2,sc3,sc4],
    [('<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#00C853" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline><polyline points="17 6 23 6 23 12"></polyline></svg>', TR["avg_savings"], "Average Savings", True),
     ('<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#2196F3" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path><polyline points="9 22 9 12 15 12 15 22"></polyline></svg>', "100M+", TR["homes_comp"], False),
     ('<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#00C853" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>', "85–92%", TR["det_accuracy"], False),
     ('<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#FF9800" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect><path d="M12 2v9"></path><path d="M8 5h8"></path></svg>', f"{symbol}0", TR["hw_cost"], True)]):
    with col:
        st.markdown(f"""
        <div class="card" style="text-align:center;">
            <div style="margin-bottom:8px; display:inline-block;">{svg_icon}</div>
            <div class="stat-val {'g' if g else ''}">{val}</div>
            <div class="stat-lbl">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="card" id="upload-section">
    <div class="card-title">{TR['upload_title']}</div>
    <div class="card-sub">{TR['upload_sub']}</div>
</div>
""", unsafe_allow_html=True)

uc1,uc2 = st.columns([3,1])
with uc1:
    uploaded = st.file_uploader("meter_csv", type=["csv"],
                                label_visibility="collapsed")
with uc2:
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.download_button(TR["sample_csv"], sample_csv(),
                       "sample.csv","text/csv", use_container_width=True)

df_raw:Optional[pd.DataFrame] = None
if uploaded:
    try:
        df_raw = pd.read_csv(uploaded, parse_dates=["timestamp"])
        if not {"timestamp","total_power_watts"}.issubset(df_raw.columns):
            st.error("Missing columns: timestamp, total_power_watts"); df_raw=None
    except Exception as e:
        st.error(f"Parse error: {e}")


if df_raw is not None:

    with st.expander(TR["preview_title"], expanded=False):
        pc1,pc2 = st.columns([4,1])
        with pc2:
            st.download_button(TR["export_raw"],uploaded.getvalue(),
                               "raw.csv","text/csv")
        st.dataframe(df_raw.head(12).style.format({"total_power_watts":"{:.1f} W"}),
                     use_container_width=True)

    # Power profile chart
    st.markdown(f"""
    <div class="card">
        <div class="card-title">{TR['profile_title']}</div>
        <div class="card-sub">{TR['profile_sub']}</div>
    </div>
    """, unsafe_allow_html=True)

    fig_pw = go.Figure()
    fig_pw.add_trace(go.Scatter(
        x=df_raw["timestamp"], y=df_raw["total_power_watts"],
        mode="lines", name="Power",
        line=dict(color=T["blue"], width=2.2),
        fill="tozeroy",
        fillcolor=f"rgba(33,150,243,{'.2' if DK else '.1'})",
        hovertemplate="%{x|%H:%M}<br><b>%{y:.1f} W</b><extra></extra>",
    ))
    fig_pw.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=T["plt_bg"],
        font=dict(color=T["plt_font"],family="Inter"),
        xaxis=dict(showgrid=True,gridcolor=T["plt_grid"],tickformat="%H:%M",
                   rangeslider=dict(visible=True,bgcolor=T["plt_bg"],
                                    bordercolor=T["border"])),
        yaxis=dict(title="Watts",showgrid=True,gridcolor=T["plt_grid"]),
        hovermode="x unified",
        margin=dict(l=15,r=15,t=10,b=5), height=380,
    )
    st.plotly_chart(fig_pw, use_container_width=True)

    # Analyse button
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    if st.button(TR["analyze_btn"], use_container_width=True):
        prog=st.progress(0); stat=st.empty()
        for i,msg in enumerate([TR["loading_model"], TR["extracting_feats"],
                                  TR["running_inf"], TR["smoothing"], TR["done"]]):
            time.sleep(.15); prog.progress((i+1)*20)
            stat.markdown(f"<span style='color:{T['tx_secondary']};font-size:.84rem;'>{msg}</span>",
                          unsafe_allow_html=True)
        try:
            model,scaler,i2l = load_model_and_scaler()
        except FileNotFoundError:
            st.error("Run `python train_model.py` first."); st.stop()
        try:
            rdf = run_nilm_inference(df_raw, model, scaler, i2l)
            st.session_state["result_df"] = rdf
        except Exception as e:
            st.error(f"Inference error: {e}"); st.stop()
        prog.empty(); stat.empty()

    # ── RESULTS ───────────────────────────────────────────────────────────────
    if "result_df" in st.session_state:
        rdf:pd.DataFrame = st.session_state["result_df"]

        st.markdown(f"""
        <div class="suc-banner">
            <div class="suc-title">{TR['complete_title']}</div>
            <div class="suc-sub">
                {TR['complete_sub'].replace("appliance categories", f"{rdf['appliance'].nunique()} " + TR.get("Standby", "Standby"))}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # metrics
        bd        = compute_breakdown(rdf)
        total_wh  = sum(bd.values())
        total_kwh = total_wh/1000
        total_cost= total_kwh*RATE
        savings   = total_cost*SAVINGS_PCT
        co2_kg    = total_kwh*0.385

        mc1,mc2,mc3,mc4 = st.columns(4)
        for col,(lbl,val,cls,tip) in zip(
            [mc1,mc2,mc3,mc4],
            [(TR["total_energy"],   f"{total_kwh:.3f} kWh", "",   TR["energy_tip"]),
             (TR["est_bill"],      f"{symbol}{total_cost:.2f} {code}",  "",   TR["bill_tip"]),
             (TR["potential_saving"], f"{symbol}{savings:.2f} {code}",     "g",  TR["saving_tip"]),
             (TR["co2_footprint"],   f"{co2_kg:.2f} kg",   "g",  TR["co2_tip"])]):
            with col:
                extra='class="pulse"' if cls=="g" else ""
                st.markdown(f"""
                <div class="mbox">
                    <div class="mbox-lbl">{lbl}
                        <span class="tip-i" title="{tip}">?</span>
                    </div>
                    <div class="mbox-val {cls}" {extra}>{val}</div>
                </div>
                """, unsafe_allow_html=True)

        # compare toggle
        cmp_c1,cmp_c2 = st.columns([4,1])
        with cmp_c1:
            st.markdown(f"""
            <div style="font-weight:700;font-size:1rem;color:{T['tx_heading']};">
                {TR['compare_title']}</div>
            <div style="font-size:.82rem;color:{T['tx_secondary']};">
                {TR['compare_sub']}</div>
            """, unsafe_allow_html=True)
        with cmp_c2:
            cmp = st.toggle(TR["show_lbl"], value=st.session_state["compare_mode"])
            st.session_state["compare_mode"] = cmp
        if cmp:
            diff=total_kwh-12.5
            dir_=TR["cmp_higher"] if diff>0 else TR["cmp_lower"]
            col_=T["orange"] if diff>0 else T["green"]
            st.markdown(f"""
            <div class="cmp-box">
                {TR['cmp_baseline']}: <b>12.500 kWh/{TR['cmp_avg'].split(' ')[0]}</b> &nbsp;·&nbsp;
                {TR['cmp_your']}: <b style="color:{col_};">{total_kwh:.3f} kWh</b>
                — <b style="color:{col_};">{abs(diff):.3f} kWh {dir_}</b> {TR['cmp_avg']}.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        # Appliance breakdown  +  Cost donut
        bc1,bc2 = st.columns([3,2])

        with bc1:
            # Build the ENTIRE card — including all rows — as ONE html block
            rows_html = ""
            for app,wh in bd.items():
                share = wh/total_wh*100
                color = APPLIANCE_COLORS.get(app,"#78909C")
                name  = TR.get(app,app)
                rows_html += f"""<div class="appl-row">
<div class="appl-dot" style="background:{color};"></div>
<div class="appl-name">{name}</div>
<div class="appl-track">
<div class="appl-fill" style="width:{share:.1f}%;background:{color};"></div>
</div>
<div class="appl-pct">{share:.1f}%</div>
<div class="appl-kwh">{wh/1000:.3f}&nbsp;kWh</div>
</div>"""

            st.markdown(f"""
            <div class="card" style="min-height:380px;">
                <div class="card-title">{TR['energy_appliance']}</div>
                <div class="card-sub">{TR['energy_appliance_sub']}</div>
                {rows_html}
            </div>
            """, unsafe_allow_html=True)

        with bc2:
            # Donut chart
            labels=[TR.get(k,k) for k in bd]
            costs =[v/1000*RATE for v in bd.values()]
            colors=[APPLIANCE_COLORS.get(k,"#78909C") for k in bd]

            st.markdown(f"""
            <div class="card-title" style="margin-bottom:4px;">{TR['cost_breakdown']}</div>
            <div class="card-sub">{TR['cost_breakdown_sub']}</div>
            """, unsafe_allow_html=True)

            fig_d = go.Figure(go.Pie(
                labels=labels, values=costs, hole=.44,
                marker=dict(colors=colors,
                            line=dict(color=T["bg_card"],width=2)),
                textinfo="percent",
                hovertemplate=f"%{{label}}<br><b>{symbol}%{{value:.3f}} {code}</b><extra></extra>",
            ))
            fig_d.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor ="rgba(0,0,0,0)",
                font=dict(color=T["plt_font"],family="Inter"),
                showlegend=True,
                legend=dict(orientation="h",yanchor="bottom",y=-0.28,
                            xanchor="center",x=0.5,font=dict(size=10)),
                margin=dict(l=5,r=5,t=5,b=5), height=270,
            )
            st.plotly_chart(fig_d, use_container_width=True)

            # Per-appliance cost rows (all in ONE html block)
            cost_rows=""
            for app,wh in list(bd.items())[:5]:
                c=wh/1000*RATE
                color = APPLIANCE_COLORS.get(app,"#78909C")
                cost_rows+=f"""<div class="cost-row">
<span class="cost-name">
<span class="appl-dot" style="background:{color}; width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:6px;"></span>
{TR.get(app,app)}
</span>
<span class="cost-val">{symbol}{c:.3f}</span>
</div>"""
            st.markdown(cost_rows, unsafe_allow_html=True)

        # Timeline
        st.markdown(f"""
        <div class="card-title" style="margin:10px 0 3px 0;">{TR['timeline_title']}</div>
        <div class="card-sub">{TR['timeline_sub']}</div>
        """, unsafe_allow_html=True)

        ptdf=rdf.iloc[::2].copy()
        ptdf["display"]=ptdf["appliance"].map(lambda x:TR.get(x,x))
        ptdf["sz"]=ptdf["total_power_watts"].clip(50,2000)

        fig_tl=px.scatter(
            ptdf, x="timestamp", y="display", color="display", size="sz",
            color_discrete_map={TR.get(k,k):v for k,v in APPLIANCE_COLORS.items()},
            labels={"display":"","timestamp":"Time"}, height=310,
        )
        fig_tl.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=T["plt_bg"],
            font=dict(color=T["plt_font"],family="Inter"),
            xaxis=dict(showgrid=True,gridcolor=T["plt_grid"],tickformat="%H:%M"),
            yaxis=dict(showgrid=False),
            legend=dict(title="",orientation="h",yanchor="bottom",
                        y=1.03,xanchor="center",x=0.5),
            margin=dict(l=10,r=10,t=8,b=8),
        )
        st.plotly_chart(fig_tl, use_container_width=True)

        # Smart Tips
        st.markdown(f"""
        <div class="sec-lbl" style="margin-top:8px;">{TR['rec_title']}</div>
        <div class="sec-sub">{TR['rec_sub']}</div>
        """, unsafe_allow_html=True)

        tips=gen_tips(bd,rdf,TR)
        tc=st.columns(min(len(tips),3))
        for col,tip in zip(tc, tips[:3]):
            with col:
                st.markdown(f"""
                <div class="tip-card">
                    <span class="tip-badge {tip['bc']}">{tip['badge']}</span>
                    <div class="tip-title">{tip['title']}</div>
                    <div class="tip-desc">{tip['desc']}</div>
                    <span class="tip-cta">{TR.get('get_started', 'View Action Plan')} &rarr;</span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        # Exports
        st.markdown(f"""
        <div class="card">
            <div class="card-title">{TR['export_title']}</div>
            <div class="card-sub">{TR['export_sub']}</div>
        </div>
        """, unsafe_allow_html=True)

        ex1,ex2=st.columns(2)
        with ex1:
            out=rdf.copy()
            out["appliance_display"]=out["appliance"].map(lambda x:TR.get(x,x))
            out["energy_wh"]=out["total_power_watts"]/60
            out["cost_local"]=out["energy_wh"]/1000*RATE
            st.download_button(TR["download_csv"],
                               out.to_csv(index=False).encode(),
                               "nilm_results.csv","text/csv",
                               use_container_width=True)
        with ex2:
            if PDF_AVAILABLE:
                with st.spinner(TR["rendering_pdf"]):
                    try:
                        pdf=generate_energy_report(
                            energy_breakdown=bd, total_kwh=total_kwh,
                            total_cost=total_cost, savings=savings,
                            smart_tips=[t["desc"] for t in gen_tips(bd, rdf, TRANSLATIONS["en"])],
                            appliance_data=rdf[["timestamp","appliance"]],
                            currency_symbol=symbol,
                            currency_code=code,
                            rate_per_kwh=RATE,
                        )
                        st.download_button(TR["download_pdf"],pdf,
                                           "Energy_Report.pdf","application/pdf",
                                           use_container_width=True)
                    except Exception as e:
                        st.warning(f"PDF error: {e}")
            else:
                st.info(TR.get("pdf_avail_info", "Install fpdf2 for PDF export."))

        st.markdown(f"""
        <div style="text-align:center;font-size:.77rem;
                    color:{T['tx_secondary']};margin-top:14px;">
            {TR['share_report']}
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown(f"""
    <div style="background:{T['bg_card']};border:2px dashed {T['empty_bdr']};
                border-radius:22px;padding:60px 40px;text-align:center;margin-top:16px;">
        <h4 style="color:{T['tx_heading']};margin-bottom:10px;">
            {TR['awaiting_title']}</h4>
        <p style="color:{T['tx_secondary']};font-size:.93rem;
                  max-width:460px;margin:0 auto;line-height:1.55;">
            {TR['awaiting_sub']}
        </p>
    </div>
    """, unsafe_allow_html=True)


# FOOTER
chips = "".join(f'<span class="tech-chip">{s}</span>' for s in ["Streamlit","scikit-learn","Plotly","Pandas","FPDF2"])
links = f'<p style="margin:0 0 8px 0;"><a href="https://huggingface.co" target="_blank" style="color:{T["blue"]};text-decoration:none;font-size:.82rem;">Hugging Face Space ↗</a></p>' \
        f'<p style="margin:0 0 8px 0;"><a href="https://github.com/peri-perii/NILM.git" target="_blank" style="color:{T["blue"]};text-decoration:none;font-size:.82rem;">GitHub Repository ↗</a></p>' \
        f'<p style="margin:0 0 8px 0;"><a href="https://github.com/peri-perii/NILM.git" target="_blank" style="color:{T["blue"]};text-decoration:none;font-size:.82rem;">Documentation ↗</a></p>'

footer_html = f'<div class="ftr">' \
              f'<div style="display:flex;flex-wrap:wrap;gap:36px;justify-content:space-between;">' \
              f'<div style="flex:1;min-width:220px;">' \
              f'<div class="ftr-logo">{TR["title"]}</div>' \
              f'<p style="font-size:.81rem;line-height:1.65;color:{T["tx_secondary"]};margin-top:10px;">{TR["footer_desc"]}</p>' \
              f'</div>' \
              f'<div style="flex:1;min-width:200px;">' \
              f'<h5 style="color:{T["tx_heading"]};margin-bottom:12px;font-size:.9rem;">Technology</h5>' \
              f'{chips}' \
              f'</div>' \
              f'<div style="flex:1;min-width:180px;">' \
              f'<h5 style="color:{T["tx_heading"]};margin-bottom:12px;font-size:.9rem;">Links</h5>' \
              f'{links}' \
              f'</div>' \
              f'</div>' \
              f'<div style="text-align:center;border-top:1px solid {T["footer_bdr"]};margin-top:28px;padding-top:18px;font-size:.74rem;color:{T["tx_secondary"]};">' \
              f'&copy; 2026 {TR["title"]} &middot; {TR["footer_copy"]}' \
              f'</div>' \
              f'</div>'

st.markdown(footer_html, unsafe_allow_html=True)
