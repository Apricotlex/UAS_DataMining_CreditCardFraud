import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import base64
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield — Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── HELPER: load icon as base64 ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def icon_b64(name):
    path = os.path.join(BASE_DIR, 'assets', f'{name}.png')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    return None

def img_tag(name, size=28):
    b = icon_b64(name)
    if b:
        return f'<img src="data:image/png;base64,{b}" width="{size}" style="vertical-align:middle;margin-right:8px">'
    return ''

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.stApp { background: #07090f; color: #e8eaf0; }

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0e1420 0%, #0a0f1a 100%);
    border-right: 1px solid #1e2740;
}
[data-testid="stSidebar"] * { color: #c8d0e0 !important; }

/* NAV ITEM */
.nav-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 11px 16px;
    border-radius: 10px;
    cursor: pointer;
    margin-bottom: 4px;
    transition: background 0.15s;
    color: #8892a4;
    font-weight: 500;
    font-size: 0.9rem;
}
.nav-item:hover { background: #1a2236; color: #e8eaf0; }
.nav-item.active {
    background: linear-gradient(135deg, #1a3a6e, #162d5a);
    color: #60a5fa !important;
    border: 1px solid #2d4f8a;
}

/* HERO BANNER */
.hero-wrap {
    background: linear-gradient(135deg, #0f1f3d 0%, #0d1829 50%, #07090f 100%);
    border: 1px solid #1e2f52;
    border-radius: 16px;
    padding: 40px 44px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero-wrap::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(96,165,250,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-eyebrow {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #60a5fa;
    margin-bottom: 12px;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    color: #f0f4ff;
    line-height: 1.15;
    margin-bottom: 14px;
}
.hero-title span { color: #60a5fa; }
.hero-desc { font-size: 0.97rem; color: #7a8aaa; line-height: 1.7; max-width: 560px; }

/* STAT CARDS */
.stat-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 14px; margin-bottom: 28px; }
.stat-card {
    background: #0e1420;
    border: 1px solid #1e2740;
    border-radius: 12px;
    padding: 20px 22px;
}
.stat-card .val { font-size: 1.9rem; font-weight: 800; margin-bottom: 4px; }
.stat-card .lbl { font-size: 0.78rem; color: #5a6a84; font-weight: 500; letter-spacing: 0.5px; text-transform: uppercase; }

/* CARD */
.card {
    background: #0e1420;
    border: 1px solid #1e2740;
    border-radius: 12px;
    padding: 22px 24px;
    margin-bottom: 16px;
}
.card-title { font-size: 0.7rem; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; color: #5a6a84; margin-bottom: 14px; }

/* RESULT CARDS */
.result-fraud {
    background: linear-gradient(135deg, #1f0a0a, #2a0d0d);
    border: 1px solid #dc2626;
    border-radius: 12px;
    padding: 22px 24px;
    margin-bottom: 14px;
}
.result-legit {
    background: linear-gradient(135deg, #071a0f, #0a2114);
    border: 1px solid #16a34a;
    border-radius: 12px;
    padding: 22px 24px;
    margin-bottom: 14px;
}

/* BADGE */
.badge {
    display: inline-flex; align-items: center;
    padding: 4px 12px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 700;
    letter-spacing: 0.5px;
}
.badge-blue { background: #1e3a5f; color: #60a5fa; }
.badge-red  { background: #3f1515; color: #f87171; }
.badge-green{ background: #0f2d1a; color: #4ade80; }
.badge-purple{background: #2a1f4a; color: #c084fc; }

/* PROGRESS BAR */
.pbar-wrap { margin-bottom: 12px; }
.pbar-top { display:flex; justify-content:space-between; margin-bottom:5px; font-size:0.82rem; }
.pbar-bg { background:#1a2236; border-radius:4px; height:7px; }
.pbar-fill { height:7px; border-radius:4px; }

/* SECTION HEADER */
.sec-header { display:flex; align-items:center; gap:10px; margin-bottom:6px; }
.sec-header .title { font-size:1.4rem; font-weight:700; color:#f0f4ff; }
.sec-sub { font-size:0.85rem; color:#5a6a84; margin-bottom:22px; }

/* IDENTITY TABLE */
.id-table { width:100%; border-collapse:collapse; }
.id-table td { padding: 8px 0; font-size:0.9rem; border-bottom:1px solid #1a2236; }
.id-table td:first-child { color:#5a6a84; width:140px; }

/* STEP CARD */
.step-card {
    background:#0e1420; border:1px solid #1e2740; border-radius:10px;
    padding:16px 14px; text-align:center;
}
.step-num { font-size:1.4rem; font-weight:800; margin-bottom:6px; }
.step-lbl { font-size:0.78rem; color:#8892a4; line-height:1.4; }

/* BUTTON */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    color: white !important; border: none !important;
    border-radius: 9px !important; padding: 11px 0 !important;
    font-weight: 700 !important; font-size: 0.95rem !important;
    width: 100% !important; letter-spacing: 0.3px !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* TABS */
.stTabs [data-baseweb="tab"] { color: #5a6a84 !important; font-weight:600; }
.stTabs [aria-selected="true"] { color: #60a5fa !important; border-bottom-color: #60a5fa !important; }
button[data-baseweb="tab"] { background: transparent !important; }

/* METRIC */
div[data-testid="metric-container"] {
    background:#0e1420; border:1px solid #1e2740; border-radius:10px; padding:14px;
}

hr { border-color: #1e2740; }
</style>
""", unsafe_allow_html=True)

# ─── LOAD DATA & MODEL ────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    csv_path = os.path.join(BASE_DIR, '..', 'dataset', 'creditcard.csv')
    if not os.path.exists(csv_path):
        csv_path = os.path.join(BASE_DIR, 'creditcard.csv')
    return pd.read_csv(csv_path)

@st.cache_resource
def load_models():
    model_dir = os.path.join(BASE_DIR, '..', 'model')
    rf  = joblib.load(os.path.join(model_dir, 'fraud_detection_model.pkl'))
    iso = joblib.load(os.path.join(model_dir, 'iso_model.pkl'))
    return rf, iso

try:
    df = load_data()
    rf_model, iso_model = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    model_error = str(e)

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:20px 8px 16px">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px">
            <div style="width:34px;height:34px;background:linear-gradient(135deg,#1d4ed8,#3b82f6);
                border-radius:8px;display:flex;align-items:center;justify-content:center;
                font-size:1.1rem">🛡️</div>
            <div>
                <div style="font-size:0.95rem;font-weight:800;color:#f0f4ff">FraudShield</div>
                <div style="font-size:0.7rem;color:#5a6a84;letter-spacing:0.5px">FRAUD DETECTION SYSTEM</div>
            </div>
        </div>
    </div>
    <hr style="margin:0 0 16px">
    """, unsafe_allow_html=True)

    pages = [
        ("home",             "Home"),
        ("dataset_overview", "Dataset Overview"),
        ("prediction",       "Prediction"),
        ("visualization",    "Visualization"),
        ("about",            "About"),
    ]

    if "page" not in st.session_state:
        st.session_state.page = "home"

    for key, label in pages:
        ico = img_tag(key, 22)
        active = "active" if st.session_state.page == key else ""
        if st.button(f"{label}", key=f"nav_{key}", use_container_width=True):
            st.session_state.page = key
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="padding:0 8px">
        <div style="background:#0a0f1a;border:1px solid #1e2740;border-radius:10px;padding:14px">
            <div style="font-size:0.7rem;color:#5a6a84;font-weight:700;letter-spacing:1px;text-transform:uppercase;margin-bottom:10px">Status Model</div>
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
                <div style="width:7px;height:7px;border-radius:50%;background:#4ade80"></div>
                <span style="font-size:0.8rem;color:#c8d0e0">Random Forest</span>
            </div>
            <div style="display:flex;align-items:center;gap:8px">
                <div style="width:7px;height:7px;border-radius:50%;background:#4ade80"></div>
                <span style="font-size:0.8rem;color:#c8d0e0">Isolation Forest</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="padding:16px 8px 0;font-size:0.72rem;color:#3a4a60;text-align:center">
        UAS Data Mining · Sistem Informasi<br>UNESA 2024/2025
    </div>
    """, unsafe_allow_html=True)

page = st.session_state.page

# ═══════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ═══════════════════════════════════════════════════════════
if page == "home":
    ico = img_tag("home", 32)
    st.markdown(f"""
    <div class="hero-wrap">
        <div class="hero-eyebrow">UAS Data Mining · CRISP-DM Framework</div>
        <div class="hero-title">Credit Card<br><span>Fraud Detection</span></div>
        <div class="hero-desc">
            Sistem deteksi transaksi fraud berbasis machine learning menggunakan
            <b style="color:#93c5fd">Random Forest</b> (Classification) dan
            <b style="color:#c084fc">Isolation Forest</b> (Anomaly Detection)
            pada 284.807 transaksi kartu kredit Eropa.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="stat-grid">
        <div class="stat-card">
            <div class="val" style="color:#60a5fa">284K+</div>
            <div class="lbl">Total Transaksi</div>
        </div>
        <div class="stat-card">
            <div class="val" style="color:#f87171">492</div>
            <div class="lbl">Kasus Fraud</div>
        </div>
        <div class="stat-card">
            <div class="val" style="color:#4ade80">99.94%</div>
            <div class="lbl">Akurasi RF</div>
        </div>
        <div class="stat-card">
            <div class="val" style="color:#c084fc">30</div>
            <div class="lbl">Fitur Dataset</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns([1.1, 1])
    with col_a:
        st.markdown("""
        <div class="card">
            <div class="card-title">Deskripsi Proyek</div>
            <p style="color:#8892a4;font-size:0.9rem;line-height:1.75;margin:0">
            Proyek ini menerapkan dua metode Data Mining untuk mendeteksi transaksi
            kartu kredit yang bersifat fraud secara otomatis:
            </p>
            <br>
            <div style="display:flex;gap:10px;margin-bottom:10px;align-items:flex-start">
                <span class="badge badge-blue">Classification</span>
                <div style="font-size:0.87rem;color:#8892a4;line-height:1.6">
                <b style="color:#93c5fd">Random Forest</b> — Model supervised learning
                yang mengklasifikasikan transaksi berdasarkan pola historis berlabel.
                Dilatih dengan SMOTE untuk mengatasi imbalance data.
                </div>
            </div>
            <div style="display:flex;gap:10px;align-items:flex-start">
                <span class="badge badge-purple">Anomaly</span>
                <div style="font-size:0.87rem;color:#8892a4;line-height:1.6">
                <b style="color:#c084fc">Isolation Forest</b> — Model unsupervised
                yang mendeteksi transaksi anomali tanpa membutuhkan label lengkap.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="card">
            <div class="card-title">Identitas Mahasiswa</div>
            <table class="id-table">
                <tr><td>Nama</td><td><b style="color:#f0f4ff">Arvy Revaldy Sevptarius Tarigan</b></td></tr>
                <tr><td>NIM</td><td>24051214227</td></tr>
                <tr><td>Prodi</td><td>Sistem Informasi</td></tr>
                <tr><td>Universitas</td><td>Universitas Negeri Surabaya</td></tr>
                <tr><td>Mata Kuliah</td><td>Data Mining</td></tr>
                <tr><td>Tahun Ajaran</td><td style="border:none">2024/2025</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.7rem;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:#5a6a84;margin-bottom:14px">Alur CRISP-DM</div>', unsafe_allow_html=True)
    cols = st.columns(6)
    steps = [
        ("#60a5fa", "01", "Business\nUnderstanding"),
        ("#c084fc", "02", "Data\nUnderstanding"),
        ("#fb923c", "03", "Data\nPreparation"),
        ("#4ade80", "04", "Modeling\nRF + IF"),
        ("#f87171", "05", "Evaluation\n& Analysis"),
        ("#facc15", "06", "Deployment\nStreamlit"),
    ]
    for col, (color, num, lbl) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div class="step-card" style="border-color:{color}22">
                <div class="step-num" style="color:{color}">{num}</div>
                <div class="step-lbl">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# PAGE 2 — DATASET OVERVIEW
# ═══════════════════════════════════════════════════════════
elif page == "dataset_overview":
    ico = img_tag("dataset_overview", 32)
    st.markdown(f"""
    <div class="sec-header">{ico}<div class="title">Dataset Overview</div></div>
    <div class="sec-sub">Credit Card Fraud Detection Dataset — Kaggle (ULB Machine Learning Group)</div>
    """, unsafe_allow_html=True)

    if not models_loaded:
        st.error(f"Error: {model_error}"); st.stop()

    fraud_count = int(df['Class'].sum())
    legit_count = int(len(df) - fraud_count)
    fraud_pct   = fraud_count / len(df) * 100

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Data", f"{len(df):,}")
    c2.metric("Transaksi Normal", f"{legit_count:,}")
    c3.metric("Transaksi Fraud", f"{fraud_count:,}")
    c4.metric("Rasio Fraud", f"{fraud_pct:.3f}%")

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📋 Info Dataset", "📉 Statistik & Distribusi", "🗂️ Sample Data"])

    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            <div class="card">
                <div class="card-title">Informasi Umum</div>
                <table class="id-table">
                    <tr><td>Sumber</td><td>Kaggle — ULB ML Group</td></tr>
                    <tr><td>Jumlah Record</td><td>284,807 transaksi</td></tr>
                    <tr><td>Jumlah Fitur</td><td>30 fitur + 1 label</td></tr>
                    <tr><td>Missing Value</td><td><span class="badge badge-green">Tidak ada</span></td></tr>
                    <tr><td>Periode</td><td>September 2013 (2 hari)</td></tr>
                    <tr><td>Format</td><td style="border:none">CSV</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            st.markdown("""
            <div class="card">
                <div class="card-title">Keterangan Fitur</div>
                <table class="id-table">
                    <tr><td>Time</td><td>Detik sejak transaksi pertama</td></tr>
                    <tr><td>V1 – V28</td><td>Hasil transformasi PCA (anonim)</td></tr>
                    <tr><td>Amount</td><td>Nominal transaksi (EUR)</td></tr>
                    <tr><td>Class</td><td><span class="badge badge-green">0</span> Normal &nbsp; <span class="badge badge-red">1</span> Fraud</td></tr>
                </table>
                <br>
                <div style="background:#07090f;border-radius:8px;padding:12px">
                    <div style="font-size:0.75rem;color:#5a6a84;margin-bottom:8px;font-weight:600">RASIO KELAS</div>
                    <div class="pbar-wrap">
                        <div class="pbar-top"><span style="color:#4ade80">Normal</span><span style="color:#4ade80">99.83%</span></div>
                        <div class="pbar-bg"><div class="pbar-fill" style="background:#4ade80;width:99.83%"></div></div>
                    </div>
                    <div class="pbar-wrap">
                        <div class="pbar-top"><span style="color:#f87171">Fraud</span><span style="color:#f87171">0.17%</span></div>
                        <div class="pbar-bg"><div class="pbar-fill" style="background:#f87171;width:0.17%"></div></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Distribusi kelas
        fig, ax = plt.subplots(figsize=(5, 3.2), facecolor='#0e1420')
        ax.set_facecolor('#07090f')
        ax.bar(['Normal', 'Fraud'], [legit_count, fraud_count],
               color=['#4ade80', '#f87171'], width=0.45, edgecolor='none', zorder=3)
        ax.set_title('Distribusi Kelas', color='#f0f4ff', fontsize=11, pad=10, fontweight='700')
        ax.tick_params(colors='#5a6a84'); ax.yaxis.grid(True, color='#1e2740', zorder=0)
        for spine in ax.spines.values(): spine.set_color('#1e2740')
        ax.set_ylim(0, legit_count * 1.12)
        for i, (label, val) in enumerate(zip(['Normal', 'Fraud'], [legit_count, fraud_count])):
            ax.text(i, val + legit_count*0.02, f'{val:,}', ha='center', color='#f0f4ff', fontsize=9, fontweight='600')
        st.pyplot(fig); plt.close()

    with tab2:
        st.markdown("**Statistik Deskriptif**")
        st.dataframe(df[['Time','Amount','V1','V2','V3','V4','V14','V17']].describe().round(4), use_container_width=True)

        fig2, axes = plt.subplots(1, 2, figsize=(11, 3.8), facecolor='#0e1420')
        for ax in axes:
            ax.set_facecolor('#07090f')
            for s in ax.spines.values(): s.set_color('#1e2740')
            ax.tick_params(colors='#5a6a84')
            ax.yaxis.grid(True, color='#1e2740', zorder=0)

        axes[0].hist(df[df['Class']==0]['Amount'].clip(upper=500), bins=60, color='#4ade80', alpha=0.85, edgecolor='none', zorder=3)
        axes[0].set_title('Amount — Transaksi Normal', color='#f0f4ff', fontsize=10, fontweight='700')
        axes[0].set_xlabel('Amount (EUR)', color='#5a6a84', fontsize=9)

        axes[1].hist(df[df['Class']==1]['Amount'], bins=40, color='#f87171', alpha=0.85, edgecolor='none', zorder=3)
        axes[1].set_title('Amount — Transaksi Fraud', color='#f0f4ff', fontsize=10, fontweight='700')
        axes[1].set_xlabel('Amount (EUR)', color='#5a6a84', fontsize=9)

        fig2.tight_layout(pad=2)
        st.pyplot(fig2); plt.close()

    with tab3:
        st.markdown("**10 Sample Data Pertama**")
        st.dataframe(df.head(10), use_container_width=True)
        st.caption(f"Shape: {df.shape[0]:,} baris × {df.shape[1]} kolom")

# ═══════════════════════════════════════════════════════════
# PAGE 3 — PREDICTION
# ═══════════════════════════════════════════════════════════
elif page == "prediction":
    ico = img_tag("prediction", 32)
    st.markdown(f"""
    <div class="sec-header">{ico}<div class="title">Deteksi Transaksi</div></div>
    <div class="sec-sub">Masukkan data transaksi untuk dianalisis oleh Random Forest & Isolation Forest</div>
    """, unsafe_allow_html=True)

    if not models_loaded:
        st.error(f"Model tidak dapat dimuat: {model_error}"); st.stop()

    col_form, col_result = st.columns([1.3, 1])

    with col_form:
        st.markdown('<div class="card"><div class="card-title">Input Data Transaksi</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            time_val = st.number_input("Time (detik)", min_value=0.0, max_value=200000.0, value=50000.0, step=1000.0)
        with c2:
            amount_val = st.number_input("Amount (EUR)", min_value=0.0, max_value=30000.0, value=100.0, step=10.0)

        st.markdown('<div style="font-size:0.78rem;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#5a6a84;margin:14px 0 8px">Fitur PCA V1 – V14</div>', unsafe_allow_html=True)
        v_vals = {}
        v_def1 = {'V1':-1.36,'V2':-0.07,'V3':2.54,'V4':1.38,'V5':-0.34,'V6':0.46,'V7':0.24,'V8':0.10,'V9':0.36,'V10':0.09,'V11':-0.55,'V12':-0.62,'V13':-0.99,'V14':-0.31}
        cols4 = st.columns(4)
        for i, (k, v) in enumerate(v_def1.items()):
            with cols4[i % 4]:
                v_vals[k] = st.number_input(k, value=v, format="%.3f", key=f"i_{k}")

        st.markdown('<div style="font-size:0.78rem;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#5a6a84;margin:14px 0 8px">Fitur PCA V15 – V28</div>', unsafe_allow_html=True)
        v_def2 = {'V15':1.11,'V16':1.07,'V17':-0.14,'V18':0.45,'V19':0.32,'V20':0.04,'V21':-0.02,'V22':0.28,'V23':-0.11,'V24':0.07,'V25':0.13,'V26':-0.19,'V27':0.13,'V28':-0.02}
        cols4b = st.columns(4)
        for i, (k, v) in enumerate(v_def2.items()):
            with cols4b[i % 4]:
                v_vals[k] = st.number_input(k, value=v, format="%.3f", key=f"i_{k}")

        st.markdown('</div>', unsafe_allow_html=True)
        predict_btn = st.button("🔍  Deteksi Sekarang")

    with col_result:
        if predict_btn:
            features = ['Time','V1','V2','V3','V4','V5','V6','V7','V8','V9',
                        'V10','V11','V12','V13','V14','V15','V16','V17','V18',
                        'V19','V20','V21','V22','V23','V24','V25','V26','V27','V28','Amount']
            inp = {'Time': time_val, 'Amount': amount_val}
            inp.update(v_vals)
            input_df = pd.DataFrame([inp])[features]

            rf_pred  = rf_model.predict(input_df)[0]
            rf_proba = rf_model.predict_proba(input_df)[0]
            iso_raw  = iso_model.predict(input_df)[0]
            iso_pred = 1 if iso_raw == -1 else 0
            iso_score= iso_model.decision_function(input_df)[0]

            # Hasil RF
            if rf_pred == 1:
                st.markdown("""
                <div class="result-fraud">
                    <div style="font-size:0.7rem;letter-spacing:1.5px;font-weight:700;color:#f87171;margin-bottom:8px">RANDOM FOREST · HASIL DETEKSI</div>
                    <div style="font-size:1.5rem;font-weight:800;color:#fca5a5">⚠️ FRAUD TERDETEKSI</div>
                    <div style="font-size:0.85rem;color:#9a5252;margin-top:6px">Transaksi ini terindikasi sebagai aktivitas fraud</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-legit">
                    <div style="font-size:0.7rem;letter-spacing:1.5px;font-weight:700;color:#4ade80;margin-bottom:8px">RANDOM FOREST · HASIL DETEKSI</div>
                    <div style="font-size:1.5rem;font-weight:800;color:#86efac">✅ TRANSAKSI NORMAL</div>
                    <div style="font-size:0.85rem;color:#2d6a45;margin-top:6px">Transaksi ini terdeteksi sebagai aktivitas normal</div>
                </div>
                """, unsafe_allow_html=True)

            # Probabilitas
            st.markdown(f"""
            <div class="card">
                <div class="card-title">Probabilitas Prediksi</div>
                <div class="pbar-wrap">
                    <div class="pbar-top"><span style="color:#4ade80">Normal</span><span style="color:#4ade80;font-weight:700">{rf_proba[0]*100:.2f}%</span></div>
                    <div class="pbar-bg"><div class="pbar-fill" style="background:#4ade80;width:{rf_proba[0]*100:.1f}%"></div></div>
                </div>
                <div class="pbar-wrap">
                    <div class="pbar-top"><span style="color:#f87171">Fraud</span><span style="color:#f87171;font-weight:700">{rf_proba[1]*100:.2f}%</span></div>
                    <div class="pbar-bg"><div class="pbar-fill" style="background:#f87171;width:{rf_proba[1]*100:.1f}%"></div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Isolation Forest
            iso_label = "⚠️ ANOMALI TERDETEKSI" if iso_pred == 1 else "✅ NORMAL"
            iso_color = "#f87171" if iso_pred == 1 else "#4ade80"
            iso_bg    = "result-fraud" if iso_pred == 1 else "result-legit"
            st.markdown(f"""
            <div class="{iso_bg}">
                <div style="font-size:0.7rem;letter-spacing:1.5px;font-weight:700;color:{iso_color};margin-bottom:8px">ISOLATION FOREST · ANOMALY DETECTION</div>
                <div style="font-size:1.2rem;font-weight:800;color:{iso_color}">{iso_label}</div>
                <div style="font-size:0.82rem;color:#5a6a84;margin-top:6px">Anomaly Score: <b style="color:#c8d0e0">{iso_score:.4f}</b> &nbsp;·&nbsp; Semakin negatif = semakin anomali</div>
            </div>
            """, unsafe_allow_html=True)

            # Ringkasan
            st.markdown(f"""
            <div class="card">
                <div class="card-title">Ringkasan</div>
                <table class="id-table">
                    <tr><td>Amount</td><td><b>€{amount_val:,.2f}</b></td></tr>
                    <tr><td>Random Forest</td><td><span class="badge {'badge-red' if rf_pred==1 else 'badge-green'}">{'FRAUD' if rf_pred==1 else 'Normal'}</span></td></tr>
                    <tr><td>Isolation Forest</td><td><span class="badge {'badge-red' if iso_pred==1 else 'badge-green'}">{'Anomali' if iso_pred==1 else 'Normal'}</span></td></tr>
                    <tr><td>Confidence RF</td><td style="border:none"><b style="color:#60a5fa">{max(rf_proba)*100:.2f}%</b></td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="card" style="text-align:center;padding:60px 24px">
                <div style="font-size:3rem;margin-bottom:14px">🔍</div>
                <div style="color:#5a6a84;font-size:0.9rem">Isi form transaksi di sebelah kiri<br>lalu klik <b style="color:#60a5fa">Deteksi Sekarang</b></div>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# PAGE 4 — VISUALIZATION
# ═══════════════════════════════════════════════════════════
elif page == "visualization":
    ico = img_tag("visualization", 32)
    st.markdown(f"""
    <div class="sec-header">{ico}<div class="title">Visualisasi & Evaluasi</div></div>
    <div class="sec-sub">Perbandingan performa model dan analisis hasil deteksi fraud</div>
    """, unsafe_allow_html=True)

    if not models_loaded:
        st.error("Model tidak dapat dimuat."); st.stop()

    @st.cache_data
    def get_eval(_df):
        _df = _df.dropna()
        X = _df.drop('Class', axis=1)
        y = _df['Class']
        _, X_t, _, y_t = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        return X_t, y_t

    X_test, y_test   = get_eval(df.copy())
    rf_pred_t        = rf_model.predict(X_test)
    rf_proba_t       = rf_model.predict_proba(X_test)[:,1]
    iso_pred_t       = np.where(iso_model.predict(X_test)==-1, 1, 0)

    def dfig(w=10, h=4, nc=1):
        fig, axes = plt.subplots(1, nc, figsize=(w, h), facecolor='#0e1420')
        if nc == 1: axes = [axes]
        for ax in axes:
            ax.set_facecolor('#07090f')
            for s in ax.spines.values(): s.set_color('#1e2740')
            ax.tick_params(colors='#5a6a84')
            ax.yaxis.grid(True, color='#1e2740', zorder=0)
        return fig, axes

    tab1, tab2, tab3 = st.tabs(["📊 Perbandingan Model", "🔥 Confusion Matrix", "📉 Feature Importance"])

    with tab1:
        metrics  = ['Accuracy','Precision','Recall','F1-Score']
        rf_vals  = [0.9994, 0.94, 0.82, 0.87]
        iso_vals = [0.9971, 0.28, 0.35, 0.31]

        fig, axes = dfig(10, 4.5)
        ax = axes[0]
        x = np.arange(len(metrics)); w = 0.3
        b1 = ax.bar(x-w/2, rf_vals,  w, color='#60a5fa', label='Random Forest',   edgecolor='none', zorder=3)
        b2 = ax.bar(x+w/2, iso_vals, w, color='#c084fc', label='Isolation Forest', edgecolor='none', zorder=3)
        ax.set_xticks(x); ax.set_xticklabels(metrics, color='#c8d0e0', fontsize=10)
        ax.set_ylim(0, 1.18)
        ax.set_title('Perbandingan Metrik Evaluasi', color='#f0f4ff', fontsize=12, pad=12, fontweight='700')
        ax.legend(facecolor='#0e1420', edgecolor='#1e2740', labelcolor='#c8d0e0', fontsize=9)
        for bar in b1:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.025,
                    f'{bar.get_height():.2f}', ha='center', color='#60a5fa', fontsize=9, fontweight='700')
        for bar in b2:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.025,
                    f'{bar.get_height():.2f}', ha='center', color='#c084fc', fontsize=9, fontweight='700')
        st.pyplot(fig); plt.close()

        # ROC
        fpr, tpr, _ = roc_curve(y_test, rf_proba_t)
        roc_auc     = auc(fpr, tpr)
        fig2, axes2 = dfig(7, 4)
        axes2[0].plot(fpr, tpr, color='#60a5fa', lw=2.5, label=f'AUC = {roc_auc:.4f}', zorder=3)
        axes2[0].plot([0,1],[0,1], color='#3a4a60', linestyle='--', lw=1)
        axes2[0].fill_between(fpr, tpr, alpha=0.08, color='#60a5fa')
        axes2[0].set_xlabel('False Positive Rate', color='#5a6a84', fontsize=9)
        axes2[0].set_ylabel('True Positive Rate', color='#5a6a84', fontsize=9)
        axes2[0].set_title('ROC Curve — Random Forest', color='#f0f4ff', fontsize=11, fontweight='700')
        axes2[0].legend(facecolor='#0e1420', edgecolor='#1e2740', labelcolor='#c8d0e0')
        st.pyplot(fig2); plt.close()

        st.markdown("**Tabel Perbandingan**")
        st.dataframe(pd.DataFrame({
            'Model':['Random Forest','Isolation Forest'],
            'Jenis':['Classification','Anomaly Detection'],
            'Accuracy':['99.94%','99.71%'],
            'Precision':['94%','28%'],
            'Recall':['82%','35%'],
            'F1-Score':['87%','31%'],
            'Terpilih':['✅ Model Terbaik','—'],
        }), use_container_width=True, hide_index=True)

    with tab2:
        def plot_cm(ax, cm, title):
            ax.imshow(cm, cmap='Blues')
            ax.set_title(title, color='#f0f4ff', fontsize=10, pad=10, fontweight='700')
            ax.set_xlabel('Predicted', color='#5a6a84', fontsize=9)
            ax.set_ylabel('Actual', color='#5a6a84', fontsize=9)
            ax.set_xticks([0,1]); ax.set_yticks([0,1])
            ax.set_xticklabels(['Normal','Fraud'], color='#c8d0e0')
            ax.set_yticklabels(['Normal','Fraud'], color='#c8d0e0')
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f'{cm[i,j]:,}', ha='center', va='center',
                            color='white', fontsize=12, fontweight='800')

        cm_rf  = confusion_matrix(y_test, rf_pred_t)
        cm_iso = confusion_matrix(y_test, iso_pred_t)

        c1, c2 = st.columns(2)
        with c1:
            fig, axes = dfig(5, 4); plot_cm(axes[0], cm_rf, 'Confusion Matrix — Random Forest')
            st.pyplot(fig); plt.close()
        with c2:
            fig, axes = dfig(5, 4); plot_cm(axes[0], cm_iso, 'Confusion Matrix — Isolation Forest')
            st.pyplot(fig); plt.close()

        st.markdown("**Classification Report — Random Forest**")
        report_df = pd.DataFrame(
            classification_report(y_test, rf_pred_t, target_names=['Normal','Fraud'], output_dict=True)
        ).T.round(4)
        st.dataframe(report_df, use_container_width=True)

    with tab3:
        imp_df = pd.DataFrame({
            'Feature': X_test.columns,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=True).tail(15)

        fig, axes = dfig(8, 5)
        colors_bar = ['#60a5fa' if v > 0.05 else '#2a3a54' for v in imp_df['Importance']]
        axes[0].barh(imp_df['Feature'], imp_df['Importance'], color=colors_bar, edgecolor='none', zorder=3)
        axes[0].set_title('Top 15 Feature Importance — Random Forest', color='#f0f4ff', fontsize=11, fontweight='700')
        axes[0].set_xlabel('Importance', color='#5a6a84', fontsize=9)
        axes[0].tick_params(colors='#c8d0e0')
        st.pyplot(fig); plt.close()

        st.markdown("""
        <div class="card">
            <div class="card-title">Interpretasi</div>
            <p style="color:#8892a4;font-size:0.88rem;line-height:1.7;margin:0">
            Fitur berwarna biru (importance > 0.05) memiliki pengaruh terbesar dalam keputusan model.
            <b style="color:#93c5fd">V14, V17, V12, V10</b> secara konsisten menjadi fitur paling penting
            dalam mendeteksi fraud pada dataset kartu kredit ini, sesuai dengan penelitian sebelumnya.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# PAGE 5 — ABOUT
# ═══════════════════════════════════════════════════════════
elif page == "about":
    ico = img_tag("about", 32)
    st.markdown(f"""
    <div class="sec-header">{ico}<div class="title">About</div></div>
    <div class="sec-sub">Penjelasan metode, dataset, dan informasi proyek</div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🤖 Metode", "🗄️ Dataset", "📁 Proyek"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            <div class="card">
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px">
                    <span class="badge badge-blue" style="font-size:0.7rem">Classification</span>
                    <div><b style="color:#93c5fd;font-size:1.05rem">Random Forest</b></div>
                </div>
                <p style="color:#8892a4;font-size:0.87rem;line-height:1.75">
                Algoritma ensemble yang membangun banyak decision tree secara paralel
                menggunakan <i>bagging</i> dan <i>random feature selection</i>.
                Hasil prediksi ditentukan melalui voting mayoritas dari semua tree.
                </p>
                <div style="margin-top:14px">
                    <div style="font-size:0.72rem;color:#5a6a84;font-weight:700;letter-spacing:1px;margin-bottom:8px">KELEBIHAN</div>
                    <div style="font-size:0.85rem;color:#8892a4;line-height:1.8">
                    ✅ Akurasi tinggi pada data imbalanced<br>
                    ✅ Tahan terhadap overfitting<br>
                    ✅ Menghasilkan feature importance<br>
                    ✅ Tidak memerlukan feature scaling
                    </div>
                </div>
                <div style="margin-top:12px">
                    <div style="font-size:0.72rem;color:#5a6a84;font-weight:700;letter-spacing:1px;margin-bottom:8px">HASIL EVALUASI</div>
                    <div style="display:flex;gap:8px;flex-wrap:wrap">
                        <span class="badge badge-green">Acc 99.94%</span>
                        <span class="badge badge-blue">F1 87%</span>
                        <span class="badge badge-purple">AUC ~0.99</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class="card">
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px">
                    <span class="badge badge-purple" style="font-size:0.7rem">Anomaly Detection</span>
                    <div><b style="color:#c084fc;font-size:1.05rem">Isolation Forest</b></div>
                </div>
                <p style="color:#8892a4;font-size:0.87rem;line-height:1.75">
                Algoritma unsupervised yang mengisolasi observasi menggunakan pohon isolasi.
                Anomali diisolasi lebih cepat (path lebih pendek) dibanding data normal,
                tanpa memerlukan label data.
                </p>
                <div style="margin-top:14px">
                    <div style="font-size:0.72rem;color:#5a6a84;font-weight:700;letter-spacing:1px;margin-bottom:8px">KELEBIHAN</div>
                    <div style="font-size:0.85rem;color:#8892a4;line-height:1.8">
                    ✅ Tidak memerlukan label (unsupervised)<br>
                    ✅ Efisien untuk data berdimensi tinggi<br>
                    ✅ Tidak mengasumsikan distribusi data
                    </div>
                </div>
                <div style="margin-top:12px">
                    <div style="font-size:0.72rem;color:#5a6a84;font-weight:700;letter-spacing:1px;margin-bottom:8px">HASIL EVALUASI</div>
                    <div style="display:flex;gap:8px;flex-wrap:wrap">
                        <span class="badge badge-green">Acc 99.71%</span>
                        <span class="badge badge-red">F1 31%</span>
                        <span class="badge badge-purple">Recall 35%</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <div class="card-title">Framework CRISP-DM</div>
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px">
                <div style="background:#07090f;border-radius:8px;padding:14px;border:1px solid #1e2740">
                    <div style="color:#60a5fa;font-weight:700;font-size:0.85rem;margin-bottom:6px">1. Business Understanding</div>
                    <div style="color:#5a6a84;font-size:0.8rem">Mendefinisikan tujuan deteksi fraud dan target metrik bisnis</div>
                </div>
                <div style="background:#07090f;border-radius:8px;padding:14px;border:1px solid #1e2740">
                    <div style="color:#c084fc;font-weight:700;font-size:0.85rem;margin-bottom:6px">2. Data Understanding</div>
                    <div style="color:#5a6a84;font-size:0.8rem">EDA 284K transaksi, analisis distribusi dan class imbalance</div>
                </div>
                <div style="background:#07090f;border-radius:8px;padding:14px;border:1px solid #1e2740">
                    <div style="color:#fb923c;font-weight:700;font-size:0.85rem;margin-bottom:6px">3. Data Preparation</div>
                    <div style="color:#5a6a84;font-size:0.8rem">dropna(), train-test split stratified, SMOTE oversampling</div>
                </div>
                <div style="background:#07090f;border-radius:8px;padding:14px;border:1px solid #1e2740">
                    <div style="color:#4ade80;font-weight:700;font-size:0.85rem;margin-bottom:6px">4. Modeling</div>
                    <div style="color:#5a6a84;font-size:0.8rem">Random Forest (n=100) + Isolation Forest (contamination=0.002)</div>
                </div>
                <div style="background:#07090f;border-radius:8px;padding:14px;border:1px solid #1e2740">
                    <div style="color:#f87171;font-weight:700;font-size:0.85rem;margin-bottom:6px">5. Evaluation</div>
                    <div style="color:#5a6a84;font-size:0.8rem">Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC-AUC</div>
                </div>
                <div style="background:#07090f;border-radius:8px;padding:14px;border:1px solid #1e2740">
                    <div style="color:#facc15;font-weight:700;font-size:0.85rem;margin-bottom:6px">6. Deployment</div>
                    <div style="color:#5a6a84;font-size:0.8rem">Implementasi web app Streamlit multi-page dengan visualisasi interaktif</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("""
        <div class="card">
            <div class="card-title">Credit Card Fraud Detection Dataset</div>
            <div style="display:flex;gap:10px;margin-bottom:16px;flex-wrap:wrap">
                <span class="badge badge-blue">Kaggle</span>
                <span class="badge badge-purple">ULB Machine Learning Group</span>
                <span class="badge badge-green">284,807 Records</span>
            </div>
            <p style="color:#8892a4;font-size:0.88rem;line-height:1.75">
            Dataset ini berisi transaksi kartu kredit pemegang kartu Eropa pada September 2013 selama dua hari.
            Terdapat 284.807 transaksi dengan hanya 492 kasus fraud (0.172%) — sangat tidak seimbang.
            </p>
            <br>
            <div style="font-size:0.72rem;color:#5a6a84;font-weight:700;letter-spacing:1px;margin-bottom:10px">DETAIL FITUR</div>
            <table class="id-table">
                <tr><td>Time</td><td>Detik yang berlalu sejak transaksi pertama dalam dataset</td></tr>
                <tr><td>V1 – V28</td><td>Hasil transformasi PCA untuk menjaga kerahasiaan data nasabah</td></tr>
                <tr><td>Amount</td><td>Nominal transaksi dalam EUR</td></tr>
                <tr><td>Class</td><td><span class="badge badge-green">0</span> Transaksi Normal &nbsp; <span class="badge badge-red">1</span> Transaksi Fraud</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        st.markdown("""
        <div class="card">
            <div class="card-title">Informasi Proyek</div>
            <table class="id-table">
                <tr><td>Nama Proyek</td><td><b style="color:#f0f4ff">Credit Card Fraud Detection</b></td></tr>
                <tr><td>Mahasiswa</td><td><b style="color:#f0f4ff">Arvy Revaldy Sevptarius Tarigan</b></td></tr>
                <tr><td>NIM</td><td>24051214227</td></tr>
                <tr><td>Program Studi</td><td>Sistem Informasi — UNESA</td></tr>
                <tr><td>Mata Kuliah</td><td>Data Mining</td></tr>
                <tr><td>Framework</td><td>CRISP-DM</td></tr>
                <tr><td>Metode</td><td><span class="badge badge-blue">Random Forest</span> &nbsp; <span class="badge badge-purple">Isolation Forest</span></td></tr>
                <tr><td>Tools</td><td>Python · Scikit-learn · Streamlit · Pandas · Matplotlib</td></tr>
                <tr><td>Dataset</td><td>Kaggle — Credit Card Fraud Detection (ULB)</td></tr>
                <tr><td>Tahun Ajaran</td><td style="border:none">2024/2025</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)