import os
import joblib
import gdown
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

# Google Drive File ID dataset creditcard.csv
GOOGLE_DRIVE_FILE_ID = "11pCerupC1jOV8_AFEITzAeafuw8C8l_f"

FEATURES = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
    "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18",
    "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27",
    "V28", "Amount"
]

# Hasil evaluasi dari notebook
RF_METRICS = {
    "Accuracy": 99.96,
    "Precision": 87.75,
    "Recall": 89.58,
    "F1-score": 88.66
}

ISO_METRICS = {
    "Accuracy": 99.71,
    "Precision": 24.32,
    "Recall": 28.13,
    "F1-score": 26.09
}


# =========================================================
# STYLE
# =========================================================
st.markdown(
    """
    <style>
    .big-title {
        font-size: 38px;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0px;
    }
    .subtitle {
        font-size: 18px;
        color: #475569;
        margin-top: 5px;
    }
    .card {
        padding: 20px;
        border-radius: 16px;
        background-color: white;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
        margin-bottom: 18px;
    }
    .success-card {
        padding: 22px;
        border-radius: 16px;
        background-color: #dcfce7;
        border-left: 6px solid #16a34a;
        color: #14532d;
        font-size: 20px;
        font-weight: 700;
    }
    .danger-card {
        padding: 22px;
        border-radius: 16px;
        background-color: #fee2e2;
        border-left: 6px solid #dc2626;
        color: #7f1d1d;
        font-size: 20px;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD DATASET FROM GOOGLE DRIVE
# =========================================================
@st.cache_data(show_spinner=True)
def load_dataset():
    dataset_dir = os.path.join(ROOT_DIR, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)

    dataset_path = os.path.join(dataset_dir, "creditcard.csv")

    # Jika dataset sudah pernah terdownload, langsung baca
    if os.path.exists(dataset_path):
        return pd.read_csv(dataset_path)

    # Download dataset dari Google Drive
    download_url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"

    try:
        gdown.download(download_url, dataset_path, quiet=False)
    except Exception as e:
        st.error(f"Gagal mengunduh dataset dari Google Drive: {e}")
        st.stop()

    if not os.path.exists(dataset_path):
        st.error("Dataset gagal diunduh. Pastikan link Google Drive sudah public.")
        st.stop()

    return pd.read_csv(dataset_path)


# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    model_paths = [
        os.path.join(ROOT_DIR, "model", "fraud_detection_model.pkl"),
        os.path.join(BASE_DIR, "..", "model", "fraud_detection_model.pkl"),
        os.path.join("model", "fraud_detection_model.pkl")
    ]

    for path in model_paths:
        if os.path.exists(path):
            return joblib.load(path)

    return None


# =========================================================
# LOAD APP DATA
# =========================================================
with st.spinner("Memuat dataset dan model..."):
    df = load_dataset()
    model = load_model()

if "Class" not in df.columns:
    st.error("Kolom 'Class' tidak ditemukan pada dataset. Pastikan file creditcard.csv sesuai dataset Kaggle.")
    st.stop()

missing_features = [col for col in FEATURES if col not in df.columns]
if missing_features:
    st.error(f"Beberapa fitur tidak ditemukan pada dataset: {missing_features}")
    st.stop()


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("💳 Fraud Detection")
menu = st.sidebar.radio(
    "Pilih Menu",
    [
        "Home",
        "Dataset Overview",
        "Prediction / Analysis",
        "Visualization",
        "About"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Aplikasi ini menggunakan model Random Forest untuk mendeteksi transaksi kartu kredit normal atau fraud."
)


# =========================================================
# HOME
# =========================================================
if menu == "Home":
    st.markdown('<p class="big-title">Deteksi Penipuan Kartu Kredit</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Classification dan Anomaly Detection Menggunakan Data Mining</p>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown(
        """
        <div class="card">
        <h3>Deskripsi Proyek</h3>
        <p>
        Aplikasi ini dibuat untuk mendeteksi transaksi penipuan kartu kredit menggunakan pendekatan
        Data Mining. Proyek ini menerapkan dua metode, yaitu <b>Classification</b> menggunakan
        Random Forest dan <b>Anomaly Detection</b> menggunakan Isolation Forest.
        </p>
        <p>
        Model terbaik berdasarkan hasil evaluasi adalah <b>Random Forest</b>, sehingga model tersebut
        digunakan pada aplikasi web berbasis Streamlit.
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Dataset", "Credit Card Fraud")
    with col2:
        st.metric("Metode Utama", "Random Forest")
    with col3:
        st.metric("Framework", "CRISP-DM")

    st.markdown("### Identitas")
    st.write("**Nama:** Bindi Achmad")
    st.write("**Proyek:** UAS Data Mining")
    st.write("**Topik:** Credit Card Fraud Detection")


# =========================================================
# DATASET OVERVIEW
# =========================================================
elif menu == "Dataset Overview":
    st.title("Dataset Overview")

    total_data = len(df)
    fraud_count = int(df["Class"].sum())
    normal_count = total_data - fraud_count
    fraud_percent = (fraud_count / total_data) * 100

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Data", f"{total_data:,}")
    with col2:
        st.metric("Transaksi Normal", f"{normal_count:,}")
    with col3:
        st.metric("Transaksi Fraud", f"{fraud_count:,}")
    with col4:
        st.metric("Persentase Fraud", f"{fraud_percent:.3f}%")

    st.markdown("### Informasi Dataset")
    info_df = pd.DataFrame({
        "Keterangan": [
            "Sumber Dataset",
            "Jumlah Record",
            "Jumlah Fitur Input",
            "Target",
            "Jenis Data",
            "Kelas Normal",
            "Kelas Fraud"
        ],
        "Nilai": [
            "Kaggle Credit Card Fraud Detection",
            f"{total_data:,} transaksi",
            "30 fitur",
            "Class",
            "Numerik",
            f"{normal_count:,} transaksi",
            f"{fraud_count:,} transaksi"
        ]
    })
    st.dataframe(info_df, use_container_width=True)

    st.markdown("### Sample Data")
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown("### Statistik Sederhana")
    st.dataframe(df.describe(), use_container_width=True)


# =========================================================
# PREDICTION / ANALYSIS
# =========================================================
elif menu == "Prediction / Analysis":
    st.title("Prediction / Analysis")

    if model is None:
        st.error(
            "Model `fraud_detection_model.pkl` tidak ditemukan. "
            "Pastikan folder `model` sudah diupload ke GitHub dan berisi file model."
        )
        st.stop()

    st.write(
        "Masukkan nilai fitur transaksi kartu kredit. "
        "Sistem akan memprediksi apakah transaksi termasuk normal atau fraud."
    )

    st.info(
        "Catatan: Fitur V1 sampai V28 merupakan fitur anonim hasil transformasi PCA dari dataset Kaggle."
    )

    st.markdown("### Gunakan Contoh Data dari Dataset")

    sample_option = st.selectbox(
        "Pilih contoh input",
        [
            "Input manual",
            "Contoh transaksi normal dari dataset",
            "Contoh transaksi fraud dari dataset"
        ]
    )

    default_values = {feature: 0.0 for feature in FEATURES}

    if sample_option == "Contoh transaksi normal dari dataset":
        sample_row = df[df["Class"] == 0].iloc[0]
        default_values = {feature: float(sample_row[feature]) for feature in FEATURES}

    elif sample_option == "Contoh transaksi fraud dari dataset":
        sample_row = df[df["Class"] == 1].iloc[0]
        default_values = {feature: float(sample_row[feature]) for feature in FEATURES}

    with st.form("prediction_form"):
        col_time, col_amount = st.columns(2)

        with col_time:
            time_value = st.number_input(
                "Time",
                value=float(default_values["Time"]),
                step=1.0
            )

        with col_amount:
            amount_value = st.number_input(
                "Amount",
                value=float(default_values["Amount"]),
                step=1.0
            )

        st.markdown("### Input Fitur V1 - V28")

        values = {}
        cols = st.columns(4)

        for i in range(1, 29):
            feature_name = f"V{i}"
            with cols[(i - 1) % 4]:
                values[feature_name] = st.number_input(
                    feature_name,
                    value=float(default_values[feature_name]),
                    step=0.01,
                    format="%.6f"
                )

        submitted = st.form_submit_button("Prediksi Transaksi")

    if submitted:
        input_data = {
            "Time": time_value,
            **values,
            "Amount": amount_value
        }

        input_df = pd.DataFrame([input_data])
        input_df = input_df[FEATURES]

        try:
            prediction = model.predict(input_df)[0]

            if hasattr(model, "predict_proba"):
                probability = model.predict_proba(input_df)[0]
                normal_prob = probability[0] * 100
                fraud_prob = probability[1] * 100
            else:
                normal_prob = None
                fraud_prob = None

            if prediction == 0:
                st.markdown(
                    """
                    <div class="success-card">
                    Hasil Prediksi: Transaksi NORMAL
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """
                    <div class="danger-card">
                    Hasil Prediksi: Transaksi FRAUD / MENCURIGAKAN
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            if normal_prob is not None and fraud_prob is not None:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Probabilitas Normal", f"{normal_prob:.2f}%")
                with col2:
                    st.metric("Probabilitas Fraud", f"{fraud_prob:.2f}%")

            st.markdown("### Data Input")
            st.dataframe(input_df, use_container_width=True)

        except Exception as e:
            st.error(f"Terjadi error saat prediksi: {e}")


# =========================================================
# VISUALIZATION
# =========================================================
elif menu == "Visualization":
    st.title("Visualization")

    total_data = len(df)
    fraud_count = int(df["Class"].sum())
    normal_count = total_data - fraud_count

    st.markdown("### Distribusi Kelas Dataset")

    class_df = pd.DataFrame({
        "Kelas": ["Normal", "Fraud"],
        "Jumlah": [normal_count, fraud_count]
    })

    fig, ax = plt.subplots()
    ax.bar(class_df["Kelas"], class_df["Jumlah"])
    ax.set_title("Distribusi Transaksi Normal dan Fraud")
    ax.set_xlabel("Kelas")
    ax.set_ylabel("Jumlah Transaksi")
    st.pyplot(fig)

    st.markdown("### Perbandingan Evaluasi Model")

    metrics_df = pd.DataFrame({
        "Metrik": list(RF_METRICS.keys()),
        "Random Forest": list(RF_METRICS.values()),
        "Isolation Forest": list(ISO_METRICS.values())
    })

    st.dataframe(metrics_df, use_container_width=True)

    fig2, ax2 = plt.subplots()
    x = np.arange(len(metrics_df["Metrik"]))
    width = 0.35

    ax2.bar(x - width / 2, metrics_df["Random Forest"], width, label="Random Forest")
    ax2.bar(x + width / 2, metrics_df["Isolation Forest"], width, label="Isolation Forest")
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics_df["Metrik"])
    ax2.set_ylabel("Persentase (%)")
    ax2.set_title("Perbandingan Performa Model")
    ax2.legend()
    st.pyplot(fig2)

    st.markdown("### Confusion Matrix Random Forest")

    cm_rf = pd.DataFrame(
        [[52569, 12], [10, 86]],
        index=["Aktual Normal", "Aktual Fraud"],
        columns=["Prediksi Normal", "Prediksi Fraud"]
    )

    st.dataframe(cm_rf, use_container_width=True)

    st.markdown("### Confusion Matrix Isolation Forest")

    cm_iso = pd.DataFrame(
        [[52497, 84], [69, 27]],
        index=["Aktual Normal", "Aktual Fraud"],
        columns=["Prediksi Normal", "Prediksi Fraud"]
    )

    st.dataframe(cm_iso, use_container_width=True)


# =========================================================
# ABOUT
# =========================================================
elif menu == "About":
    st.title("About Project")

    st.markdown(
        """
        ### Tentang Proyek

        Proyek ini dibuat untuk memenuhi tugas UAS Data Mining. Topik yang digunakan adalah
        deteksi penipuan kartu kredit menggunakan dataset Credit Card Fraud Detection dari Kaggle.

        ### Dataset

        Dataset yang digunakan adalah Credit Card Fraud Detection dari Kaggle. Dataset ini berisi
        transaksi kartu kredit yang telah dianonimkan. Fitur V1 sampai V28 merupakan hasil
        transformasi PCA, sedangkan Time dan Amount tetap tersedia sebagai fitur numerik.

        ### Metode yang Digunakan

        **1. Random Forest**

        Random Forest digunakan sebagai metode Classification untuk memprediksi apakah transaksi
        termasuk normal atau fraud. Model ini dipilih sebagai model terbaik karena memperoleh
        nilai F1-score tertinggi dibandingkan metode pembanding.

        **2. Isolation Forest**

        Isolation Forest digunakan sebagai metode Anomaly Detection. Metode ini mendeteksi data
        yang memiliki pola berbeda dari mayoritas transaksi normal.

        **3. SMOTE**

        SMOTE digunakan untuk mengatasi ketidakseimbangan kelas pada data training. Teknik ini
        membantu model mengenali pola transaksi fraud yang jumlahnya sangat sedikit.

        ### Framework

        Penelitian ini menggunakan framework CRISP-DM, yaitu:

        1. Business Understanding  
        2. Data Understanding  
        3. Data Preparation  
        4. Modeling  
        5. Evaluation  
        6. Deployment  

        ### Kesimpulan

        Berdasarkan hasil evaluasi, Random Forest menjadi model terbaik dengan accuracy 99,96%,
        precision 87,75%, recall 89,58%, dan F1-score 88,66%. Oleh karena itu, Random Forest
        digunakan sebagai model utama pada aplikasi web ini.
        """
    )
