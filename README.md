# Deteksi Penipuan Kartu Kredit Menggunakan Classification dan Anomaly Detection

## Deskripsi Proyek

Proyek ini merupakan tugas UAS Data Mining yang bertujuan untuk mendeteksi transaksi penipuan kartu kredit menggunakan pendekatan Machine Learning. Permasalahan yang diangkat adalah credit card fraud detection, yaitu proses mengidentifikasi apakah suatu transaksi termasuk transaksi normal atau transaksi fraud.

Dataset yang digunakan memiliki karakteristik imbalanced class karena jumlah transaksi fraud jauh lebih sedikit dibandingkan transaksi normal. Oleh karena itu, proyek ini menerapkan teknik preprocessing dan evaluasi model yang sesuai agar hasil analisis tidak hanya bergantung pada accuracy.

## Dataset

Dataset yang digunakan adalah Credit Card Fraud Detection dari Kaggle.

Link dataset:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Informasi dataset:

* Jumlah data: 284.807 transaksi
* Jumlah fitur input: 30 fitur
* Target: Class
* Class 0: Transaksi normal
* Class 1: Transaksi fraud
* Jumlah transaksi normal: 284.315
* Jumlah transaksi fraud: 492
* Persentase fraud: sekitar 0,172%

## Metode yang Digunakan

Proyek ini menggunakan dua metode Data Mining, yaitu:

1. Classification menggunakan Random Forest
2. Anomaly Detection menggunakan Isolation Forest

Selain itu, digunakan teknik SMOTE untuk menangani ketidakseimbangan kelas pada data training.

## Framework Penelitian

Framework yang digunakan adalah CRISP-DM, dengan tahapan:

1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
6. Deployment

## Hasil Evaluasi Model

Hasil evaluasi menunjukkan bahwa Random Forest memiliki performa terbaik dibandingkan Isolation Forest.

| Model            | Accuracy | Precision | Recall | F1-score |
| ---------------- | -------: | --------: | -----: | -------: |
| Random Forest    |   99,96% |    87,75% | 89,58% |   88,66% |
| Isolation Forest |   99,71% |    24,32% | 28,13% |   26,09% |

Berdasarkan hasil tersebut, Random Forest dipilih sebagai model terbaik untuk diimplementasikan ke aplikasi web berbasis Streamlit.

## Struktur Folder Project

```text
UAS_DataMining
│
├── app
│   ├── assets
│   └── app.py
│
├── dataset
│   └── creditcard.csv
│
├── laporan
│   └── Artikel_UAS_Data_Mining_Deteksi_Fraud_Kartu_Kredit.pdf
│
├── model
│   ├── feature_names.pkl
│   ├── fraud_detection_model.pkl
│   └── iso_model.pkl
│
├── notebook
│   └── Credit_Card_Fraud_Detection_Final_.ipynb
│
├── requirements.txt
└── README.md
```

## Cara Menjalankan Aplikasi

1. Pastikan Python sudah terpasang pada perangkat.
2. Buka folder project menggunakan VS Code atau terminal.
3. Install library yang dibutuhkan:

```bash
pip install -r requirements.txt
```

4. Masuk ke folder aplikasi:

```bash
cd app
```

5. Jalankan aplikasi Streamlit:

```bash
streamlit run app.py
```

6. Aplikasi akan terbuka melalui browser lokal.

## Fitur Aplikasi

Aplikasi web memiliki beberapa fitur utama, yaitu:

1. Home
   Berisi judul proyek, deskripsi singkat, dan identitas anggota.

2. Dataset Overview
   Berisi informasi dataset, jumlah data, statistik sederhana, dan distribusi kelas.

3. Prediction / Analysis
   Berisi form input transaksi dan hasil prediksi apakah transaksi termasuk normal atau fraud.

4. Visualization
   Berisi grafik pendukung dan visualisasi hasil analisis.

5. About
   Berisi penjelasan metode, dataset, framework penelitian, dan informasi proyek.

## File yang Dikumpulkan

File yang dikumpulkan dalam repository project meliputi:

* Dataset
* Notebook analisis dan modeling
* Source code aplikasi Streamlit
* Model machine learning
* Laporan artikel PDF
* requirements.txt
* README.md

## Anggota

Nama: Arvy Revaldy Sevptarius Tarigan
Proyek: UAS Data Mining
Topik: Credit Card Fraud Detection
