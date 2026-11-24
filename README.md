# Solar Power Forecasting Web App & Deep Learning Ramp-Down Detection

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Framework-Flask-000000.svg)](https://flask.palletsprojects.com/)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-LSTM%20%7C%20GRU%20%7C%20BiLSTM-orange.svg)](https://keras.io/)
[![Scikit-Learn & XGBoost](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20XGBoost-f7931e.svg)](https://scikit-learn.org/)
[![Published Conference Papers](https://img.shields.io/badge/Published-4th%20Sivas%20Int.%20Conference%202025-success.svg)](#publications--academic-certifications)

## 📌 Project Overview
Solar energy generation fluctuates significantly due to cloud movements, weather changes, and environmental factors. Sudden and sharp drops in power output (**ramp-down events**) can destabilize power grids and disrupt supply-demand balance.

This repository brings together two complementary research studies conducted on a **4-year real-world field dataset (35,055 hourly records, 2019–2022)** from the **İbrahim Taşyurdu Solar Power Plant (469.2 kWe)** in Pazarcık, Kahramanmaraş, Türkiye:
1. **Interactive ML Forecasting Web Application (`app.py`, `templates/index.html`):** A Flask-based web platform for automated Exploratory Data Analysis (EDA) and solar power forecasting across 7 machine learning regression algorithms.
2. **Deep Learning Ramp-Down Event Detection (`lstmanidüşüş.ipynb`, `gruanidüşüş.ipynb`, `BiLSTM.ipynb`):** Recurrent neural network architectures (`LSTM`, `GRU`, `BiLSTM`) trained with 48-hour sliding windows to predict power output and detect critical ramp-down anomalies.

---

## 🏆 Publications & Academic Certifications
Both studies were accepted as peer-reviewed papers and presented orally at the **4th Sivas International Conference on Scientific and Innovation Research** *(May 30–31, 2025 — Sivas University of Science and Technology & IKSAD Institute, Sivas, Türkiye)*. Official participation and oral presentation certificates are included in this repository (`01-Serap YÜCEL.pdf` & `02-Serap YÜCEL.pdf`):

1. **Paper 1:** *A Web-Based Machine Learning Application for Solar Power Generation Prediction*
2. **Paper 2:** *Detection of Ramp-Down Events in Solar Power Generation Data*

---

## 👥 Research Team & Academic Supervision
Developed at **Kahramanmaraş Sütçü İmam University** (Departments of Computer Engineering & Electrical-Electronics Engineering):
* **Authors:** **Serap YÜCEL**, Renim HUSSIN, Beyda ALAGÖZ, Fatma KAYKI
* **Academic Advisors:** Assoc. Prof. Dr. Furkan DİNÇER & Assist. Prof. Dr. Yavuz CANBAY
* **Data Partner:** İbrahim Taşyurdu Solar Power Plant (469.2 kWe), Kahramanmaraş, Türkiye

---

## 📊 Experimental Results

### 1. Web-Based Machine Learning Forecasting (`app.py`)
Evaluated across 7 regression models using autoregressive lag features (`lag_1`, `lag_2`, `lag_3`) with chronological 80% Train / 20% Test splitting:

| Algorithm | MAE | RMSE | MSE | $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: |
| **Gradient Boosting (Best Model)** | **13.55** | **26.30** | **691.90** | **0.9583** |
| **XGBoost Regressor** | 15.05 | 26.80 | 718.38 | 0.9567 |
| **Support Vector Regression (SVR)** | 16.49 | 26.92 | 724.53 | 0.9563 |
| **K-Nearest Neighbors (KNN)** | 13.34 | 28.32 | 802.19 | 0.9516 |
| **Random Forest Regressor** | 13.83 | 28.59 | 817.27 | 0.9507 |
| **Linear Regression** | 20.95 | 32.31 | 1043.88 | 0.9370 |
| **Decision Tree Regressor** | 15.16 | 33.03 | 1090.78 | 0.9342 |

### 2. Deep Learning Ramp-Down Event Detection (`.ipynb` Notebooks)
Trained on daytime generation records (`07:00–19:00`) using a 48-step sliding window (`64x32x32` 3-layer RNN architecture, `Adam` optimizer, `20 epochs`) across 509 ramp-down events (413 train / 96 test):

| Deep Learning Model | MAE | MSE | $R^2$ Score | Mean Ramp-Down Time Deviation |
| :--- | :---: | :---: | :---: | :---: |
| **LSTM (Best Model)** | **9.60** | **482.03** | **0.971** | **1.29 hours** |
| **GRU** | 9.72 | 489.60 | 0.970 | 1.47 hours |
| **BiLSTM** | 10.34 | 498.28 | 0.970 | 1.29 hours |

---

## 📂 Repository Structure
```text
├── app.py                      # Flask backend, automated EDA & 7 ML regression pipelines
├── templates/
│   └── index.html              # Responsive Tailwind CSS dashboard & visualization UI
├── lstmanidüşüş.ipynb          # LSTM time-series forecasting & ramp-down detection notebook
├── gruanidüşüş.ipynb           # GRU time-series forecasting & ramp-down detection notebook
├── BiLSTM.ipynb                # Bidirectional LSTM forecasting & ramp-down detection notebook
├── 01-Serap YÜCEL.pdf          # Oral Presentation Certificate (Web-Based ML Application)
└── 02-Serap YÜCEL.pdf          # Oral Presentation Certificate (Ramp-Down Event Detection)
