# 🏥 EpiSentinel AI
### Next-Generation Epidemiological Intelligence & ILI Multi-Horizon Forecasting Platform

[![Python 3.14](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Async-005571.svg)](https://fastapi.tiangolo.com/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-CSS-38BDF8.svg)](https://tailwindcss.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-LSTM-EE4C2C.svg)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosted-18998D.svg)](https://xgboost.readthedocs.io/)

---

## 📌 Overview

**EpiSentinel AI** is an enterprise-grade epidemiological surveillance and artificial intelligence forecasting platform designed to anticipate respiratory virus surges (**Influenza-Like Illness - ILI**) up to **14 days before clinical saturation occurs**.

Traditional public health reporting is retrospective—documenting infection rates after a wave has already peaked. EpiSentinel AI shifts the paradigm toward proactive intervention by integrating **CDC ILINet surveillance benchmarks (2014–2017)** with physical meteorological covariates (ambient temperature, barometric pressure, relative humidity, wind velocity). It compares three machine learning paradigms—**XGBoost**, **Deep PyTorch LSTM Recurrent Neural Networks**, and **ARIMA**—to deliver actionable early warnings for hospital capacity managers and healthcare leadership.

---

## 🏛️ System Architecture

The platform follows a decoupled, asynchronous 3-tier enterprise architecture:
```bash
┌─────────────────────────────────────────────────────────────────────────┐
│                           PRESENTATION LAYER                            │
│  Pristine Light Theme SaaS (HTML5 + Tailwind CSS + Chart.js Engine)     │
│  - Landing Page (/)               - Surveillance Command (/dashboard)   │
│  - Research Lab (/analytics)      - Interactive Drag & Drop Dropzone    │
└────────────────────────────────────┬────────────────────────────────────┘
│ Async REST / Multipart Fetch
┌────────────────────────────────    ▼────────────────────────────────────┐
│                           APPLICATION LAYER                             │
│                  FastAPI Asynchronous Microservice                      │
│  - GET /           - GET /dashboard         - GET /analytics            │
│  - GET /api/default_data   - GET /api/model_metrics                     │
│  - POST /api/predict_csv (Robust Ingestion & Multi-Variant CDC Parser)  │
└────────────────────────────────────┬────────────────────────────────────┘
│ PyTorch & Scikit-Learn Ingestion
┌────────────────────────────────    ▼────────────────────────────────────┐
│                        ANALYTICS & ML INFERENCE                         │
│  - XGBoost Gradient Boosted Trees (Champion: RMSE 0.1534, MAPE 7.43%)   │
│  - Stacked PyTorch LSTM Regressor (Runner-Up: RMSE 0.2200, MAPE 8.64%)  │
│  - Classical Autoregressive Integrated Moving Average (ARIMA Baseline)  │
│  - 4-Tier Epidemiological Risk Classification Engine                    │
└─────────────────────────────────────────────────────────────────────────┘
```bash
---

## ✨ Key Features

- **Pristine Corporate SaaS UI:** Clean light-theme enterprise interface styled with *Plus Jakarta Sans* typography, structured bento grids, and interactive Chart.js visualizations.
- **Multi-Page Navigation:** Seamless transition between the **Enterprise Landing Page (`/`)**, **Surveillance Command Center (`/dashboard`)**, and **Research Model Leaderboard (`/analytics`)**.
- **Robust Asynchronous CDC Ingestion (`/api/predict_csv`):** Automatically strips CDC metadata comments, skips title banners, handles suppressed cell counts (`'X'`), auto-detects column variants, and aggregates multi-state records into unified national timelines.
- **Multi-Model AI Forecasting:** Evaluates and displays predictions across XGBoost, PyTorch LSTM, and ARIMA across 7-day and 14-day horizons.
- **4-Tier Epidemic Risk Engine:** Automatically classifies incoming surveillance data into **Low (<2.0%)**, **Moderate (2.0–3.5%)**, **High (3.5–4.5%)**, and **Critical Surge (≥4.5%)** with responsive alert badges.

---

## 📂 Repository Directory Structure

* **`main.py`** — Production FastAPI Server & REST APIs. Serves templates, mounts static assets, and hosts the ingestion engine.
* **`templates/index.html`** — Enterprise Landing Page. Value proposition, architecture bento, and direct workspace links.
* **`templates/dashboard.html`** — Surveillance Command Center. KPI summary cards, live charts, CSV dropzone, and data preview table.
* **`templates/analytics.html`** — Model Analytics Leaderboard. Comparative evaluation table, metric bar charts, and empirical research plots.
* **`train_lstm.py`** — Deep Learning Training Script. PyTorch stacked LSTM regressor training pipeline.
* **`compare_models.py`** — Model Comparison & Aggregation. Compiles cross-team performance metrics and evaluation plots.
* **`preprocess_master.py`** — Dataset Cleaning Pipeline. Aggregates state-level data, imputes missing values, and calculates lag features.
* **`cleaned_ili_master.csv`** — Raw Multi-State CDC Dataset. 9,020 records across 50 U.S. states.
* **`final_processed_ili_weekly.csv`** — Ground Truth Gold Standard. 164 chronological national weekly records (2014–2017).
* **`predictions.csv`** — Out-of-Sample Test Predictions. Actual ground truth (`y_true`) vs. model forecasts (`y_pred`).
* **`final_model_comparison.csv`** — Model Evaluation Matrix. Formal CSV summary documenting RMSE, MAE, and MAPE.
* **`static/`** — Research Figures. Embedded empirical plots (`master_model_comparison.png`, `correlation_matrix.png`, etc.).

---

## ⚡ Quickstart & Installation

### Prerequisites
- Python 3.14 (`py -3.14`)[cite: 2]
- Required packages: `fastapi`, `uvicorn`, `pandas`, `numpy`, `jinja2`, `python-multipart`[cite: 2]

### 1. Clone the Repository
```bash
git clone [https://github.com/NiharikaMKodige/Influenza-like-Illness-Prediction.git](https://github.com/NiharikaMKodige/Influenza-like-Illness-Prediction.git)
cd EpiSentinel-AI
``` 

### 2. Install Dependencies
```bash
py -3.14 -m pip install fastapi uvicorn pandas numpy jinja2 python-multipart
``` 

### 3. Launch the Application Server
```bash 
py -3.14 -m uvicorn main:app --reload
``` 

### 4. Access the Platform
Open your browser and navigate to:

🌐 Landing Page: http://127.0.0.1:8000/
📊 Surveillance Dashboard: http://127.0.0.1:8000/dashboard
🔬 Model Leaderboard: http://127.0.0.1:8000/analytics

## 🏆 Model Performance Benchmark

* **🥇 XGBoost (Champion)**
  * **RMSE:** 0.1534 | **MAE:** 0.1059 | **MAPE:** 7.43%
  * *Advantage:* Superior non-linear capture of seasonal epidemic spikes and meteorological covariates.
* **🥈 Stacked LSTM (Runner-Up)**
  * **RMSE:** 0.2200 | **MAE:** 0.1256 | **MAPE:** 8.64%
  * *Advantage:* Robust sequential memory across 4-week recurrent windows.
* **🥉 ARIMA (Baseline)**
  * **RMSE:** 0.6473 | **MAE:** 0.4810 | **MAPE:** 50.46%
  * *Advantage:* Standard autoregressive statistical baseline; shows significant decay on multi-week horizons.