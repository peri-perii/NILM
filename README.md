---
title: Beyond The Space
emoji: 🌍
colorFrom: yellow
colorTo: indigo
sdk: streamlit
sdk_version: "1.35.0"
python_version: "3.11"
app_file: app.py
pinned: false
---

#  Beyond the One Number

> **From aggregate to itemized — tamper-proof, automated billing for every home.**

---

## 🚨 Problem Statement

Traditional electricity meters deliver a single aggregate reading, giving homeowners zero visibility into individual appliance usage. This opacity fuels three critical failures: inability to pinpoint high-consumption devices, reliance on intermediaries that enable meter tampering and billing fraud, and manual processes that introduce errors and delayed discrepancy discovery.

## 💡 Solution

**Beyond the One Number** is a software-only AI system that transforms any existing electricity meter into a smart, tamper-proof metering solution using **Non-Intrusive Load Monitoring (NILM)** — no hardware upgrades required. By analysing the aggregate power signal, it identifies individual appliance usage, generates itemised bills, and surfaces actionable savings recommendations.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📂 **CSV Upload** | Upload standard meter data (timestamp + total_power_watts) |
| 🤖 **ML Disaggregation** | Random Forest identifies 6 appliances from power signatures |
| 📈 **Interactive Charts** | 24-h power profile, energy breakdown, activity timeline (Plotly) |
| 💰 **Cost Analysis** | Per-appliance cost at $0.15/kWh with 20% savings estimate |
| 💡 **Smart Tips Engine** | Context-aware recommendations (evening AC, heater scheduling…) |
| 📄 **CSV Export** | Download full minute-by-minute labelled results |
| 📑 **PDF Report** ⭐ | One-page professional report with embedded charts, table, and tips |
| 🌐 **HF Spaces Ready** | Deployable to Hugging Face Spaces with zero config |

---

## 🚀 How to Run Locally

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate synthetic data
```bash
python generate_data.py
```
Creates `household_power.csv` (1 440 rows, 1-minute resolution) and `appliance_ground_truth.csv`.

### 3. Train the NILM model
```bash
python train_model.py
```
Outputs `nilm_model.joblib`, `nilm_scaler.joblib`, and `label_map.json`.  
Expect **≥ 80% test accuracy** on the generated data.

### 4. Launch the app
```bash
streamlit run app.py
```
Opens `http://localhost:8501` in your browser.

### 5. Use the app
1. Upload `household_power.csv` (or click **Download Sample CSV**)
2. Click **Analyse Appliances**
3. Explore charts and cost breakdown
4. Click **Download PDF Report** for the professional one-pager

---

## 🛠️ Tech Stack

| Layer | Library | Purpose |
|---|---|---|
| Frontend | **Streamlit** ≥ 1.32 | Interactive web UI |
| ML | **scikit-learn** (Random Forest) | Appliance classification |
| Features | **NumPy / Pandas** | Signal windowing + stats |
| Charts | **Plotly** | Interactive visualisations |
| PDF | **fpdf2** + Matplotlib | Report generation |
| Model I/O | **joblib** | Model serialisation |

---

## 🤗 Deploy to Hugging Face Spaces

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Select **Streamlit** as the SDK
3. Push your files (including the trained `.joblib` files and `label_map.json`):

```bash
git init
git remote add origin https://huggingface.co/spaces/<your-username>/<space-name>
git add .
git commit -m "Initial commit: Beyond the One Number NILM"
git push origin main
```

> **Note:** Generate and commit the model files locally first (`generate_data.py` → `train_model.py`), then push everything together. The Space will auto-install `requirements.txt`.

---

## 📁 File Structure

```
beyond-the-one-number/
├── requirements.txt          # Python dependencies
├── generate_data.py          # Synthetic data generator
├── train_model.py            # Random Forest NILM trainer
├── pdf_generator.py          # PDF report module
├── app.py                    # Streamlit application
├── README.md                 # This file
├── household_power.csv       # [generated] meter data
├── appliance_ground_truth.csv# [generated] ground truth
├── nilm_model.joblib         # [generated] trained model
├── nilm_scaler.joblib        # [generated] feature scaler
└── label_map.json            # [generated] label mapping
```

---

## 📸 Screenshot

*Upload your meter CSV → click Analyse → see your appliances broken down.*

![App Screenshot](screenshot.png)

---

## 📜 License

MIT — free for personal and commercial use.
