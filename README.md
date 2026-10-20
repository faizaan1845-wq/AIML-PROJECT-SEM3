# 🛡️ TrustRate AI — Electronics Review Authenticity & Rating Deflator

TrustRate AI is a machine learning system that detects deceptive and fake reviews on electronic devices listed on online e-commerce platforms, recalculates the **Actual Authentic Rating**, and displays the **Rating Inflation Delta** (e.g. Given: **4.5★** vs Actual: **3.8★** -> **Inflated by +0.7★**).

---

## 🌟 Key Features

1. **🏬 Online Electronics Storefront & Catalog:**
   - 300+ indexed electronic devices across 7 categories (Headphones, Laptops, Phone Chargers, Smartwatches, Speakers, Cameras, Power Banks).
   - Side-by-side comparison of Given Rating vs Actual Rating with instant Trust Grade badges (A to F).

2. **🔍 Deep-Dive Forensic Report:**
   - 4 Hero metric cards: Given Rating, Actual Genuine Rating, Rating Inflation Delta, and Trust Grade.
   - Interactive Star Distribution Shift chart (visualizing the artificial 5-star spike and the real customer curve).
   - Side-by-side review breakdown (Verified Genuine vs Flagged Suspicious).

3. **🌐 Live E-Commerce URL & Custom Review Auditor:**
   - Paste any online product URL or paste customer reviews to audit in real-time.
   - 1-Click benchmark presets (including the **Budget Wireless Earbuds: 4.5★ -> 3.8★ = +0.7★** benchmark).

4. **⚡ Single Review Instant Scanner:**
   - Real-time fraud probability score, uppercase shouting detection, and promotional buzzword extraction.

5. **📊 AI Model Studio & Viva Lab:**
   - Comprehensive model benchmark comparison (Accuracy, Precision, Recall, F1, ROC-AUC).
   - Interactive confusion matrix on 6,681 held-out test reviews.
   - Top predictive tokens and viva defense examination guide.

---

## 🚀 Quickstart Guide

### 1. Environment Setup
```bash
cd /Users/ataussalamfaizaan/Desktop/TrustRate_AI
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. (Optional) Re-run Data Curation & Model Training
```bash
# 1. Build curated training dataset and index 300+ electronics devices
python3 prepare_data.py

# 2. Train and benchmark candidate ML models (outputs model.joblib & metrics.json)
python3 train_model.py
```

### 3. Launch the Interactive Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📈 Model Performance Benchmark

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Balanced)** 🏆 | **72.16%** | **71.55%** | **73.56%** | **0.7254** | **0.7932** |
| **Multinomial Naive Bayes** | 70.17% | 67.78% | 76.86% | 0.7204 | 0.7788 |
| **Complement Naive Bayes** | 70.18% | 67.80% | 76.86% | 0.7205 | 0.7788 |
| **Calibrated SGD (Modified Huber)** | 71.83% | 71.67% | 72.19% | 0.7193 | 0.7844 |

---

## 📂 Project Structure

```
TrustRate_AI/
├── app.py                         # Streamlit interactive forensic dashboard
├── train_model.py                 # Multi-model benchmarking and training pipeline
├── prepare_data.py                # Dataset curation and electronics indexing script
├── model_utils.py                 # Core inference, explainability and deflator logic
├── model.joblib                   # Best trained pipeline (TF-IDF + Logistic Regression)
├── metrics.json                   # Full evaluation results and confusion matrix
├── requirements.txt               # Dependencies
├── README.md                      # Quickstart documentation
├── PROJECT_REPORT_AND_VIVA.md     # Full technical report and viva Q&A
└── data/
    ├── curated_training_dataset.csv # Balanced 26,722 training reviews
    ├── electronics_catalog.json     # 311 indexed electronics products
    ├── demo_reviews.csv             # Benchmark presets
    └── electronics_reviews_sample.csv # 5,000 electronics sample reviews
```
