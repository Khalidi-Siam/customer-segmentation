# 🎯 Customer Segmentation

> An end-to-end machine learning pipeline for customer segmentation using unsupervised learning, served through a Django web application.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Online-brightgreen)](https://customer-segmentation-zjvi.onrender.com)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Django](https://img.shields.io/badge/Django-5.0%2B-092E20?logo=django)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Pipeline Architecture](#-pipeline-architecture)
- [Django Web App](#-django-web-app)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)
- [Future Improvements](#-roadmap)
- [Author](#-author)

---

## 📖 Overview

This project delivers a complete customer segmentation solution that:

1. **Trains** an unsupervised KMeans model on the Mall Customers dataset
2. **Automatically selects** the optimal number of clusters using silhouette analysis
3. **Generates human-readable labels** for each segment (e.g., *High Value*, *Budget Conservative*)
4. **Serves predictions** through an intuitive Django web interface

Users can input a customer's annual income and spending score to instantly receive their cluster assignment and segment profile.

---

## ✨ Features

| Category | Capabilities |
|----------|-------------|
| **ML Pipeline** | Modular, end-to-end components with artifact persistence |
| **Auto K-Selection** | Silhouette score optimization across K=2–10 |
| **Cluster Evaluation** | Silhouette, Calinski-Harabasz, and Davies-Bouldin metrics |
| **Segment Profiling** | Rule-based naming with income/spending/age analysis |
| **Web Interface** | Real-time prediction via Django with clean UI |
| **Data Processing** | Median imputation + standard scaling via `ColumnTransformer` |

---

## 🛠 Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Language** | Python 3.10+ |
| **ML / Data** | Scikit-learn, Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Web Framework** | Django |
| **Serialization** | Dill |

---

## 📁 Project Structure

```
clustering/
├── 📄 README.md                  # Project documentation
├── 📄 requirements.txt           # Python dependencies
├── 📄 setup.py                   # Package configuration
│
├── 📓 notebook/
│   ├── 📓 EDA.ipynb              # Exploratory data analysis
│   └── 📂 data/
│       └── 📄 Mall_Customers.csv # Source dataset
│
├── 📂 src/
│   ├── 📂 component/             # Pipeline building blocks
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── data_validation.py
│   │   ├── model_trainer.py
│   │   └── build_cluster_profile.py
│   ├── 📂 pipeline/              # Orchestrated workflows
│   │   ├── train_pipeline.py
│   │   └── prediction_pipeline.py
│   ├── exception.py              # Custom error handling
│   ├── logger.py                 # Logging configuration
│   └── utils.py                  # Shared utilities
│
├── 📂 customer_segmentation/     # Django application
│   ├── 📄 manage.py
│   ├── 📂 customer_segmentation/ # Project settings
│   ├── 📂 prediction/            # Prediction app
│   └── 📂 templates/             # HTML templates
│
├── 📂 artifacts/                 # Generated after training ⚙️
└── 📂 logs/                      # Runtime logs 📝
```

> 💡 The `artifacts/` and `logs/` directories are generated automatically during runtime.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

### 1. Clone & Navigate

```bash
git clone <your-repo-url>
cd clustering
```

### 2. Set Up Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv env
.\env\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
python -m venv env
env\Scripts\activate.bat
```

**macOS / Linux:**
```bash
python -m venv env
source env/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 4. Train the Model

```bash
python -m src.pipeline.train_pipeline
```

This generates all required artifacts in the `artifacts/` directory.

### 5. Launch the Web App

```bash
python customer_segmentation/manage.py runserver
```

Open your browser and navigate to **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 🔄 Pipeline Architecture

### Step 1: Data Ingestion

| Input | Output |
|-------|--------|
| `notebook/data/Mall_Customers.csv` | `artifacts/data.csv` |
| | `artifacts/train.csv` |
| | `artifacts/test.csv` |

- Reads the source CSV dataset
- Persists raw data copies to `artifacts/`
- Splits into train/test sets using `train_test_split`

### Step 2: Data Transformation

| Features Used | Features Dropped |
|---------------|------------------|
| `Annual Income (k$)` | `CustomerID` |
| `Spending Score (1-100)` | `Gender`, `Age` |

- Applies **median imputation** for missing values
- Applies **standard scaling** via Scikit-learn `ColumnTransformer`
- Saves transformed arrays and the fitted preprocessor

**Outputs:** `artifacts/train.npy`, `artifacts/test.npy`, `artifacts/preprocessor.pkl`

### Step 3: Model Training

```
K ∈ [2, 3, 4, 5, 6, 7, 8, 9, 10]
       ↓
  Silhouette Score
       ↓
  Best K Selected
       ↓
  Final KMeans Model
```

- Evaluates K values from 2 to 10
- Selects optimal K via **silhouette score**
- Computes cluster quality metrics:
  - ✅ Silhouette Score
  - ✅ Calinski-Harabasz Score
  - ✅ Davies-Bouldin Score

**Output:** `artifacts/model.pkl`

### Step 4: Cluster Profiling & Naming

Each cluster is profiled by:
- Average annual income
- Average spending score
- Average age
- Customer count

**Example segment labels:**

| Segment | Profile |
|---------|---------|
| `High Value` | High income, high spending |
| `Careful Affluent` | High income, low spending |
| `High Spend Low Income` | Low income, high spending |
| `Budget Conservative` | Low income, low spending |
| `Mainstream` | Average income and spending |

**Outputs:** `artifacts/cluster_profile.csv`, `artifacts/cluster_labels.json`

---

## 🌐 Django Web App

### Endpoints

| Route | Description |
|-------|-------------|
| `/` | Home page with prediction form |
| `/result/` | Displays cluster prediction results |

### Prediction Flow

```
User Input (Income + Spending)
         ↓
   CustomData DataFrame
         ↓
   Load preprocessor.pkl
         ↓
   Transform features
         ↓
   Load model.pkl → Predict cluster
         ↓
   Load cluster_labels.json → Map to segment name
         ↓
   Display result (Cluster ID + Segment Label)
```

---

## ⚙️ Configuration

### Development Defaults

The project ships with development-friendly settings:

| Setting | Value |
|---------|-------|
| `DEBUG` | `True` |
| `ALLOWED_HOSTS` | `["*"]` |

> ⚠️ Update these in `customer_segmentation/settings.py` before deploying to production.

### Required Artifacts

The Django app requires these files (generated by training):

- `artifacts/model.pkl` — Trained KMeans model
- `artifacts/preprocessor.pkl` — Fitted feature preprocessor
- `artifacts/cluster_labels.json` — Cluster ID to name mapping

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| **Artifact not found** | Run `python -m src.pipeline.train_pipeline` first |
| **Module import errors** | Ensure `pip install -e .` completed successfully |
| **Django won't start** | Run `manage.py` from root directory |
| **Environment issues** | Verify the virtual environment is activated |

---

## Future Improvements

- [ ] Implement data validation logic in `data_validation.py`
- [ ] Add model versioning and experiment tracking (MLflow)
- [ ] Write automated tests for pipeline components and views
- [ ] Expose REST API endpoints for programmatic predictions
- [ ] Add Docker support for reproducible deployments
- [ ] Integrate additional clustering algorithms (DBSCAN, Agglomerative)

---

## 👤 Author

**Khalidi Siam**

---

<p align="center">
  <em>Built with ❤️ using Python, Scikit-learn, and Django</em>
</p>

