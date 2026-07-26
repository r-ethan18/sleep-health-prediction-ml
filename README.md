# Sleep Disorder Risk Prediction

A machine learning project designed to evaluate and predict an individual's **Sleep Disorder Risk** (`Healthy`, `Mild`, `Moderate`, `Severe`) using demographic, physiological, and daily lifestyle factors.

---

## Project Overview

Sleep disorders significantly impact physical health, cognitive performance, and overall quality of life. This project leverages a comprehensive dataset of 100,000 records to build and compare multiple machine learning classifiers for early risk detection.

Key highlights of the project:
- Preprocessing and feature engineering on high-dimensional tabular data.
- Addressing class imbalance through synthetic oversampling (SMOTE) and custom class-weighting strategies.
- Comparative evaluation of baseline models, linear classifiers, ensemble methods, and gradient boosting algorithms.
- Tracking model parameters and metrics using **MLflow**.

---

## Recommended Environment

> [!IMPORTANT]
> **This project is best run in a Google Colab environment.**

Running in Google Colab provides several benefits:
- Native integration with `kagglehub` for automatic dataset retrieval.
- Pre-installed dependencies and GPU/CPU acceleration.
- Built-in `google.colab.files` module to easily zip and download the MLflow tracking directory (`mlruns.zip`).

---

## Dataset Information

- **Source:** [Kaggle - Sleep Health and Daily Performance Dataset](https://www.kaggle.com/datasets/mohankrishnathalla/sleep-health-and-daily-performance-dataset)
- **Size:** 100,000 entries
- **Target Variable:** `sleep_disorder_risk` (Multiclass: `Healthy`, `Mild`, `Moderate`, `Severe`)
- **Key Features:**
  - **Demographics & Lifestyle:** `gender`, `age`, `occupation`, `country`, `work_hours_per_week`
  - **Sleep Metrics:** `sleep_duration_hrs`, `sleep_quality_score`, `rem_percentage`, `deep_sleep_percentage`, `sleep_latency_mins`, `wake_episodes_per_night`
  - **Behavioral & Health:** `bmi`, `caffeine_mg_before_bed`, `alcohol_units_before_bed`, `screen_time_before_bed_mins`, `steps_that_day`, `stress_score`, `mental_health_status`

---

## Machine Learning Techniques Used

1. **Data Preprocessing & Feature Transformation:**
   - Excluded non-predictive features (`id`, `cognitive_performance_score`, `felt_rested`).
   - Built a scikit-learn `ColumnTransformer` pipeline:
     - `StandardScaler` for numerical feature scaling.
     - `OneHotEncoder` (handling unknown categories) for categorical variables.

2. **Class Imbalance Mitigation:**
   - **SMOTE (Synthetic Minority Over-sampling Technique):** Applied via `imblearn` pipelines to balance minority classes.
   - **Cost-Sensitive Learning / Class Weighting:** Implemented customized multi-class weights in LightGBM and Random Forest to heavily penalize misclassification of severe/moderate disorder risks.

3. **Machine Learning Algorithms:**
   - **K-Nearest Neighbors (KNN):** Distance-based baseline classifier.
   - **SMOTE + KNN:** KNN classifier trained on SMOTE-oversampled training data.
   - **Logistic Regression:** Linear multi-class baseline.
   - **PCA + Logistic Regression:** Dimensionality reduction via Principal Component Analysis (retaining 95% variance) combined with Logistic Regression.
   - **Random Forest Classifier:** Ensemble tree classifier utilizing balanced class weights (`n_estimators=300`).
   - **LightGBM Classifier:** Optimized Gradient Boosted Decision Tree (GBDT) framework utilizing custom objective weighting (`Healthy: 1`, `Mild: 1.5`, `Moderate: 6`, `Severe: 12`).

---

## Results & Model Performance

### Model Comparison (Macro Average)

| Model | Macro Precision | Macro Recall | Macro F1-Score |
| :--- | :---: | :---: | :---: |
| **K-NN Classifier** | 0.66 | 0.45 | 0.48 |
| **SMOTE + KNN** | 0.54 | 0.62 | 0.56 |
| **Logistic Regression** | 0.75 | 0.71 | 0.73 |
| **Random Forest** | 0.84 | 0.76 | 0.79 |
| **LightGBM (Validation)** | 0.86 | 0.89 | 0.87 |
| **LightGBM (Final Test Set)** | **0.86** | **0.90** | **0.88** |

### Final Test Evaluation (LightGBM)

- **Overall Test Accuracy:** **95%**
- **Macro F1-Score:** **0.88**

| Risk Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Healthy** | 1.00 | 0.99 | 0.99 | 10,937 |
| **Mild** | 0.96 | 0.91 | 0.93 | 6,590 |
| **Moderate** | 0.68 | 0.82 | 0.75 | 1,665 |
| **Severe** | 0.82 | 0.87 | 0.85 | 808 |

### Key Findings

- **Top Predictors:** Feature importance analysis revealed that **Sleep Duration**, **BMI**, and **Sleep Latency** are the primary factors driving predictions for sleep disorder risk.
- **Class Balancing Impact:** Weighting smaller risk classes significantly boosted the recall for high-risk categories (`Moderate` and `Severe`), which is critical for early healthcare intervention.

---

## Experiment Tracking with MLflow

MLflow is lightly used throughout the notebook for experiment management:
- Logs hyperparameters (e.g., model types, PCA component counts).
- Tracks classification metrics (`precision`, `recall`, `f1-score` per category, and overall `accuracy`).
- Stores generated artifacts, including **confusion matrices** and **feature importance plots**.
- Saves the final retrained model pipeline for quick serialization.

*All experiment logs are stored in a local `./mlruns` directory and can be downloaded as a `.zip` file at the end of the notebook.*

---

## How to Run

1. Open **Google Colab**.
2. Upload `Sleep_Health_Project.ipynb` or open it directly in Colab.
3. Make sure you have your Kaggle credentials configured if prompted, or let `kagglehub` handle dataset downloading automatically.
4. Execute all cells sequentially (`Runtime > Run all`).
5. (Optional) Inspect or download `mlruns.zip` to review MLflow logs.
