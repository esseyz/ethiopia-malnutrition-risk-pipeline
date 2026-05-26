# Ethiopia Child Malnutrition & Stunting Risk Pipeline

An enterprise-grade, end-to-end data processing and predictive machine learning pipeline designed to stratify malnutrition risk in children under the age of 5. Utilizing real-world microdata from the **Ethiopia Demographic and Health Survey (EDHS)**, this system isolates key socio-demographic and maternal determinants to predict clinical height-for-age Z-score categories mapped directly to World Health Organization (WHO) Child Growth Standards.

## 📊 Analytical Baseline & Performance
* **Target Balance Profile:** Cleaned clinical outcomes sit at **63.67% Normal**, **20.04% Moderate Malnutrition**, and **16.30% Severe Malnutrition**.
* **Clinical Prioritization (Recall):** To maximize real-world utility for medical intervention targeting, both models incorporate explicit class-balancing (`class_weight='balanced'`) to prioritize **Recall** over raw accuracy for high-risk cohorts:
  * **Production Random Forest Pipeline:** **63% Recall** on Severe Malnutrition cases (Macro F1: `0.41`).
  * **Multinomial Logistic Regression Baseline:** **62% Recall** on Severe Malnutrition cases (Macro F1: `0.40`).
* **Statistical Verification:** Chi-Square Tests of Independence confirm a deterministic association between socioeconomic indicators and stunting velocity, rejecting the null hypothesis with a p-value of **$4.8093 \times 10^{-40}$**.

## 🛠️ Repository Architecture
The workspace is engineered following clean code and decoupling principles, segregating exploratory scripts, model parameters, and tracking artifacts:

```text
ethiopia-malnutrition-risk-pipeline/
├── artifacts/              # Serialized training artifacts (.pkl)
├── data/
│   ├── processed/          # Fully decoded targets engineered for Tableau
│   └── raw/                # Immutable tracking vault for raw EDHS data
├── dashboard/              # Tableau visual layout definitions
├── notebooks/              # Scratchpad ad-hoc analysis notebooks
├── src/
│   ├── models/             # Encapsulated modeling and evaluation engines
│   └── preprocessing/      # Preprocessing steps, pipelines, and inference tests
├── .gitignore              # Production deployment version control filters
├── app.py                  # Single-entry Master Workflow Orchestrator
└── README.md               # Executive Technical Briefing
```

## 🚀 Execution & Reproducibility Guide

### 1. Environment Initialization

Clone this repository to your local machine and ensure standard python execution dependencies are satisfied:

```bash
git clone [https://github.com/YOUR_USERNAME/ethiopia-malnutrition-risk-pipeline.git](https://github.com/YOUR_USERNAME/ethiopia-malnutrition-risk-pipeline.git)
cd ethiopia-malnutrition-risk-pipeline
pip install pandas numpy scikit-learn scipy statsmodels joblib
```

### 2. Data Acquisition & Legal Placement
Due to human subject confidentiality constraints and the DHS Program User Agreement, raw microdata cannot be re-distributed directly within public repositories.

1. Register for a research data access account at The DHS Program Portal.

2. Request access to the Ethiopia 2016 Demographic and Health Survey dataset.

3. Download the Children's Recode (KR) dataset in Stata format (.DTA).

4. Extract and move the file ETKR71FL.DTA directly into your local data/raw/ directory.

### 3. Automated Monolithic Pipeline Execution
The complete lifecycle—spanning raw data parsing, structural anomaly filtering (DHS 9998 measurements), stratified training segmentation, interaction feature engineering, statistical evaluation, and model serialization—is managed by a single master orchestrator file. Run the script from the root directory:

```bash
python app.py
```

### 4. Output Data Mapping
Successful execution generates two primary deployment artifacts:

artifacts/random_forest_pipeline.pkl: The frozen Scikit-Learn Pipeline object containing fitted StandardScaler weights, OneHotEncoder maps, and tree parameters.

data/processed/tableau_malnutrition_reporting.csv: A fully string-decoded test matrix containing true clinical states, model prediction classifications, and calculated continuous probability vectors optimized for corporate Business Intelligence (BI) software tools like Tableau.

## 📈 Executive Tableau Dashboard System
The generated processed CSV is structured to seamlessly plug into Tableau Desktop or Public to drive strategic health resource allocation.

### Design and Formatting Requirements:
- UI/UX Aesthetic: Minimalist Corporate SaaS / Premium B2B Canvas.

- Palette Mapping: Base canvas utilizes a crisp off-white/light-gray background (#F9F9FB), framework grids are bounded in Deep Chocolate (#2C1A14), and severe patient alerts are isolated exclusively in Metallic Gold (#C5A059).
  
- Interactivity: Configure a Dashboard Filter Action on the Socio-Demographic Grid (Wealth_Quintile $\times$ Mother_Education) to dynamically update regional demographic bar charts on a simple mouse click.