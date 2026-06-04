# Ethiopia Child Malnutrition & Stunting Risk Pipeline

> An end-to-end machine learning pipeline for predicting child stunting severity 
> using the 2016 Ethiopia Demographic and Health Survey (EDHS).

---

## Overview

Child stunting — defined by the WHO as a height-for-age Z-score (HAZ) below −2 
standard deviations — affects nearly **38% of Ethiopian children under five**, 
one of the highest prevalence rates in Sub-Saharan Africa.

This project builds a reproducible ML pipeline on real-world DHS microdata to:

- Characterise the sociodemographic determinants of stunting
- Validate feature-target associations through statistical inference (Chi-Square + Cramér's V)
- Train and evaluate models that stratify malnutrition risk into three clinical classes
- Export a production-ready pipeline and Tableau-ready reporting dataset

---

## Results

| Model | CV Macro F1 | Test Macro F1 | ROC-AUC (OvR) |
|-------|------------|---------------|---------------|
| Logistic Regression | 0.399 ± 0.013 | 0.388 | 0.597 |
| **Random Forest** | **0.412 ± 0.006** | **0.391** | **0.631** |

> **Note on performance:** Macro F1 of ~0.41 reflects the inherent difficulty 
> of three-class separation using 6 sociodemographic features with small 
> individual effect sizes (Cramér's V: 0.10–0.14). The model demonstrates 
> meaningful discriminative ability (ROC-AUC 0.63) and prioritises recall 
> on the Severe class — the clinically critical outcome.

---

## Key Findings

### 1. Child Age Dominates (Feature Importance: 46.9%)
Stunting is a cumulative deficit. Older children have had longer exposure 
to risk factors — the model captures this growth trajectory signal more 
strongly than any sociodemographic variable.

### 2. Maternal Age is Second (26.5%)
Younger mothers face compounded disadvantages — lower nutritional knowledge, 
reduced economic autonomy, and potential adolescent growth competition with 
the fetus.

### 3. Compounded Rural Poverty is the Strongest Policy Signal
Children who are simultaneously **rural and in the lowest two wealth quintiles** 
face a severe stunting rate of **21.1%** — nearly double the 11.4% rate 
in all other groups. This group represents **51.9% of the survey sample**.

### 4. Maternal Education Shows the Steepest Gradient
Severe stunting among children of uneducated mothers (19.5%) is **6× higher** 
than among children of highly educated mothers (3.1%).

---

## Visualisations

### Clinical Target Distribution
![Class Distribution](notebooks/figures/01_class_distribution.png)

### Severe Stunting Rate by Sociodemographic Category
![Severe Stunting Rates](notebooks/figures/03_severe_stunting_rates.png)

### Stunting Z-Score by Wealth & Education
![Z-Score Boxplots](notebooks/figures/05_zscore_boxplots.png)

### Rural Poverty Index — Interaction Feature
![Rural Poverty Index](notebooks/figures/06_rural_poverty_index.png)

### Statistical Inference — Cramér's V Effect Sizes
![Cramers V](notebooks/figures/07_cramers_v.png)

### Cross-Validation Stability
![Cross Validation](notebooks/figures/08_cross_validation.png)

### Confusion Matrices
![Confusion Matrices](notebooks/figures/09_confusion_matrices.png)

### Feature Importance
![Feature Importance](notebooks/figures/11_feature_importance.png)

---

## Project Structure
```
ethiopia-malnutrition-risk-pipeline/
├── data/
│   ├── raw/                  # ETKR71FL.DTA (not committed — DHS access required)
│   └── processed/            # tableau_malnutrition_reporting.csv
├── notebooks/
│   ├── analysis.ipynb        # Full narrative analysis notebook
│   └── figures/              # All output visualisations (11 charts)
├── src/
│   ├── preprocessing/
│   │   ├── inspect_dhs.py
│   │   ├── target_analysis.py
│   │   ├── statistical_inference.py
│   │   └── build_pipeline.py
│   └── models/
│       ├── train_evaluate.py
│       └── export_artifacts.py
├── artifacts/
│   └── random_forest_pipeline.pkl
├── app.py
├── requirements.txt
└── README.md
```
---

## Reproducing This Project

### 1. Data Access
The raw DHS data file (`ETKR71FL.DTA`) requires registration at 
[dhsprogram.com](https://dhsprogram.com). Request the **Ethiopia 2016 
Children's Recode (KR)** dataset. Place the `.DTA` file in `data/raw/`.

### 2. Install dependencies
```powershell
pip install -r requirements.txt
```

### 3. Run the notebook
Open `notebooks/analysis.ipynb` in VS Code or Jupyter and run all cells.

### 4. Run the full pipeline
```powershell
python app.py
```
This executes all preprocessing, trains both models, and exports the 
Tableau reporting CSV to `data/processed/`.

---

## Dataset

| Property | Detail |
|---|---|
| Source | DHS Program — Ethiopia 2016 |
| Recode | Children's Recode (KR) — ETKR71FL.DTA |
| Raw records | 10,641 children |
| After cleaning | 8,855 children |
| Features used | 7 (6 sociodemographic + 1 engineered) |
| Target | Height-for-Age Z-Score → 3-class stunting severity |

---

## Clinical Class Definitions (WHO)

| Class | HAZ Threshold | Prevalence in Sample |
|-------|--------------|----------------------|
| Normal | ≥ −2.0 SD | 63.6% |
| Moderate Stunting | −3.0 to −2.0 SD | 19.9% |
| Severe Stunting | < −3.0 SD | 16.5% |

---

## Limitations & Future Work

- **Feature scope:** Only 7 features from 1,251 available columns. 
  Adding birth weight, breastfeeding duration, dietary diversity, 
  and regional identifiers could push Macro F1 above 0.55.
- **Algorithm:** XGBoost or LightGBM with `GridSearchCV` tuning 
  would likely outperform the current Random Forest.
- **Geography:** Regional variation within Ethiopia is not captured. 
  A geographically-aware model would be more actionable for policy.
- **Temporality:** Cross-sectional data cannot capture the longitudinal 
  nature of stunting accumulation.

---

## Author
Essey Zebene Degefu