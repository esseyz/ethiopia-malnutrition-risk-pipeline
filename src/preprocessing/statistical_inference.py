import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import statsmodels.api as sm
from src.preprocessing.build_pipeline import load_and_split_data

def run_statistical_inference():
    """
    Executes Phase 4: Conducts feature engineering and performs robust
    statistical inference tests to validate feature-to-target significance.
    """
    print("=== Phase 4: Statistical Inference & Feature Engineering ===")
    
    # Ingest our un-leaked training split from Phase 3
    raw_data_path = "data/raw/ETKR71FL.DTA"
    X_train, _, y_train, _ = load_and_split_data(raw_data_path)
    
    # Create an analysis dataframe by combining features and targets
    analysis_df = X_train.copy()
    analysis_df['stunting_class'] = y_train
    
    # 1. Feature Engineering: Risk Interaction Terms
    # Rural Poverty Index: Identifies households that are both rural (residence == 2) and in the lowest two wealth quintiles (1 or 2)
    analysis_df['rural_poverty_index'] = np.where(
        (analysis_df['residence_urban_rural'] == 2) & (analysis_df['wealth_index_quintile'] <= 2), 
        1, 0
    )
    
    print("[OK] Engineered Feature: 'rural_poverty_index' successfully generated.")
    
    # 2. Statistical Inference: Chi-Square Test of Independence
    # Testing relationship between Wealth Index and Stunting Severity
    contingency_table = pd.crosstab(analysis_df['wealth_index_quintile'], analysis_df['stunting_class'])
    
    chi2, p_val, dof, expected = chi2_contingency(contingency_table)
    
    print("\n--- Chi-Square Test: Wealth Index vs. Stunting Severity ---")
    print(f"Chi-Square Statistic : {chi2:.4f}")
    print(f"P-value              : {p_val:.4e}")
    print(f"Degrees of Freedom   : {dof}")
    
    if p_val < 0.05:
        print("[CONCLUSION] Reject H0: There is a statistically significant association between wealth and stunting.")
    else:
        print("[CONCLUSION] Fail to reject H0: No statistically significant association found.")
        
    # 3. Statistical Inference: Rural Poverty Index vs Stunting
    contingency_table_rural = pd.crosstab(analysis_df['rural_poverty_index'], analysis_df['stunting_class'])
    chi2_r, p_val_r, _, _ = chi2_contingency(contingency_table_rural)
    print(f"\n--- Chi-Square Test: Rural Poverty Index vs. Stunting Severity ---")
    print(f"P-value              : {p_val_r:.4e}")

if __name__ == "__main__":
    run_statistical_inference()