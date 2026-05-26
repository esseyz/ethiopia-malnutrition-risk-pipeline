import pandas as pd
import numpy as np
import sys
import os
import joblib

# Ensure the root directory is in the path to import our partition engine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.models.train_evaluate import train_and_evaluate_production

def export_production_assets():
    print("=== Phase 6: Exporting Production Artifacts & Tableau Assets ===")
    
    # 1. Train and extract our fitted pipeline objects
    log_reg_pipeline, rf_pipeline = train_and_evaluate_production()
    
    # Save the models to disk
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(rf_pipeline, "artifacts/random_forest_pipeline.pkl")
    print("[OK] Serialized Random Forest model saved to 'artifacts/random_forest_pipeline.pkl'")
    
    # 2. Extract fresh, raw test data partitions
    from src.preprocessing.build_pipeline import load_and_split_data
    raw_data_path = "data/raw/ETKR71FL.DTA"
    _, X_test, _, y_test = load_and_split_data(raw_data_path)
    
    # Create the identical enriched version for the final predictions
    X_test_enriched = X_test.copy()
    X_test_enriched['rural_poverty_index'] = np.where(
        (X_test_enriched['residence_urban_rural'] == 2) & (X_test_enriched['wealth_index_quintile'] <= 2), 
        1, 0
    )
    
    # 3. Generate predictions using the enriched dataframe structure
    print("\n[INFO] Generating model predictions and probability vectors...")
    probabilities = rf_pipeline.predict_proba(X_test_enriched)
    predictions = rf_pipeline.predict(X_test_enriched)
    
    # 4. Compile the executive reporting table with clean, human-readable labels
    tableau_df = pd.DataFrame({
        'Child_Age_Months': X_test_enriched['child_age_months'].astype(int),
        'Child_Sex': X_test_enriched['child_sex'].map({1: 'Male', 2: 'Female'}),
        'Mother_Age': X_test_enriched['mother_age'].astype(int),
        'Mother_Education': X_test_enriched['mother_education'].map({0: 'No Education', 1: 'Primary', 2: 'Secondary', 3: 'Higher'}),
        'Wealth_Quintile': X_test_enriched['wealth_index_quintile'].map({1: 'Poorest', 2: 'Poorer', 3: 'Middle', 4: 'Richer', 5: 'Richest'}),
        'Residence_Type': X_test_enriched['residence_urban_rural'].map({1: 'Urban', 2: 'Rural'}),
        'Rural_Poverty_Indicator': X_test_enriched['rural_poverty_index'].map({1: 'High Vulnerability', 0: 'Standard'}),
        'True_Clinical_Status': y_test.map({0: 'Normal', 1: 'Moderate', 2: 'Severe'}),
        'Predicted_Clinical_Status': pd.Series(predictions, index=y_test.index).map({0: 'Normal', 1: 'Moderate', 2: 'Severe'}),
        'Model_Confidence_Severe_Risk': probabilities[:, 2]  # Probability vector for Severe class
    })
    
    # Save file out to our processed data vault
    os.makedirs("data/processed", exist_ok=True)
    tableau_df.to_csv("data/processed/tableau_malnutrition_reporting.csv", index=False)
    print("\n[OK] Success! Tableau reporting source generated.")
    print(f"     File saved to: data/processed/tableau_malnutrition_reporting.csv ({tableau_df.shape[0]} rows)")

if __name__ == "__main__":
    export_production_assets()